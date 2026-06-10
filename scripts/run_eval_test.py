"""从 test/ 题库运行 Agent 评测：重置记忆、入库 PDF、多轮随机顺序测试。"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.pipeline import InsuranceAgentPipeline

QUESTIONS_FILE = ROOT / "test" / "测试问题.txt"
TEST_PDF_DIR = ROOT / "test"
OUTPUT_DIR = ROOT / "test" / "results"


def parse_questions(text: str) -> dict[str, str]:
    """解析 Q01–Q40 题目文本。"""
    pattern = re.compile(
        r"【(Q\d{2})】[^\n]*\n(.+?)(?=\n\n【Q|\Z)",
        re.DOTALL,
    )
    questions: dict[str, str] = {}
    for qid, body in pattern.findall(text):
        lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
        divider = re.compile(r"^═+$")
        for line in lines:
            if divider.match(line) or re.match(r"^[一二三四五六]、", line):
                continue
            questions[qid] = line
            break
        if qid not in questions and lines:
            questions[qid] = lines[0]
    return questions


def build_run_order(
    all_ids: list[str],
    *,
    repeat_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """生成单次测试顺序：覆盖全部题目 + 随机 repeat_count 题重复一次。"""
    rng = random.Random(seed)
    repeat_ids = rng.sample(all_ids, repeat_count)
    order = list(all_ids) + list(repeat_ids)
    rng.shuffle(order)
    return order, repeat_ids


async def reset_memory_and_ingest(pipeline: InsuranceAgentPipeline) -> None:
    """清空长期记忆与向量库，重新入库 test 目录 PDF。"""
    await pipeline.clear_policy_library(clear_long_term=True)

    pdf_files = sorted(TEST_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"未找到 PDF：{TEST_PDF_DIR}")

    for pdf_path in pdf_files:
        content = pdf_path.read_bytes()
        result = await pipeline.upload_policy_pdf(content, pdf_path.name)
        print(f"  已入库: {result['filename']} ({result['chunks']} 片段)")


async def clear_session_and_ltm(
    pipeline: InsuranceAgentPipeline,
    session_id: str,
) -> int:
    """清空指定 session 短期记忆与长期记忆（LTM）。"""
    pipeline.clear_session(session_id)
    if not pipeline.memory_manager:
        return 0
    async with pipeline.memory_manager.long_term:
        return await pipeline.memory_manager.long_term.clear_all()


async def run_single_question(
    pipeline: InsuranceAgentPipeline,
    question_text: str,
    session_id: str,
) -> tuple[str, dict | None, float | None, float, str | None]:
    """通过 chat_stream 作答，返回 (回答, 合规, 首token秒, 总耗时秒, 错误)。"""
    t0 = time.perf_counter()
    first_token_seconds: float | None = None
    reply = ""
    compliance: dict | None = None
    error: str | None = None

    try:
        async for event in pipeline.chat_stream(question_text, session_id=session_id):
            if event.get("type") == "answer" and event.get("content") and first_token_seconds is None:
                first_token_seconds = round(time.perf_counter() - t0, 2)
            if event.get("type") == "done":
                reply = event.get("answer", "") or ""
                compliance = event.get("compliance")
    except Exception as exc:
        error = str(exc)

    elapsed = round(time.perf_counter() - t0, 2)
    return reply, compliance, first_token_seconds, elapsed, error


async def run_eval(
    rounds: int = 3,
    repeat_min: int = 5,
    repeat_max: int = 10,
    base_seed: int = 20260608,
    *,
    pipeline: InsuranceAgentPipeline | None = None,
    questions: dict[str, str] | None = None,
    fixed_order: tuple[list[str], list[str]] | None = None,
    clear_between_rounds: bool = False,
    reuse_session: bool = False,
    skip_initial_reset: bool = False,
    experiment_label: str = "",
    output_stem: str | None = None,
) -> tuple[Path, dict]:
    settings = get_settings()
    settings.validate()

    if questions is None:
        raw = QUESTIONS_FILE.read_text(encoding="utf-8")
        questions = parse_questions(raw)
    if len(questions) != 40:
        raise RuntimeError(f"期望 40 题，实际解析到 {len(questions)} 题")

    all_ids = sorted(questions.keys())
    owns_pipeline = pipeline is None
    if owns_pipeline:
        pipeline = InsuranceAgentPipeline(settings)
        print("=" * 60)
        print("1. 初始化 Pipeline …")
        await pipeline.initialize()

    if not skip_initial_reset:
        print("2. 重置记忆并入库 test PDF …")
        await reset_memory_and_ingest(pipeline)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    label_suffix = f"_{experiment_label}" if experiment_label else ""
    stem = output_stem or f"eval_{timestamp}{label_suffix}"
    out_json = OUTPUT_DIR / f"{stem}.json"
    out_md = OUTPUT_DIR / f"{stem}.md"

    if fixed_order is not None:
        shared_order, shared_repeat_ids = fixed_order
        shared_repeat_count = len(shared_repeat_ids)
    else:
        shared_order = shared_repeat_ids = shared_repeat_count = None

    report: dict = {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "rounds": rounds,
            "repeat_range": [repeat_min, repeat_max],
            "base_seed": base_seed,
            "question_count": len(all_ids),
            "memory_reset": not skip_initial_reset,
            "clear_between_rounds": clear_between_rounds,
            "reuse_session": reuse_session,
            "fixed_order": fixed_order is not None,
            "experiment_label": experiment_label,
            "pdfs_ingested": [p.name for p in sorted(TEST_PDF_DIR.glob("*.pdf"))],
        },
        "runs": [],
    }
    if fixed_order is not None:
        report["meta"]["shared_question_order"] = shared_order
        report["meta"]["shared_repeat_question_ids"] = shared_repeat_ids

    group_session_id = f"eval_{experiment_label or 'default'}_{timestamp}"

    for run_idx in range(1, rounds + 1):
        if fixed_order is not None:
            order, repeat_ids = shared_order, shared_repeat_ids
            repeat_count = shared_repeat_count
            seed = base_seed
        else:
            seed = base_seed + run_idx * 1000
            repeat_count = random.Random(seed).randint(repeat_min, repeat_max)
            order, repeat_ids = build_run_order(all_ids, repeat_count=repeat_count, seed=seed)

        if reuse_session:
            session_id = group_session_id
        else:
            session_id = f"eval_run_{run_idx}_{timestamp}"

        if clear_between_rounds:
            deleted = await clear_session_and_ltm(pipeline, session_id)
            print(f"\n  [轮前清空] session={session_id}, LTM 删除 {deleted} 条")

        print(f"\n{'=' * 60}")
        print(f"第 {run_idx}/{rounds} 轮 | session={session_id}")
        print(f"  题目总数={len(order)}（含重复 {repeat_count} 题: {', '.join(repeat_ids)}）")

        run_record: dict = {
            "run_index": run_idx,
            "session_id": session_id,
            "seed": seed,
            "repeat_count": repeat_count,
            "repeat_question_ids": repeat_ids,
            "question_order": order,
            "cleared_before_run": clear_between_rounds,
            "answers": [],
        }

        for seq, qid in enumerate(order, start=1):
            question_text = questions[qid]
            print(f"  [{run_idx}/{rounds}] {seq}/{len(order)} {qid} …", flush=True)
            reply, compliance, ttft, elapsed, error = await run_single_question(
                pipeline, question_text, session_id
            )

            entry = {
                "sequence": seq,
                "question_id": qid,
                "question": question_text,
                "answer": reply,
                "elapsed_seconds": elapsed,
                "first_token_seconds": ttft,
                "compliance": compliance,
                "error": error,
            }
            run_record["answers"].append(entry)

            # 增量写入，防止长跑中断丢数据
            report["runs"] = report["runs"][: run_idx - 1] + [run_record]
            out_json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(f"  第 {run_idx} 轮完成。")

    report["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, report, questions)
    print(f"\n结果已保存:\n  {out_json}\n  {out_md}")

    from scripts.postprocess_eval import postprocess

    post_paths = postprocess(out_json)
    print("\n后处理完成:")
    for name, path in post_paths.items():
        print(f"  {name}: {path}")
    return out_json, report


def _write_markdown(out_md: Path, report: dict, questions: dict[str, str]) -> None:
    lines: list[str] = [
        "# Insurance Agent 评测结果",
        "",
        f"- 开始时间: {report['meta'].get('started_at', '')}",
        f"- 结束时间: {report['meta'].get('finished_at', '')}",
        f"- 轮次: {report['meta']['rounds']}",
        f"- 每轮重复题数: {report['meta']['repeat_range'][0]}–{report['meta']['repeat_range'][1]}",
        f"- 记忆已重置: {report['meta']['memory_reset']}",
        f"- 轮间清空 session+LTM: {report['meta'].get('clear_between_rounds', False)}",
        f"- 复用同一 session: {report['meta'].get('reuse_session', False)}",
        f"- 固定题集: {report['meta'].get('fixed_order', False)}",
        f"- 入库 PDF: {', '.join(report['meta']['pdfs_ingested'])}",
        "",
    ]

    for run in report["runs"]:
        lines.extend(
            [
                f"## 第 {run['run_index']} 轮",
                "",
                f"- Session: `{run['session_id']}`",
                f"- 随机种子: {run['seed']}",
                f"- 重复题目 ({run['repeat_count']}): {', '.join(run['repeat_question_ids'])}",
                f"- 出题顺序: {', '.join(run['question_order'])}",
                "",
            ]
        )

        seen: dict[str, int] = {}
        for item in run["answers"]:
            qid = item["question_id"]
            seen[qid] = seen.get(qid, 0) + 1
            occurrence = seen[qid]
            occ_label = f"（第 {occurrence} 次出现）" if occurrence > 1 else ""

            lines.extend(
                [
                    f"### {item['sequence']}. 【{qid}】{occ_label}",
                    "",
                    "**问题：**",
                    "",
                    item["question"],
                    "",
                    "**Agent 完整回答：**",
                    "",
                    item["answer"] if item["answer"] else "（无回答）",
                    "",
                ]
            )
            if item.get("error"):
                lines.extend([f"**错误：** {item['error']}", ""])
            ttft = item.get("first_token_seconds")
            ttft_txt = f"{ttft}s" if ttft is not None else "N/A"
            lines.extend(
                [
                    f"*总耗时 {item['elapsed_seconds']}s · 首 token {ttft_txt}*",
                    "",
                    "---",
                    "",
                ]
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 test/ 题库评测")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat-min", type=int, default=5)
    parser.add_argument("--repeat-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    try:
        _out_json, _report = asyncio.run(
            run_eval(
                rounds=args.rounds,
                repeat_min=args.repeat_min,
                repeat_max=args.repeat_max,
                base_seed=args.seed,
            )
        )
        print(_out_json)
    except Exception as exc:
        print(f"评测失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
