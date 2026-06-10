"""记忆消融实验：固定题集 × 3 轮，对比「轮间清空」与「累积记忆」两组。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.postprocess_eval import build_score_report, postprocess
from scripts.run_eval_test import (
    OUTPUT_DIR,
    build_run_order,
    parse_questions,
    run_eval,
)
from src.config import get_settings
from src.pipeline import InsuranceAgentPipeline

SUMMARY_FILE = ROOT / "test" / "实验总结"


def _round_stats(score_report: dict) -> list[dict]:
    timing = score_report["summary"].get("timing", {})
    per_run = {r["run_index"]: r for r in timing.get("per_run", [])}
    rows = []
    for run in score_report["runs"]:
        t = per_run.get(run["run_index"], {})
        rows.append(
            {
                "run_index": run["run_index"],
                "total_score": run["total_score"],
                "max_score": run["max_score"],
                "accuracy_pct": run["accuracy_pct"],
                "avg_elapsed_seconds": t.get("avg_elapsed_seconds"),
                "avg_first_token_seconds": t.get("avg_first_token_seconds"),
                "full_correct": sum(1 for s in run["scores"] if s["score"] == 2),
                "partial": sum(1 for s in run["scores"] if s["score"] == 1),
                "wrong": sum(1 for s in run["scores"] if s["score"] == 0),
            }
        )
    return rows


def _score_by_round_question(score_report: dict) -> dict[tuple[int, str], int]:
    out: dict[tuple[int, str], int] = {}
    for run in score_report["runs"]:
        seen: dict[str, int] = {}
        for item in run["scores"]:
            qid = item["question_id"]
            seen[qid] = seen.get(qid, 0) + 1
            key = (run["run_index"], qid, seen[qid])
            out[key] = item["score"] if item["score"] is not None else -1
    return out


def write_comparison(
    *,
    isolated_json: Path,
    persistent_json: Path,
    isolated_score: dict,
    persistent_score: dict,
    fixed_order: tuple[list[str], list[str]],
    out_path: Path,
) -> None:
    order, repeat_ids = fixed_order
    iso_rows = _round_stats(isolated_score)
    per_rows = _round_stats(persistent_score)
    iso_s = isolated_score["summary"]
    per_s = persistent_score["summary"]

    lines = [
        "# 记忆消融实验对比",
        "",
        f"- 生成时间: {datetime.now(timezone.utc).astimezone().isoformat()}",
        f"- 固定题集: {len(order)} 题（40 题 + 重复 {len(repeat_ids)} 题）",
        f"- 重复题: {', '.join(repeat_ids)}",
        "",
        "## 实验设计",
        "",
        "| 组别 | 轮间 session | 轮间 LTM | 题集 |",
        "|------|-------------|----------|------|",
        "| **A 隔离组** | 每轮 `clear_session` | 每轮清空 | 固定相同 |",
        "| **B 累积组** | 不清空，复用同一 session | 不清空，持续累积 | 固定相同 |",
        "",
        "两组实验开始前均执行一次向量库重置 + PDF 入库；组间独立，各从干净基线开始。",
        "",
        "## 总览对比",
        "",
        "| 指标 | A 隔离组 | B 累积组 | 差值 (B−A) |",
        "|------|----------|----------|------------|",
        f"| 总得分 | {iso_s['total_score']}/{iso_s['max_score']} | "
        f"{per_s['total_score']}/{per_s['max_score']} | "
        f"{per_s['total_score'] - iso_s['total_score']:+d} |",
        f"| 正确率 | {iso_s['overall_pct']}% | {per_s['overall_pct']}% | "
        f"{per_s['overall_pct'] - iso_s['overall_pct']:+.1f}% |",
        f"| 完全正确 | {iso_s['full_correct']} | {per_s['full_correct']} | "
        f"{per_s['full_correct'] - iso_s['full_correct']:+d} |",
        f"| 部分正确 | {iso_s['partial']} | {per_s['partial']} | "
        f"{per_s['partial'] - iso_s['partial']:+d} |",
        f"| 错误 | {iso_s['wrong']} | {per_s['wrong']} | "
        f"{per_s['wrong'] - iso_s['wrong']:+d} |",
        f"| 平均耗时 | {iso_s['timing']['avg_elapsed_seconds']}s | "
        f"{per_s['timing']['avg_elapsed_seconds']}s | "
        f"{per_s['timing']['avg_elapsed_seconds'] - iso_s['timing']['avg_elapsed_seconds']:+.2f}s |",
        f"| 平均首 token | {iso_s['timing']['avg_first_token_seconds']}s | "
        f"{per_s['timing']['avg_first_token_seconds']}s | "
        f"{per_s['timing']['avg_first_token_seconds'] - iso_s['timing']['avg_first_token_seconds']:+.2f}s |",
        "",
        "## 分轮得分",
        "",
        "| 轮次 | A 隔离组 | B 累积组 |",
        "|------|----------|----------|",
    ]
    for iso, per in zip(iso_rows, per_rows):
        lines.append(
            f"| 第 {iso['run_index']} 轮 | {iso['total_score']}/{iso['max_score']} "
            f"({iso['accuracy_pct']}%) | {per['total_score']}/{per['max_score']} "
            f"({per['accuracy_pct']}%) |"
        )

    lines.extend(["", "## 分轮耗时", "", "| 轮次 | A 平均耗时 | A 首 token | B 平均耗时 | B 首 token |", "|------|------------|------------|------------|------------|"])
    for iso, per in zip(iso_rows, per_rows):
        lines.append(
            f"| 第 {iso['run_index']} 轮 | {iso['avg_elapsed_seconds']}s | "
            f"{iso['avg_first_token_seconds']}s | {per['avg_elapsed_seconds']}s | "
            f"{per['avg_first_token_seconds']}s |"
        )

    # 逐题三轮均分对比
    lines.extend(["", "## 逐题均分对比", "", "| 题号 | A 均分 | B 均分 | Δ | A 各轮 | B 各轮 |", "|------|--------|--------|---|--------|--------|"])
    iso_by_q = isolated_score["summary"]["by_question"]
    per_by_q = {q["question_id"]: q for q in persistent_score["summary"]["by_question"]}
    for q in iso_by_q:
        qid = q["question_id"]
        p = per_by_q.get(qid, {})
        a_avg = q.get("avg")
        b_avg = p.get("avg")
        delta = round(b_avg - a_avg, 2) if a_avg is not None and b_avg is not None else "N/A"
        lines.append(
            f"| {qid} | {a_avg} | {b_avg} | {delta} | {q.get('scores', [])} | {p.get('scores', [])} |"
        )

    # B 组逐轮提升（同题集重复跑）
    lines.extend(["", "## B 累积组：同题重复作答变化", ""])
    repeat_set = set(repeat_ids)
    for qid in repeat_set:
        per_scores = [s["score"] for s in persistent_score["_all_scores"] if s["question_id"] == qid]
        iso_scores = [s["score"] for s in isolated_score["_all_scores"] if s["question_id"] == qid]
        if len(per_scores) >= 2:
            trend = " → ".join(str(s) for s in per_scores)
            lines.append(f"- **{qid}** 累积组: {trend} | 隔离组: {' → '.join(str(s) for s in iso_scores)}")

    lines.extend(
        [
            "",
            "## 结论提示",
            "",
            "- 若 B 组第 2/3 轮明显高于 A 组同轮次，说明 session/LTM 累积对答题有正向帮助。",
            "- 若 B 组耗时逐轮上升而 A 组稳定，说明上下文变长带来推理开销。",
            "- 若两组差异不大，说明当前任务主要依赖 RAG 检索，记忆层影响有限。",
            "",
            "## 原始数据",
            "",
            f"- A 隔离组: `{isolated_json.name}`",
            f"- B 累积组: `{persistent_json.name}`",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


async def run_ablation(
    *,
    rounds: int = 3,
    repeat_count: int = 7,
    base_seed: int = 20260608,
) -> None:
    settings = get_settings()
    settings.validate()

    raw = (ROOT / "test" / "测试问题.txt").read_text(encoding="utf-8")
    questions = parse_questions(raw)
    all_ids = sorted(questions.keys())
    fixed_order = build_run_order(all_ids, repeat_count=repeat_count, seed=base_seed)

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("固定题集")
    print(f"  顺序: {', '.join(fixed_order[0])}")
    print(f"  重复题 ({repeat_count}): {', '.join(fixed_order[1])}")
    print("=" * 70)

    # --- A 隔离组 ---
    print("\n\n" + "#" * 70)
    print("# 实验 A：每轮 clear_session + 清空 LTM")
    print("#" * 70)
    pipeline_a = InsuranceAgentPipeline(settings)
    await pipeline_a.initialize()
    iso_json, _ = await run_eval(
        rounds=rounds,
        base_seed=base_seed,
        pipeline=pipeline_a,
        questions=questions,
        fixed_order=fixed_order,
        clear_between_rounds=True,
        reuse_session=True,
        skip_initial_reset=False,
        experiment_label="isolated",
        output_stem=f"ablation_{timestamp}_isolated",
    )
    postprocess(iso_json)
    iso_data = json.loads(iso_json.read_text(encoding="utf-8"))
    iso_score = build_score_report(iso_json, iso_data)

    # --- B 累积组 ---
    print("\n\n" + "#" * 70)
    print("# 实验 B：不清空 session 和 LTM，复用同一 session")
    print("#" * 70)
    pipeline_b = InsuranceAgentPipeline(settings)
    await pipeline_b.initialize()
    per_json, _ = await run_eval(
        rounds=rounds,
        base_seed=base_seed,
        pipeline=pipeline_b,
        questions=questions,
        fixed_order=fixed_order,
        clear_between_rounds=False,
        reuse_session=True,
        skip_initial_reset=False,
        experiment_label="persistent",
        output_stem=f"ablation_{timestamp}_persistent",
    )
    postprocess(per_json)
    per_data = json.loads(per_json.read_text(encoding="utf-8"))
    per_score = build_score_report(per_json, per_data)

    compare_path = OUTPUT_DIR / f"ablation_{timestamp}_compare.md"
    write_comparison(
        isolated_json=iso_json,
        persistent_json=per_json,
        isolated_score=iso_score,
        persistent_score=per_score,
        fixed_order=fixed_order,
        out_path=compare_path,
    )

    print("\n" + "=" * 70)
    print("消融实验完成")
    print(f"  对比报告: {compare_path}")
    print(f"  实验总结: {SUMMARY_FILE}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆消融：隔离 vs 累积")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat-count", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    try:
        asyncio.run(
            run_ablation(
                rounds=args.rounds,
                repeat_count=args.repeat_count,
                base_seed=args.seed,
            )
        )
    except Exception as exc:
        print(f"消融实验失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
