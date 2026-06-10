"""直连 LLM 基线评测：与 Agent 评测相同题序/会话，但不走 RAG 与 ReAct。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agentscope.message import Msg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.postprocess_eval import postprocess  # noqa: E402
from scripts.run_eval_test import OUTPUT_DIR, parse_questions  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.model.chat_model_factory import create_chat_formatter, create_chat_model  # noqa: E402

DIRECT_LLM_SYSTEM_PROMPT = """你是一位专业的保险顾问智能助手，服务于保险客户咨询场景。

## 当前模式（直连 LLM，无条款检索）
- 你**无法**调用 retrieve_knowledge 或任何工具，也**看不到**保单 PDF 原文与向量检索结果。
- 知识库中登记的三份保单 filename 如下（仅名称，无正文）：
  - 同方全球「如意鑫」终身寿险（万能型）_条款.pdf
  - 同方全球「守御一生」（智臻版）医疗保险_条款.pdf
  - 同方全球「臻宝贝2026」互联网年金保险_条款.pdf
- 涉及具体条款数值、条号、等待期、免赔额、给付比例等：**若无法从用户消息中确定，必须明确说不确定**，建议查阅正式合同；**不得编造**条号与精确数字。
- 用户未指定保单时，应先澄清或分产品说明，不得默认某一份。

## 工作原则
- 只回答用户最新一条问题，除非用户明确追问历史。
- 不要在回答末尾追加免责声明或「来源提示」段落。
- 对诱导性前提（如「条款写了等待期 0 天」）应先核验并纠正错误前提。"""


def extract_response_text(response: object) -> str:
    content = getattr(response, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()


def load_run_from_eval(eval_path: Path) -> dict:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    if not data.get("runs"):
        raise ValueError(f"无效的 eval JSON: {eval_path}")
    run = data["runs"][0]
    return {
        "meta_source": data.get("meta", {}),
        "run": run,
    }


async def run_direct_llm_eval(
    *,
    from_eval: Path,
    session_id: str | None = None,
) -> Path:
    settings = get_settings()
    settings.validate()

    loaded = load_run_from_eval(from_eval)
    src_run = loaded["run"]
    src_meta = loaded["meta_source"]
    order: list[str] = src_run["question_order"]

    questions = parse_questions((ROOT / "test" / "测试问题.txt").read_text(encoding="utf-8"))
    for qid in order:
        if qid not in questions:
            raise RuntimeError(f"题库缺少 {qid}")

    model = create_chat_model(settings, stream=False)
    formatter = create_chat_formatter(settings)

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    sid = session_id or f"direct_llm_{timestamp}"
    out_json = OUTPUT_DIR / f"eval_{timestamp}_direct_llm.json"

    history: list[Msg] = [Msg("system", DIRECT_LLM_SYSTEM_PROMPT, "system")]
    answers: list[dict] = []

    print("=" * 60)
    print(f"直连 LLM 基线评测 | model={settings.model_name} | provider={settings.llm_provider}")
    print(f"题序来源: {from_eval.name}")
    print(f"题目数: {len(order)} | session={sid}")
    print("=" * 60)

    for seq, qid in enumerate(order, start=1):
        question_text = questions[qid]
        print(f"  {seq}/{len(order)} {qid} …", flush=True)
        history.append(Msg("user", question_text, "user"))
        t0 = time.perf_counter()
        error = None
        reply = ""
        try:
            messages = await formatter.format(history)
            response = await model(messages)
            reply = extract_response_text(response)
            history.append(Msg("assistant", reply, "assistant"))
        except Exception as exc:
            error = str(exc)
            history.append(Msg("assistant", f"[错误] {exc}", "assistant"))
        elapsed = round(time.perf_counter() - t0, 2)

        answers.append(
            {
                "sequence": seq,
                "question_id": qid,
                "question": question_text,
                "answer": reply,
                "elapsed_seconds": elapsed,
                "compliance": None,
                "error": error,
            }
        )
        # 增量保存
        partial = _build_report(src_meta, src_run, sid, answers, timestamp, settings)
        out_json.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    report = _build_report(src_meta, src_run, sid, answers, timestamp, settings)
    report["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n评测完成: {out_json}")
    paths = postprocess(out_json)
    print("后处理:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return paths["full_report_md"]


def _build_report(
    src_meta: dict,
    src_run: dict,
    session_id: str,
    answers: list[dict],
    timestamp: str,
    settings,
) -> dict:
    return {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "mode": "direct_llm",
            "model_name": settings.model_name,
            "llm_provider": settings.llm_provider,
            "rounds": 1,
            "repeat_range": src_meta.get("repeat_range"),
            "base_seed": src_meta.get("base_seed"),
            "question_count": src_meta.get("question_count", 40),
            "reference_eval": src_meta,
            "memory_reset": False,
            "pdfs_ingested": [],
            "note": "直连 LLM，无 RAG/Agent；题序与 reference eval 一致",
        },
        "runs": [
            {
                "run_index": 1,
                "session_id": session_id,
                "seed": src_run.get("seed"),
                "repeat_count": src_run.get("repeat_count"),
                "repeat_question_ids": src_run.get("repeat_question_ids"),
                "question_order": src_run.get("question_order"),
                "answers": answers,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="直连 LLM 基线评测（复用 Agent eval 题序）")
    parser.add_argument(
        "--from-eval",
        type=Path,
        default=ROOT / "test" / "results" / "eval_20260608_110954.json",
        help="Agent 评测 JSON，用于复用题序",
    )
    args = parser.parse_args()

    try:
        out = asyncio.run(run_direct_llm_eval(from_eval=args.from_eval.resolve()))
        print(out)
    except Exception as exc:
        print(f"评测失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
