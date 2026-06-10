"""评测后处理：评分 + 与参考答案逐题 diff + 综合报告。"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_eval import (  # noqa: E402
    CATEGORIES,
    parse_questions,
    score_answer,
)

ANSWERS_FILE = ROOT / "test" / "测试答案.txt"
QUESTIONS_FILE = ROOT / "test" / "测试问题.txt"


def parse_reference_answers(text: str) -> dict[str, str]:
    """解析 A01–A40，映射到 Q01–Q40。"""
    pattern = re.compile(
        r"【A(\d{2})】对应 (Q\d{2})\n(.+?)(?=\n\n【A|\Z)",
        re.DOTALL,
    )
    refs: dict[str, str] = {}
    for _aid, qid, body in pattern.findall(text):
        lines = []
        for line in body.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("引用："):
                lines.append(line)
                continue
            if re.match(r"^═+", line) or re.match(r"^[一二三四五六]、", line):
                continue
            lines.append(line)
        refs[qid] = "\n".join(lines).strip()
    return refs


def make_unified_diff(reference: str, agent: str, *, qid: str) -> str:
    ref_lines = (reference or "（无参考答案）").splitlines()
    agent_lines = (agent or "（无回答）").splitlines()
    diff = difflib.unified_diff(
        ref_lines,
        agent_lines,
        fromfile=f"{qid} 参考答案",
        tofile=f"{qid} Agent回答",
        lineterm="",
    )
    body = "\n".join(diff)
    return body if body.strip() else "（逐行 diff 无差异：Agent 回答与参考答案在文本层面完全一致或仅格式不同）"


def _timing_summary(data: dict) -> dict:
    """从 eval JSON 汇总耗时指标。"""
    all_elapsed: list[float] = []
    all_ttft: list[float] = []
    per_run: list[dict] = []

    for run in data["runs"]:
        run_elapsed = [a["elapsed_seconds"] for a in run["answers"] if a.get("elapsed_seconds") is not None]
        run_ttft = [
            a["first_token_seconds"]
            for a in run["answers"]
            if a.get("first_token_seconds") is not None
        ]
        all_elapsed.extend(run_elapsed)
        all_ttft.extend(run_ttft)
        per_run.append(
            {
                "run_index": run["run_index"],
                "count": len(run_elapsed),
                "avg_elapsed_seconds": round(sum(run_elapsed) / len(run_elapsed), 2) if run_elapsed else None,
                "avg_first_token_seconds": round(sum(run_ttft) / len(run_ttft), 2) if run_ttft else None,
                "total_elapsed_seconds": round(sum(run_elapsed), 2) if run_elapsed else None,
            }
        )

    return {
        "total_answers": len(all_elapsed),
        "avg_elapsed_seconds": round(sum(all_elapsed) / len(all_elapsed), 2) if all_elapsed else None,
        "avg_first_token_seconds": round(sum(all_ttft) / len(all_ttft), 2) if all_ttft else None,
        "total_elapsed_seconds": round(sum(all_elapsed), 2) if all_elapsed else None,
        "per_run": per_run,
    }


def build_score_report(eval_path: Path, data: dict) -> dict:
    questions = parse_questions(QUESTIONS_FILE.read_text(encoding="utf-8"))
    all_scores: list[dict] = []
    cat_scores: dict[str, list[int]] = {}

    report = {
        "source_eval": eval_path.name,
        "scoring_method": "规则匹配（关键事实 + 行为合规 + 引用）",
        "score_scale": "0=错误/不合格, 1=部分正确, 2=正确",
        "runs": [],
        "summary": {},
    }

    for run in data["runs"]:
        run_scores = []
        for item in run["answers"]:
            qid = item["question_id"]
            s = score_answer(qid, item["answer"], question_sent=item.get("question", ""))
            s["expected_question"] = questions.get(qid, "")
            s.update(
                {
                    "run_index": run["run_index"],
                    "sequence": item["sequence"],
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "first_token_seconds": item.get("first_token_seconds"),
                }
            )
            run_scores.append(s)
            all_scores.append(s)
            if s["score"] is not None:
                cat_scores.setdefault(s["category"], []).append(s["score"])

        valid = [x for x in run_scores if x["score"] is not None]
        total = sum(x["score"] for x in valid)
        max_total = sum(x["max_score"] for x in valid)
        report["runs"].append(
            {
                "run_index": run["run_index"],
                "items": len(run_scores),
                "valid_items": len(valid),
                "invalid_items": len(run_scores) - len(valid),
                "total_score": total,
                "max_score": max_total,
                "accuracy_pct": round(100 * total / max_total, 1) if max_total else 0,
                "scores": run_scores,
            }
        )

    q_summary = []
    for qid in sorted(CATEGORIES.keys()):
        scores = [s for s in all_scores if s["question_id"] == qid and s["score"] is not None]
        invalid_n = sum(1 for s in all_scores if s["question_id"] == qid and s["score"] is None)
        q_summary.append(
            {
                "question_id": qid,
                "category": CATEGORIES[qid],
                "attempts": len(scores) + invalid_n,
                "scores": [s["score"] for s in scores],
                "avg": round(sum(s["score"] for s in scores) / len(scores), 2) if scores else None,
            }
        )

    valid_scores = [s for s in all_scores if s["score"] is not None]
    total_score = sum(s["score"] for s in valid_scores)
    max_score = sum(s["max_score"] for s in valid_scores)
    invalid_count = sum(1 for s in all_scores if s["score"] is None)

    report["summary"] = {
        "total_attempts": len(all_scores),
        "valid_attempts": len(valid_scores),
        "invalid_attempts": invalid_count,
        "total_score": total_score,
        "max_score": max_score,
        "overall_pct": round(100 * total_score / max_score, 1) if max_score else 0,
        "full_correct": sum(1 for s in valid_scores if s["score"] == 2),
        "partial": sum(1 for s in valid_scores if s["score"] == 1),
        "wrong": sum(1 for s in valid_scores if s["score"] == 0),
        "timing": _timing_summary(data),
        "by_category": {
            cat: {
                "count": len(scores),
                "avg": round(sum(scores) / len(scores), 2),
                "pct": round(100 * sum(scores) / (2 * len(scores)), 1),
            }
            for cat, scores in sorted(cat_scores.items())
        },
        "by_question": q_summary,
    }
    report["_all_scores"] = all_scores
    return report


def write_score_md(report: dict, out_md: Path) -> None:
    s = report["summary"]
    timing = s.get("timing", {})
    lines = [
        "# Agent 评测评分报告",
        "",
        f"- 数据来源: `{report['source_eval']}`",
        f"- 评分方法: {report['scoring_method']}",
        f"- 分值: {report['score_scale']}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 答题次数 | {s['valid_attempts']} |",
        f"| 总得分 | **{s['total_score']} / {s['max_score']}** |",
        f"| 正确率 | **{s['overall_pct']}%** |",
        f"| 完全正确 (2分) | {s['full_correct']} |",
        f"| 部分正确 (1分) | {s['partial']} |",
        f"| 错误/不合格 (0分) | {s['wrong']} |",
        "",
        "## 耗时汇总",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 平均回答耗时 | **{timing.get('avg_elapsed_seconds', 'N/A')}s** |",
        f"| 平均首 token 时间 | **{timing.get('avg_first_token_seconds', 'N/A')}s** |",
        f"| 总耗时 | {timing.get('total_elapsed_seconds', 'N/A')}s |",
        "",
    ]
    if s["invalid_attempts"]:
        lines.append(f"| 无效答题 | {s['invalid_attempts']} |")
        lines.append("")

    for run in report["runs"]:
        run_timing = next(
            (r for r in timing.get("per_run", []) if r["run_index"] == run["run_index"]),
            {},
        )
        lines.extend(
            [
                f"## 第 {run['run_index']} 轮",
                "",
                f"- 得分: **{run['total_score']}/{run['max_score']}** ({run['accuracy_pct']}%)",
                f"- 平均耗时: {run_timing.get('avg_elapsed_seconds', 'N/A')}s",
                f"- 平均首 token: {run_timing.get('avg_first_token_seconds', 'N/A')}s",
                "",
            ]
        )

    lines.extend(["", "## 分题型", "", "| 题型 | 答题数 | 均分/2 | 正确率 |", "|------|--------|--------|--------|"])
    for cat, info in s["by_category"].items():
        lines.append(f"| {cat} | {info['count']} | {info['avg']} | {info['pct']}% |")

    lines.extend(["", "## 逐题均分", "", "| 题号 | 题型 | 得分 | 均分 |", "|------|------|------|------|"])
    for q in s["by_question"]:
        lines.append(
            f"| {q['question_id']} | {q['category']} | {q['scores']} | {q['avg']} |"
        )

    lines.extend(["", "## 得分点与扣分点明细", ""])
    for run in report["runs"]:
        lines.extend([f"### 第 {run['run_index']} 轮", ""])
        for sc in run["scores"]:
            if sc.get("score") is None:
                continue
            passed = sc.get("passed") or []
            issues = sc.get("issues") or []
            lines.append(
                f"- **{sc['question_id']}** (seq={sc['sequence']}) "
                f"**{sc['score']}/2 · {sc['label']}**"
            )
            if passed:
                lines.append(f"  - 得分点: {'; '.join(passed)}")
            if issues:
                lines.append(f"  - 扣分点: {'; '.join(issues)}")
            elif sc["score"] == 2:
                lines.append("  - 扣分点: 无")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_full_report(
    eval_path: Path,
    data: dict,
    score_report: dict,
    references: dict[str, str],
    out_md: Path,
) -> None:
    s = score_report["summary"]
    timing = s.get("timing", {})
    score_by_seq: dict[tuple[int, int], dict] = {}
    for run in score_report["runs"]:
        for sc in run["scores"]:
            score_by_seq[(run["run_index"], sc["sequence"])] = sc

    lines = [
        "# Insurance Agent 完整评测报告",
        "",
        f"- 评测数据: `{eval_path.name}`",
        f"- 总得分: **{s['total_score']} / {s['max_score']}（{s['overall_pct']}%）**",
        f"- 完全正确/部分/错误: {s['full_correct']} / {s['partial']} / {s['wrong']}",
        f"- 平均回答耗时: **{timing.get('avg_elapsed_seconds', 'N/A')}s**",
        f"- 平均首 token 时间: **{timing.get('avg_first_token_seconds', 'N/A')}s**",
        "",
        "本报告包含：**评分**、**得分/扣分明细**、**Agent 完整回答**、**与参考答案的逐题 unified diff**、**耗时与首 token 时间**。",
        "",
        "---",
        "",
    ]

    for run in data["runs"]:
        run_timing = next(
            (r for r in timing.get("per_run", []) if r["run_index"] == run["run_index"]),
            {},
        )
        lines.extend(
            [
                f"## 第 {run['run_index']} 轮",
                "",
                f"- 平均耗时: {run_timing.get('avg_elapsed_seconds', 'N/A')}s · "
                f"平均首 token: {run_timing.get('avg_first_token_seconds', 'N/A')}s",
                "",
            ]
        )
        seen: dict[str, int] = {}
        for item in run["answers"]:
            qid = item["question_id"]
            seen[qid] = seen.get(qid, 0) + 1
            occ = f"（第 {seen[qid]} 次）" if seen[qid] > 1 else ""
            sc = score_by_seq.get((run["run_index"], item["sequence"]), {})
            score_txt = (
                f"{sc.get('score', '?')}/2 · {sc.get('label', '?')}"
                if sc.get("score") is not None
                else "无效"
            )
            ref = references.get(qid, "（未找到参考答案）")
            diff = make_unified_diff(ref, item.get("answer", ""), qid=qid)
            passed = sc.get("passed") or []
            issues = sc.get("issues") or []
            ttft = item.get("first_token_seconds")
            ttft_txt = f"{ttft}s" if ttft is not None else "N/A"

            lines.extend(
                [
                    f"### {item['sequence']}. 【{qid}】{occ} · {score_txt}",
                    "",
                    f"**题型：** {CATEGORIES.get(qid, '')}",
                    "",
                    "**问题：**",
                    "",
                    item.get("question", ""),
                    "",
                    "**评分明细：**",
                    "",
                ]
            )
            if passed:
                lines.extend(["**得分点：**", ""] + [f"- {p}" for p in passed] + [""])
            if issues:
                lines.extend(["**扣分点：**", ""] + [f"- {i}" for i in issues] + [""])
            elif sc.get("score") == 2:
                lines.extend(["**扣分点：** 无", ""])

            lines.extend(
                [
                    "**参考答案：**",
                    "",
                    ref,
                    "",
                    "**Agent 完整回答：**",
                    "",
                    item.get("answer") or "（无回答）",
                    "",
                    "**与参考答案 Diff（`-` 参考 / `+` Agent）：**",
                    "",
                    "```diff",
                    diff,
                    "```",
                    "",
                    f"*总耗时 {item.get('elapsed_seconds', '?')}s · 首 token {ttft_txt}*",
                    "",
                    "---",
                    "",
                ]
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def postprocess(eval_path: Path) -> dict[str, Path]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    references = parse_reference_answers(ANSWERS_FILE.read_text(encoding="utf-8"))
    stem = eval_path.stem

    score_report = build_score_report(eval_path, data)
    all_scores = score_report.pop("_all_scores", [])
    score_report["items"] = all_scores

    score_json = eval_path.parent / f"{stem}_score.json"
    score_md = eval_path.parent / f"{stem}_score.md"
    full_md = eval_path.parent / f"{stem}_full_report.md"

    score_json.write_text(
        json.dumps({k: v for k, v in score_report.items() if k != "items"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_score_md(score_report, score_md)
    write_full_report(eval_path, data, score_report, references, full_md)

    return {
        "score_json": score_json,
        "score_md": score_md,
        "full_report_md": full_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评测后处理：评分 + diff 报告")
    parser.add_argument("eval_json", type=Path, nargs="?", help="eval_*.json 路径")
    args = parser.parse_args()

    if args.eval_json:
        eval_path = args.eval_json.resolve()
    else:
        candidates = sorted((ROOT / "test" / "results").glob("eval_*.json"))
        candidates = [p for p in candidates if "_score" not in p.name]
        if not candidates:
            print("未找到 eval JSON", file=sys.stderr)
            sys.exit(1)
        eval_path = candidates[-1]

    if not eval_path.is_file():
        print(f"文件不存在: {eval_path}", file=sys.stderr)
        sys.exit(1)

    paths = postprocess(eval_path)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
