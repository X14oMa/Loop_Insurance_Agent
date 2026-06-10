"""对照 test/测试答案.txt 对 eval JSON 自动评分。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_JSON = ROOT / "test" / "results" / "eval_20260608_102118.json"
OUT_MD = ROOT / "test" / "results" / "eval_20260608_102118_score.md"
OUT_JSON = ROOT / "test" / "results" / "eval_20260608_102118_score.json"

CATEGORIES = {
    **{f"Q{i:02d}": "基础事实" for i in range(1, 9)},
    **{f"Q{i:02d}": "条款解释" for i in range(9, 17)},
    **{f"Q{i:02d}": "多条件判断" for i in range(17, 25)},
    **{f"Q{i:02d}": "风险提示" for i in range(25, 31)},
    **{f"Q{i:02d}": "模糊问题" for i in range(31, 36)},
    **{f"Q{i:02d}": "诱导性问题" for i in range(36, 41)},
}


def parse_questions(text: str) -> dict[str, str]:
    """解析 Q01–Q40 题目文本（取【Qx】后第一个非空、非分隔线行）。"""
    pattern = re.compile(
        r"【(Q\d{2})】[^\n]*\n(.+?)(?=\n\n【Q|\Z)",
        re.DOTALL,
    )
    divider = re.compile(r"^═+$")
    questions: dict[str, str] = {}
    for qid, body in pattern.findall(text):
        for line in body.strip().splitlines():
            line = line.strip()
            if not line or divider.match(line):
                continue
            if re.match(r"^[一二三四五六]、", line):
                continue
            questions[qid] = line
            break
    return questions


def is_invalid_question_sent(question: str) -> bool:
    return bool(re.match(r"^═+$", (question or "").strip()))


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def has_any(text: str, patterns: list[str]) -> bool:
    t = norm(text)
    return any(norm(p) in t for p in patterns)


def has_all(text: str, patterns: list[str]) -> bool:
    t = norm(text)
    return all(norm(p) in t for p in patterns)


def has_clause_ref(text: str) -> bool:
    return bool(
        re.search(r"第[\d\.]+条", text)
        or re.search(r"[\u4e00-\u9fff]+_条款\.pdf", text)
        or "附表" in text
    )


def is_negated_context(text: str, pattern: str) -> bool:
    """判断关键词是否出现在否定/纠错语境中。"""
    t = text or ""
    for m in re.finditer(re.escape(pattern), t, flags=re.IGNORECASE):
        start = max(0, m.start() - 40)
        window = t[start : m.start()]
        if re.search(r"不对|不是|并非|错误|不正确|并未|没有写|并未写|不能|无法|勿|切勿|混淆", window):
            return True
    return False


def has_wrong(text: str, patterns: list[str], *, need_reject: bool = False) -> bool:
    for p in patterns:
        if not has_any(text, [p]):
            continue
        if need_reject and is_negated_context(text, p):
            continue
        return True
    return False


@dataclass
class Rubric:
    qid: str
    max_score: int = 2
    must: list[str] = field(default_factory=list)
    must_any: list[list[str]] = field(default_factory=list)
    must_any_extra: list[str] = field(default_factory=list)
    should: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    need_clarify: bool = False
    need_reject: bool = False
    forbid_default_single: bool = False
    note: str = ""


RUBRICS: dict[str, Rubric] = {
    "Q01": Rubric("Q01", must=["15"], should=["犹豫"], forbid=["30", "10"]),
    "Q02": Rubric("Q02", must_any=[["75", "七十五"], ["出生满七日"]]),
    "Q03": Rubric("Q03", must=["1%"], forbid=["5%"]),
    "Q04": Rubric("Q04", must=["2%"], forbid=["3%", "5%"]),
    "Q05": Rubric("Q05", must_any=[["30", "三十"]], should=["等待"]),
    "Q06": Rubric("Q06", must_any=[["一年", "1年"]], must=["不保证续保"]),
    "Q07": Rubric(
        "Q07",
        must_any=[["600", "600万"], ["600万元"]],
        should=["计划"],
        note="应给出计划一600万或说明需确认计划",
    ),
    "Q08": Rubric("Q08", must_any=[["17", "十七"]], should=["出生满七日"]),
    "Q09": Rubric(
        "Q09",
        must_any=[["保单年度", "投保年龄"]],
        should=["12.7", "到达年龄"],
    ),
    "Q10": Rubric(
        "Q10",
        must=["1%"],
        must_any=[["转入", "第五个"]],
        should=["趸交", "追加"],
    ),
    "Q11": Rubric(
        "Q11",
        must_any=[["个人账户", "个人支付"]],
        should=["补偿"],
    ),
    "Q12": Rubric(
        "Q12",
        must_any=[["50%", "50％", "百分之五十"]],
        must=["18"],
        should=["第六", "较迟"],
    ),
    "Q13": Rubric("Q13", must_any=[["月", "每月"]], should=["结算", "宣告"]),
    "Q14": Rubric(
        "Q14",
        must_any=[["观察", "门急诊观察"]],
        should=["24", "住院"],
    ),
    "Q15": Rubric("Q15", must=["60%"], forbid=[]),
    "Q16": Rubric(
        "Q16",
        must_any=[["100%", "100％"]],
        must=["25"],
    ),
    "Q17": Rubric(
        "Q17",
        must=["140%"],
        must_any=[["较高", "取高"], ["个人账户"]],
    ),
    "Q18": Rubric(
        "Q18",
        must_any=[["不承担", "不赔", "不给付", "不支付"]],
        must=["等待"],
    ),
    "Q19": Rubric("Q19", must=["60%"]),
    "Q20": Rubric(
        "Q20",
        must=["18"],
        should=["第六", "较迟", "8"],
    ),
    "Q21": Rubric("Q21", must=["3%"], forbid=["5%"]),
    "Q22": Rubric(
        "Q22",
        must_any=[["不承担", "不赔", "不给付", "免除"]],
        should=["既往"],
    ),
    "Q23": Rubric(
        "Q23",
        must_any=[["满足", "可以", "能够", "属于", "豁免"]],
        should=["意外"],
    ),
    "Q24": Rubric(
        "Q24",
        must_any=[["共用", "共享"]],
        should=["免赔"],
    ),
    "Q25": Rubric(
        "Q25",
        must_any=[["现金价值", "个人账户价值"]],
        should=["解除", "损失"],
    ),
    "Q26": Rubric(
        "Q26",
        must=["不保证续保"],
        should=["停售", "重新"],
    ),
    "Q27": Rubric(
        "Q27",
        must_any=[["不承担", "不承担保险责任"]],
        must_any_extra=["不结算", "不计息", "不结算利息"],
        note="需同时提到不承担责任与不结算利息",
    ),
    "Q28": Rubric(
        "Q28",
        must_any=[["不承担", "不赔"]],
        must_any_extra=["不退", "不返还", "不退还"],
    ),
    "Q29": Rubric(
        "Q29",
        must_any=[["现金价值"]],
        should=["损失", "犹豫期后"],
    ),
    "Q30": Rubric("Q30", must=["既往"], should=["如实告知", "除外"]),
    "Q31": Rubric(
        "Q31",
        need_clarify=True,
        must_any=[
            ["如意鑫", "守御", "臻宝贝"],
            ["哪份", "哪一款", "具体", "分别", "不同"],
        ],
        forbid_default_single=True,
        note="须区分三份保单或追问",
    ),
    "Q32": Rubric(
        "Q32",
        need_clarify=True,
        must_any=[["守御", "医疗"]],
        should=["等待", "既往", "如意鑫", "臻宝贝"],
        note="须分产品说明，避免简单能/不能",
    ),
    "Q33": Rubric(
        "Q33",
        need_clarify=True,
        must_any=[
            ["计划", "选项", "约定", "载明", "投保"],
            ["0", "1万", "100"],
        ],
        forbid=["免赔额是", "免赔额为10000"],  # 单一固定值且未说明计划
        note="须说明取决于计划/选项",
    ),
    "Q34": Rubric(
        "Q34",
        must=["不保证续保"],
        must_any=[["不能", "无法", "不能保证", "不可以保证"]],
    ),
    "Q35": Rubric(
        "Q35",
        need_clarify=True,
        must_any=[["80%", "80％"]],
        should=["如意鑫", "臻宝贝", "守御"],
        note="须区分产品；守御无贷款",
    ),
    "Q36": Rubric(
        "Q36",
        need_reject=True,
        must=["30"],
        forbid=["等待期是0", "等待期为0", "等待期0天", "0天等待"],
    ),
    "Q37": Rubric(
        "Q37",
        need_reject=True,
        must_any=[["不对", "不是", "错误", "并非"]],
        must=["18"],
        forbid=["一出生", "出生就可以", "出生起"],
    ),
    "Q38": Rubric(
        "Q38",
        need_reject=True,
        must=["1%"],
        forbid=["保证5%", "至少5%", "不低于5%"],
    ),
    "Q39": Rubric(
        "Q39",
        need_reject=True,
        must_any=[["不对", "不是", "错误", "并不能"]],
        should=["二级", "指定", "认可"],
    ),
    "Q40": Rubric(
        "Q40",
        need_reject=True,
        must_any=[["不对", "不是", "不承担", "不豁免", "免除"]],
        must=["毒品", "吸毒"],
    ),
}


def _collect_passed_checks(qid: str, answer: str, r: Rubric) -> list[str]:
    """汇总已满足的得分点。"""
    passed: list[str] = []

    for p in r.must:
        if has_any(answer, [p]):
            passed.append(f"关键事实: {p}")

    for group in r.must_any:
        if has_any(answer, group):
            passed.append(f"要点: {' / '.join(group)}")

    if r.must_any_extra and has_any(answer, r.must_any_extra):
        passed.append(f"附加要点: {' / '.join(r.must_any_extra)}")

    for p in r.should:
        if has_any(answer, [p]):
            passed.append(f"补充说明: {p}")

    if r.need_reject and has_any(
        answer, ["不对", "不是", "错误", "并非", "不正确", "并不能", "否"]
    ):
        passed.append("明确纠正错误前提")

    if r.need_clarify and has_any(
        answer,
        [
            "哪份",
            "哪一款",
            "具体",
            "分别",
            "不同",
            "需确认",
            "请确认",
            "取决于",
            "计划",
            "选项",
            "保单载明",
        ],
    ):
        passed.append("澄清/区分产品或条件")

    if has_clause_ref(answer):
        passed.append("包含条款引用")

    if qid == "Q27" and has_any(answer, ["不承担", "不承担保险责任"]) and has_any(
        answer, ["不结算", "不计息", "不结算利息"]
    ):
        passed.append("同时说明不承担责任与不结算利息")

    if qid == "Q28" and has_any(answer, ["不承担", "不赔"]) and has_any(
        answer, ["不退", "不返还", "不退还"]
    ):
        passed.append("同时说明不赔且不退费")

    if qid == "Q35" and has_any(answer, ["守御", "没有", "未约定", "不提供", "无保单贷款"]):
        passed.append("说明守御一生无保单贷款")

    return passed


def score_answer(qid: str, answer: str, *, question_sent: str = "") -> dict:
    if is_invalid_question_sent(question_sent):
        return {
            "question_id": qid,
            "score": None,
            "max_score": 2,
            "label": "无效（题干解析错误）",
            "category": CATEGORIES[qid],
            "has_clause_ref": False,
            "passed": [],
            "issues": ["评测脚本未发送真实题干，Agent 回答不可计入能力分"],
            "note": "需修复 run_eval_test.py 后重测",
        }

    r = RUBRICS[qid]
    issues: list[str] = []
    score = r.max_score

    if qid == "Q15" and has_any(answer, ["60%"]):
        pass
    elif r.forbid and has_wrong(answer, r.forbid, need_reject=r.need_reject):
        score = 0
        issues.append(f"含错误信息: {r.forbid}")

    if getattr(r, "forbid_default_single", False):
        if has_any(answer, ["等待期是30", "等待期为30"]) and not has_any(
            answer, ["守御", "分别", "如意鑫", "臻宝贝", "哪份"]
        ):
            score = min(score, 1)
            issues.append("未澄清保单即给出单一等待期")

    for p in r.must:
        if not has_any(answer, [p]):
            score -= 1
            issues.append(f"缺少关键事实: {p}")

    for group in r.must_any:
        if not has_any(answer, group):
            score -= 1
            issues.append(f"缺少要点组: {' / '.join(group)}")

    if r.must_any_extra and not has_any(answer, r.must_any_extra):
        score -= 1
        issues.append(f"缺少附加要点: {' / '.join(r.must_any_extra)}")

    if r.need_reject:
        if not has_any(answer, ["不对", "不是", "错误", "并非", "不正确", "并不能", "否"]):
            score = min(score, 1)
            issues.append("未明确纠正错误前提")

    if r.need_clarify:
        clarify_ok = has_any(
            answer,
            [
                "哪份",
                "哪一款",
                "具体",
                "分别",
                "不同",
                "需确认",
                "请确认",
                "取决于",
                "计划",
                "选项",
                "保单载明",
            ],
        )
        if not clarify_ok and qid in {"Q31", "Q35"}:
            score -= 1
            issues.append("未充分澄清/区分产品")

    if qid == "Q27":
        if not (has_any(answer, ["不承担", "不承担保险责任"]) and has_any(
            answer, ["不结算", "不计息", "不结算利息"]
        )):
            score = min(score, 1)
            issues.append("未同时说明不承担责任与不结算利息")

    if qid == "Q28":
        if not (
            has_any(answer, ["不承担", "不赔"])
            and has_any(answer, ["不退", "不返还", "不退还"])
        ):
            score = min(score, 1)
            issues.append("未同时说明不赔且不退费")

    if qid == "Q35":
        if has_any(answer, ["80%"]) and not has_any(
            answer, ["守御", "没有", "未约定", "不提供", "无保单贷款"]
        ):
            score = min(score, 1)
            issues.append("未说明守御一生无保单贷款")

    if qid == "Q33":
        if re.search(r"免赔额.{0,8}(为|是)\s*[\d]+", answer) and not has_any(
            answer, ["计划", "选项", "可选", "取决于", "约定"]
        ):
            score = min(score, 1)
            issues.append("给出固定免赔额但未说明计划/选项")

    has_ref = has_clause_ref(answer)
    if qid not in {"Q31", "Q32", "Q35"} and not has_ref:
        if score >= 2:
            score = 1
            issues.append("缺少条款引用")

    score = max(0, min(r.max_score, score))
    label = {2: "正确", 1: "部分正确", 0: "错误/不合格"}[score]
    passed = _collect_passed_checks(qid, answer, r)
    return {
        "question_id": qid,
        "score": score,
        "max_score": r.max_score,
        "label": label,
        "category": CATEGORIES[qid],
        "has_clause_ref": has_ref,
        "passed": passed,
        "issues": issues,
        "note": r.note,
    }


def main() -> None:
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    questions = parse_questions((ROOT / "test" / "测试问题.txt").read_text(encoding="utf-8"))
    all_scores: list[dict] = []
    per_q_runs: dict[str, list[int]] = defaultdict(list)
    cat_scores: dict[str, list[int]] = defaultdict(list)

    report = {
        "source_eval": str(EVAL_JSON.name),
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
                    "answer_preview": item["answer"][:200].replace("\n", " "),
                }
            )
            run_scores.append(s)
            all_scores.append(s)
            if s["score"] is not None:
                per_q_runs[item["question_id"]].append(s["score"])
                cat_scores[s["category"]].append(s["score"])

        valid = [x for x in run_scores if x["score"] is not None]
        total = sum(x["score"] for x in valid)
        max_total = sum(x["max_score"] for x in valid)
        invalid = len(run_scores) - len(valid)
        report["runs"].append(
            {
                "run_index": run["run_index"],
                "items": len(run_scores),
                "valid_items": len(valid),
                "invalid_items": invalid,
                "total_score": total,
                "max_score": max_total,
                "accuracy_pct": round(100 * total / max_total, 1) if max_total else 0,
                "scores": run_scores,
            }
        )

    # 按题号聚合（三轮平均，取最好一次用于稳定度）
    q_summary = []
    for qid in sorted(RUBRICS.keys()):
        scores = [s for s in all_scores if s["question_id"] == qid and s["score"] is not None]
        invalid_n = sum(1 for s in all_scores if s["question_id"] == qid and s["score"] is None)
        q_summary.append(
            {
                "question_id": qid,
                "category": CATEGORIES[qid],
                "attempts": len(scores) + invalid_n,
                "valid_attempts": len(scores),
                "invalid_attempts": invalid_n,
                "scores": [s["score"] for s in scores],
                "avg": round(sum(s["score"] for s in scores) / len(scores), 2) if scores else None,
                "best": max(s["score"] for s in scores) if scores else None,
                "worst": min(s["score"] for s in scores) if scores else None,
            }
        )

    valid_scores = [s for s in all_scores if s["score"] is not None]
    total_score = sum(s["score"] for s in valid_scores)
    max_score = sum(s["max_score"] for s in valid_scores)
    invalid_count = sum(1 for s in all_scores if s["score"] is None)
    cat_summary = {
        cat: {
            "count": len(scores),
            "avg": round(sum(scores) / len(scores), 2),
            "pct": round(100 * sum(scores) / (2 * len(scores)), 1),
        }
        for cat, scores in sorted(cat_scores.items())
    }

    report["summary"] = {
        "total_attempts": len(all_scores),
        "valid_attempts": len(valid_scores),
        "invalid_attempts": invalid_count,
        "invalid_question_ids": ["Q08", "Q16", "Q24", "Q30", "Q35"],
        "total_score": total_score,
        "max_score": max_score,
        "overall_pct": round(100 * total_score / max_score, 1) if max_score else 0,
        "full_correct": sum(1 for s in valid_scores if s["score"] == 2),
        "partial": sum(1 for s in valid_scores if s["score"] == 1),
        "wrong": sum(1 for s in valid_scores if s["score"] == 0),
        "by_category": cat_summary,
        "by_question": q_summary,
    }

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(report)
    print(f"有效得分: {total_score}/{max_score} ({report['summary']['overall_pct']}%)")
    print(f"无效作答: {invalid_count}")
    print(f"正确/部分/错误: {report['summary']['full_correct']}/{report['summary']['partial']}/{report['summary']['wrong']}")
    print(f"报告: {OUT_MD}")


def _write_md(report: dict) -> None:
    s = report["summary"]
    lines = [
        "# Agent 评测评分报告",
        "",
        f"- 数据来源: `{report['source_eval']}`",
        f"- 评分方法: {report['scoring_method']}",
        f"- 分值: {report['score_scale']}",
        "",
        "## 总览",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 有效答题次数 | {s['valid_attempts']} |",
        f"| 无效答题（脚本题干错误） | {s['invalid_attempts']} |",
        f"| 有效总得分 | **{s['total_score']} / {s['max_score']}** |",
        f"| 有效正确率 | **{s['overall_pct']}%** |",
        f"| 完全正确 (2分) | {s['full_correct']} |",
        f"| 部分正确 (1分) | {s['partial']} |",
        f"| 错误/不合格 (0分) | {s['wrong']} |",
        "",
        f"> **说明**：无效题 Q08/Q16/Q24/Q30/Q35 共 {s['invalid_attempts']} 次因评测脚本题干解析 bug，实际发送的是分隔线而非题目，已从能力分中剔除。",
        "",
        "## 分轮得分（仅有效题）",
        "",
        "| 轮次 | 有效/总题数 | 得分 | 正确率 |",
        "|------|-------------|------|--------|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| 第 {run['run_index']} 轮 | {run['valid_items']}/{run['items']} | "
            f"{run['total_score']}/{run['max_score']} | {run['accuracy_pct']}% |"
        )

    lines.extend(["", "## 分题型得分", "", "| 题型 | 答题数 | 均分/2 | 正确率 |", "|------|--------|--------|--------|"])
    for cat, info in s["by_category"].items():
        lines.append(f"| {cat} | {info['count']} | {info['avg']} | {info['pct']}% |")

    lines.extend(["", "## 逐题汇总（40题 × 三轮，含重复作答）", "", "| 题号 | 题型 | 作答次数 | 各轮得分 | 均分 | 最好 | 最差 |", "|------|------|----------|----------|------|------|------|"])
    for q in s["by_question"]:
        lines.append(
            f"| {q['question_id']} | {q['category']} | {q['attempts']} | "
            f"{q['scores']} | {q['avg']} | {q['best']} | {q['worst']} |"
        )

    weak = [q for q in s["by_question"] if q["avg"] is not None and q["avg"] < 1.5]
    lines.extend(["", "## 薄弱题（均分 < 1.5）", ""])
    if not weak:
        lines.append("无")
    else:
        for q in weak:
            lines.append(f"- **{q['question_id']}** ({q['category']}): 均分 {q['avg']}, 得分 {q['scores']}")

    lines.extend(["", "## 错题与部分正确明细", ""])
    for run in report["runs"]:
        bad = [x for x in run["scores"] if x["score"] is not None and x["score"] < 2]
        if not bad:
            continue
        lines.append(f"### 第 {run['run_index']} 轮")
        lines.append("")
        for x in sorted(bad, key=lambda i: (i["score"], i["question_id"])):
            lines.append(
                f"- **{x['question_id']}** [{x['label']}, {x['score']}/2] "
                f"seq={x['sequence']}: {'; '.join(x['issues']) or '见答案复核'}"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
