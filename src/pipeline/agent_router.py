"""对话轮次配置：复杂度与检索预算（无 SubAgent 路由）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from agentscope.message import Msg

from src.config import Settings
from src.model.chat_model_factory import create_chat_formatter, create_chat_model

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")

Complexity = Literal["low", "medium", "high"]

_CLAUSE_KEYWORDS = (
    "等待期", "免赔", "免责", "理赔", "赔付", "保障", "除外", "续保",
    "退保", "保额", "保费", "受益人", "合同", "条款", "报销", "给付",
)
_COMPLEXITY_MARKERS = (
    "若", "是否", "能否", "对比", "比较", "分别", "同时", "以及", "还有",
    "哪些", "哪种", "情形", "条件", "满足", "不符合",
)
@dataclass(frozen=True)
class TurnConfig:
    """单次用户消息的主 Agent 轮次配置。"""

    complexity: Complexity
    max_retrievals: int
    reason: str


def _heuristic_split(user_input: str) -> list[str]:
    text = user_input.strip()
    if not text:
        return []

    parts = re.split(r"(?<=[？?])\s+", text)
    questions = [
        part.strip() for part in parts if part.strip() and len(part.strip()) >= 8
    ]
    if len(questions) >= 2:
        return questions

    numbered = re.split(
        r"(?=(?:\n|^)\s*(?:\d+[.、．]|[-*])\s*)",
        text,
    )
    numbered = [
        part.strip() for part in numbered if part.strip() and len(part.strip()) >= 8
    ]
    if len(numbered) >= 2:
        return numbered

    return [text]


def _extract_response_text(response: object) -> str:
    content = getattr(response, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def score_complexity(user_input: str, question_count: int) -> Complexity:
    text = user_input.strip()
    score = 0
    score += min(question_count, 3) * 2
    score += sum(1 for kw in _CLAUSE_KEYWORDS if kw in text)
    score += sum(1 for marker in _COMPLEXITY_MARKERS if marker in text)
    if len(text) > 180:
        score += 2
    if len(text) > 350:
        score += 2

    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def main_agent_retrieval_budget(settings: Settings, complexity: Complexity) -> int:
    if complexity == "high":
        return settings.max_main_agent_retrievals_high
    if complexity == "medium":
        return settings.max_main_agent_retrievals
    return settings.max_main_agent_retrievals_low


def _parse_complexity_payload(raw: str) -> dict:
    match = _JSON_BLOCK.search(raw)
    if not match:
        return {}
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_turn_config(
    settings: Settings,
    *,
    complexity: Complexity,
    reason: str,
) -> TurnConfig:
    return TurnConfig(
        complexity=complexity,
        max_retrievals=main_agent_retrieval_budget(settings, complexity),
        reason=reason,
    )


async def resolve_turn_config(user_input: str, settings: Settings) -> TurnConfig:
    """解析本轮主 Agent 的检索预算。"""
    text = user_input.strip()
    if not text:
        return _build_turn_config(
            settings,
            complexity="low",
            reason="空输入",
        )

    heuristic = _heuristic_split(text)
    complexity = score_complexity(text, len(heuristic))

    need_llm = complexity in {"medium", "high"} or len(heuristic) >= 2

    if not need_llm:
        return _build_turn_config(
            settings,
            complexity=complexity,
            reason=f"启发式复杂度 {complexity}",
        )

    model = create_chat_model(
        settings,
        model_name=settings.decompose_model_name,
        stream=False,
        multimodality=False,
    )
    prompt = (
        "你是保险咨询复杂度评估助手。仅输出 JSON，不要其他文字。\n"
        '字段：{"complexity":"low|medium|high","reason":"简短中文"}\n'
        f"用户输入：\n{text}"
    )
    formatter = create_chat_formatter(settings)
    messages = await formatter.format([Msg("user", prompt, "user")])
    response = await model(messages)
    payload = _parse_complexity_payload(_extract_response_text(response))

    llm_complexity = str(payload.get("complexity", complexity)).lower()
    if llm_complexity in {"low", "medium", "high"}:
        complexity = llm_complexity  # type: ignore[assignment]
    reason = str(payload.get("reason", "")).strip() or f"LLM 评估复杂度 {complexity}"

    return _build_turn_config(
        settings,
        complexity=complexity,
        reason=reason,
    )


# 向后兼容旧名称
RouteDecision = TurnConfig


async def route_user_query(user_input: str, settings: Settings) -> TurnConfig:
    """已废弃语义：等同于 resolve_turn_config。"""
    return await resolve_turn_config(user_input, settings)
