"""用户画像 / 咨询偏好等长期记忆（不记录对话过程）。"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from agentscope.message import Msg
from agentscope.model import ChatModelBase

from src.model.chat_model_factory import create_chat_formatter

if TYPE_CHECKING:
    from src.config import Settings

PROFILE_PREFIXES = ("[用户画像]", "[咨询偏好]", "[合同上下文]")
LEGACY_DIALOGUE_PREFIXES = ("用户问：", "助手答：")

_EXTRACT_PROMPT = """你是用户画像提取器。根据本轮用户问题与助手回答，提取**可跨会话复用**的简短事实。

只输出以下三类条目（每行一条，无则输出 NONE）：
- [用户画像] …（姓名、保单号、已选计划/免赔额、健康状况、家庭情况等）
- [咨询偏好] …（回答风格、语言、是否需计算步骤等）
- [合同上下文] …（用户明确声明的投保状态、已购产品名，非条款原文）

禁止：
- 复述用户整段问题或助手完整回答
- 记录条款内容、检索结果、SubAgent 过程
- 记录一次性考题或模拟题全文

若无任何值得长期记住的新增事实，只输出：NONE"""

_PROFILE_TRIGGER = (
    "记住", "叫我", "我的名字", "我的保单", "保单号", "已选", "我投保",
    "我买", "计划一", "计划二", "既往", "我有", "不要啰嗦", "简洁",
    "请用", "称呼我", "我是", "今年", "岁", "性别",
)


def is_legacy_dialogue_record(content: str) -> bool:
    text = content.strip()
    return any(text.startswith(prefix) for prefix in LEGACY_DIALOGUE_PREFIXES)


def is_profile_record(content: str) -> bool:
    text = content.strip()
    return any(text.startswith(prefix) for prefix in PROFILE_PREFIXES)


def normalize_profile_entries(raw_lines: list[str]) -> list[str]:
    """规范化待写入的长期记忆条目。"""
    entries: list[str] = []
    for line in raw_lines:
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        if is_legacy_dialogue_record(text):
            continue
        if len(text) > 400:
            text = text[:400] + "…"
        if not is_profile_record(text):
            text = f"[用户画像] {text}"
        entries.append(text)
    return entries


def should_extract_profile(user_text: str) -> bool:
    """启发式：是否值得尝试从本轮对话提取画像/偏好。"""
    text = user_text.strip()
    if not text:
        return False
    if len(text) > 600:
        return False
    if any(marker in text for marker in _PROFILE_TRIGGER):
        return True
    q_marks = text.count("？") + text.count("?")
    if q_marks >= 4:
        return False
    return len(text) <= 120


def _parse_extractor_output(raw: str) -> list[str]:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines or (len(lines) == 1 and lines[0].upper() == "NONE"):
        return []
    return normalize_profile_entries(lines)


async def extract_profile_updates(
    user_text: str,
    assistant_text: str,
    *,
    model: ChatModelBase,
    settings: Settings,
) -> list[str]:
    """从单轮对话中提取 0～N 条画像/偏好条目。"""
    if not should_extract_profile(user_text):
        return []

    formatter = create_chat_formatter(settings)
    user_blob = user_text.strip()[:2000]
    assistant_blob = (assistant_text or "").strip()[:800]
    messages = await formatter.format(
        [
            Msg("system", _EXTRACT_PROMPT, "system"),
            Msg(
                "user",
                f"【用户问题】\n{user_blob}\n\n"
                f"【助手回答摘要】\n{assistant_blob or '（无）'}",
                "user",
            ),
        ],
    )
    response = await model(messages)
    parts: list[str] = []
    if response.content:
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return _parse_extractor_output("".join(parts))
