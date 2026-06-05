"""OpenAI 兼容 Formatter：发往 API 前剥离 thinking 块，避免 skip 警告与无效 token。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock


def _strip_thinking_blocks(msg: Msg) -> Msg:
    """复制消息并移除 thinking 块（保留 text / tool 等）。"""
    if not msg.content:
        return msg

    # 纯字符串 content 不能对 msg.content 做 for 循环（会按字符拆开）
    if isinstance(msg.content, str):
        return msg

    new_content: list[Any] = []
    for block in msg.content:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                continue
            new_content.append(deepcopy(block))
        elif isinstance(block, str):
            new_content.append(TextBlock(type="text", text=block))
        else:
            new_content.append(block)

    if not new_content:
        return Msg(
            name=msg.name,
            content=[],
            role=msg.role,
            metadata=msg.metadata,
        )

    return Msg(
        name=msg.name,
        content=new_content,
        role=msg.role,
        metadata=msg.metadata,
    )


class ThinkingAwareOpenAIFormatter(OpenAIChatFormatter):
    """DeepSeek 等推理模型：会话内保留 thinking，调用 API 时不发送 thinking 块。"""

    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        cleaned = [_strip_thinking_blocks(m) for m in msgs]
        cleaned = [m for m in cleaned if m.content]
        return await super().format(cleaned)
