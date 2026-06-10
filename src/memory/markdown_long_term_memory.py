"""Markdown 长期记忆：按 user_id 存为单个 .md 文件，由 Agent 工具读写。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentscope.memory import LongTermMemoryBase
from agentscope.message import Msg, TextBlock
from agentscope.tool import ToolResponse

_MAX_READ_CHARS = 6000


def _safe_user_id(user_id: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", user_id.strip())
    return safe or "default_user"


class MarkdownLongTermMemory(LongTermMemoryBase):
    """按用户 ID 持久化一份 Markdown 长期记忆文本。"""

    def __init__(self, memory_dir: Path, user_id: str = "default_user") -> None:
        super().__init__()
        self.user_id = user_id
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.memory_dir / f"{_safe_user_id(user_id)}.md"

    async def __aenter__(self) -> MarkdownLongTermMemory:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def read_text(self) -> str:
        if not self.file_path.exists():
            return ""
        return self.file_path.read_text(encoding="utf-8").strip()

    def write_text(self, content: str) -> None:
        text = content.strip()
        if not text:
            if self.file_path.exists():
                self.file_path.unlink()
            return
        if not text.lstrip().startswith("#"):
            text = f"# 长期记忆\n\n{text}"
        self.file_path.write_text(text + "\n", encoding="utf-8")

    @staticmethod
    def _append_entries(existing: str, entries: list[str]) -> str:
        lines = [e.strip() for e in entries if e.strip()]
        if not lines:
            return existing
        block = "\n".join(f"- {line}" for line in lines)
        if not existing.strip():
            return f"# 长期记忆\n\n{block}\n"
        return existing.rstrip() + f"\n\n{block}\n"

    async def record(
        self,
        msgs: list[Msg | None],
        **kwargs: Any,
    ) -> None:
        """AgentScope 回合结束回调：不自动写入会话全文。"""
        del msgs, kwargs

    async def retrieve(
        self,
        msg: Msg | list[Msg] | None,
        limit: int = 5,
        **kwargs: Any,
    ) -> str:
        del msg, limit, kwargs
        result = await self.retrieve_from_memory(keywords=[])
        texts: list[str] = []
        for block in result.content:
            if isinstance(block, dict):
                texts.append(str(block.get("text", "")))
        return "\n".join(texts)

    async def record_to_memory(
        self,
        thinking: str,
        content: list[str],
        **kwargs: Any,
    ) -> ToolResponse:
        del thinking, kwargs
        entries = [c.strip() for c in content if c and c.strip()]
        if not entries:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="未写入：请提供可长期保存的用户事实或偏好（如称呼、持有保单、"
                        "选过的计划/选项、沟通偏好），每条一句；勿记录条款细节或整段问答。",
                    ),
                ],
            )

        merged = self._append_entries(self.read_text(), entries)
        self.write_text(merged)
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"已追加 {len(entries)} 条到长期记忆（Markdown）。",
                ),
            ],
        )

    async def retrieve_from_memory(
        self,
        keywords: list[str],
        limit: int = 5,
        **kwargs: Any,
    ) -> ToolResponse:
        del keywords, limit, kwargs
        content = self.read_text()
        if not content:
            return ToolResponse(
                content=[TextBlock(type="text", text="长期记忆为空。")],
            )

        display = content
        if len(display) > _MAX_READ_CHARS:
            display = display[:_MAX_READ_CHARS] + "\n…(内容已截断)"

        return ToolResponse(
            content=[TextBlock(type="text", text=display)],
        )

    async def clear_all(self) -> int:
        if self.file_path.exists():
            self.file_path.unlink()
            return 1
        return 0
