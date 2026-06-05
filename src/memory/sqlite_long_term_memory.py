"""SQLite 长期用户记忆：仅存用户画像 / 偏好 / 合同上下文摘要。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentscope.memory import LongTermMemoryBase
from agentscope.message import Msg, TextBlock
from agentscope.tool import ToolResponse

from src.memory.profile_memory import (
    is_legacy_dialogue_record,
    is_profile_record,
    normalize_profile_entries,
)


class SQLiteLongTermMemory(LongTermMemoryBase):
    """基于 SQLite 的长期用户记忆（画像类条目，不存对话全文）。"""

    def __init__(self, db_path: Path, user_id: str = "default_user") -> None:
        super().__init__()
        self.db_path = db_path
        self.user_id = user_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_memories_user "
                "ON user_memories(user_id)",
            )

    async def __aenter__(self) -> SQLiteLongTermMemory:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def _extract_keywords(self, text: str) -> list[str]:
        import re

        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]{2,}", text)
        return tokens[:20]

    async def record(
        self,
        msgs: list[Msg | None],
        **kwargs: Any,
    ) -> None:
        """AgentScope 回合结束回调：不将整段会话写入长期记忆。"""
        del msgs, kwargs

    async def record_profile_entries(self, entries: list[str]) -> int:
        """写入规范化后的画像/偏好条目。"""
        normalized = normalize_profile_entries(entries)
        if not normalized:
            return 0
        await self.record_to_memory(thinking="", content=normalized)
        return len(normalized)

    async def retrieve(
        self,
        msg: Msg | list[Msg] | None,
        limit: int = 5,
        **kwargs: Any,
    ) -> str:
        if isinstance(msg, list):
            query = " ".join(
                m.get_text_content() or "" for m in msg if m
            )
        elif msg:
            query = msg.get_text_content() or ""
        else:
            return ""

        keywords = self._extract_keywords(query)[:5]
        if not keywords:
            return ""

        result = await self.retrieve_from_memory(keywords=keywords, limit=limit)
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
        normalized = normalize_profile_entries(content)
        if not normalized:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="未写入：长期记忆仅接受用户画像/咨询偏好/合同上下文摘要，"
                        "请勿记录整段问答或条款原文。",
                    ),
                ],
            )

        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for item in normalized:
                keywords = json.dumps(
                    self._extract_keywords(item),
                    ensure_ascii=False,
                )
                conn.execute(
                    "INSERT INTO user_memories "
                    "(user_id, content, keywords, created_at) VALUES (?, ?, ?, ?)",
                    (self.user_id, item, keywords, now),
                )

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"已记录 {len(normalized)} 条用户画像/偏好信息。",
                ),
            ],
        )

    async def retrieve_from_memory(
        self,
        keywords: list[str],
        limit: int = 5,
        **kwargs: Any,
    ) -> ToolResponse:
        del kwargs
        if not keywords:
            return ToolResponse(
                content=[TextBlock(type="text", text="未提供检索关键词。")],
            )

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT content, keywords FROM user_memories "
                "WHERE user_id = ? ORDER BY id DESC LIMIT 100",
                (self.user_id,),
            ).fetchall()

        scored: list[tuple[float, str]] = []
        for content, kw_json in rows:
            if is_legacy_dialogue_record(content):
                continue
            stored_keywords = json.loads(kw_json)
            score = sum(
                1
                for kw in keywords
                if any(kw in sk or sk in kw for sk in stored_keywords)
                or kw in content
            )
            if is_profile_record(content):
                score += 0.5
            if score > 0:
                scored.append((score, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [text for _, text in scored[:limit]]

        if not top:
            return ToolResponse(
                content=[TextBlock(type="text", text="未找到相关用户画像记忆。")],
            )

        return ToolResponse(
            content=[
                TextBlock(type="text", text=f"- {item}") for item in top
            ],
        )

    async def clear_all(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM user_memories WHERE user_id = ?",
                (self.user_id,),
            )
            return cursor.rowcount
