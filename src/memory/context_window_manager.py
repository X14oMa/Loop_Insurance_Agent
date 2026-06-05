"""两层上下文窗口管理：80% 动态视图 / 93% 或 window-13k 硬重写。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase
from agentscope.token import CharTokenCounter, TokenCounterBase

from src.config import Settings
from src.memory.context_attachments import SessionContextAttachments
from src.model.context_window_registry import (
    compute_thresholds,
    resolve_context_window,
)

logger = logging.getLogger(__name__)

ContextTier = Literal["normal", "soft_view", "hard_rewrite"]
_HINT_MARK = "hint"

_SOFT_SUMMARY_PROMPT = """请将以下对话片段压缩为简洁中文摘要。
要求：保留核心事实与结论；不重复；不遗漏关键条款号、数字、保单名；不要编造。
只输出摘要正文，不要标题或前后缀。"""

_HARD_REWRITE_PROMPT = """你是对话历史重写器。请将「完整对话记录」压缩为以下四段，严格使用 XML 标签包裹：

1. <boundary>…</boundary>
   记录：压缩前估算 token 数、最后一条消息的 id、消息条数、压缩时间(UTC)、压缩层级(tier2_hard_rewrite)。

2. <full_summary>…</full_summary>
   全量摘要：覆盖所有轮次的核心信息，尽量精简，不重复，不遗漏。

3. <attachments>…</attachments>
   原样保留下方「附件上下文」中的信息（可略作整理，勿删除条目）。

4. <recent_turns>…</recent_turns>
   最近一轮用户最新问题与助手最新回答的原文（若对话中不存在助手回答可只写用户问题）。

只输出上述 XML 四段，不要其他文字。"""


@dataclass
class CompressionBoundaryRecord:
    """硬压缩边界元数据。"""

    tokens_before: int
    last_message_id: str
    message_count: int
    compressed_at: str
    tier: str = "tier2_hard_rewrite"
    archive_index: int = 0


@dataclass
class LayeredMemoryState:
    """分层记忆运行状态（与 canonical 日志分离）。"""

    tier: ContextTier = "normal"
    context_window: int = 64_000
    soft_threshold: int = 0
    hard_threshold: int = 0
    last_token_count: int = 0
    soft_early_summary: str = ""
    soft_middle_summary: str = ""
    soft_summary_source_hash: str = ""
    boundaries: list[CompressionBoundaryRecord] = field(default_factory=list)
    hard_rewrite_count: int = 0


def _msg_text(msg: Msg) -> str:
    return msg.get_text_content() or ""


def _is_hint(msg: Msg, marks: list[str]) -> bool:
    return _HINT_MARK in marks


def _is_tool_related(msg: Msg) -> bool:
    return bool(
        msg.get_content_blocks("tool_use")
        or msg.get_content_blocks("tool_result"),
    )


def _messages_fingerprint(msgs: list[Msg]) -> str:
    parts = [f"{m.id}:{m.role}:{_msg_text(m)[:200]}" for m in msgs]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _select_tail_messages(
    msgs: list[Msg],
    *,
    keep_recent: int,
) -> tuple[list[Msg], list[Msg]]:
    """保留尾部消息，尽量保持 tool_use/tool_result 成对。"""
    if not msgs or keep_recent <= 0:
        return msgs, []

    tail: list[Msg] = []
    pending_tool_ids: set[str] = set()
    i = len(msgs) - 1

    while i >= 0 and len(tail) < keep_recent + 4:
        msg = msgs[i]
        tail.insert(0, msg)
        for block in msg.get_content_blocks("tool_result"):
            pending_tool_ids.add(block.get("id", ""))
        for block in msg.get_content_blocks("tool_use"):
            tid = block.get("id", "")
            if tid in pending_tool_ids:
                pending_tool_ids.discard(tid)
        if len(tail) >= keep_recent and not pending_tool_ids:
            break
        i -= 1

    head = msgs[: max(0, len(msgs) - len(tail))]
    return head, tail


def _split_early_middle(non_tool_msgs: list[Msg]) -> tuple[list[Msg], list[Msg]]:
    if len(non_tool_msgs) <= 4:
        return non_tool_msgs, []
    early_len = max(1, len(non_tool_msgs) // 3)
    return non_tool_msgs[:early_len], non_tool_msgs[early_len:]


def _parse_hard_rewrite_output(text: str) -> dict[str, str]:
    def _extract(tag: str) -> str:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    return {
        "boundary": _extract("boundary"),
        "full_summary": _extract("full_summary"),
        "attachments": _extract("attachments"),
        "recent_turns": _extract("recent_turns"),
    }


class ContextWindowManager:
    """两层上下文窗口控制器。"""

    def __init__(
        self,
        settings: Settings,
        *,
        formatter: FormatterBase,
        summarizer_model: ChatModelBase,
        token_counter: TokenCounterBase | None = None,
    ) -> None:
        self.settings = settings
        self.formatter = formatter
        self.summarizer = summarizer_model
        self.token_counter = token_counter or CharTokenCounter()
        self.state = LayeredMemoryState()

    def bind_model_name(self, model_name: str) -> None:
        window = resolve_context_window(
            model_name,
            override=self.settings.model_context_window or None,
        )
        soft, hard = compute_thresholds(
            window,
            soft_ratio=self.settings.context_soft_ratio,
            hard_ratio=self.settings.context_hard_ratio,
            hard_reserve_tokens=self.settings.context_hard_reserve_tokens,
        )
        self.state.context_window = window
        self.state.soft_threshold = soft
        self.state.hard_threshold = hard

    async def count_tokens(
        self,
        messages: list[Msg],
        *,
        sys_prompt: str = "",
    ) -> int:
        prompt = await self.formatter.format(
            [Msg("system", sys_prompt, "system"), *messages],
        )
        return await self.token_counter.count(prompt)

    def _canonical_messages(
        self,
        content: list[tuple[Msg, list[str]]],
    ) -> list[Msg]:
        return [
            msg
            for msg, marks in content
            if not _is_hint(msg, marks)
        ]

    async def refresh_tier(
        self,
        content: list[tuple[Msg, list[str]]],
        *,
        sys_prompt: str,
    ) -> ContextTier:
        """根据 canonical 日志更新 tier（不修改原始消息）。"""
        canonical = self._canonical_messages(content)
        if not canonical:
            self.state.tier = "normal"
            self.state.last_token_count = 0
            return "normal"

        tokens = await self.count_tokens(canonical, sys_prompt=sys_prompt)
        self.state.last_token_count = tokens

        if tokens >= self.state.hard_threshold:
            self.state.tier = "hard_rewrite"
        elif tokens >= self.state.soft_threshold:
            self.state.tier = "soft_view"
        else:
            self.state.tier = "normal"

        logger.info(
            "上下文 tier=%s tokens=%d soft=%d hard=%d window=%d",
            self.state.tier,
            tokens,
            self.state.soft_threshold,
            self.state.hard_threshold,
            self.state.context_window,
        )
        return self.state.tier

    async def _summarize_chunk(self, msgs: list[Msg], *, label: str) -> str:
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            role = m.role or "unknown"
            text = _msg_text(m)
            if text.strip():
                lines.append(f"[{role}] {text[:4000]}")
            for block in m.get_content_blocks("tool_use"):
                lines.append(
                    f"[tool_use] {block.get('name', '')} "
                    f"{json.dumps(block.get('input', {}), ensure_ascii=False)[:500]}",
                )
            for block in m.get_content_blocks("tool_result"):
                out = block.get("output", [])
                preview = str(out)[:800]
                lines.append(f"[tool_result] {preview}")

        user_body = "\n".join(lines)
        prompt = await self.formatter.format(
            [
                Msg("system", _SOFT_SUMMARY_PROMPT, "system"),
                Msg("user", f"片段类型：{label}\n\n{user_body}", "user"),
            ],
        )
        res = await self.summarizer(prompt)
        text = ""
        if res.content:
            for block in res.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text", ""))
        return text.strip()

    async def _ensure_soft_summaries(
        self,
        content: list[tuple[Msg, list[str]]],
    ) -> None:
        canonical = self._canonical_messages(content)
        fingerprint = _messages_fingerprint(canonical)
        if fingerprint == self.state.soft_summary_source_hash:
            return

        _, tail = _select_tail_messages(
            canonical,
            keep_recent=self.settings.agent_compression_keep_recent,
        )
        body = canonical[: max(0, len(canonical) - len(tail))]
        tools = [m for m in body if _is_tool_related(m)]
        non_tool = [m for m in body if m not in tools]
        early, middle = _split_early_middle(non_tool)

        self.state.soft_early_summary = await self._summarize_chunk(
            early,
            label="早期历史",
        )
        self.state.soft_middle_summary = await self._summarize_chunk(
            middle,
            label="中间历史",
        )
        self.state.soft_summary_source_hash = fingerprint

    async def build_soft_view(
        self,
        content: list[tuple[Msg, list[str]]],
    ) -> list[Msg]:
        """第一层：动态压缩视图（不修改 canonical）。"""
        await self._ensure_soft_summaries(content)
        canonical = self._canonical_messages(content)
        head, tail = _select_tail_messages(
            canonical,
            keep_recent=self.settings.agent_compression_keep_recent,
        )
        body = canonical[: max(0, len(canonical) - len(tail))]
        tool_msgs = [m for m in body if _is_tool_related(m)]

        view: list[Msg] = [
            Msg(
                "user",
                "<context_view mode=\"soft\" note=\"以下为动态压缩视图，"
                "完整原文已在本地保留\">",
                "user",
            ),
        ]
        if self.state.soft_early_summary:
            view.append(
                Msg(
                    "user",
                    f"<early_history_summary>\n"
                    f"{self.state.soft_early_summary}\n"
                    f"</early_history_summary>",
                    "user",
                ),
            )
        for tool_msg in tool_msgs:
            view.append(tool_msg)
        if self.state.soft_middle_summary:
            view.append(
                Msg(
                    "user",
                    f"<middle_history_summary>\n"
                    f"{self.state.soft_middle_summary}\n"
                    f"</middle_history_summary>",
                    "user",
                ),
            )
        view.extend(tail)
        return view

    async def apply_hard_rewrite(
        self,
        content: list[tuple[Msg, list[str]]],
        *,
        sys_prompt: str,
        attachments: SessionContextAttachments,
        archives: list[list[tuple[Msg, list[str]]]],
    ) -> list[tuple[Msg, list[str]]]:
        """第二层：整段重写并替换工作区 content；canonical 存入 archives。"""
        canonical = self._canonical_messages(content)
        if not canonical:
            return content

        tokens_before = await self.count_tokens(canonical, sys_prompt=sys_prompt)
        last_id = canonical[-1].id
        archives.append(list(content))

        transcript_lines = []
        for msg in canonical:
            transcript_lines.append(
                f"[id={msg.id} role={msg.role}]\n{_msg_text(msg)}",
            )
        transcript = "\n\n".join(transcript_lines)

        attachment_text = attachments.to_markdown()
        rewrite_prompt = await self.formatter.format(
            [
                Msg("system", _HARD_REWRITE_PROMPT, "system"),
                Msg(
                    "user",
                    f"压缩前 token 估算: {tokens_before}\n"
                    f"消息条数: {len(canonical)}\n"
                    f"最后消息 id: {last_id}\n\n"
                    f"--- 附件上下文 ---\n{attachment_text}\n\n"
                    f"--- 完整对话记录 ---\n{transcript}",
                    "user",
                ),
            ],
        )
        res = await self.summarizer(rewrite_prompt)
        raw = ""
        if res.content:
            for block in res.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    raw += str(block.get("text", ""))

        sections = _parse_hard_rewrite_output(raw)
        if not sections["boundary"]:
            sections["boundary"] = (
                f"tokens_before={tokens_before}; "
                f"last_message_id={last_id}; "
                f"message_count={len(canonical)}; "
                f"compressed_at={datetime.now(timezone.utc).isoformat()}; "
                f"tier=tier2_hard_rewrite"
            )

        boundary_record = CompressionBoundaryRecord(
            tokens_before=tokens_before,
            last_message_id=last_id,
            message_count=len(canonical),
            compressed_at=datetime.now(timezone.utc).isoformat(),
            archive_index=len(archives) - 1,
        )
        self.state.boundaries.append(boundary_record)
        self.state.hard_rewrite_count += 1
        self.state.soft_early_summary = ""
        self.state.soft_middle_summary = ""
        self.state.soft_summary_source_hash = ""
        self.state.tier = "hard_rewrite"

        new_content: list[tuple[Msg, list[str]]] = [
            (
                Msg(
                    "user",
                    [
                        TextBlock(
                            type="text",
                            text=(
                                "<compression_boundary>\n"
                                f"{sections['boundary']}\n"
                                "</compression_boundary>"
                            ),
                        ),
                    ],
                    "user",
                ),
                [],
            ),
            (
                Msg(
                    "user",
                    [
                        TextBlock(
                            type="text",
                            text=(
                                "<full_summary>\n"
                                f"{sections['full_summary'] or '（摘要生成失败，请结合附件）'}\n"
                                "</full_summary>"
                            ),
                        ),
                    ],
                    "user",
                ),
                [],
            ),
            (
                Msg(
                    "user",
                    [
                        TextBlock(
                            type="text",
                            text=(
                                "<attachments>\n"
                                f"{sections['attachments'] or attachment_text}\n"
                                "</attachments>"
                            ),
                        ),
                    ],
                    "user",
                ),
                [],
            ),
        ]
        if sections["recent_turns"]:
            new_content.append(
                (
                    Msg(
                        "user",
                        [
                            TextBlock(
                                type="text",
                                text=(
                                    "<recent_turns>\n"
                                    f"{sections['recent_turns']}\n"
                                    "</recent_turns>"
                                ),
                            ),
                        ],
                        "user",
                    ),
                    [],
                ),
            )

        logger.info(
            "硬压缩完成 archive_index=%d tokens_before=%d",
            boundary_record.archive_index,
            tokens_before,
        )
        return new_content

    async def get_api_messages(
        self,
        content: list[tuple[Msg, list[str]]],
        *,
        sys_prompt: str,
    ) -> list[Msg]:
        """返回应送入模型的消息列表（软层动态视图或完整 canonical）。"""
        await self.refresh_tier(content, sys_prompt=sys_prompt)

        if self.state.tier == "soft_view":
            return await self.build_soft_view(content)

        return self._canonical_messages(content)

    async def maybe_apply_hard_rewrite(
        self,
        content: list[tuple[Msg, list[str]]],
        *,
        sys_prompt: str,
        attachments: SessionContextAttachments,
        archives: list[list[tuple[Msg, list[str]]]],
    ) -> bool:
        """第二层硬压缩：超 hard 阈值时归档并重写工作区。返回是否已执行。"""
        await self.refresh_tier(content, sys_prompt=sys_prompt)
        if self.state.tier != "hard_rewrite":
            return False

        canonical = self._canonical_messages(content)
        if len(canonical) <= 2:
            return False

        new_content = await self.apply_hard_rewrite(
            content,
            sys_prompt=sys_prompt,
            attachments=attachments,
            archives=archives,
        )
        content.clear()
        content.extend(new_content)
        self.state.tier = "normal"
        self.state.last_token_count = await self.count_tokens(
            self._canonical_messages(content),
            sys_prompt=sys_prompt,
        )
        return True
