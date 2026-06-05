"""分层会话记忆：本地完整保留 canonical，API 调用使用动态视图。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

from src.config import Settings
from src.memory.context_attachments import SessionContextAttachments
from src.memory.context_window_manager import ContextWindowManager
from src.model.chat_model_factory import create_chat_formatter, create_chat_model


class LayeredSessionMemory(InMemoryMemory):
    """两层上下文窗口会话记忆。

    - ``content``：当前工作区（硬压缩后可能被替换为四段式摘要）
    - ``archives``：硬压缩前完整快照（只增不改）
    - ``attachments``：硬压缩附件段数据源
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        formatter = create_chat_formatter(settings)
        summarizer = create_chat_model(
            settings,
            model_name=settings.compression_model_name,
            stream=False,
            multimodality=False,
        )
        self.settings = settings
        self.context_manager = ContextWindowManager(
            settings,
            formatter=formatter,
            summarizer_model=summarizer,
        )
        self.context_manager.bind_model_name(settings.model_name)
        self.attachments = SessionContextAttachments()
        self.archives: list[list[tuple[Msg, list[str]]]] = []
        self._agent_sys_prompt: str = ""

        self.register_state("archives")

    def set_agent_context(self, *, sys_prompt: str, model_name: str) -> None:
        """绑定当前 Agent 的 system prompt 与模型（用于窗口计算）。"""
        self._agent_sys_prompt = sys_prompt
        self.context_manager.bind_model_name(model_name)

    async def on_iteration_start(self) -> bool:
        """ReAct 每轮迭代前：检测硬压缩阈值并可能重写。"""
        return await self.context_manager.maybe_apply_hard_rewrite(
            self.content,
            sys_prompt=self._agent_sys_prompt,
            attachments=self.attachments,
            archives=self.archives,
        )

    async def get_memory(
        self,
        mark: str | None = None,
        exclude_mark: str | None = None,
        prepend_summary: bool = True,
        **kwargs: Any,
    ) -> list[Msg]:
        """API 用消息：软层返回动态视图；不修改 canonical 原文。"""
        del prepend_summary

        if kwargs.get("raw_canonical"):
            return await super().get_memory(
                mark=mark,
                exclude_mark=exclude_mark,
                prepend_summary=False,
            )

        api_msgs = await self.context_manager.get_api_messages(
            self.content,
            sys_prompt=self._agent_sys_prompt,
        )

        if mark is not None:
            api_msgs = [
                m
                for m in api_msgs
                if mark in self._marks_for_msg(m)
            ]
        if exclude_mark is not None:
            api_msgs = [
                m
                for m in api_msgs
                if exclude_mark not in self._marks_for_msg(m)
            ]

        return api_msgs

    def _marks_for_msg(self, msg: Msg) -> list[str]:
        for stored, marks in self.content:
            if stored.id == msg.id:
                return marks
        return []

    async def add(
        self,
        memories: Msg | list[Msg] | None,
        marks: str | list[str] | None = None,
        allow_duplicates: bool = False,
        **kwargs: Any,
    ) -> None:
        """追加到工作区（硬压缩后新消息继续追加）。"""
        await super().add(
            memories,
            marks=marks,
            allow_duplicates=allow_duplicates,
            **kwargs,
        )

    def get_archive_count(self) -> int:
        return len(self.archives)

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            "archives": [
                [[msg.to_dict(), marks] for msg, marks in snapshot]
                for snapshot in self.archives
            ],
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        super().load_state_dict(state_dict, strict=strict)
        self.archives = []
        for snapshot in state_dict.get("archives", []):
            restored: list[tuple[Msg, list[str]]] = []
            for item in snapshot:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    msg_dict, marks = item
                    restored.append((Msg.from_dict(msg_dict), deepcopy(marks)))
            self.archives.append(restored)
