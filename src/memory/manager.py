"""记忆管理器：会话短期记忆 + Markdown 长期记忆。"""

from __future__ import annotations

from agentscope.message import Msg

from src.config import Settings
from src.memory.ltm_merge import merge_ltm_update
from src.memory.markdown_long_term_memory import MarkdownLongTermMemory
from src.model.chat_model_factory import create_chat_model


class MemoryManager:
    """记忆管理模块。

    - 多轮对话上下文：由 ``SessionManager`` / ``LayeredSessionMemory`` 负责
    - 长期记忆：按 user_id 存 Markdown 文本；读取由 Agent 工具自决，写入由关键词+LLM 合并
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.long_term = MarkdownLongTermMemory(
            memory_dir=settings.long_term_memory_path,
            user_id=settings.user_id,
        )
        self._merge_model = create_chat_model(
            settings,
            model_name=settings.profile_model_name,
            stream=False,
            multimodality=False,
        )

    async def update_after_response(
        self,
        user_msg: Msg,
        assistant_msg: Msg,
    ) -> None:
        """对话结束后：关键词命中时用 LLM 合并更新 Markdown 长期记忆。"""
        if not self.settings.long_term_auto_profile:
            return

        user_text = user_msg.get_text_content() or ""
        assistant_text = assistant_msg.get_text_content() or ""

        async with self.long_term:
            existing = self.long_term.read_text()

        new_markdown = await merge_ltm_update(
            user_text,
            assistant_text,
            existing,
            model=self._merge_model,
            settings=self.settings,
        )
        if not new_markdown:
            return

        async with self.long_term:
            self.long_term.write_text(new_markdown)
