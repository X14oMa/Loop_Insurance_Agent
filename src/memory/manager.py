"""记忆管理器：会话短期记忆 + 用户画像类长期记忆。"""

from __future__ import annotations

from agentscope.message import Msg
from agentscope.model import ChatModelBase

from src.config import Settings
from src.memory.profile_memory import extract_profile_updates
from src.memory.retrieval_decision import MemoryRetrievalDecisionMaker
from src.memory.sqlite_long_term_memory import SQLiteLongTermMemory
from src.model.chat_model_factory import create_chat_formatter, create_chat_model


class MemoryManager:
    """记忆管理模块。

    - 多轮对话上下文：由 ``SessionManager`` / ``LayeredSessionMemory`` 负责
    - 长期记忆：仅持久化用户画像、咨询偏好、合同上下文等摘要（不存问答全文）
    """

    def __init__(
        self,
        settings: Settings,
        model: ChatModelBase | None = None,
    ) -> None:
        self.settings = settings
        db_path = settings.long_term_memory_path / f"{settings.user_id}.db"
        self.long_term = SQLiteLongTermMemory(
            db_path=db_path,
            user_id=settings.user_id,
        )
        chat_model = model or create_chat_model(settings, stream=False)
        self._profile_model = create_chat_model(
            settings,
            model_name=settings.profile_model_name,
            stream=False,
            multimodality=False,
        )
        self.decision_maker = MemoryRetrievalDecisionMaker(
            model=chat_model,
            formatter=create_chat_formatter(settings),
            long_term_memory=self.long_term,
        )

    async def retrieve_user_profile_context(self, user_query: str) -> str | None:
        """判断是否注入已存储的用户画像/偏好（非对话记录）。"""
        async with self.long_term:
            return await self.decision_maker.decide_and_retrieve(user_query)

    async def update_after_response(
        self,
        user_msg: Msg,
        assistant_msg: Msg,
    ) -> None:
        """对话结束后：可选提取画像条目写入长期记忆（不记录问答全文）。"""
        if not self.settings.long_term_auto_profile:
            return

        user_text = user_msg.get_text_content() or ""
        assistant_text = assistant_msg.get_text_content() or ""
        entries = await extract_profile_updates(
            user_text,
            assistant_text,
            model=self._profile_model,
            settings=self.settings,
        )
        if not entries:
            return

        async with self.long_term:
            await self.long_term.record_profile_entries(entries)
