"""记忆检索决策：仅在需要用户画像/偏好时注入长期记忆。"""

from __future__ import annotations

import json

from agentscope.formatter import FormatterBase
from agentscope.memory import LongTermMemoryBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase

from src.memory.profile_memory import _PROFILE_TRIGGER


class MemoryRetrievalDecisionMaker:
    """决定是否从长期记忆检索用户画像类信息。"""

    DECISION_PROMPT = """你是记忆检索决策器。长期记忆中**只存**用户画像、咨询偏好、合同上下文摘要，不存历史问答全文。

需要检索 (need_retrieval=true) 的情况：
- 用户引用既往个人信息或你的记录（「上次说的」「我的计划」「记住的」）
- 问题依赖用户专属背景（已选计划、免赔额设定、保单号、既往病史等）且当前问题未写全

不需要检索 (need_retrieval=false) 的情况：
- 纯条款/理赔/产品通用知识问答（应走知识库检索）
- 模拟题、卷面多子题、计算题且用户未声明个人投保信息
- 问题已自带完整计算或计划前提

请以 JSON 回复：
{"need_retrieval": true|false, "keywords": ["关键词1", ...]}
keywords 最多 3 个，聚焦用户身份/计划/偏好，不要用宽泛条款词（如「等待期」「免赔」）。
只输出 JSON。"""

    def __init__(
        self,
        model: ChatModelBase,
        formatter: FormatterBase,
        long_term_memory: LongTermMemoryBase,
    ) -> None:
        self.model = model
        self.formatter = formatter
        self.long_term_memory = long_term_memory

    async def decide_and_retrieve(self, user_query: str) -> str | None:
        messages = await self.formatter.format(
            msgs=[
                Msg("system", self.DECISION_PROMPT, "system"),
                Msg("user", f"用户问题：{user_query}", "user"),
            ],
        )
        response = await self.model(messages)
        decision_text = ""
        if response.content:
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    decision_text += block.get("text", "")

        try:
            decision = json.loads(decision_text.strip())
        except json.JSONDecodeError:
            need_retrieval = any(kw in user_query for kw in _PROFILE_TRIGGER) or any(
                kw in user_query
                for kw in ("上次", "之前", "记得我说过", "我的保单", "我叫")
            )
            decision = {
                "need_retrieval": need_retrieval,
                "keywords": [],
            }

        if not decision.get("need_retrieval"):
            return None

        keywords = decision.get("keywords", [])
        if not keywords:
            keywords = [
                token
                for token in ("用户画像", "咨询偏好", "合同上下文")
                if token in user_query
            ]
        if not keywords:
            keywords = [user_query[:40]]

        result = await self.long_term_memory.retrieve_from_memory(
            keywords=keywords,
            limit=5,
        )
        memory_texts: list[str] = []
        for block in result.content:
            if isinstance(block, dict) and block.get("type") == "text":
                memory_texts.append(str(block.get("text", "")))

        if not memory_texts:
            return None

        return "\n".join(memory_texts)
