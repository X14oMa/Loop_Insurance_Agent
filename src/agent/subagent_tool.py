"""主 Agent 可调用的 SubAgent 委托工具。"""

from __future__ import annotations

from collections.abc import Callable

from agentscope.message import Msg, TextBlock
from agentscope.tool import ToolResponse

from src.compliance import ComplianceValidator
from src.config import Settings
from src.model.chat_model_factory import create_chat_formatter, create_chat_model
from src.pipeline.subagent_runner import run_subagents, synthesize_subagent_brief
from src.rag import HybridKnowledge
from src.rag.tool_response_limits import truncate_with_notice


class SubAgentDelegation:
    """并行运行 SubAgent 并返回带条款引用的简短汇总（供主 Agent tool_result）。"""

    def __init__(
        self,
        settings: Settings,
        knowledge: HybridKnowledge,
        compliance: ComplianceValidator,
        policy_context_provider: Callable[[], str | None],
    ) -> None:
        self.settings = settings
        self.knowledge = knowledge
        self.compliance = compliance
        self._policy_context_provider = policy_context_provider

    async def delegate_subagents(
        self,
        questions: list[str],
        user_intent: str = "",
        doc_id: str | None = None,
    ) -> ToolResponse:
        """将多个独立子问题委托给 SubAgent 并行分析，返回简短汇总（含章节/条款引用）。

        Args:
            questions: 2～{max_subagents} 个互不相同的子问题，每个应可独立检索条款。
            user_intent: 用户本轮整体意图（可选，用于汇总时对齐主题）。
            doc_id: 可选保单 filename，与子 Agent 检索一致。
        """
        del doc_id  # 子 Agent 输入已含 policy_library；保留参数供模型显式传 doc

        if not self.settings.subagent_enabled:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="错误：SubAgent 委托功能已关闭（SUBAGENT_ENABLED=false）。"
                        "请改用 retrieve_knowledge 自行检索。",
                    ),
                ],
            )

        cleaned = [str(q).strip() for q in questions if str(q).strip()]
        if len(cleaned) < self.settings.subagent_min_questions:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"错误：delegate_subagents 至少需要 "
                            f"{self.settings.subagent_min_questions} 个非空子问题。"
                            "单一问题请直接使用 retrieve_knowledge。"
                        ),
                    ),
                ],
            )

        trimmed = cleaned[: self.settings.max_subagents]
        policy_context = self._policy_context_provider()

        sub_results = await run_subagents(
            trimmed,
            settings=self.settings,
            knowledge=self.knowledge,
            compliance=self.compliance,
            policy_context=policy_context,
            max_retrievals=self.settings.max_subagent_retrievals,
        )

        brief = await synthesize_subagent_brief(
            trimmed,
            sub_results,
            user_intent=user_intent or "（未提供）",
            settings=self.settings,
            max_chars=self.settings.subagent_tool_max_chars,
        )

        brief = truncate_with_notice(
            brief,
            self.settings.subagent_tool_max_chars,
        )

        header = (
            f"【SubAgent 汇总】共 {len(trimmed)} 个子问题。"
            "以下为简短结论，引用格式为 [保单filename | 章节/条款标识]。"
            "请据此组织面向用户的答复，勿编造条号。\n\n"
        )
        return ToolResponse(
            content=[TextBlock(type="text", text=header + brief)],
        )
