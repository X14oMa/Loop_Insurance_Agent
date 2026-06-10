"""保险 Agent 工厂：基于 AgentScope ReActAgent 构建主智能体。"""

from __future__ import annotations

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from collections.abc import Callable

from src.agent.insurance_react_agent import InsuranceReActAgent
from src.agent.subagent_tool import SubAgentDelegation
from src.compliance import ComplianceValidator
from src.config import Settings
from src.memory import MemoryManager
from src.model.chat_model_factory import create_chat_formatter, create_chat_model
from src.rag import HybridKnowledge


SYSTEM_PROMPT = """你是一位专业的保险顾问智能助手，服务于保险客户咨询场景。

## 知识库保单
- 每条用户消息可能附带 `<policy_library>`，列出当前已入库的保单 filename
- 用户问「这份保单叫什么/是哪份合同」时，**直接根据该列表回答**，无需检索
- 仅一份保单时，用户说「这份保单」默认指该保单；检索条款时请传入完整 doc_id
- 不得编造列表中不存在的保单或产品名称

## 核心职责
1. **Query 分析**：深入理解用户问题的语义和意图
2. **工具调用决策**
   - **retrieve_knowledge**：涉及具体条款时必须使用；每次 query 简短；**同一用户问题最多 {max_main_agent_retrievals} 次**
   - **delegate_subagents**：当用户一次提出 **{subagent_min_questions} 个及以上** 独立子问题（或单问但跨等待期/免责/理赔等多主题需分别检索）时使用；传入清晰子问题列表；工具返回为**已汇总短文**（含 [保单filename | 第X章/第X条] 引用），请据此作答，勿再逐条复述工具全文
   - 简单单一事实问（如「等待期几天」）→ 只用 retrieve_knowledge，不要 delegate_subagents
3. **草稿回答生成**：基于工具结果生成准确、有条理的回答；面向用户的引用须与工具中的章节/条款标识一致，不得编造条号

## 工作原则
- 回答必须基于知识库检索结果，不得编造条款内容
- 多维度条款题至少 2～3 次不同关键词检索；看到 ...(内容已截断) 必须换 query 再检。
- 引用条款时必须在正文中标注来源（格式如 [保单filename | 第X章/第X条] 或「依据某某条款第X条」）；**不要**在回答末尾追加免责声明或「来源提示」段落（界面会统一展示）
- 不确定时应明确告知，并建议用户查阅正式合同或联系客服
- 只回答用户最新一条问题，除非用户明确说「刚才/上次」，否则不要重复回答会话中已解决的历史问题
- 当用户未指定具体保单时，需要分析用户意图，确认用户想要咨询的保单，如果用户意图不明确，则首先需要询问用户想要咨询的保单，然后简要介绍所有保单的相关内容，如果部分保单不包含该问题相关内容，也应明确告知用户。

## 选择题与带选项问题
- 当用户问题含有选项（如 A/B/C/D、①②③、判断题「对/错」等）时，先完成推理，再**逐项对照题干**中的选项全文，确认哪一项与结论一致，再写出答案字母或选项编号。
- **禁止**将推理得到的数值/结论与错误的选项字母拼接（例如：题干 A=15%、B=30% 时，不得写「A.30%」；应写「B」或「B.30%」并说明 B 的表述为 30%）。
- 推荐格式：**答案：B**（或 正确/错误）；**理由：…**；必要时逐条简述为何排除 A/C/D。
- 若题干选项不完整、序号混乱或无法唯一对应，应明确说明，不要猜测选项字母。


## 长期记忆（Markdown，按用户持久化）
- **读取**：当用户引用既往信息、或问题依赖其个人投保背景且当前未写全时，调用 retrieve_from_memory
- **写入**：当用户明确要求记住或透露可跨会话复用的事实/偏好时，调用 record_to_memory
- **只记用户上下文**：身份、持有保单名、用户选过的计划/选项、沟通偏好；**不记**条款细节（等待期、比例、医院范围等应靠 retrieve_knowledge）
- **record_to_memory 示例**（每条一句、用户级事实）：
  - 「用户希望称呼为张先生」
  - 「用户持有守御一生医疗险，选择计划一」
  - 「用户偏好简洁回答」
- **不要写入**：条款解释、检索摘要、「用户曾误解…现已澄清」、整段问答

## 禁止事项
- 不得做出超出合同约定的承诺
- 不得提供虚假理赔信息
- 不得替代正式法律意见"""


SUBAGENT_PROMPT = """你是保险条款 SubAgent，只负责回答分配给你的**单一子问题**。

## 规则
- 必须调用 retrieve_knowledge 检索条款后再回答
- **最多调用 retrieve_knowledge {max_subagent_retrievals} 次**；每次 query 简短精准，禁止拼接超长 query
- doc_id 必须使用 `<policy_library>` 中的完整 filename
- 回答须含条款依据（章节号/条款号），不得编造
- 多维度条款题至少 2～3 次不同关键词检索；看到 ...(内容已截断) 必须换 query 再检。
- 若子问题含选项：先对照题干选项再写答案字母，禁止把正确结论贴到错误选项上。
- 只回答当前子问题，不要扩展其他话题"""


class InsuranceAgentFactory:
    """主 Agent 工厂（AgentScope ReAct Loop）。"""

    @staticmethod
    def _register_retrieve_tool(
        toolkit: Toolkit,
        knowledge: HybridKnowledge,
        *,
        max_retrievals: int,
        for_subagent: bool = False,
        allow_override: bool = False,
    ) -> None:
        role = "SubAgent 子问题" if for_subagent else "主对话"
        toolkit.register_tool_function(
            knowledge.retrieve_knowledge,
            func_description=(
                f"从保险条款知识库混合检索相关条款（{role}）。"
                f"最多调用 {max_retrievals} 次。"
                "query 应简短；doc_id 传 policy_library 中的完整 filename。"
            ),
            namesake_strategy="override" if allow_override else "raise",
        )

    @staticmethod
    def _register_delegate_tool(
        toolkit: Toolkit,
        settings: Settings,
        knowledge: HybridKnowledge,
        compliance: ComplianceValidator,
        policy_context_provider: Callable[[], str | None],
        *,
        allow_override: bool = False,
    ) -> None:
        if not settings.subagent_enabled:
            return
        delegation = SubAgentDelegation(
            settings=settings,
            knowledge=knowledge,
            compliance=compliance,
            policy_context_provider=policy_context_provider,
        )
        toolkit.register_tool_function(
            delegation.delegate_subagents,
            func_description=(
                "将 2 个及以上独立子问题委托给并行 SubAgent 分析，返回**简短汇总**"
                "（每条结论含 [保单filename | 第X章/第X条] 引用）。"
                f"参数 questions 为子问题列表（{settings.subagent_min_questions}～"
                f"{settings.max_subagents} 个）；user_intent 填用户本轮整体意图。"
                "单一简单问题请用 retrieve_knowledge，勿用本工具。"
            ),
            namesake_strategy="override" if allow_override else "raise",
        )

    @staticmethod
    def apply_main_agent_turn_config(
        agent: ReActAgent | InsuranceReActAgent,
        knowledge: HybridKnowledge,
        settings: Settings,
        compliance: ComplianceValidator,
        max_retrievals: int,
        policy_context_provider: Callable[[], str | None],
    ) -> None:
        """同步更新主 Agent 的 system prompt 与工具说明。"""
        agent._sys_prompt = SYSTEM_PROMPT.format(
            max_main_agent_retrievals=max_retrievals,
            subagent_min_questions=settings.subagent_min_questions,
        )
        InsuranceAgentFactory._register_retrieve_tool(
            agent.toolkit,
            knowledge,
            max_retrievals=max_retrievals,
            for_subagent=False,
            allow_override=True,
        )
        InsuranceAgentFactory._register_delegate_tool(
            agent.toolkit,
            settings,
            knowledge,
            compliance,
            policy_context_provider,
            allow_override=True,
        )

    @staticmethod
    def create(
        settings: Settings,
        knowledge: HybridKnowledge,
        memory_manager: MemoryManager,
        compliance: ComplianceValidator,
        policy_context_provider: Callable[[], str | None],
    ) -> InsuranceReActAgent:
        model = create_chat_model(settings, stream=True)

        toolkit = Toolkit()
        InsuranceAgentFactory._register_retrieve_tool(
            toolkit,
            knowledge,
            max_retrievals=settings.max_main_agent_retrievals,
        )
        InsuranceAgentFactory._register_delegate_tool(
            toolkit,
            settings,
            knowledge,
            compliance,
            policy_context_provider,
        )

        agent = InsuranceReActAgent(
            name=settings.agent_name,
            sys_prompt=SYSTEM_PROMPT.format(
                max_main_agent_retrievals=settings.max_main_agent_retrievals,
                subagent_min_questions=settings.subagent_min_questions,
            ),
            model=model,
            formatter=create_chat_formatter(settings),
            memory=InMemoryMemory(),
            toolkit=toolkit,
            long_term_memory=memory_manager.long_term,
            long_term_memory_mode="agent_control",
            compression_config=None,
        )
        agent._disable_console_output = True

        agent.register_instance_hook(
            hook_type="post_reply",
            hook_name="compliance_validation",
            hook=compliance.create_post_reply_hook(),
        )

        return agent

    @staticmethod
    def create_subagent(
        settings: Settings,
        knowledge: HybridKnowledge,
        compliance: ComplianceValidator,
        subagent_index: int,
        *,
        max_retrievals: int | None = None,
    ) -> ReActAgent:
        retrieval_limit = max_retrievals or settings.max_subagent_retrievals
        model = create_chat_model(settings, stream=False)
        toolkit = Toolkit()
        InsuranceAgentFactory._register_retrieve_tool(
            toolkit,
            knowledge,
            max_retrievals=retrieval_limit,
            for_subagent=True,
        )

        agent = ReActAgent(
            name=f"SubAgent-{subagent_index}",
            sys_prompt=SUBAGENT_PROMPT.format(
                max_subagent_retrievals=retrieval_limit,
            ),
            model=model,
            formatter=create_chat_formatter(settings),
            memory=InMemoryMemory(),
            toolkit=toolkit,
            long_term_memory=None,
            compression_config=None,
        )
        agent._disable_console_output = True
        agent.register_instance_hook(
            hook_type="post_reply",
            hook_name="compliance_validation",
            hook=compliance.create_post_reply_hook(),
        )
        return agent

