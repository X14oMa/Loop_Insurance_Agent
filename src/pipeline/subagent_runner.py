"""SubAgent 并行执行与简短汇总（供主 Agent delegate_subagents 工具）。"""

from __future__ import annotations

import asyncio

from agentscope.message import Msg

from src.compliance import ComplianceValidator
from src.config import Settings
from src.model.chat_model_factory import create_chat_formatter, create_chat_model
from src.rag import HybridKnowledge

SYNTHESIZER_BRIEF_PROMPT = """你是保险条款分析汇总器。根据各子问题的分析结果，输出供主 Agent 使用的**简短中文汇总**。

硬性要求：
1. 总篇幅不超过 {max_chars} 字（含引用标注）
2. 只保留与用户整体意图相关的核心结论，不重复、不啰嗦
3. **每条**结论末尾必须附带引用，格式严格为：[保单filename | 章节/条款标识]
   - 章节/条款标识必须来自子分析中已出现的「第X章」「第X条」「第X节」等文字，不得编造
   - 若子分析未给出条号，写 [保单filename | 依据子分析] 并说明不确定
4. 按子问题顺序组织；单主题可合并为一段
5. 某子问题信息不足时，单独一句标明
6. 禁止提及 SubAgent、工具、检索过程；禁止 Markdown 大标题"""


async def run_subagents(
    questions: list[str],
    *,
    settings: Settings,
    knowledge: HybridKnowledge,
    compliance: ComplianceValidator,
    policy_context: str | None,
    max_retrievals: int | None = None,
) -> list[tuple[int, str, str]]:
    """并行运行 SubAgent，返回 (序号, 子问题, 回答)。"""
    sem = asyncio.Semaphore(settings.max_subagent_concurrency)

    async def _run_one(index: int, question: str) -> tuple[int, str, str]:
        from src.agent.insurance_agent import InsuranceAgentFactory

        async with sem:
            agent = InsuranceAgentFactory.create_subagent(
                settings=settings,
                knowledge=knowledge,
                compliance=compliance,
                subagent_index=index + 1,
                max_retrievals=max_retrievals,
            )
            sections: list[str] = []
            if policy_context:
                sections.append(policy_context)
            sections.append(f"子问题 {index + 1}：{question}")
            user_msg = Msg("User", "\n\n".join(sections), "user")
            response = await agent(user_msg)
            return index, question, response.get_text_content() or ""

    tasks = [_run_one(i, q) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda item: item[0])


def _format_sub_results_for_synthesis(
    sub_results: list[tuple[int, str, str]],
    *,
    per_answer_cap: int = 2500,
) -> str:
    blocks: list[str] = []
    for index, question, answer in sub_results:
        text = answer if len(answer) <= per_answer_cap else answer[: per_answer_cap] + "…"
        blocks.append(
            f"### 子问题 {index + 1}\n问题：{question}\n\n分析：\n{text}\n",
        )
    return "\n".join(blocks)


async def synthesize_subagent_brief(
    questions: list[str],
    sub_results: list[tuple[int, str, str]],
    *,
    user_intent: str,
    settings: Settings,
    max_chars: int,
) -> str:
    """将 SubAgent 完整分析压缩为带引用标注的短文。"""
    model = create_chat_model(settings, stream=False)
    formatter = create_chat_formatter(settings)
    body = _format_sub_results_for_synthesis(sub_results)
    prompt = SYNTHESIZER_BRIEF_PROMPT.format(max_chars=max_chars)
    messages = await formatter.format(
        [
            Msg("system", prompt, "system"),
            Msg(
                "user",
                f"【用户整体意图】\n{user_intent}\n\n"
                f"【子问题列表】\n"
                + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
                + f"\n\n【各子问题分析】\n{body}",
                "user",
            ),
        ],
    )
    response = await model(messages)
    parts: list[str] = []
    if response.content:
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return "".join(parts).strip() or "（汇总失败：未能生成简短结论，请主 Agent 自行检索。）"