"""长期记忆写入：关键词硬触发 + LLM 语义归纳合并为 Markdown。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentscope.message import Msg
from agentscope.model import ChatModelBase

from src.model.chat_model_factory import create_chat_formatter

if TYPE_CHECKING:
    from src.config import Settings

_WRITE_TRIGGER = (
    "记住", "叫我", "我的名字", "我的保单", "保单号", "已选", "我投保",
    "我买", "计划一", "计划二", "既往", "我有", "不要啰嗦", "简洁",
    "请用", "称呼我", "我是", "今年", "岁", "性别", "记下来", "别忘了",
)

_MERGE_PROMPT = """你是长期记忆维护助手。根据「已有 Markdown 记忆」与「本轮对话」，输出更新后的完整 Markdown 文档。

## 应存什么（用户上下文，非条款知识库）
- 用户自称信息（称呼、年龄、性别等，仅用户明确透露的）
- 投保上下文（持有哪几份保单、用户选过的计划/选项/免赔额等**用户侧选择**）
- 跨会话复用的咨询偏好（如「回答要简洁」「用您称呼」）
- 合并时去重、删矛盾、去掉已失效项；**不要**保留「用户曾误解…现已澄清」类对话过程

## 不应存什么
- 条款参数与规则细节（等待期天数、给付比例、医院范围、领取规则等——这些应靠检索，不是用户画像）
- 检索结果、SubAgent 过程、助手完整回答、一次性考题全文

## 正确示例（篇幅与粒度请参考，勿机械复制内容）
```markdown
# 长期记忆

## 用户身份
- 称呼：张先生
- 年龄：42 岁（用户自报）

## 投保上下文
- 持有：同方全球「守御一生」医疗险、「臻宝贝2026」年金
- 守御一生：用户投保时选择计划一、一般医疗 1 万元免赔额

## 咨询偏好
- 希望回答简洁，少重复条款原文
```

## 输出要求
- 保留仍有效的旧事实，修正矛盾，整体保持精炼（小节 + 短列表即可）
- 若无任何值得长期保存的新增或变更，只输出：NONE"""


def should_write_ltm(user_text: str) -> bool:
    """关键词硬识别：是否值得尝试合并写入长期记忆。"""
    text = user_text.strip()
    if not text:
        return False
    if len(text) > 600:
        return False
    if any(marker in text for marker in _WRITE_TRIGGER):
        return True
    q_marks = text.count("？") + text.count("?")
    if q_marks >= 4:
        return False
    return len(text) <= 120


def _parse_merge_output(raw: str) -> str | None:
    text = raw.strip()
    if not text or text.upper() == "NONE":
        return None
    return text


async def merge_ltm_update(
    user_text: str,
    assistant_text: str,
    existing_markdown: str,
    *,
    model: ChatModelBase,
    settings: Settings,
) -> str | None:
    """关键词命中后，由 LLM 将本轮信息语义归纳进 Markdown；无更新则返回 None。"""
    if not should_write_ltm(user_text):
        return None

    formatter = create_chat_formatter(settings)
    user_blob = user_text.strip()[:2000]
    assistant_blob = (assistant_text or "").strip()[:800]
    existing_blob = (existing_markdown or "（空）").strip()[:4000]

    messages = await formatter.format(
        [
            Msg("system", _MERGE_PROMPT, "system"),
            Msg(
                "user",
                f"【已有记忆】\n{existing_blob}\n\n"
                f"【用户问题】\n{user_blob}\n\n"
                f"【助手回答摘要】\n{assistant_blob or '（无）'}",
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
    return _parse_merge_output("".join(parts))
