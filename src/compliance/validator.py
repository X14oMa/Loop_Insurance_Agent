"""合规校验：约束 Agent 正文，来源提示与免责声明由前端展示。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentscope.agent import AgentBase
from agentscope.message import Msg, TextBlock

from src.config import Settings


@dataclass(frozen=True)
class ComplianceMeta:
    """供前端展示的合规 UI 标记（不写入 Agent 正文）。"""

    applies: bool
    """本轮回答是否涉及条款/规则类内容。"""
    show_source_notice: bool
    """是否在气泡下展示来源提示。"""
    has_source_citation: bool
    """正文中是否已含条款来源类表述。"""

    def to_dict(self) -> dict[str, bool]:
        return {
            "applies": self.applies,
            "show_source_notice": self.show_source_notice,
            "has_source_citation": self.has_source_citation,
        }


class ComplianceValidator:
    """合规校验器。

    - Agent 回答：须保留条款引用约束（system prompt），但不在正文追加免责声明/来源提示块
    - 前端：固定展示免责声明；涉及条款时在消息下展示来源提示
    """

    CLAUSE_KEYWORDS = (
        "条款", "保障", "理赔", "免责", "等待期", "免赔额",
        "保额", "保费", "赔付", "报销", "保险责任", "除外责任",
        "合同", "保单", "续保", "退保", "受益人",
    )

    SOURCE_PATTERN = re.compile(
        r"\[来源[:：].+?\]|\[条款来源[:：].+?\]|"
        r"\[[^\]]+\s*\|\s*第[^\]]+\]|"
        r"根据[^。\n]{0,80}?条款|依据[^。\n]{0,80}?规定",
    )

    _SOURCE_REMINDER_PATTERN = re.compile(
        r"\n+>\s*\*\*来源提示\*\*[\s\S]*$",
        re.IGNORECASE,
    )

    def __init__(self, settings: Settings) -> None:
        self.disclaimer = settings.disclaimer_text.strip()
        self.source_notice = settings.source_notice_text.strip()

    def needs_compliance(self, text: str) -> bool:
        """判断回答是否涉及保险条款/规则。"""
        return any(kw in text for kw in self.CLAUSE_KEYWORDS)

    def has_source_citation(self, text: str) -> bool:
        return bool(self.SOURCE_PATTERN.search(text))

    def strip_display_footers(self, text: str) -> str:
        """移除正文中误带的免责声明或来源提示块（历史逻辑/模型复述）。"""
        cleaned = text
        if self.disclaimer and self.disclaimer in cleaned:
            cleaned = cleaned.replace(self.disclaimer, "")
        if self.source_notice and self.source_notice in cleaned:
            cleaned = cleaned.replace(self.source_notice, "")
        cleaned = self._SOURCE_REMINDER_PATTERN.sub("", cleaned)
        return cleaned.strip()

    def process(self, response: Msg) -> tuple[Msg, ComplianceMeta]:
        """清洗 Agent 正文并生成本轮合规 UI 元数据。"""
        text = response.get_text_content() or ""
        if not text.strip():
            return response, ComplianceMeta(
                applies=False,
                show_source_notice=False,
                has_source_citation=False,
            )

        cleaned = self.strip_display_footers(text)
        applies = self.needs_compliance(cleaned)
        has_source = self.has_source_citation(cleaned) if applies else False

        meta = ComplianceMeta(
            applies=applies,
            show_source_notice=applies,
            has_source_citation=has_source,
        )

        if cleaned == text:
            return response, meta

        return Msg(
            name=response.name,
            content=[TextBlock(type="text", text=cleaned)],
            role=response.role,
            metadata=response.metadata,
        ), meta

    def validate(self, response: Msg) -> Msg:
        """向后兼容：仅返回清洗后的消息（Hook 使用）。"""
        processed, _ = self.process(response)
        return processed

    def ui_config(self) -> dict[str, str]:
        """前端固定展示的文案。"""
        return {
            "disclaimer": self.disclaimer,
            "source_notice": self.source_notice,
        }

    def create_post_reply_hook(self) -> Any:
        """post_reply：只清洗正文，不追加页脚类文字。"""

        def compliance_post_hook(
            self_agent: AgentBase,
            kwargs: dict[str, Any],
            output: Msg,
        ) -> Msg:
            del self_agent, kwargs
            return self.validate(output)

        return compliance_post_hook
