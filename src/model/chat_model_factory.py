"""聊天模型工厂：按 LLM_PROVIDER 创建 DashScope 或 DeepSeek（OpenAI 兼容）模型。"""

from __future__ import annotations

from agentscope.formatter import DashScopeChatFormatter, FormatterBase
from agentscope.model import ChatModelBase, OpenAIChatModel

from src.config import Settings
from src.model.dashscope_factory import create_dashscope_chat_model
from src.model.thinking_aware_formatter import ThinkingAwareOpenAIFormatter

LLM_PROVIDER_DEEPSEEK = "deepseek"
LLM_PROVIDER_DASHSCOPE = "dashscope"

_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


def normalize_llm_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in {LLM_PROVIDER_DEEPSEEK, LLM_PROVIDER_DASHSCOPE}:
        return value
    raise ValueError(
        f"不支持的 LLM_PROVIDER={provider!r}，"
        f"请使用 {LLM_PROVIDER_DEEPSEEK} 或 {LLM_PROVIDER_DASHSCOPE}",
    )


def create_chat_formatter(settings: Settings) -> FormatterBase:
    """与当前 LLM 提供商匹配的对话 Formatter。"""
    if settings.llm_provider == LLM_PROVIDER_DEEPSEEK:
        return ThinkingAwareOpenAIFormatter()
    return DashScopeChatFormatter()


def create_chat_model(
    settings: Settings,
    *,
    model_name: str | None = None,
    stream: bool = True,
    multimodality: bool | None = None,
) -> ChatModelBase:
    """创建主/子 Agent、路由、压缩等使用的聊天模型。"""
    if settings.llm_provider == LLM_PROVIDER_DEEPSEEK:
        resolved = (model_name or settings.model_name).strip()
        return OpenAIChatModel(
            model_name=resolved,
            api_key=settings.deepseek_api_key,
            stream=stream,
            client_kwargs={"base_url": settings.deepseek_base_url},
        )

    return create_dashscope_chat_model(
        settings,
        model_name=model_name,
        stream=stream,
        multimodality=multimodality,
    )
