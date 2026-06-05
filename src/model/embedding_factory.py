"""向量嵌入模型工厂：按 EMBEDDING_PROVIDER 从 .env 创建 Embedding。"""

from __future__ import annotations

from agentscope.embedding import (
    DashScopeTextEmbedding,
    EmbeddingModelBase,
    OpenAITextEmbedding,
)

from src.config import Settings

EMBEDDING_PROVIDER_DASHSCOPE = "dashscope"
EMBEDDING_PROVIDER_OPENAI = "openai"


def create_embedding_model(settings: Settings) -> EmbeddingModelBase:
    """创建 RAG 使用的文本嵌入模型。

    - dashscope: 通义 text-embedding-v4 等（默认）
    - openai: OpenAI 兼容 API（如 DeepSeek Embedding、OpenAI）
    """
    provider = settings.embedding_provider
    if provider == EMBEDDING_PROVIDER_DASHSCOPE:
        api_key = settings.embedding_api_key or settings.dashscope_api_key
        if not api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=dashscope 时需配置 "
                "DASHSCOPE_API_KEY 或 EMBEDDING_API_KEY。",
            )
        return DashScopeTextEmbedding(
            api_key=api_key,
            model_name=settings.embedding_model_name,
            dimensions=settings.embedding_dimensions,
        )

    if provider == EMBEDDING_PROVIDER_OPENAI:
        api_key = (
            settings.embedding_api_key
            or settings.deepseek_api_key
            or settings.dashscope_api_key
        )
        if not api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai 时需配置 EMBEDDING_API_KEY、"
                "DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY 之一。",
            )
        client_kwargs: dict = {}
        if settings.embedding_base_url:
            client_kwargs["base_url"] = settings.embedding_base_url
        return OpenAITextEmbedding(
            api_key=api_key,
            model_name=settings.embedding_model_name,
            dimensions=settings.embedding_dimensions,
            **client_kwargs,
        )

    raise ValueError(
        f"不支持的 EMBEDDING_PROVIDER={settings.embedding_provider!r}，"
        f"请使用 {EMBEDDING_PROVIDER_DASHSCOPE} 或 {EMBEDDING_PROVIDER_OPENAI}。",
    )
