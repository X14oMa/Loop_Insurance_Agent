"""DashScope 模型创建：自动选择文本/多模态 API 与区域 base URL。"""

from __future__ import annotations

from agentscope.model import DashScopeChatModel

from src.config import Settings

# 常见误写 → 推荐可用 ID（百炼 Model Studio）
_MODEL_ALIASES: dict[str, str] = {
    "qwen3.7-plus": "qwen3.6-plus",
    "qwen3.7-plus-latest": "qwen3.6-plus",
}


def resolve_model_name(model_name: str) -> str:
    key = model_name.strip()
    return _MODEL_ALIASES.get(key.lower(), key)


def infer_multimodality(model_name: str) -> bool:
    """判断模型是否应走 MultiModalConversation API。"""
    name = resolve_model_name(model_name).lower()
    if name.startswith("qvq") or "-vl" in name:
        return True
    # qwen3.x-plus 系列需多模态接口（纯文本输入也可用）
    if name.startswith("qwen3") and "plus" in name:
        return True
    return False


def create_dashscope_chat_model(
    settings: Settings,
    *,
    model_name: str | None = None,
    stream: bool = True,
    multimodality: bool | None = None,
) -> DashScopeChatModel:
    """创建 DashScopeChatModel，避免模型名/端点不匹配导致 url error。"""
    resolved = resolve_model_name(model_name or settings.model_name)
    kwargs: dict = {
        "model_name": resolved,
        "api_key": settings.dashscope_api_key,
        "stream": stream,
    }
    if settings.dashscope_base_url:
        kwargs["base_http_api_url"] = settings.dashscope_base_url

    use_multimodal = (
        multimodality
        if multimodality is not None
        else infer_multimodality(resolved)
    )
    if use_multimodal:
        kwargs["multimodality"] = True

    return DashScopeChatModel(**kwargs)
