"""模型上下文窗口（token）注册表。"""

from __future__ import annotations

import os

# 常见模型默认上下文（token）。可通过 MODEL_CONTEXT_WINDOW 覆盖。
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v4-flash": 128_000,
    "deepseek-v3": 64_000,
    "qwen-plus": 131_072,
    "qwen-turbo": 131_072,
    "qwen-max": 131_072,
    "qwen3.6-plus": 131_072,
    "qwen3.5-plus": 131_072,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}

_DEFAULT_CONTEXT_WINDOW = 64_000


def resolve_context_window(
    model_name: str,
    *,
    override: int | None = None,
) -> int:
    """解析模型上下文窗口大小（token）。"""
    if override and override > 0:
        return override

    env_override = os.getenv("MODEL_CONTEXT_WINDOW", "").strip()
    if env_override.isdigit():
        return int(env_override)

    key = model_name.strip().lower()
    if key in _MODEL_CONTEXT_WINDOWS:
        return _MODEL_CONTEXT_WINDOWS[key]

    for prefix, size in (
        ("deepseek", 64_000),
        ("qwen", 131_072),
        ("gpt-", 128_000),
    ):
        if key.startswith(prefix):
            return size

    return _DEFAULT_CONTEXT_WINDOW


def compute_thresholds(
    context_window: int,
    *,
    soft_ratio: float = 0.8,
    hard_ratio: float = 0.93,
    hard_reserve_tokens: int = 13_000,
) -> tuple[int, int]:
    """返回 (soft_threshold, hard_threshold)，hard 取 ratio 与 reserve 的较大值。"""
    soft = max(1, int(context_window * soft_ratio))
    hard = max(
        max(1, int(context_window * hard_ratio)),
        context_window - hard_reserve_tokens,
    )
    return soft, hard
