"""向后兼容：问题拆分提示（主 Agent 自行调用 delegate_subagents）。"""

from __future__ import annotations

from src.pipeline.agent_router import _heuristic_split


def split_user_questions(user_input: str) -> list[str]:
    """启发式拆句，供测试或外部脚本使用。"""
    parts = _heuristic_split(user_input)
    return parts if parts else ([user_input.strip()] if user_input.strip() else [])
