"""retrieve_knowledge 工具返回长度限制。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResponseLimits:
    max_chunks: int = 3
    max_chunk_chars: int = 2000
    max_total_chars: int = 8000


def truncate_with_notice(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...(内容已截断)"


def build_tool_response_texts(
    blocks: list[str],
    limits: ToolResponseLimits,
) -> list[str]:
    """按块数、单块长度、总长度截断工具返回。"""
    trimmed: list[str] = []
    total = 0
    for raw in blocks[: limits.max_chunks]:
        piece = truncate_with_notice(raw, limits.max_chunk_chars)
        if total + len(piece) > limits.max_total_chars:
            remaining = limits.max_total_chars - total
            if remaining <= 0:
                break
            piece = truncate_with_notice(piece, remaining)
        trimmed.append(piece)
        total += len(piece)
        if total >= limits.max_total_chars:
            break
    return trimmed
