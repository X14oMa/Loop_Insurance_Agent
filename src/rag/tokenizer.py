"""RAG 中英文混合分词。"""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """中文按字/双字切分，英文按单词切分。"""
    tokens: list[str] = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        if re.match(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(list(segment))
            if len(segment) >= 2:
                for i in range(len(segment) - 1):
                    tokens.append(segment[i : i + 2])
        else:
            tokens.append(segment)
    return tokens
