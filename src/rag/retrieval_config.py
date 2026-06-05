"""RAG 检索策略配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    """RAG 检索行为配置（默认全部开启）。"""

    use_doc_id_filter: bool = True
    use_parent_return: bool = True
    use_rerank: bool = True
    recall_multiplier: int = 4
    """混合检索候选池 = limit * recall_multiplier。"""
