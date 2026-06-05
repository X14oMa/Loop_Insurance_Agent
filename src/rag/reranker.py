"""轻量重排序：融合 RRF、语义分、BM25 分与词面重叠。"""

from __future__ import annotations

from agentscope.rag import Document

from src.rag.tokenizer import tokenize


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    doc_tokens = set(tokenize(text))
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}


def rerank_documents(
    query: str,
    documents: list[Document],
    rrf_scores: dict[str, float],
    semantic_scores: dict[str, float],
    bm25_scores: dict[str, float],
    display_texts: dict[str, str],
    *,
    w_rrf: float = 0.35,
    w_semantic: float = 0.35,
    w_bm25: float = 0.15,
    w_lexical: float = 0.15,
) -> list[Document]:
    """对候选文档重排序并写回 score。"""
    if not documents:
        return []

    norm_rrf = _normalize(rrf_scores)
    norm_sem = _normalize(semantic_scores)
    norm_bm25 = _normalize(bm25_scores)

    scored: list[tuple[float, Document]] = []
    for doc in documents:
        text = display_texts.get(doc.id, doc.metadata.content["text"])
        lexical = _lexical_overlap(query, text)
        final_score = (
            w_rrf * norm_rrf.get(doc.id, 0.0)
            + w_semantic * norm_sem.get(doc.id, 0.0)
            + w_bm25 * norm_bm25.get(doc.id, 0.0)
            + w_lexical * lexical
        )
        doc.score = final_score
        scored.append((final_score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored]
