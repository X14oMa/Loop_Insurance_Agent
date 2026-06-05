"""混合检索知识库：语义 + BM25 + RRF，支持 doc 过滤、parent 返回与 rerank。"""

from __future__ import annotations

from typing import Any

from agentscope.message import TextBlock
from agentscope.rag import Document, SimpleKnowledge
from agentscope.tool import ToolResponse
from rank_bm25 import BM25Okapi

from src.rag.parent_store import ParentChunkStore
from src.rag.reranker import rerank_documents
from src.rag.retrieval_config import RetrievalConfig
from src.rag.tokenizer import tokenize
from src.rag.tool_response_limits import ToolResponseLimits, build_tool_response_texts


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridKnowledge(SimpleKnowledge):
    """混合检索知识库。"""

    def __init__(
        self,
        *args: Any,
        parent_store: ParentChunkStore | None = None,
        retrieval_config: RetrievalConfig | None = None,
        tool_response_limits: ToolResponseLimits | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.parent_store = parent_store
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.tool_response_limits = tool_response_limits or ToolResponseLimits()
        self._documents: list[Document] = []
        self._doc_index: dict[str, Document] = {}
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    def set_parent_store(self, parent_store: ParentChunkStore) -> None:
        self.parent_store = parent_store

    async def add_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> None:
        await super().add_documents(documents, **kwargs)
        self._documents.extend(documents)
        for doc in documents:
            self._doc_index[doc.id] = doc
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        self._corpus_tokens = [
            tokenize(doc.metadata.content["text"]) for doc in self._documents
        ]
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def reset_index(self) -> None:
        self._documents = []
        self._doc_index = {}
        self._bm25 = None
        self._corpus_tokens = []

    async def restore_bm25_from_vector_store(self) -> int:
        client = self.embedding_store.get_client()
        collection = self.embedding_store.collection_name

        if not await client.collection_exists(collection):
            self.reset_index()
            return 0

        from agentscope.rag._document import DocMetadata

        documents: list[Document] = []
        offset = None
        while True:
            records, offset = await client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in records:
                documents.append(
                    Document(
                        id=str(point.id),
                        metadata=DocMetadata(**point.payload),
                    ),
                )
            if offset is None:
                break

        self._documents = documents
        self._doc_index = {doc.id: doc for doc in documents}
        self._rebuild_bm25_index()
        return len(documents)

    async def delete_by_doc_id(self, doc_id: str) -> None:
        from qdrant_client import models

        client = self.embedding_store.get_client()
        collection = self.embedding_store.collection_name

        if await client.collection_exists(collection):
            await client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id),
                            ),
                        ],
                    ),
                ),
            )

        self._documents = [
            doc for doc in self._documents if doc.metadata.doc_id != doc_id
        ]
        self._doc_index = {doc.id: doc for doc in self._documents}
        self._rebuild_bm25_index()
        if self.parent_store:
            self.parent_store.remove_doc(doc_id)

    async def clear_vector_store(self) -> None:
        client = self.embedding_store.get_client()
        collection = self.embedding_store.collection_name
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
        self.reset_index()
        if self.parent_store:
            self.parent_store.clear()

    def _build_qdrant_filter(self, doc_id: str | None) -> Any:
        if not doc_id:
            return None
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                ),
            ],
        )

    async def _semantic_search(
        self,
        query: str,
        limit: int,
        score_threshold: float | None,
        doc_id: str | None,
    ) -> list[Document]:
        res_embedding = await self.embedding_model(
            [TextBlock(type="text", text=query)],
        )
        query_filter = self._build_qdrant_filter(doc_id)
        kwargs: dict[str, Any] = {}
        if query_filter is not None:
            kwargs["query_filter"] = query_filter

        return await self.embedding_store.search(
            res_embedding.embeddings[0],
            limit=limit,
            score_threshold=score_threshold,
            **kwargs,
        )

    def _keyword_search(
        self,
        query: str,
        limit: int,
        doc_id: str | None = None,
    ) -> list[Document]:
        if not self._bm25 or not self._documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidate_indices = [
            idx
            for idx, doc in enumerate(self._documents)
            if doc_id is None or doc.metadata.doc_id == doc_id
        ]
        if not candidate_indices:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            candidate_indices,
            key=lambda i: scores[i],
            reverse=True,
        )[:limit]

        results: list[Document] = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            doc = self._documents[idx]
            doc.score = float(scores[idx])
            results.append(doc)
        return results

    def get_display_text(
        self,
        doc: Document,
        config: RetrievalConfig | None = None,
    ) -> str:
        cfg = config or self.retrieval_config
        child_text = doc.metadata.content["text"]
        if not cfg.use_parent_return or not self.parent_store:
            return child_text
        return self.parent_store.display_text(
            doc.metadata.doc_id,
            doc.metadata.chunk_id,
            child_text,
        )

    def get_clause_label(
        self,
        doc: Document,
        config: RetrievalConfig | None = None,
    ) -> str:
        if not self.parent_store:
            return ""
        return self.parent_store.clause_label(doc.metadata.doc_id, doc.metadata.chunk_id)

    def _parent_dedupe_key(self, doc: Document) -> str:
        if not self.parent_store:
            return doc.id
        entry = self.parent_store.get(doc.metadata.doc_id, doc.metadata.chunk_id)
        if entry is None or not entry.use_parent_return:
            return doc.id
        return entry.parent_id

    def _dedupe_by_parent(
        self,
        docs: list[Document],
        limit: int,
        config: RetrievalConfig,
    ) -> list[Document]:
        if not config.use_parent_return or not self.parent_store:
            return docs[:limit]

        seen_parents: set[str] = set()
        deduped: list[Document] = []
        for doc in docs:
            parent_key = self._parent_dedupe_key(doc)
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)
            deduped.append(doc)
            if len(deduped) >= limit:
                break
        return deduped

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        doc_id: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        del kwargs
        cfg = self.retrieval_config
        filter_doc_id = doc_id if cfg.use_doc_id_filter else None
        pool_size = max(limit * cfg.recall_multiplier, limit * 2)

        semantic_docs = await self._semantic_search(
            query=query,
            limit=pool_size,
            score_threshold=score_threshold,
            doc_id=filter_doc_id,
        )
        keyword_docs = self._keyword_search(
            query=query,
            limit=pool_size,
            doc_id=filter_doc_id,
        )

        semantic_scores = {doc.id: float(doc.score or 0.0) for doc in semantic_docs}
        bm25_scores = {doc.id: float(doc.score or 0.0) for doc in keyword_docs}

        semantic_ranked = [doc.id for doc in semantic_docs]
        keyword_ranked = [doc.id for doc in keyword_docs]
        fused_scores = _reciprocal_rank_fusion([semantic_ranked, keyword_ranked])

        all_docs = {doc.id: doc for doc in semantic_docs + keyword_docs}
        candidate_ids = sorted(
            fused_scores.keys(),
            key=lambda item: fused_scores[item],
            reverse=True,
        )

        candidates = [all_docs[item] for item in candidate_ids if item in all_docs]
        display_texts = {
            doc.id: self.get_display_text(doc, cfg) for doc in candidates
        }

        if cfg.use_rerank:
            candidates = rerank_documents(
                query=query,
                documents=candidates,
                rrf_scores=fused_scores,
                semantic_scores=semantic_scores,
                bm25_scores=bm25_scores,
                display_texts=display_texts,
            )
        else:
            for doc in candidates:
                doc.score = fused_scores.get(doc.id, 0.0)

        return self._dedupe_by_parent(candidates, limit, cfg)

    async def retrieve_knowledge(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        doc_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResponse:
        del kwargs
        if not query or not str(query).strip():
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="错误：query 不能为空。请提供具体检索关键词。",
                    ),
                ],
            )

        effective_limit = min(limit, self.tool_response_limits.max_chunks)
        docs = await self.retrieve(
            query=str(query).strip(),
            limit=effective_limit,
            score_threshold=score_threshold,
            doc_id=doc_id,
        )

        if not docs:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="未检索到相关条款。请尝试换用更具体的关键词，"
                        "或指定 doc_id 过滤到具体保单。",
                    ),
                ],
            )

        raw_blocks: list[str] = []
        for doc in docs:
            label = self.get_clause_label(doc)
            label_prefix = f"{label} | " if label else ""
            display = self.get_display_text(doc)
            raw_blocks.append(
                f"[来源: {doc.metadata.doc_id} | "
                f"{label_prefix}"
                f"相关度: {doc.score:.4f}]\n"
                f"{display}",
            )

        trimmed_blocks = build_tool_response_texts(raw_blocks, self.tool_response_limits)

        return ToolResponse(
            content=[TextBlock(type="text", text=block) for block in trimmed_blocks],
        )
