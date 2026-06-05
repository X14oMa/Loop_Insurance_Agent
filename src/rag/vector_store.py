"""Windows 兼容的 Qdrant 向量存储。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from agentscope.rag import QdrantStore
from agentscope.rag._store._qdrant_store import QdrantStore as QdrantStoreBase

from src.config import Settings

logger = logging.getLogger(__name__)

META_INFO_FILENAME = "meta.json"
LOCK_FILENAME = ".lock"

_DEPRECATED_COLLECTION_FIELDS = frozenset({"metadata"})


def repair_qdrant_meta(store_path: str | Path) -> None:
    """修复 Qdrant 本地存储 meta.json 中的废弃字段。"""
    meta_path = Path(store_path) / META_INFO_FILENAME
    if not meta_path.exists():
        return

    with meta_path.open(encoding="utf-8") as file:
        meta = json.load(file)

    changed = False
    for config in meta.get("collections", {}).values():
        for field in _DEPRECATED_COLLECTION_FIELDS:
            if field in config:
                del config[field]
                changed = True

    if changed:
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False)


def release_stale_qdrant_lock(store_path: str | Path) -> bool:
    """尝试释放因异常退出遗留的 Qdrant 锁文件。"""
    lock_path = Path(store_path) / LOCK_FILENAME
    if not lock_path.exists():
        return False

    try:
        import portalocker

        with lock_path.open("a+b") as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
            portalocker.unlock(lock_file)
        lock_path.unlink(missing_ok=True)
        logger.info("已清理遗留的 Qdrant 锁文件: %s", lock_path)
        return True
    except Exception:
        return False


class LocalQdrantStore(QdrantStoreBase):
    """使用 path 参数的本地 Qdrant 存储。"""

    def __init__(
        self,
        store_path: str,
        collection_name: str,
        dimensions: int,
        distance: Literal["Cosine", "Euclid", "Dot", "Manhattan"] = "Cosine",
        client_kwargs: dict[str, Any] | None = None,
        collection_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as e:
            raise ImportError(
                "Qdrant client is not installed. Run: pip install 'agentscope[rag]'",
            ) from e

        normalized_path = store_path.replace("\\", "/")
        repair_qdrant_meta(normalized_path)

        client_kwargs = client_kwargs or {}
        self._client = AsyncQdrantClient(path=normalized_path, **client_kwargs)

        self.collection_name = collection_name
        self.dimensions = dimensions
        self.distance = distance
        self.collection_kwargs = collection_kwargs or {}


def create_vector_store(settings: Settings) -> QdrantStoreBase:
    """创建持久化向量存储。条款文本与元数据保存在 Qdrant payload 中。"""
    collection_kwargs = {
        "collection_name": "insurance_clauses",
        "dimensions": settings.embedding_dimensions,
    }

    if settings.vector_store_persist:
        store_path = str(settings.vector_store_path).replace("\\", "/")
        repair_qdrant_meta(store_path)
        try:
            return LocalQdrantStore(store_path=store_path, **collection_kwargs)
        except RuntimeError as exc:
            if "already accessed by another instance" not in str(exc):
                raise
            if release_stale_qdrant_lock(settings.vector_store_path):
                return LocalQdrantStore(store_path=store_path, **collection_kwargs)
            logger.warning(
                "向量库目录被其他进程占用，已自动切换为内存模式。"
                "如需持久化，请先关闭其他 python -m src.web_main 进程。",
            )

    return QdrantStore(location=":memory:", **collection_kwargs)
