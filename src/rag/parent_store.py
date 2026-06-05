"""Parent chunk 侧车存储：小块索引、大块返回。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParentChunkEntry:
    doc_id: str
    chunk_id: int
    parent_id: str
    clause_label: str
    parent_text: str
    use_parent_return: bool = True


class ParentChunkStore:
    """按 (doc_id, chunk_id) 映射到 parent 条款全文。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, ParentChunkEntry] = {}
        self.load()

    @staticmethod
    def _key(doc_id: str, chunk_id: int) -> str:
        return f"{doc_id}::{chunk_id}"

    def load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._entries = {}
            return
        with self.path.open(encoding="utf-8") as file:
            raw: dict[str, dict[str, Any]] = json.load(file)
        self._entries = {}
        for key, value in raw.items():
            self._entries[key] = ParentChunkEntry(
                doc_id=value["doc_id"],
                chunk_id=value["chunk_id"],
                parent_id=value["parent_id"],
                clause_label=value["clause_label"],
                parent_text=value["parent_text"],
                use_parent_return=value.get("use_parent_return", True),
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: asdict(entry) for key, entry in self._entries.items()}
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def upsert_batch(self, entries: list[ParentChunkEntry]) -> None:
        for entry in entries:
            self._entries[self._key(entry.doc_id, entry.chunk_id)] = entry
        self.save()

    def get(self, doc_id: str, chunk_id: int) -> ParentChunkEntry | None:
        return self._entries.get(self._key(doc_id, chunk_id))

    def remove_doc(self, doc_id: str) -> None:
        keys = [key for key in self._entries if key.startswith(f"{doc_id}::")]
        for key in keys:
            del self._entries[key]
        self.save()

    def clear(self) -> None:
        self._entries = {}
        self.save()

    def display_text(self, doc_id: str, chunk_id: int, child_text: str) -> str:
        entry = self.get(doc_id, chunk_id)
        if entry is None or not entry.use_parent_return:
            return child_text
        return entry.parent_text

    def clause_label(self, doc_id: str, chunk_id: int) -> str:
        entry = self.get(doc_id, chunk_id)
        return entry.clause_label if entry else ""

    def should_use_parent_return(self, doc_id: str, chunk_id: int) -> bool:
        entry = self.get(doc_id, chunk_id)
        return entry.use_parent_return if entry else False
