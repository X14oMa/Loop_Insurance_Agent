"""保单元数据清单（仅存于向量库目录，不保存 PDF 源文件）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PolicyManifest:
    """记录已入库保单的元数据，与 Qdrant payload 中的 doc_id 对应。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            return
        with self.path.open(encoding="utf-8") as file:
            self._entries = json.load(file)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self._entries, file, ensure_ascii=False, indent=2)

    def add(
        self,
        filename: str,
        size_kb: float,
        chunks: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._entries[filename] = {
            "filename": filename,
            "size_kb": size_kb,
            "chunks": chunks,
            "uploaded_at": now,
        }
        self.save()

    def remove(self, filename: str) -> None:
        self._entries.pop(filename, None)
        self.save()

    def clear(self) -> None:
        self._entries = {}
        self.save()

    def list_all(self) -> list[dict[str, Any]]:
        return sorted(
            self._entries.values(),
            key=lambda item: item.get("uploaded_at", ""),
            reverse=True,
        )

    def get(self, filename: str) -> dict[str, Any] | None:
        return self._entries.get(filename)
