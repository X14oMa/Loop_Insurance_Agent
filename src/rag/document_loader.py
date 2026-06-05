"""PDF 内存解析与按条款结构分块。"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from agentscope.rag import Document

from src.rag.clause_chunking import chunk_text_by_clauses
from src.rag.parent_store import ParentChunkEntry


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^\w\u4e00-\u9fff.\-]", "_", name)
    if not name.lower().endswith(".pdf"):
        name = f"{Path(name).stem}.pdf"
    return name or "policy.pdf"


async def load_pdf_bytes(
    content: bytes,
    filename: str,
    child_size: int = 256,
    child_overlap: int = 64,
) -> tuple[list[Document], list[ParentChunkEntry]]:
    """从 PDF 字节流解析，按条款切分 parent/child，不写入磁盘。"""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("请安装 pypdf：pip install pypdf") from e

    safe_name = _sanitize_filename(filename)
    reader = PdfReader(BytesIO(content))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages_text).strip()

    if not full_text:
        return [], []

    return chunk_text_by_clauses(
        full_text,
        safe_name,
        child_size=child_size,
        child_overlap=child_overlap,
    )
