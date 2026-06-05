"""按保险条款/章节结构切分，并生成 parent/child 双层 chunk。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from agentscope.message import TextBlock
from agentscope.rag._document import DocMetadata
from agentscope.rag import Document

from src.rag.parent_store import ParentChunkEntry

_CN_NUM = r"[零一二三四五六七八九十百\d]+"

# 多种章节标题模式（优先级：条 > 章/节 > 数字编号 > 中文序号）
SECTION_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str], int]] = [
    (
        re.compile(rf"(?:(?<=\n)|^)(第{_CN_NUM}条)\s*"),
        lambda m: m.group(1),
        100,
    ),
    (
        re.compile(rf"(?:(?<=\n)|^)(第{_CN_NUM}[章节])\s*"),
        lambda m: m.group(1),
        90,
    ),
    (
        re.compile(r"(?:(?<=\n)|^)(\d+\.\d+(?:\.\d+)?)(?:[ \t]+|\s*)(?=\S)"),
        lambda m: m.group(1),
        80,
    ),
    (
        re.compile(r"(?:(?<=\n)|^)([一二三四五六七八九十百]+[、．.])\s*"),
        lambda m: m.group(1).rstrip("、．."),
        70,
    ),
]

TOC_LINE_PATTERN = re.compile(r"\.{4,}|…{2,}")


@dataclass(frozen=True)
class SectionBlock:
    label: str
    text: str
    use_parent_return: bool


def _is_toc_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if TOC_LINE_PATTERN.search(stripped):
        return True
    # 形如 "2.8 .............. 12"
    return bool(re.match(r"^\d+(?:\.\d+)+\s+[\.·\s]{6,}\d+\s*$", stripped))


def _match_line_start(text: str, position: int) -> str:
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def _choose_section_pattern(text: str) -> tuple[list[re.Match[str]], Callable[[re.Match[str]], str]] | None:
    best: tuple[int, int, list[re.Match[str]], Callable[[re.Match[str]], str]] | None = None

    for pattern, label_fn, priority in SECTION_PATTERNS:
        matches = [
            match
            for match in pattern.finditer(text)
            if not _is_toc_line(_match_line_start(text, match.start()))
        ]
        if len(matches) < 2:
            continue
        score = (priority, len(matches))
        if best is None or score > (best[0], best[1]):
            best = (priority, len(matches), matches, label_fn)

    if best is None:
        return None
    return best[2], best[3]


def _split_into_sections(text: str) -> list[SectionBlock]:
    """将全文切分为章节块，并标记是否启用 parent 返回。"""
    stripped = text.strip()
    if not stripped:
        return []

    chosen = _choose_section_pattern(stripped)
    if chosen is None:
        # 无法识别结构：仅索引 child，检索时不展开 parent
        return [SectionBlock(label="全文", text=stripped, use_parent_return=False)]

    matches, label_fn = chosen
    blocks: list[SectionBlock] = []

    if matches[0].start() > 0:
        preamble = stripped[: matches[0].start()].strip()
        if preamble:
            blocks.append(
                SectionBlock(
                    label="前言",
                    text=preamble,
                    use_parent_return=False,
                ),
            )

    for index, match in enumerate(matches):
        label = label_fn(match)
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        body = stripped[start:end].strip()
        if not body:
            continue
        blocks.append(
            SectionBlock(
                label=label,
                text=body,
                use_parent_return=True,
            ),
        )

    return blocks or [SectionBlock(label="全文", text=stripped, use_parent_return=False)]


def _split_child_chunks(
    text: str,
    child_size: int,
    overlap: int,
) -> list[str]:
    if len(text) <= child_size:
        return [text]

    chunks: list[str] = []
    step = max(child_size - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start : start + child_size]
        if piece.strip():
            chunks.append(piece)
        if start + child_size >= len(text):
            break
    return chunks


def chunk_text_by_clauses(
    text: str,
    doc_id: str,
    child_size: int = 256,
    child_overlap: int = 64,
) -> tuple[list[Document], list[ParentChunkEntry]]:
    """按章节切 parent，再切 child 用于向量索引。"""
    sections = _split_into_sections(text)
    documents: list[Document] = []
    parent_entries: list[ParentChunkEntry] = []
    chunk_id = 0

    for section_index, section in enumerate(sections):
        parent_id = f"{doc_id}::section_{section_index}"
        child_texts = _split_child_chunks(section.text, child_size, child_overlap)

        for child_text in child_texts:
            documents.append(
                Document(
                    metadata=DocMetadata(
                        content=TextBlock(type="text", text=child_text),
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        total_chunks=0,
                    ),
                ),
            )
            parent_entries.append(
                ParentChunkEntry(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    clause_label=section.label,
                    parent_text=section.text,
                    use_parent_return=section.use_parent_return,
                ),
            )
            chunk_id += 1

    total_chunks = len(documents)
    for doc in documents:
        doc.metadata.total_chunks = total_chunks

    return documents, parent_entries
