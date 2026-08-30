"""Markdown chunking for the credit policy knowledge base.

Splits each policy document into section-level chunks on `##` headers so
retrieval can cite a specific document + section, per the project's
anti-hallucination requirement that every RAG answer point to concrete
evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    document: str
    section: str
    text: str
    chunk_id: str


_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def chunk_markdown_file(path: Path) -> list[DocumentChunk]:
    content = path.read_text(encoding="utf-8")
    document_name = path.stem

    h1_match = _H1_RE.search(content)
    doc_title = h1_match.group(1).strip() if h1_match else document_name

    splits = list(_H2_RE.finditer(content))
    chunks: list[DocumentChunk] = []

    if not splits:
        chunks.append(
            DocumentChunk(
                document=doc_title,
                section="Full document",
                text=content.strip(),
                chunk_id=f"{document_name}::full",
            )
        )
        return chunks

    preamble = content[: splits[0].start()].strip()
    if preamble:
        chunks.append(
            DocumentChunk(
                document=doc_title,
                section="Preamble",
                text=preamble,
                chunk_id=f"{document_name}::preamble",
            )
        )

    for i, match in enumerate(splits):
        section_title = match.group(1).strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(content)
        section_text = content[start:end].strip()
        chunks.append(
            DocumentChunk(
                document=doc_title,
                section=section_title,
                text=f"## {section_title}\n\n{section_text}",
                chunk_id=f"{document_name}::{i}",
            )
        )
    return chunks


def chunk_knowledge_base(directory: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown_file(path))
    return chunks
