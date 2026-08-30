"""Unit tests for src.rag (chunking, embeddings, retrieval)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.chunking import chunk_knowledge_base, chunk_markdown_file
from src.rag.embeddings import TfidfEmbeddingFunction
from src.rag.retriever import build_index, retrieve
from src.utils.config import get_settings

SAMPLE_MD = """# Sample Policy

> **SYNTHETIC DEMONSTRATION POLICY.**

## Section One

Some content about debt to income ratios and eligibility.

## Section Two

Some content about utilization thresholds.
"""


@pytest.fixture
def sample_doc(tmp_path: Path) -> Path:
    path = tmp_path / "sample_policy.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    return path


def test_chunk_markdown_file_splits_on_h2(sample_doc: Path) -> None:
    chunks = chunk_markdown_file(sample_doc)
    sections = [c.section for c in chunks]
    assert "Section One" in sections
    assert "Section Two" in sections
    assert any(c.document == "Sample Policy" for c in chunks)


def test_chunk_knowledge_base_reads_all_md_files(tmp_path: Path, sample_doc: Path) -> None:
    (tmp_path / "another.md").write_text("# Another\n\n## X\n\nText.", encoding="utf-8")
    chunks = chunk_knowledge_base(tmp_path)
    docs = {c.document for c in chunks}
    assert "Sample Policy" in docs
    assert "Another" in docs


def test_tfidf_embedding_function_roundtrip(tmp_path: Path) -> None:
    corpus = ["debt to income ratio", "credit utilization threshold", "another unrelated topic"]
    ef = TfidfEmbeddingFunction.fit(corpus)
    ef.save(tmp_path)
    loaded = TfidfEmbeddingFunction.load(tmp_path)
    vectors = loaded(["debt to income ratio"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_build_index_and_retrieve_real_knowledge_base(tmp_path: Path) -> None:
    settings = get_settings()
    kb_dir = settings.knowledge_base_path
    persist_dir = tmp_path / "chroma_test"

    n_chunks = build_index(knowledge_base_dir=kb_dir, persist_dir=persist_dir)
    assert n_chunks > 0

    results = retrieve(
        "Which policy discusses customers with high debt commitment?",
        top_k=3,
        persist_dir=persist_dir,
    )
    assert len(results) > 0
    assert any("debt" in r.text.lower() or "commitment" in r.text.lower() for r in results)
    for r in results:
        assert r.document
        assert r.section


def test_retrieval_returns_source_citation_fields(tmp_path: Path) -> None:
    settings = get_settings()
    persist_dir = tmp_path / "chroma_test2"
    build_index(knowledge_base_dir=settings.knowledge_base_path, persist_dir=persist_dir)
    results = retrieve("responsible AI human oversight", top_k=2, persist_dir=persist_dir)
    assert len(results) > 0
    top = results[0]
    assert isinstance(top.document, str) and top.document
    assert isinstance(top.section, str) and top.section
    assert isinstance(top.chunk_id, str) and top.chunk_id
