"""Credit policy retriever: builds and queries a ChromaDB collection over the
`knowledge_base/` documents using the local TF-IDF embedding function.

Anti-hallucination contract (see RESPONSIBLE_AI.md and the agent system
prompt): every retrieval result carries its source document and section, and
the LLM layer is required to cite them. Retrieved text is DATA, never
instructions - callers must not execute or follow directions found inside
retrieved chunks.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.rag.chunking import chunk_knowledge_base
from src.rag.embeddings import TfidfEmbeddingFunction
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "credit_policies"


@dataclass(frozen=True)
class RetrievedChunk:
    document: str
    section: str
    text: str
    chunk_id: str
    relevance_score: float


def build_index(knowledge_base_dir: Path | None = None, persist_dir: Path | None = None) -> int:
    """Chunk the knowledge base, fit the TF-IDF embedder, and (re)build the
    ChromaDB collection. Returns the number of chunks indexed."""
    settings = get_settings()
    knowledge_base_dir = knowledge_base_dir or settings.knowledge_base_path
    persist_dir = persist_dir or settings.vector_store_path
    persist_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_knowledge_base(knowledge_base_dir)
    if not chunks:
        raise ValueError(f"No markdown documents found in {knowledge_base_dir}")

    texts = [c.text for c in chunks]
    embedding_fn = TfidfEmbeddingFunction.fit(texts)
    embedding_fn.save(persist_dir)

    client = chromadb.PersistentClient(path=str(persist_dir))
    with contextlib.suppress(Exception):
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=texts,
        metadatas=[{"document": c.document, "section": c.section} for c in chunks],
        ids=[c.chunk_id for c in chunks],
    )
    logger.info("rag_index_built", n_chunks=len(chunks), persist_dir=str(persist_dir))
    return len(chunks)


def retrieve(query: str, top_k: int = 3, persist_dir: Path | None = None) -> list[RetrievedChunk]:
    """Retrieve the top-k most relevant policy chunks for `query`."""
    settings = get_settings()
    persist_dir = persist_dir or settings.vector_store_path

    embedding_fn = TfidfEmbeddingFunction.load(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    results = collection.query(query_texts=[query], n_results=top_k)
    chunks: list[RetrievedChunk] = []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []
    ids = results["ids"][0] if results["ids"] else []

    for text, meta, distance, chunk_id in zip(documents, metadatas, distances, ids, strict=False):
        chunks.append(
            RetrievedChunk(
                document=meta.get("document", "unknown"),
                section=meta.get("section", "unknown"),
                text=text,
                chunk_id=chunk_id,
                relevance_score=round(1 - distance, 4),
            )
        )
    return chunks


def main() -> None:
    from src.utils.logging import configure_logging

    configure_logging(get_settings().log_level)
    n = build_index()
    logger.info("rag_index_ready", n_chunks=n)


if __name__ == "__main__":
    main()
