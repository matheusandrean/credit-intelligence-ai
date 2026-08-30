"""Local, offline TF-IDF embedding function for the RAG vector store.

The project deliberately avoids depending on a downloaded neural embedding
model: the platform must work fully offline in "Demo Mode" (see
docs/DEMO_MODE.md) without any network call or paid API. A fitted
`TfidfVectorizer` gives deterministic, reproducible, dependency-light
vectors that are entirely sufficient for a small (~5 document) policy
knowledge base, while still exercising a real vector-store workflow
(ChromaDB) end to end.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"


class TfidfEmbeddingFunction(chromadb.EmbeddingFunction):
    """A chromadb-compatible embedding function backed by a fitted TF-IDF vectorizer."""

    def __init__(self, vectorizer: TfidfVectorizer):
        self._vectorizer = vectorizer

    def __call__(self, input: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(input)
        return matrix.toarray().astype(np.float64).tolist()

    @staticmethod
    def name() -> str:
        return "tfidf_local"

    @classmethod
    def fit(cls, corpus: list[str]) -> TfidfEmbeddingFunction:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=4096,
        )
        vectorizer.fit(corpus)
        return cls(vectorizer)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, directory / VECTORIZER_FILENAME)

    @classmethod
    def load(cls, directory: Path) -> TfidfEmbeddingFunction:
        vectorizer = joblib.load(directory / VECTORIZER_FILENAME)
        return cls(vectorizer)
