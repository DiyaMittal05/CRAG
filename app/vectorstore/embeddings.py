"""Embeddings provider selection.

Mirrors the LLM client split: a real OpenAI embedding model in production, and
a deterministic hashing-based embedding in tests/CI so Chroma's actual
similarity search is exercised without network calls or an API key.
"""
from __future__ import annotations

import hashlib

from app.text_utils import keywords as _tokenize

_VECTOR_SIZE = 256

# _tokenize (app.text_utils.keywords) does regex word-extraction + stemming +
# stopword removal, not str.split(): splitting on whitespace alone leaves
# trailing punctuation attached ("password?" != "password"), and raw
# term-frequency hashing with no IDF weighting lets common words ("to", "the",
# "do"...) dominate the normalized vector, swamping the one content word that
# actually distinguishes a relevant chunk from an irrelevant one. Using the
# same tokenizer the grader uses also keeps retrieval and grading consistent.


class HashingEmbeddings:
    """Deterministic bag-of-words embedding via feature hashing.

    Not semantically meaningful (no synonym/paraphrase understanding), but a
    real lexical-overlap signal — good enough to validate retrieval plumbing
    (indexing, nearest-neighbor search, metadata) in CI without network calls.
    """

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * _VECTOR_SIZE
        for token in _tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % _VECTOR_SIZE
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings):
    if settings.use_fake_backends:
        return HashingEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
