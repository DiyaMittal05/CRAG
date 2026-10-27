"""Shared tokenization for the deterministic mock backends (FakeLLMClient's
grading/critique heuristics and HashingEmbeddings). Not NLP-grade — a cheap
suffix-stripping stemmer so "subscriptions" matches "subscription" and
"canceling" matches "cancel", which plain exact-token matching misses and
which otherwise causes real documents to be graded irrelevant for questions
they actually answer."""
from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "the", "is", "it", "i", "my", "me", "do", "does", "to", "of", "on", "in",
    "for", "and", "or", "how", "what", "when", "can", "you", "your", "with", "this",
    "that", "are", "be", "if", "so", "will", "not", "we", "go", "get", "have", "has",
}

_SUFFIXES = ("ies", "ing", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def words(text: str) -> set[str]:
    """Raw lowercase tokens, unstemmed, unfiltered."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def keywords(text: str) -> set[str]:
    """Stemmed, stopword-filtered tokens, for overlap/relevance scoring."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in raw if w not in STOPWORDS}
