from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    chat_model: str = os.getenv("CRAG_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("CRAG_EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_persist_dir: str = os.getenv("CRAG_CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))
    docs_dir: str = os.getenv("CRAG_DOCS_DIR", str(BASE_DIR / "data" / "docs"))
    retriever_k: int = int(os.getenv("CRAG_RETRIEVER_K", "4"))
    relevance_threshold: float = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", "0.6"))
    max_retries: int = int(os.getenv("CRAG_MAX_RETRIES", "2"))
    # When no API key is configured (e.g. in CI), fall back to deterministic fakes.
    use_fake_backends: bool = os.getenv("CRAG_USE_FAKE_BACKENDS", "").lower() in {"1", "true", "yes"} or not os.getenv(
        "OPENAI_API_KEY"
    )


settings = Settings()
