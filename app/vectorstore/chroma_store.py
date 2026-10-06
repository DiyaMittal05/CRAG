from __future__ import annotations

from langchain_chroma import Chroma

from app.config import Settings
from app.vectorstore.embeddings import build_embeddings


def build_vectorstore(settings: Settings, *, collection_name: str = "support_docs") -> Chroma:
    embeddings = build_embeddings(settings)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(vectorstore: Chroma, k: int):
    return vectorstore.as_retriever(search_kwargs={"k": k})
