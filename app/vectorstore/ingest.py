"""Load the sample support docs into Chroma.

Run with:

    python -m app.vectorstore.ingest
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore.chroma_store import build_vectorstore


def load_documents(docs_dir: str) -> list[Document]:
    paths = sorted(Path(docs_dir).glob("*.md"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            documents.append(Document(page_content=chunk, metadata={"source": path.name}))
    return documents


def main() -> None:
    documents = load_documents(settings.docs_dir)
    if not documents:
        print(f"No .md documents found in {settings.docs_dir}", file=sys.stderr)
        return

    vectorstore = build_vectorstore(settings)
    vectorstore.add_documents(documents)
    print(f"Ingested {len(documents)} chunks from {settings.docs_dir} into {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
