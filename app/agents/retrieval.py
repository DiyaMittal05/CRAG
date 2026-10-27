from __future__ import annotations

from langchain_core.documents import Document

from app.llm.client import LLMClient
from app.schemas import DocumentGrade

GRADER_SYSTEM_PROMPT = """You are a strict relevance-grading agent in a Corrective RAG pipeline.
Given a user question and a single retrieved document, decide whether the
document contains information that would help answer the question. Be
conservative: a document that is only superficially related is not relevant."""

REWRITE_SYSTEM_PROMPT = """You rewrite support questions to improve retrieval recall.
Rewrite the user's question using broader phrasing, synonyms, and likely support
terminology, while preserving the original intent. Return only the rewritten
question, with no preamble."""


class RetrievalAgent:
    """Handles vector-store retrieval, per-document relevance grading, and the
    query-rewrite step used to self-correct when retrieval quality is weak."""

    def __init__(self, llm_client: LLMClient, vectorstore, k: int) -> None:
        self.llm_client = llm_client
        self.vectorstore = vectorstore
        self.k = k

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Returns (document, distance) pairs. Lower distance = more similar;
        this is Chroma's raw metric space, not a normalized 0-1 similarity."""
        return self.vectorstore.similarity_search_with_score(query, k=self.k)

    def retrieve(self, query: str) -> list[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query)]

    def grade(self, question: str, document: Document) -> DocumentGrade:
        user = f"Question: {question}\nDocument: {document.page_content}"
        return self.llm_client.structured(GRADER_SYSTEM_PROMPT, user, DocumentGrade)

    def rewrite_query(self, question: str) -> str:
        return self.llm_client.complete(REWRITE_SYSTEM_PROMPT, f"Question: {question}")
