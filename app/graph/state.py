from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document

from app.schemas import Critique, DocumentGrade


class GraphState(TypedDict, total=False):
    question: str
    original_question: str
    route: str
    documents: list[Document]
    graded_documents: list[tuple[Document, DocumentGrade]]
    relevant_ratio: float
    answer: str
    critique: Critique
    retries: int
    confidence: float
    escalated: bool
    trace_id: str
