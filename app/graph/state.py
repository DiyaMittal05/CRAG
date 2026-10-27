from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document

from app.schemas import Critique, DocumentGrade


class GraphState(TypedDict, total=False):
    question: str
    original_question: str
    route: str
    documents: list[Document]
    scored_documents: list[tuple[Document, float]]
    graded_documents: list[tuple[Document, DocumentGrade]]
    relevant_ratio: float
    answer: str
    critique: Critique
    retries: int
    queries_tried: list[str]
    confidence: float
    escalated: bool
    escalation_reason: str
    trace_id: str
    mode: str
