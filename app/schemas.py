"""Pydantic models shared across agents, the graph, and the API layer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output of the router agent."""

    route: Literal["vectorstore", "direct_answer", "escalate"] = Field(
        description="Where the query should be handled next."
    )
    reasoning: str = Field(description="Short justification for the chosen route.")


class DocumentGrade(BaseModel):
    """Structured output of the grader agent for a single retrieved document."""

    relevant: bool = Field(description="Whether the document helps answer the question.")
    score: float = Field(ge=0.0, le=1.0, description="Confidence that the document is relevant.")
    reasoning: str = Field(description="Short justification for the grade.")


class Critique(BaseModel):
    """Structured output of the critique agent over a draft answer."""

    grounded: bool = Field(description="Whether the answer is fully supported by the sources.")
    addresses_question: bool = Field(description="Whether the answer actually answers the question.")
    confidence: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(description="Actionable feedback used to drive a retry.")


class SupportQuery(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class Source(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    relevance_score: float = Field(ge=0.0, le=1.0)


class SupportResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    route: str
    escalated: bool
    retries: int
    trace_id: str
