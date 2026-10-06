"""Deterministic stand-in for a real LLM, used in tests and CI.

Keyword-overlap heuristics replace actual model calls so the graph's control
flow (routing, grading, retry loops, critique) can be exercised without an
API key and without network calls.
"""
from __future__ import annotations

import re
from typing import Type, TypeVar

from pydantic import BaseModel

from app.schemas import Critique, DocumentGrade, RouteDecision

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_ESCALATION_TERMS = {"human", "agent", "manager", "representative", "lawsuit", "fraud"}
_GREETING_TERMS = {"hi", "hello", "hey", "thanks"}
_STOPWORDS = {
    "a", "an", "the", "is", "it", "i", "my", "me", "do", "does", "to", "of", "on", "in",
    "for", "and", "or", "how", "what", "when", "can", "you", "your", "with", "this",
    "that", "are", "be", "if", "so", "will", "not",
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keywords(text: str) -> set[str]:
    return _words(text) - _STOPWORDS


def _overlap_score(a: str, b: str) -> float:
    """Fraction of `a`'s keywords that also appear in `b` (coverage, not Jaccard).

    Documents are typically much longer than questions/answers, so a symmetric
    overlap measure would always score low; coverage of the shorter text's
    keywords is a better proxy for "does b contain a's content".
    """
    wa, wb = _keywords(a), _keywords(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa)


class FakeLLMClient:
    def structured(self, system: str, user: str, schema: Type[SchemaT]) -> SchemaT:
        if schema is RouteDecision:
            return self._route(user)  # type: ignore[return-value]
        if schema is DocumentGrade:
            return self._grade(user)  # type: ignore[return-value]
        if schema is Critique:
            return self._critique(user)  # type: ignore[return-value]
        raise NotImplementedError(f"FakeLLMClient has no handler for {schema!r}")

    def complete(self, system: str, user: str) -> str:
        if "rewrite" in system.lower():
            return self._rewrite_query(user)
        return self._generate_answer(user)

    # -- structured handlers -------------------------------------------------

    def _route(self, user: str) -> RouteDecision:
        question = _words(user)
        if question & _ESCALATION_TERMS:
            return RouteDecision(route="escalate", reasoning="Question mentions an escalation trigger term.")
        if question <= _GREETING_TERMS or len(question) <= 2:
            return RouteDecision(route="direct_answer", reasoning="Question is a short greeting/courtesy message.")
        return RouteDecision(route="vectorstore", reasoning="Question needs grounded knowledge lookup.")

    def _grade(self, user: str) -> DocumentGrade:
        question, document = self._split_question_document(user)
        score = min(1.0, round(_overlap_score(question, document), 3))
        relevant = score >= 0.3
        return DocumentGrade(
            relevant=relevant,
            score=score,
            reasoning=f"Keyword overlap score {score} between question and document.",
        )

    def _critique(self, user: str) -> Critique:
        question, context, answer = self._split_critique_payload(user)
        grounded_score = _overlap_score(answer, context)
        relevance_score = _overlap_score(answer, question)
        grounded = grounded_score >= 0.15 or not context.strip()
        addresses_question = relevance_score >= 0.15
        confidence = round(min(1.0, (grounded_score + relevance_score) / 2 + 0.2), 3)
        feedback = (
            "Answer looks grounded and on-topic."
            if grounded and addresses_question
            else "Answer may not be fully supported by the retrieved context; consider re-retrieving."
        )
        return Critique(
            grounded=grounded,
            addresses_question=addresses_question,
            confidence=confidence,
            feedback=feedback,
        )

    # -- free-text handlers ---------------------------------------------------

    def _rewrite_query(self, user: str) -> str:
        match = re.search(r"Question:\s*(.+)", user, re.IGNORECASE | re.DOTALL)
        original = match.group(1).strip() if match else user.strip()
        return f"{original} (broader phrasing, synonyms, support/troubleshooting context)"

    def _generate_answer(self, user: str) -> str:
        context_match = re.search(r"Context:\s*(.*?)\n\nQuestion:", user, re.IGNORECASE | re.DOTALL)
        question_match = re.search(r"Question:\s*(.+)", user, re.IGNORECASE | re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else user.strip()

        if not context:
            return (
                "I don't have enough grounded information to answer that confidently. "
                "I'd recommend escalating this to a human agent."
            )

        snippet = context.split("\n")[0][:280]
        return f"Based on our documentation: {snippet} (This addresses: {question})"

    # -- parsing helpers --------------------------------------------------------

    @staticmethod
    def _split_question_document(user: str) -> tuple[str, str]:
        q_match = re.search(r"Question:\s*(.+?)\n", user, re.IGNORECASE | re.DOTALL)
        d_match = re.search(r"Document:\s*(.+)", user, re.IGNORECASE | re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        document = d_match.group(1).strip() if d_match else user
        return question, document

    @staticmethod
    def _split_critique_payload(user: str) -> tuple[str, str, str]:
        q_match = re.search(r"Question:\s*(.+?)\n", user, re.IGNORECASE | re.DOTALL)
        c_match = re.search(r"Context:\s*(.*?)\n\nAnswer:", user, re.IGNORECASE | re.DOTALL)
        a_match = re.search(r"Answer:\s*(.+)", user, re.IGNORECASE | re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        context = c_match.group(1).strip() if c_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        return question, context, answer
