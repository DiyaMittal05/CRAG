from __future__ import annotations

from app.llm.client import LLMClient
from app.schemas import Critique

CRITIQUE_SYSTEM_PROMPT = """You are a critique agent in a Corrective RAG pipeline.
Check a draft answer against the retrieved context for two things:
1. Groundedness: every factual claim is supported by the context.
2. Relevance: the answer actually addresses the question asked.
Be skeptical of answers that hedge, invent specifics, or drift off-topic."""


class CritiqueAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def critique(self, question: str, context: str, answer: str) -> Critique:
        user = f"Question: {question}\nContext: {context}\n\nAnswer: {answer}"
        return self.llm_client.structured(CRITIQUE_SYSTEM_PROMPT, user, Critique)
