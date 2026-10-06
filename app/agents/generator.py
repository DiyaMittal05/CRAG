from __future__ import annotations

from app.llm.client import LLMClient

GENERATE_SYSTEM_PROMPT = """You are a helpful customer support agent. Answer the
user's question using only the provided context. If the context is insufficient
or empty, say so plainly rather than guessing."""


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate(self, context: str, question: str) -> str:
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return self.llm_client.complete(GENERATE_SYSTEM_PROMPT, user)
