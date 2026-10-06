from __future__ import annotations

from app.llm.client import LLMClient
from app.schemas import RouteDecision

ROUTER_SYSTEM_PROMPT = """You are the routing agent for a customer support system.
Classify the user's question into exactly one route:
- "vectorstore": the question needs grounded lookup in product documentation.
- "direct_answer": the message is a greeting, thanks, or otherwise needs no lookup.
- "escalate": the user is explicitly asking for a human, or the issue is sensitive
  (legal threats, fraud claims) and should not be auto-answered.
Always pick the single best route."""


class RouterAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def route(self, question: str) -> RouteDecision:
        return self.llm_client.structured(ROUTER_SYSTEM_PROMPT, f"Question: {question}", RouteDecision)
