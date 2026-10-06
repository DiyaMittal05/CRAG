"""LLM client abstraction.

Agents depend on the `LLMClient` protocol rather than a concrete provider so the
graph can run against a real OpenAI-backed model in production and against a
deterministic fake in tests and CI, without any branching in agent code.
"""
from __future__ import annotations

from typing import Protocol, Type, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient(Protocol):
    def structured(self, system: str, user: str, schema: Type[SchemaT]) -> SchemaT:
        """Call the model and parse its output into `schema`."""

    def complete(self, system: str, user: str) -> str:
        """Call the model and return raw text."""


class OpenAIChatClient:
    """LLMClient backed by `langchain_openai.ChatOpenAI`."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0) -> None:
        from langchain_openai import ChatOpenAI

        self._chat = ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

    def structured(self, system: str, user: str, schema: Type[SchemaT]) -> SchemaT:
        structured_llm = self._chat.with_structured_output(schema)
        return structured_llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    def complete(self, system: str, user: str) -> str:
        response = self._chat.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return response.content


def build_llm_client(settings) -> "LLMClient":
    if settings.use_fake_backends:
        from app.llm.fake import FakeLLMClient

        return FakeLLMClient()
    return OpenAIChatClient(model=settings.chat_model, api_key=settings.openai_api_key)
