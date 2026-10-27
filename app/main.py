from __future__ import annotations

from typing import Union

from fastapi import Depends, FastAPI

from app.config import Settings, settings as default_settings
from app.graph.build import build_graph
from app.llm.client import LLMClient, build_llm_client
from app.schemas import EscalationResponse, Source, SupportQuery, SupportResponse
from app.vectorstore.chroma_store import build_vectorstore

app = FastAPI(title="CRAG Support Agent", version="0.1.0")


def build_app_state(settings: Settings, llm_client: LLMClient | None = None) -> dict:
    """Builds one shared LLM client + vectorstore, and both the crag and
    baseline graphs on top of them, so `llm_client.call_count` and the indexed
    documents are identical between modes — the only difference is the graph
    topology (see app/graph/build.py)."""
    llm_client = llm_client or build_llm_client(settings)
    vectorstore = build_vectorstore(settings)
    return {
        "llm_client": llm_client,
        "graphs": {
            "crag": build_graph(llm_client, vectorstore, settings, mode="crag"),
            "baseline": build_graph(llm_client, vectorstore, settings, mode="baseline"),
        },
    }


_state: dict | None = None


def get_app_state() -> dict:
    global _state
    if _state is None:
        _state = build_app_state(default_settings)
    return _state


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=Union[SupportResponse, EscalationResponse])
def query(payload: SupportQuery, state: dict = Depends(get_app_state)):
    llm_client: LLMClient = state["llm_client"]
    graph = state["graphs"][payload.mode]
    trace_id = payload.session_id or "trace"

    calls_before = llm_client.call_count
    result = graph.invoke({"question": payload.question, "trace_id": trace_id})
    llm_calls = llm_client.call_count - calls_before

    documents = result.get("documents", [])
    graded = {id(doc): grade for doc, grade in result.get("graded_documents", [])}
    sources = [
        Source(
            content=doc.page_content,
            metadata=doc.metadata,
            relevance_score=graded[id(doc)].score if id(doc) in graded else 1.0,
        )
        for doc in documents
    ]

    if result.get("escalation_reason"):
        return EscalationResponse(
            reason=result["escalation_reason"],
            queries_tried=result.get("queries_tried", []),
            top_sources=sources,
            trace_id=trace_id,
            mode=payload.mode,
            retries=result.get("retries", 0),
            llm_calls=llm_calls,
        )

    return SupportResponse(
        answer=result.get("answer", ""),
        sources=sources,
        confidence=result.get("confidence", 0.5),
        route=result.get("route", "unknown"),
        escalated=result.get("escalated", False),
        retries=result.get("retries", 0),
        trace_id=trace_id,
        mode=payload.mode,
        llm_calls=llm_calls,
    )
