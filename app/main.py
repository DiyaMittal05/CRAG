from __future__ import annotations

from fastapi import Depends, FastAPI

from app.config import Settings, settings as default_settings
from app.graph.build import build_graph
from app.llm.client import LLMClient, build_llm_client
from app.observability.tracer import Tracer
from app.schemas import Source, SupportQuery, SupportResponse
from app.vectorstore.chroma_store import build_vectorstore, get_retriever

app = FastAPI(title="CRAG Support Agent", version="0.1.0")


def build_app_graph(settings: Settings, llm_client: LLMClient | None = None):
    llm_client = llm_client or build_llm_client(settings)
    vectorstore = build_vectorstore(settings)
    retriever = get_retriever(vectorstore, k=settings.retriever_k)
    tracer = Tracer()
    return build_graph(llm_client, retriever, settings, tracer)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_app_graph(default_settings)
    return _graph


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=SupportResponse)
def query(payload: SupportQuery, graph=Depends(get_graph)) -> SupportResponse:
    trace_id = payload.session_id or "trace"
    tracer = Tracer(trace_id=trace_id)
    result = graph.invoke({"question": payload.question, "trace_id": tracer.trace_id})

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

    return SupportResponse(
        answer=result.get("answer", ""),
        sources=sources,
        confidence=result.get("confidence", 0.5),
        route=result.get("route", "unknown"),
        escalated=result.get("escalated", False),
        retries=result.get("retries", 0),
        trace_id=tracer.trace_id,
    )
