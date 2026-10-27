"""Assembles the CRAG LangGraph, or a plain-RAG baseline, from the same nodes.

`mode="crag"` wires the full self-correcting pipeline: router -> retrieve ->
grade (retry loop: weak grading triggers query rewrite + re-retrieval, bounded
by `settings.max_retries`, falling through to an escalation node rather than
answering from context the grader has rejected) -> generate -> critique
(a second, independent retry loop: an ungrounded/off-topic draft triggers
another correction cycle) -> finalize.

`mode="baseline"` wires a single retrieve-then-answer pass over the *same*
router, retriever, LLM, and generation prompt, with no grader, no query
rewriting/re-retrieval, and no critique. It exists so the corrective logic can
be measured against a fair, otherwise-identical comparison point (see
eval/compare_baseline_crag.py) rather than asserted without evidence.
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents.critique import CritiqueAgent
from app.agents.generator import AnswerGenerator
from app.agents.retrieval import RetrievalAgent
from app.agents.router import RouterAgent
from app.config import Settings
from app.graph.state import GraphState
from app.llm.client import LLMClient
from app.observability.tracer import Tracer

Mode = Literal["baseline", "crag"]


def build_graph(llm_client: LLMClient, vectorstore, settings: Settings, mode: Mode = "crag"):
    router_agent = RouterAgent(llm_client)
    retrieval_agent = RetrievalAgent(llm_client, vectorstore, settings.retriever_k)
    generator = AnswerGenerator(llm_client)
    critique_agent = CritiqueAgent(llm_client)

    def _tracer(state: GraphState) -> Tracer:
        # Built per-node-call from the trace_id carried in state, rather than
        # closed over at graph-build time, so each request's events land under
        # its own trace_id even though the compiled graph is a long-lived singleton.
        return Tracer(trace_id=state.get("trace_id") or "unknown")

    def route_query(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("route_query", question=state["question"], mode=mode) as detail:
            decision = router_agent.route(state["question"])
            detail["route"] = decision.route
            detail["reasoning"] = decision.reasoning
            return {
                "route": decision.route,
                "original_question": state["question"],
                "retries": 0,
                "queries_tried": [state["question"]],
                "mode": mode,
            }

    def retrieve(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("retrieve", query=state["question"]) as detail:
            scored = retrieval_agent.retrieve_with_scores(state["question"])
            detail["retrieved"] = [
                {"source": doc.metadata.get("source"), "distance": float(score)} for doc, score in scored
            ]
            return {"documents": [doc for doc, _ in scored], "scored_documents": scored}

    def grade_documents(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("grade_documents") as detail:
            documents = state.get("documents", [])
            distances = {id(doc): distance for doc, distance in state.get("scored_documents", [])}
            graded = [(doc, retrieval_agent.grade(state["original_question"], doc)) for doc in documents]
            relevant_docs = [doc for doc, grade in graded if grade.relevant]
            relevant_ratio = (len(relevant_docs) / len(graded)) if graded else 0.0
            detail["relevant_ratio"] = relevant_ratio
            detail["relevant_count"] = len(relevant_docs)
            detail["chunk_grades"] = [
                {
                    "source": doc.metadata.get("source"),
                    "retrieval_distance": distances.get(id(doc)),
                    "grader_score": grade.score,
                    "relevant": grade.relevant,
                }
                for doc, grade in graded
            ]
            return {
                "documents": relevant_docs,
                "graded_documents": graded,
                "relevant_ratio": relevant_ratio,
            }

    def transform_query(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("transform_query") as detail:
            rewritten = retrieval_agent.rewrite_query(state["original_question"])
            retries = state.get("retries", 0) + 1
            queries_tried = state.get("queries_tried", []) + [rewritten]
            detail["rewritten_question"] = rewritten
            detail["retries"] = retries
            return {"question": rewritten, "retries": retries, "queries_tried": queries_tried}

    def generate(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("generate") as detail:
            documents = state.get("documents", [])
            context = "\n".join(doc.page_content for doc in documents)
            answer = generator.generate(context=context, question=state["original_question"])
            detail["answer_preview"] = answer[:160]
            return {"answer": answer}

    def critique(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("critique") as detail:
            documents = state.get("documents", [])
            context = "\n".join(doc.page_content for doc in documents)
            result = critique_agent.critique(
                question=state["original_question"], context=context, answer=state["answer"]
            )
            detail["grounded"] = result.grounded
            detail["addresses_question"] = result.addresses_question
            detail["confidence"] = result.confidence
            return {"critique": result}

    def finalize(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("finalize") as detail:
            result = state.get("critique")
            if result is None:
                confidence, escalated = 1.0, False
            else:
                confidence = result.confidence
                escalated = not (result.grounded and result.addresses_question)
            detail["confidence"] = confidence
            detail["escalated"] = escalated
            return {"confidence": confidence, "escalated": escalated}

    def escalate(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("escalate", reason="router_escalate"):
            return {
                "answer": "I'm connecting you with a human support agent who can help with this.",
                "documents": [],
                "confidence": 1.0,
                "escalated": True,
            }

    def escalate_weak_context(state: GraphState) -> dict:
        tracer = _tracer(state)
        with tracer.node_span("escalate_weak_context") as detail:
            reason = (
                f"Grader rejected retrieved context after {state.get('retries', 0)} retries "
                f"(relevant_ratio={state.get('relevant_ratio', 0.0):.2f}, "
                f"threshold={settings.relevance_threshold:.2f})"
            )
            detail["reason"] = reason
            return {
                "escalation_reason": reason,
                "escalated": True,
                "confidence": 0.0,
                "answer": "",
            }

    def decide_after_route(state: GraphState) -> str:
        return state["route"]

    def decide_after_grading(state: GraphState) -> str:
        if state.get("relevant_ratio", 0.0) >= settings.relevance_threshold:
            return "generate"
        if state.get("retries", 0) >= settings.max_retries:
            return "escalate_weak_context"
        return "transform_query"

    def decide_after_generate(state: GraphState) -> str:
        # Critique checks a drafted answer against retrieved context; the
        # direct_answer route never retrieves anything, so there is nothing
        # to critique - without this, "generate" fed critique_answer
        # unconditionally, and a canned no-context answer never "addressed"
        # a greeting, spuriously driving even "hi" into the retry loop.
        return "critique" if state.get("route") == "vectorstore" else "finalize"

    def decide_after_critique(state: GraphState) -> str:
        result = state.get("critique")
        if result and result.grounded and result.addresses_question:
            return "finalize"
        if state.get("retries", 0) >= settings.max_retries:
            return "finalize"
        return "retry"

    graph = StateGraph(GraphState)
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("finalize", finalize)
    graph.add_node("escalate", escalate)

    graph.set_entry_point("route_query")
    graph.add_conditional_edges(
        "route_query",
        decide_after_route,
        {"vectorstore": "retrieve", "direct_answer": "generate", "escalate": "escalate"},
    )
    graph.add_edge("escalate", END)

    if mode == "baseline":
        # Single retrieve-then-answer pass: no grader, no rewrite/re-retrieval, no critique.
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "finalize")
        graph.add_edge("finalize", END)
    else:
        graph.add_node("grade_documents", grade_documents)
        graph.add_node("transform_query", transform_query)
        graph.add_node("critique_answer", critique)
        graph.add_node("escalate_weak_context", escalate_weak_context)

        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            decide_after_grading,
            {
                "generate": "generate",
                "transform_query": "transform_query",
                "escalate_weak_context": "escalate_weak_context",
            },
        )
        graph.add_edge("transform_query", "retrieve")
        graph.add_conditional_edges(
            "generate",
            decide_after_generate,
            {"critique": "critique_answer", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "critique_answer",
            decide_after_critique,
            {"finalize": "finalize", "retry": "transform_query"},
        )
        graph.add_edge("finalize", END)
        graph.add_edge("escalate_weak_context", END)

    return graph.compile()
