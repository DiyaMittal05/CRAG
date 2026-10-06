"""Assembles the CRAG LangGraph: router -> retrieve/grade (self-correcting
retry loop) -> generate -> critique (second retry loop) -> finalize/escalate.

This replaces a linear retrieve-and-answer chain with explicit branching so
weak retrieval or an ungrounded answer triggers a bounded, automatic retry
instead of returning a low-quality response.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.critique import CritiqueAgent
from app.agents.generator import AnswerGenerator
from app.agents.retrieval import RetrievalAgent
from app.agents.router import RouterAgent
from app.config import Settings
from app.graph.state import GraphState
from app.llm.client import LLMClient
from app.observability.tracer import Tracer


def build_graph(llm_client: LLMClient, retriever, settings: Settings, tracer: Tracer):
    router_agent = RouterAgent(llm_client)
    retrieval_agent = RetrievalAgent(llm_client, retriever)
    generator = AnswerGenerator(llm_client)
    critique_agent = CritiqueAgent(llm_client)

    def route_query(state: GraphState) -> dict:
        with tracer.node_span("route_query", question=state["question"]) as detail:
            decision = router_agent.route(state["question"])
            detail["route"] = decision.route
            return {"route": decision.route, "original_question": state["question"], "retries": 0}

    def retrieve(state: GraphState) -> dict:
        with tracer.node_span("retrieve", query=state["question"]) as detail:
            documents = retrieval_agent.retrieve(state["question"])
            detail["num_documents"] = len(documents)
            return {"documents": documents}

    def grade_documents(state: GraphState) -> dict:
        with tracer.node_span("grade_documents") as detail:
            documents = state.get("documents", [])
            graded = [(doc, retrieval_agent.grade(state["original_question"], doc)) for doc in documents]
            relevant_docs = [doc for doc, grade in graded if grade.relevant]
            relevant_ratio = (len(relevant_docs) / len(graded)) if graded else 0.0
            detail["relevant_ratio"] = relevant_ratio
            detail["relevant_count"] = len(relevant_docs)
            return {
                "documents": relevant_docs,
                "graded_documents": graded,
                "relevant_ratio": relevant_ratio,
            }

    def transform_query(state: GraphState) -> dict:
        with tracer.node_span("transform_query") as detail:
            rewritten = retrieval_agent.rewrite_query(state["original_question"])
            retries = state.get("retries", 0) + 1
            detail["rewritten_question"] = rewritten
            detail["retries"] = retries
            return {"question": rewritten, "retries": retries}

    def generate(state: GraphState) -> dict:
        with tracer.node_span("generate") as detail:
            documents = state.get("documents", [])
            context = "\n".join(doc.page_content for doc in documents)
            answer = generator.generate(context=context, question=state["original_question"])
            detail["answer_preview"] = answer[:160]
            return {"answer": answer}

    def critique(state: GraphState) -> dict:
        with tracer.node_span("critique") as detail:
            documents = state.get("documents", [])
            context = "\n".join(doc.page_content for doc in documents)
            result = critique_agent.critique(
                question=state["original_question"], context=context, answer=state["answer"]
            )
            detail["grounded"] = result.grounded
            detail["addresses_question"] = result.addresses_question
            return {"critique": result}

    def finalize(state: GraphState) -> dict:
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
        with tracer.node_span("escalate"):
            return {
                "answer": "I'm connecting you with a human support agent who can help with this.",
                "documents": [],
                "confidence": 1.0,
                "escalated": True,
            }

    def decide_after_route(state: GraphState) -> str:
        return state["route"]

    def decide_after_grading(state: GraphState) -> str:
        if state.get("relevant_ratio", 0.0) >= settings.relevance_threshold:
            return "generate"
        if state.get("retries", 0) >= settings.max_retries:
            return "generate"
        return "transform_query"

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
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)
    graph.add_node("critique_answer", critique)
    graph.add_node("finalize", finalize)
    graph.add_node("escalate", escalate)

    graph.set_entry_point("route_query")

    graph.add_conditional_edges(
        "route_query",
        decide_after_route,
        {"vectorstore": "retrieve", "direct_answer": "generate", "escalate": "escalate"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", "critique_answer")
    graph.add_conditional_edges(
        "critique_answer",
        decide_after_critique,
        {"finalize": "finalize", "retry": "transform_query"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("escalate", END)

    return graph.compile()
