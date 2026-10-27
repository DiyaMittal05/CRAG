"""Shared eval machinery: loads eval_set.jsonl, runs a graph (baseline or
crag) over it with the deterministic mock backend, and computes metrics from
the actual run - nothing here is an estimate. Used by both run_eval.py (the
CI gate) and compare_baseline_crag.py (the baseline-vs-crag report).

Methodology notes (read before trusting a number from this module):

- All runs use FakeLLMClient + HashingEmbeddings (app/llm/fake.py,
  app/vectorstore/embeddings.py) because this environment has no
  OPENAI_API_KEY. These are real, deterministic programs - the numbers below
  are measured from actually running them, not invented - but they measure
  the mock backend's behavior, not a production LLM's. Re-run with
  OPENAI_API_KEY set (and CRAG_USE_FAKE_BACKENDS unset) to get numbers for
  the real model; nothing here does that automatically.
- "Groundedness" is a lexical-overlap proxy (stemmed keyword coverage of the
  answer's content inside the retrieved context), not a semantic entailment
  or LLM-judge check. This codebase has no formal chunk-citation mechanism in
  generated answers, so a stricter check isn't available without a real LLM
  judge, which this environment can't run. See groundedness_score().
- An LLM-judge pass (optional per the task spec) is intentionally skipped:
  the only LLM available here is FakeLLMClient itself, so using it as a
  "judge" would just be the same heuristic grading itself - not an
  independent check. If a real key is added later, a judge pass using a
  different model than the generator would be a real addition, not a
  cosmetic one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import Settings
from app.graph.build import build_graph
from app.llm.fake import FakeLLMClient
from app.observability.tracer import Tracer
from app.text_utils import keywords
from app.vectorstore.chroma_store import build_vectorstore
from app.vectorstore.ingest import load_documents

EVAL_SET_PATH = Path(__file__).parent / "eval_set.jsonl"
GROUNDEDNESS_THRESHOLD = 0.4


def load_eval_cases() -> list[dict]:
    cases = []
    with EVAL_SET_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_eval_settings(chroma_dir: str, **overrides) -> Settings:
    base = dict(
        use_fake_backends=True,
        chroma_persist_dir=chroma_dir,
        relevance_threshold=0.2,  # matches the existing test/CI harness convention
        max_retries=2,
        retriever_k=4,
    )
    base.update(overrides)
    return Settings(**base)


def build_eval_graph(settings: Settings, mode: str, collection_name: str):
    llm_client = FakeLLMClient()
    documents = load_documents(settings.docs_dir)
    vectorstore = build_vectorstore(settings, collection_name=collection_name)
    vectorstore.add_documents(documents)
    graph = build_graph(llm_client, vectorstore, settings, mode=mode)
    return graph, llm_client


def _contains_any(answer: str, facts: list[str]) -> bool:
    answer_l = (answer or "").lower()
    return any(f.lower() in answer_l for f in facts)


def groundedness_score(answer: str, context: str) -> float:
    """Stemmed keyword coverage of the answer inside the retrieved context.

    A lexical proxy, not semantic entailment: see the module docstring.
    """
    a = keywords(answer)
    if not a:
        return 1.0  # an empty/declined answer makes no unsupported claims
    c = keywords(context)
    if not c:
        return 0.0
    return len(a & c) / len(a)


def evaluate_case(graph, llm_client, case: dict, mode: str) -> dict:
    # Tracer._store is process-lifetime and keyed by trace_id; since
    # compare_baseline_crag.py runs baseline then crag with the same case
    # ids in one process, the trace_id must carry mode too or crag's trace
    # lookup would also return baseline's (node-disjoint, but not worth
    # relying on that).
    trace_id = f"eval-{mode}-{case['id']}"
    calls_before = llm_client.call_count
    start = time.perf_counter()
    result = graph.invoke({"question": case["question"], "trace_id": trace_id})
    latency_ms = (time.perf_counter() - start) * 1000
    llm_calls = llm_client.call_count - calls_before
    events = Tracer.events_for(trace_id)

    answer = result.get("answer", "") or ""
    route = result.get("route")
    retries = result.get("retries", 0)
    escalated_weak_context = bool(result.get("escalation_reason"))
    documents = result.get("documents", [])
    context = "\n".join(doc.page_content for doc in documents)

    answerable = case.get("answerable", True)
    key_facts = case.get("key_facts") or []
    expected_route = case.get("expected_route")

    if not answerable:
        declined = escalated_weak_context or bool(result.get("escalated")) or not answer.strip()
        correctness = None
        refused_correctly = declined
    else:
        refused_correctly = None
        correctness = _contains_any(answer, key_facts) if key_facts else None

    grounded = groundedness_score(answer, context)
    re_retrieval_triggered = any(e.node == "transform_query" and e.phase == "end" for e in events)

    return {
        "id": case["id"],
        "category": case.get("category"),
        "question": case["question"],
        "answerable": answerable,
        "route": route,
        "expected_route": expected_route,
        "route_correct": (route == expected_route) if expected_route else None,
        "answer": answer,
        "correctness": correctness,
        "refused_correctly": refused_correctly,
        "groundedness_score": grounded,
        "grounded": grounded >= GROUNDEDNESS_THRESHOLD,
        "escalated": bool(result.get("escalated")),
        "escalated_weak_context": escalated_weak_context,
        "retries": retries,
        "re_retrieval_triggered": re_retrieval_triggered,
        "llm_calls": llm_calls,
        "latency_ms": latency_ms,
    }


def run_eval(mode: str, chroma_dir: str, settings_overrides: dict | None = None) -> list[dict]:
    cases = load_eval_cases()
    settings = build_eval_settings(chroma_dir, **(settings_overrides or {}))
    graph, llm_client = build_eval_graph(settings, mode=mode, collection_name=f"eval_{mode}_docs")
    return [evaluate_case(graph, llm_client, case, mode) for case in cases]


def _pct(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def summarize(rows: list[dict]) -> dict:
    latencies = sorted(r["latency_ms"] for r in rows)
    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    scored_correctness_rows = [r for r in answerable_rows if r["correctness"] is not None]

    return {
        "n": len(rows),
        "correctness": _pct([r["correctness"] for r in scored_correctness_rows]),
        "correctness_n_scored": len(scored_correctness_rows),
        "groundedness": _pct([r["grounded"] for r in rows]),
        "refusal_accuracy": _pct([r["refused_correctly"] for r in unanswerable_rows]),
        "refusal_accuracy_n": len(unanswerable_rows),
        "route_accuracy": _pct([r["route_correct"] for r in rows if r["route_correct"] is not None]),
        "escalation_rate": _pct([r["escalated"] for r in rows]),
        "weak_context_escalation_rate": _pct([r["escalated_weak_context"] for r in rows]),
        "re_retrieval_rate": _pct([r["re_retrieval_triggered"] for r in rows]),
        "avg_llm_calls": (sum(r["llm_calls"] for r in rows) / len(rows)) if rows else None,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }
