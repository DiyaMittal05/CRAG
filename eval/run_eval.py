"""Automated eval-set runner, wired into CI on every push.

Runs the full CRAG graph (with deterministic fake LLM/embedding backends, so
no API key is required) against a fixed set of support questions and checks
routing + grounding expectations. Exits non-zero if the pass rate drops below
`PASS_THRESHOLD`, which fails the GitHub Actions job.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.graph.build import build_graph
from app.llm.fake import FakeLLMClient
from app.observability.tracer import Tracer
from app.vectorstore.chroma_store import build_vectorstore, get_retriever
from app.vectorstore.ingest import load_documents

EVAL_SET_PATH = Path(__file__).parent / "eval_set.jsonl"
PASS_THRESHOLD = 0.75


def load_eval_cases() -> list[dict]:
    cases = []
    with EVAL_SET_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_eval_graph(chroma_dir: str):
    settings = Settings(
        use_fake_backends=True,
        chroma_persist_dir=chroma_dir,
        relevance_threshold=0.2,
        max_retries=2,
        retriever_k=4,
    )
    documents = load_documents(settings.docs_dir)
    vectorstore = build_vectorstore(settings, collection_name="eval_support_docs")
    vectorstore.add_documents(documents)
    retriever = get_retriever(vectorstore, k=settings.retriever_k)
    llm_client = FakeLLMClient()
    tracer = Tracer()
    return build_graph(llm_client, retriever, settings, tracer)


def check_case(case: dict, result: dict) -> tuple[bool, str]:
    expected_route = case.get("expect_route")
    if expected_route and result.get("route") != expected_route:
        return False, f"expected route={expected_route!r}, got {result.get('route')!r}"

    must_include_any = case.get("must_include_any") or []
    if must_include_any:
        answer = (result.get("answer") or "").lower()
        if not any(term.lower() in answer for term in must_include_any):
            return False, f"answer did not contain any of {must_include_any}: {answer!r}"

    if not case.get("allow_escalation") and expected_route == "vectorstore":
        if result.get("escalated") and case.get("expect_route") != "escalate":
            return False, "unexpectedly escalated a question that should have been answered"

    return True, "ok"


def main() -> int:
    cases = load_eval_cases()
    with tempfile.TemporaryDirectory() as chroma_dir:
        graph = build_eval_graph(chroma_dir)

        passed = 0
        for case in cases:
            result = graph.invoke({"question": case["question"]})
            ok, detail = check_case(case, result)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {case['question']!r} -> {detail}")
            if ok:
                passed += 1

    pass_rate = passed / len(cases) if cases else 0.0
    print(f"\nPass rate: {passed}/{len(cases)} ({pass_rate:.0%})")

    if pass_rate < PASS_THRESHOLD:
        print(f"FAILED: pass rate below threshold ({PASS_THRESHOLD:.0%})")
        return 1

    print("Eval suite passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
