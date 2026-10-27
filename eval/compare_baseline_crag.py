"""Runs the full eval set against BOTH baseline and crag modes (same
documents, same retriever settings, same LLM/prompts - mode is the only
difference) and writes results/eval_results.json + results/RESULTS.md.

Uses the deterministic mock backend (see eval/metrics.py's module docstring
for exactly what that does and doesn't tell you). Run with:

    python -m eval.compare_baseline_crag
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval.metrics import load_eval_cases, run_eval, summarize

RESULTS_DIR = Path(__file__).parent.parent / "results"


def recovery_rate(baseline_rows: list[dict], crag_rows: list[dict]) -> dict:
    """Of the questions where CRAG's grader triggered a re-retrieval, how
    often was baseline's single-pass answer wrong/ungrounded while CRAG's
    answer (after correction) was right/grounded? Only computed over
    questions with a scorable correctness check (non-empty key_facts)."""
    by_id_baseline = {r["id"]: r for r in baseline_rows}
    triggered = [r for r in crag_rows if r["re_retrieval_triggered"]]
    scorable = [r for r in triggered if r["correctness"] is not None]
    recovered = 0
    for crag_row in scorable:
        base_row = by_id_baseline[crag_row["id"]]
        baseline_failed = not (base_row["correctness"] and base_row["grounded"])
        crag_succeeded = bool(crag_row["correctness"] and crag_row["grounded"])
        if baseline_failed and crag_succeeded:
            recovered += 1
    return {
        "re_retrieval_triggered_n": len(triggered),
        "scorable_n": len(scorable),
        "recovered_n": recovered,
        "recovery_rate": (recovered / len(scorable)) if scorable else None,
    }


def per_category(rows: list[dict]) -> dict:
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {cat: summarize(category_rows) for cat, category_rows in cats.items()}


def main() -> None:
    cases = load_eval_cases()
    with tempfile.TemporaryDirectory() as d:
        baseline_rows = run_eval("baseline", d)
        crag_rows = run_eval("crag", d)

    baseline_summary = summarize(baseline_rows)
    crag_summary = summarize(crag_rows)
    recovery = recovery_rate(baseline_rows, crag_rows)

    RESULTS_DIR.mkdir(exist_ok=True)

    results = {
        "n_cases": len(cases),
        "backend": "mock (FakeLLMClient + HashingEmbeddings, no OPENAI_API_KEY in this environment)",
        "baseline": baseline_summary,
        "crag": crag_summary,
        "recovery": recovery,
        "baseline_rows": baseline_rows,
        "crag_rows": crag_rows,
        "baseline_by_category": per_category(baseline_rows),
        "crag_by_category": per_category(crag_rows),
    }
    (RESULTS_DIR / "eval_results.json").write_text(json.dumps(results, indent=2, default=str))

    def fmt(v, suffix=""):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.1%}{suffix}" if suffix == "" else f"{v:.1f}{suffix}"
        return str(v)

    lines = [
        "# Baseline vs. CRAG — eval results",
        "",
        f"Backend: **{results['backend']}**",
        "",
        f"Eval set: {len(cases)} questions ({', '.join(sorted(per_category(crag_rows).keys()))})",
        "",
        "## Headline metrics",
        "",
        "| Metric | Baseline | CRAG |",
        "|---|---|---|",
        f"| Correctness (of {crag_summary['correctness_n_scored']} scorable) | "
        f"{fmt(baseline_summary['correctness'])} | {fmt(crag_summary['correctness'])} |",
        f"| Groundedness (lexical proxy) | {fmt(baseline_summary['groundedness'])} | {fmt(crag_summary['groundedness'])} |",
        f"| Refusal accuracy (n={crag_summary['refusal_accuracy_n']} unanswerable) | "
        f"{fmt(baseline_summary['refusal_accuracy'])} | {fmt(crag_summary['refusal_accuracy'])} |",
        f"| Route accuracy | {fmt(baseline_summary['route_accuracy'])} | {fmt(crag_summary['route_accuracy'])} |",
        f"| Escalation rate | {fmt(baseline_summary['escalation_rate'])} | {fmt(crag_summary['escalation_rate'])} |",
        f"| Avg LLM calls / query | {fmt(baseline_summary['avg_llm_calls'], 'x')} | {fmt(crag_summary['avg_llm_calls'], 'x')} |",
        f"| Latency p50 (ms) | {fmt(baseline_summary['latency_p50_ms'], 'ms')} | {fmt(crag_summary['latency_p50_ms'], 'ms')} |",
        f"| Latency p95 (ms) | {fmt(baseline_summary['latency_p95_ms'], 'ms')} | {fmt(crag_summary['latency_p95_ms'], 'ms')} |",
        "",
        "## Correction behavior (CRAG only)",
        "",
        f"- Re-retrieval triggered on {recovery['re_retrieval_triggered_n']}/{len(cases)} questions "
        f"({fmt(crag_summary['re_retrieval_rate'])})",
        f"- Weak-context escalation (gave up rather than answer from rejected context): "
        f"{fmt(crag_summary['weak_context_escalation_rate'])}",
        f"- Recovery rate (of {recovery['scorable_n']} re-retrieval cases with a scorable fact check, "
        f"fraction where baseline was wrong/ungrounded but CRAG was right): {fmt(recovery['recovery_rate'])}",
        "",
        "## By category (CRAG)",
        "",
        "| Category | n | Correctness | Groundedness | Re-retrieval rate |",
        "|---|---|---|---|---|",
    ]
    for cat, summ in sorted(per_category(crag_rows).items()):
        lines.append(
            f"| {cat} | {summ['n']} | {fmt(summ['correctness'])} | {fmt(summ['groundedness'])} | {fmt(summ['re_retrieval_rate'])} |"
        )

    lines += [
        "",
        "## Methodology caveats",
        "",
        "- Both modes share one LLM client and one vectorstore (same documents, same retriever "
        "settings, same generation prompt); mode only changes which graph nodes run.",
        "- This run used the deterministic mock backend (no OPENAI_API_KEY in this environment). "
        "These are real measured numbers from actually running the code, not estimates - but they "
        "characterize the mock's behavior, not a production LLM's. Re-run with a real key for "
        "production numbers.",
        "- Groundedness is a stemmed-keyword-overlap proxy between the answer and the retrieved "
        "context, not semantic entailment or an LLM-judge pass (no independent judge model was "
        "available in this environment).",
        "- `ambiguous` category questions have no key_facts (the question is intentionally "
        "underspecified), so they're excluded from the correctness denominator; they still count "
        "toward groundedness and routing.",
    ]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'eval_results.json'} and {RESULTS_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
