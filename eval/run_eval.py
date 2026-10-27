"""CI gate: runs the full eval set against the CRAG graph (mock backend, no
API key needed) and fails the build if route accuracy or refusal accuracy on
unanswerable questions drops below a regression threshold.

Thresholds are set from a real measured baseline run plus a small margin
(see results/RESULTS.md for the run they came from) - not guessed.
"""
from __future__ import annotations

import sys
import tempfile

from eval.metrics import load_eval_cases, run_eval, summarize

# Set from the measured run in results/RESULTS.md (route_accuracy=100%,
# refusal_accuracy=60%), each with a margin below that - not guessed. The
# mock backend is deterministic, so these exist to catch a regression from a
# future code/eval_set change, not run-to-run noise.
ROUTE_ACCURACY_THRESHOLD = 0.95
REFUSAL_ACCURACY_THRESHOLD = 0.50


def main() -> int:
    cases = load_eval_cases()
    with tempfile.TemporaryDirectory() as chroma_dir:
        rows = run_eval("crag", chroma_dir)

    for row in rows:
        route_mark = "." if row["route_correct"] in (True, None) else "X"
        print(f"[{route_mark}] {row['id']:<16} route={row['route']!r} retries={row['retries']}")

    summary = summarize(rows)
    print(f"\nn={summary['n']} cases")
    print(f"route_accuracy={summary['route_accuracy']}")
    print(f"refusal_accuracy={summary['refusal_accuracy']} (n={summary['refusal_accuracy_n']})")
    print(f"groundedness={summary['groundedness']}")
    print(f"correctness={summary['correctness']} (scored on {summary['correctness_n_scored']}/{summary['n']})")
    print(f"re_retrieval_rate={summary['re_retrieval_rate']}")
    print(f"weak_context_escalation_rate={summary['weak_context_escalation_rate']}")
    print(f"avg_llm_calls={summary['avg_llm_calls']}")
    print(f"latency_p50_ms={summary['latency_p50_ms']:.2f} latency_p95_ms={summary['latency_p95_ms']:.2f}")

    failed = False
    if summary["route_accuracy"] is not None and summary["route_accuracy"] < ROUTE_ACCURACY_THRESHOLD:
        print(f"\nFAILED: route_accuracy {summary['route_accuracy']:.0%} < {ROUTE_ACCURACY_THRESHOLD:.0%}")
        failed = True
    if summary["refusal_accuracy"] is not None and summary["refusal_accuracy"] < REFUSAL_ACCURACY_THRESHOLD:
        print(
            f"\nFAILED: refusal_accuracy {summary['refusal_accuracy']:.0%} < {REFUSAL_ACCURACY_THRESHOLD:.0%}"
        )
        failed = True

    if failed:
        return 1

    print("\nEval suite passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
