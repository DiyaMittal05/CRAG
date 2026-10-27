# Baseline vs. CRAG — eval results

Backend: **mock (FakeLLMClient + HashingEmbeddings, no OPENAI_API_KEY in this environment)**

Eval set: 58 questions (account_access, ambiguous, api_usage, billing, data_export, escalate, greeting, integrations, multi_hop, notifications, security, unanswerable)

## Headline metrics

| Metric | Baseline | CRAG |
|---|---|---|
| Correctness (of 38 scorable) | 55.3% | 60.5% |
| Groundedness (lexical proxy) | 89.7% | 89.7% |
| Refusal accuracy (n=10 unanswerable) | 0.0% | 60.0% |
| Route accuracy | 100.0% | 100.0% |
| Escalation rate | 5.2% | 17.2% |
| Avg LLM calls / query | 1.9x | 7.4x |
| Latency p50 (ms) | 1.5ms | 2.4ms |
| Latency p95 (ms) | 1.7ms | 4.6ms |

## Correction behavior (CRAG only)

- Re-retrieval triggered on 7/58 questions (12.1%)
- Weak-context escalation (gave up rather than answer from rejected context): 12.1%
- Recovery rate (of 0 re-retrieval cases with a scorable fact check, fraction where baseline was wrong/ungrounded but CRAG was right): n/a

## By category (CRAG)

| Category | n | Correctness | Groundedness | Re-retrieval rate |
|---|---|---|---|---|
| account_access | 6 | 50.0% | 100.0% | 0.0% |
| ambiguous | 4 | n/a | 100.0% | 25.0% |
| api_usage | 6 | 66.7% | 100.0% | 0.0% |
| billing | 6 | 66.7% | 100.0% | 0.0% |
| data_export | 6 | 66.7% | 100.0% | 0.0% |
| escalate | 3 | n/a | 0.0% | 0.0% |
| greeting | 3 | n/a | 0.0% | 0.0% |
| integrations | 3 | 33.3% | 100.0% | 0.0% |
| multi_hop | 5 | 60.0% | 100.0% | 0.0% |
| notifications | 3 | 66.7% | 100.0% | 0.0% |
| security | 3 | 66.7% | 100.0% | 0.0% |
| unanswerable | 10 | n/a | 100.0% | 60.0% |

## Methodology caveats

- Both modes share one LLM client and one vectorstore (same documents, same retriever settings, same generation prompt); mode only changes which graph nodes run.
- This run used the deterministic mock backend (no OPENAI_API_KEY in this environment). These are real measured numbers from actually running the code, not estimates - but they characterize the mock's behavior, not a production LLM's. Re-run with a real key for production numbers.
- Groundedness is a stemmed-keyword-overlap proxy between the answer and the retrieved context, not semantic entailment or an LLM-judge pass (no independent judge model was available in this environment).
- `ambiguous` category questions have no key_facts (the question is intentionally underspecified), so they're excluded from the correctness denominator; they still count toward groundedness and routing.
