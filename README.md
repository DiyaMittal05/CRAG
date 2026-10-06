# CRAG Support Agent

A self-correcting Retrieval-Augmented Generation (Corrective RAG / CRAG)
support agent. Instead of a linear retrieve-then-answer chain, it grades the
relevance of retrieved documents and conditionally rewrites the query and
re-retrieves when results are weak, then critiques its own draft answer for
groundedness before responding.

## Architecture

Three agents, wired together as a branching [LangGraph](https://github.com/langchain-ai/langgraph) state machine:

- **Router** — classifies each question as `vectorstore` (needs doc lookup),
  `direct_answer` (greeting/small talk), or `escalate` (explicit human
  request or sensitive issue).
- **Retrieval / Grader** — retrieves candidate documents from Chroma, grades
  each one's relevance with a structured `DocumentGrade`, and rewrites the
  query for a bounded number of retries when too few documents are relevant.
- **Critique** — checks the drafted answer against the retrieved context for
  groundedness and relevance; an ungrounded answer triggers another
  retrieve-rewrite-generate cycle (up to `CRAG_MAX_RETRIES`) before the agent
  gives up and flags the response as escalated.

```
route_query ──escalate──────────────────────────────► escalate ─► END
     │
     ├──direct_answer───────────────────────────────► generate
     │
     └──vectorstore──► retrieve ──► grade_documents ──┬─(relevant enough)──► generate
                           ▲                            │
                           │                   (weak, retries left)
                           └──────── transform_query ◄──┘
                                                            generate ──► critique ──┬─(grounded)──► finalize ─► END
                                                                                     └─(not grounded, retries left)──► transform_query
```

All structured agent outputs (`RouteDecision`, `DocumentGrade`, `Critique`)
are Pydantic models, so routing/grading/critique decisions are validated,
typed, and safe to log.

## Observability

Every graph node is wrapped in a `Tracer.node_span`, emitting structured JSON
start/end events (node name, duration, key decision data) keyed by
`trace_id`. See [`app/observability/tracer.py`](app/observability/tracer.py).

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY

python -m app.vectorstore.ingest   # index data/docs into Chroma
uvicorn app.main:app --reload
```

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I reset my password?"}'
```

Without an `OPENAI_API_KEY`, the app automatically falls back to a
deterministic fake LLM/embedding backend (see `app/llm/fake.py`) — useful for
exploring the graph's control flow without any API cost.

## Tests & eval

```bash
pytest
python -m eval.run_eval
```

`tests/` covers schema validation, graph control flow (routing, the
self-correction retry loop, escalation), and the FastAPI endpoint.
`eval/run_eval.py` runs a fixed set of support questions end-to-end against
the deterministic fake backend and fails if the pass rate drops below 75% —
this runs on every push via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Project layout

```
app/
  agents/        router, retrieval/grader, critique, answer generator
  graph/         LangGraph state + branching wiring
  llm/           LLM client protocol + OpenAI/fake implementations
  vectorstore/   Chroma store, embeddings, doc ingestion
  observability/ structured trace logging
  main.py        FastAPI app
data/docs/       sample support knowledge base (markdown)
eval/            eval-set + CI-gated runner
tests/           pytest suite
```
