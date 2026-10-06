"""Lightweight structured tracing for graph node execution.

Each node emits one JSON event on entry and one on exit (with duration and a
truncated view of the state delta) to the `crag.trace` logger. In production
this can be pointed at any log sink; for local/dev runs it also keeps events
in memory, retrievable by `trace_id`, so the API can echo a summary back.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("crag.trace")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class TraceEvent:
    trace_id: str
    node: str
    phase: str  # "start" | "end"
    timestamp: float
    duration_ms: float | None = None
    detail: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "trace_id": self.trace_id,
                "node": self.node,
                "phase": self.phase,
                "timestamp": self.timestamp,
                "duration_ms": self.duration_ms,
                **self.detail,
            },
            default=str,
        )


class Tracer:
    """Collects events for a single trace_id and emits them as structured logs."""

    _store: dict[str, list[TraceEvent]] = {}

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        Tracer._store.setdefault(self.trace_id, [])

    @contextmanager
    def node_span(self, node_name: str, **start_detail):
        start = time.perf_counter()
        start_event = TraceEvent(
            trace_id=self.trace_id, node=node_name, phase="start", timestamp=time.time(), detail=start_detail
        )
        self._emit(start_event)
        result_detail: dict = {}
        try:
            yield result_detail
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            end_event = TraceEvent(
                trace_id=self.trace_id,
                node=node_name,
                phase="end",
                timestamp=time.time(),
                duration_ms=round(duration_ms, 2),
                detail=result_detail,
            )
            self._emit(end_event)

    def _emit(self, event: TraceEvent) -> None:
        Tracer._store[self.trace_id].append(event)
        logger.info(event.to_json())

    @classmethod
    def events_for(cls, trace_id: str) -> list[TraceEvent]:
        return cls._store.get(trace_id, [])
