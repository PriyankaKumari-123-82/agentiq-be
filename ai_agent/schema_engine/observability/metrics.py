"""Structured metrics and tracing for every schema-engine operation.

All timing / counting is done via a lightweight ``SchemaMetrics`` collector
that writes structured log records and optionally forwards to Langfuse or
any OpenTelemetry-compatible backend.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class MetricEvent:
    """Single metric data-point."""
    name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SchemaMetrics:
    """Collect and expose schema-engine metrics."""

    def __init__(self):
        self._events: List[MetricEvent] = []
        self._counters: Dict[str, float] = {}

    # ── Counters ─────────────────────────────────────────────────────────

    def inc(self, name: str, amount: float = 1, **tags: str) -> None:
        self._counters[name] = self._counters.get(name, 0) + amount
        self._events.append(MetricEvent(name=name, value=amount, tags=tags))

    # ── Timers ───────────────────────────────────────────────────────────

    @contextmanager
    def timer(self, name: str, **tags: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._events.append(
                MetricEvent(name=name, value=elapsed_ms, unit="ms", tags=tags)
            )
            log.info("METRIC %s=%.1fms %s", name, elapsed_ms, tags or "")

    @asynccontextmanager
    async def async_timer(self, name: str, **tags: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._events.append(
                MetricEvent(name=name, value=elapsed_ms, unit="ms", tags=tags)
            )
            log.info("METRIC %s=%.1fms %s", name, elapsed_ms, tags or "")

    # ── Structured log helpers ───────────────────────────────────────────

    def record_retrieval(
        self,
        server: str,
        query: str,
        tables_returned: int,
        token_estimate: int,
        latency_ms: float,
        confidence: float,
    ) -> None:
        self.inc("retrieval.count", server=server)
        self._events.append(MetricEvent(
            name="retrieval",
            value=latency_ms,
            unit="ms",
            tags={
                "server": server,
                "tables_returned": str(tables_returned),
                "token_estimate": str(token_estimate),
                "confidence": f"{confidence:.2f}",
            },
        ))
        log.info(
            "RETRIEVAL server=%s tables=%d tokens=~%d latency=%.0fms confidence=%.2f query=%s",
            server, tables_returned, token_estimate, latency_ms, confidence,
            query[:80],
        )

    def record_validation(
        self,
        is_valid: bool,
        errors: int,
        warnings: int,
        complexity: str,
    ) -> None:
        self.inc("validation.count")
        if not is_valid:
            self.inc("validation.failures")
        log.info(
            "VALIDATION valid=%s errors=%d warnings=%d complexity=%s",
            is_valid, errors, warnings, complexity,
        )

    def record_sql_execution(
        self,
        server: str,
        latency_ms: float,
        row_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        self.inc("execution.count", server=server)
        if error:
            self.inc("execution.errors", server=server)
        log.info(
            "SQL_EXEC server=%s latency=%.0fms rows=%s error=%s",
            server, latency_ms, row_count, error,
        )

    def record_token_reduction(
        self,
        full_schema_tokens: int,
        context_tokens: int,
    ) -> None:
        reduction = (1 - context_tokens / max(full_schema_tokens, 1)) * 100
        log.info(
            "TOKEN_REDUCTION full=%d context=%d reduction=%.1f%%",
            full_schema_tokens, context_tokens, reduction,
        )
        self.inc("token_reduction.total_saved", full_schema_tokens - context_tokens)

    # ── Export ────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "recent_events": len(self._events),
        }

    def flush(self) -> List[MetricEvent]:
        events = list(self._events)
        self._events.clear()
        return events


# ── Singleton ────────────────────────────────────────────────────────────

_metrics: Optional[SchemaMetrics] = None


def get_metrics() -> SchemaMetrics:
    global _metrics
    if _metrics is None:
        _metrics = SchemaMetrics()
    return _metrics
