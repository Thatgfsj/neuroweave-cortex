"""Observability tracing — lightweight span-based instrumentation.

Provides OpenTelemetry-compatible span wrappers for debugging
"why was this memory recalled?" and measuring system performance.

No external dependencies — self-contained implementation that
emits OTel-compatible span dicts. Can be upgraded to full
OpenTelemetry SDK by setting TRACING_BACKEND = "otel".

Traces captured per recall:
- Query text, response anchor IDs, scores
- Phase values, resonance strength, PageRank scores
- Retrieval layers visited (exact_cache / raw_buffer / community / cortex / hub)
- Timing breakdown per layer
- Gating decisions and lateral inhibition effects

Usage:
    from .tracing import TraceContext, get_tracer

    tracer = get_tracer()

    @tracer.trace("recall")
    def recall(query, context):
        ...

    # Or manual spans:
    with tracer.span("gate.select") as span:
        span.set_attribute("candidates", 42)
        result = gate.gate(...)
        span.set_attribute("selected", len(result))
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Callable


# ── Data structures ───────────────────────────────────────────

@dataclass
class TraceSpan:
    """A single span within a trace — one operation (e.g., a retrieval layer)."""
    name: str
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    status: str = "ok"  # "ok", "error"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    children: list[TraceSpan] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    def set_attribute(self, key: str, value):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: str):
        self.status = status

    def finish(self):
        self.end_time = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": self._serialize_attrs(self.attributes),
            "events": self.events,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def _serialize_attrs(attrs: dict) -> dict:
        result = {}
        for k, v in attrs.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            elif isinstance(v, (list, tuple)):
                result[k] = str(v)[:200]
            else:
                result[k] = str(v)[:200]
        return result


@dataclass
class Trace:
    """A full trace — root span with all child spans."""
    root: TraceSpan
    trace_id: str = ""
    started_at: float = field(default_factory=time.time)

    def finish(self):
        self.root.finish()

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "root": self.root.to_dict(),
            "total_duration_ms": round(self.root.duration_ms, 3),
        }


@dataclass
class TraceSummary:
    """Lightweight summary of recent trace activity."""
    total_traces: int = 0
    recent_traces: list[dict] = field(default_factory=list)
    avg_duration_ms: float = 0.0
    error_count: int = 0


# ── Tracer ────────────────────────────────────────────────────

class MemoryTracer:
    """Lightweight span-based tracer for star-graph operations.

    No external dependencies. Emits structured trace dicts that can
    be sent to any observability backend.

    Usage:
        tracer = MemoryTracer(max_traces=100)

        with tracer.span("recall.full") as span:
            span.set_attribute("query", "Redis timeout")
            # ... do retrieval ...
            span.set_attribute("results", 5)

        # Read recent traces
        for trace in tracer.recent():
            print(trace.to_dict())
    """

    def __init__(self, max_traces: int = 100, enabled: bool = True):
        self.max_traces = max_traces
        self.enabled = enabled
        self._traces: list[Trace] = []
        self._current: Trace | None = None
        self._span_stack: list[TraceSpan] = []
        self._total_traces: int = 0
        self._error_count: int = 0
        self._on_trace_callbacks: list[Callable] = []

    @contextmanager
    def span(self, name: str, attributes: dict | None = None):
        """Create a span context manager.

        Usage:
            with tracer.span("retrieval.gate") as span:
                span.set_attribute("k", 20)
                result = gate.gate(candidates, ...)
        """
        if not self.enabled:
            yield _NoopSpan()
            return

        span = TraceSpan(
            name=name,
            trace_id=self._current.trace_id if self._current else _new_id(),
            span_id=_new_id(),
            parent_span_id=self._span_stack[-1].span_id if self._span_stack else "",
        )
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)

        is_root = not self._span_stack

        if is_root:
            self._current = Trace(root=span, trace_id=span.trace_id)

        self._span_stack.append(span)

        try:
            yield span
            span.set_status("ok")
        except Exception as exc:
            span.set_status("error")
            span.set_attribute("error", str(exc))
            self._error_count += 1
            raise
        finally:
            span.finish()
            self._span_stack.pop()

            if is_root and self._current:
                self._current.finish()
                self._traces.append(self._current)
                self._total_traces += 1

                # Evict oldest if full
                while len(self._traces) > self.max_traces:
                    self._traces.pop(0)

                # Fire callbacks
                for cb in self._on_trace_callbacks:
                    try:
                        cb(self._current)
                    except Exception:
                        pass

                self._current = None

    def record(self, name: str, attributes: dict | None = None,
               status: str = "ok", duration_ms: float = 0.0) -> TraceSpan:
        """Record a completed span directly (no context manager needed).

        Convenient for one-shot recording after an operation completes.
        """
        span = TraceSpan(
            name=name,
            trace_id=_new_id(),
            span_id=_new_id(),
            attributes=attributes or {},
            status=status,
        )
        span.end_time = span.start_time + (duration_ms / 1000)
        self._traces.append(Trace(root=span, trace_id=span.trace_id))
        self._total_traces += 1

        while len(self._traces) > self.max_traces:
            self._traces.pop(0)

        return span

    def trace(self, name: str):
        """Decorator: trace a function call as a span.

        Usage:
            @tracer.trace("recall")
            def recall(query, context):
                ...
        """
        def decorator(fn):
            def wrapper(*args, **kwargs):
                with self.span(name) as span:
                    span.set_attribute("function", fn.__name__)
                    return fn(*args, **kwargs)
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper
        return decorator

    def event(self, name: str, attributes: dict | None = None):
        """Record an instantaneous event on the current span."""
        if not self.enabled or not self._span_stack:
            return
        self._span_stack[-1].add_event(name, attributes)

    def set_attribute(self, key: str, value):
        """Set an attribute on the current span."""
        if not self.enabled or not self._span_stack:
            return
        self._span_stack[-1].set_attribute(key, value)

    def recent(self, n: int | None = None) -> list[Trace]:
        """Get recent traces (most recent last)."""
        traces = self._traces[-n:] if n else self._traces
        return list(traces)

    def clear(self):
        """Clear all stored traces."""
        self._traces.clear()

    def on_trace(self, callback: Callable):
        """Register a callback invoked on each completed trace."""
        self._on_trace_callbacks.append(callback)

    @property
    def summary(self) -> TraceSummary:
        durations = [t.root.duration_ms for t in self._traces]
        avg = sum(durations) / len(durations) if durations else 0.0
        return TraceSummary(
            total_traces=self._total_traces,
            recent_traces=[t.to_dict() for t in self._traces[-5:]],
            avg_duration_ms=round(avg, 3),
            error_count=self._error_count,
        )


class _NoopSpan:
    """No-op span when tracing is disabled."""
    def set_attribute(self, key, value): pass
    def add_event(self, name, attributes=None): pass
    def set_status(self, status): pass
    def finish(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ── Global tracer ─────────────────────────────────────────────

_tracer: MemoryTracer | None = None


def get_tracer(enabled: bool = True) -> MemoryTracer:
    """Get or create the global memory tracer."""
    global _tracer
    if _tracer is None:
        _tracer = MemoryTracer(enabled=enabled)
    return _tracer


def reset_tracer():
    """Reset the global tracer."""
    global _tracer
    _tracer = None


# ── Helper: trace a recall operation with full detail ────────

@contextmanager
def trace_recall(tracer: MemoryTracer, query: str, max_items: int = 10):
    """Convenience: trace a full recall operation with standard attributes.

    Usage:
        with trace_recall(tracer, query) as ctx:
            ctx["exact_cache"] = exact_hits
            ctx["graph"] = graph_items
            ctx["raw"] = raw_items
            ...merge...
            ctx["final"] = merged
    """
    ctx: dict = {}
    with tracer.span("recall") as span:
        span.set_attribute("query", query[:200])
        span.set_attribute("max_items", max_items)
        try:
            yield ctx
            span.set_attribute("exact_hits", len(ctx.get("exact_cache", [])))
            span.set_attribute("graph_items", len(ctx.get("graph", [])))
            span.set_attribute("raw_chunks", len(ctx.get("raw", [])))
            span.set_attribute("final_count", len(ctx.get("final", [])))
            if ctx.get("final"):
                top_scores = [f"{item.relevance_score:.3f}" for item in ctx["final"][:3]]
                span.set_attribute("top3_scores", top_scores)
                span.set_attribute("layers_visited", ctx.get("layers_visited", []))
        except Exception:
            span.set_status("error")
            raise


# ── ID generation ─────────────────────────────────────────────

def _new_id() -> str:
    return uuid.uuid4().hex[:16]
