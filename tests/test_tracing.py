"""Tests for tracing module — TraceSpan, Trace, MemoryTracer, helpers."""

import time

import pytest

from star_graph.tracing import (
    TraceSpan,
    Trace,
    TraceSummary,
    MemoryTracer,
    _NoopSpan,
    get_tracer,
    reset_tracer,
    trace_recall,
    _new_id,
)


class TestTraceSpan:
    def test_defaults(self):
        span = TraceSpan(name="test.span")
        assert span.name == "test.span"
        assert span.status == "ok"
        assert span.attributes == {}
        assert span.events == []
        assert span.children == []
        assert span.duration_ms == 0.0

    def test_with_ids(self):
        span = TraceSpan(
            name="test", trace_id="t1", span_id="s1",
            parent_span_id="p1",
        )
        assert span.trace_id == "t1"
        assert span.span_id == "s1"
        assert span.parent_span_id == "p1"

    def test_set_attribute(self):
        span = TraceSpan(name="test")
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_set_attribute_int(self):
        span = TraceSpan(name="test")
        span.set_attribute("count", 42)
        assert span.attributes["count"] == 42

    def test_add_event(self):
        span = TraceSpan(name="test")
        span.add_event("cache.hit", {"key": "val"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "cache.hit"

    def test_add_event_no_attrs(self):
        span = TraceSpan(name="test")
        span.add_event("simple.event")
        assert span.events[0]["attributes"] == {}

    def test_set_status(self):
        span = TraceSpan(name="test")
        span.set_status("error")
        assert span.status == "error"

    def test_finish(self):
        span = TraceSpan(name="test", start_time=100.0)
        span.finish()
        assert span.end_time > 0

    def test_duration_ms(self):
        span = TraceSpan(name="test", start_time=100.0)
        span.end_time = 100.05
        assert span.duration_ms == pytest.approx(50.0)

    def test_to_dict_basic(self):
        span = TraceSpan(name="test", trace_id="t1", span_id="s1")
        span.finish()
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "ok"
        assert "duration_ms" in d
        assert d["children"] == []

    def test_to_dict_with_children(self):
        parent = TraceSpan(name="parent", trace_id="t1", span_id="p1")
        child = TraceSpan(name="child", trace_id="t1", span_id="c1",
                          parent_span_id="p1")
        parent.children.append(child)
        parent.finish()
        child.finish()
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "child"

    def test_to_dict_with_attributes(self):
        span = TraceSpan(name="test", trace_id="t1", span_id="s1")
        span.set_attribute("int_val", 42)
        span.set_attribute("str_val", "hello")
        span.set_attribute("bool_val", True)
        span.set_attribute("none_val", None)
        span.set_attribute("list_val", [1, 2, 3])
        span.finish()
        d = span.to_dict()
        attrs = d["attributes"]
        assert attrs["int_val"] == 42
        assert attrs["str_val"] == "hello"
        assert attrs["bool_val"] is True
        assert attrs["none_val"] is None
        assert isinstance(attrs["list_val"], str)  # serialized to string

    def test_serialize_attrs_custom_object(self):
        span = TraceSpan(name="test")
        span.set_attribute("obj", object())
        span.finish()
        d = span.to_dict()
        assert isinstance(d["attributes"]["obj"], str)


class TestTrace:
    def test_defaults(self):
        root = TraceSpan(name="root")
        t = Trace(root=root, trace_id="abc123")
        assert t.root is root
        assert t.trace_id == "abc123"

    def test_finish(self):
        root = TraceSpan(name="root")
        t = Trace(root=root)
        t.finish()
        assert root.end_time > 0

    def test_to_dict(self):
        root = TraceSpan(name="root", trace_id="t1", span_id="s1")
        t = Trace(root=root, trace_id="t1")
        root.finish()
        d = t.to_dict()
        assert d["trace_id"] == "t1"
        assert "root" in d
        assert "total_duration_ms" in d


class TestTraceSummary:
    def test_defaults(self):
        ts = TraceSummary()
        assert ts.total_traces == 0
        assert ts.recent_traces == []
        assert ts.avg_duration_ms == 0.0
        assert ts.error_count == 0


class TestMemoryTracer:
    def test_init(self):
        mt = MemoryTracer()
        assert mt.max_traces == 100
        assert mt.enabled is True
        assert len(mt._traces) == 0

    def test_init_custom(self):
        mt = MemoryTracer(max_traces=10, enabled=False)
        assert mt.max_traces == 10
        assert mt.enabled is False

    def test_span_disabled(self):
        mt = MemoryTracer(enabled=False)
        with mt.span("test") as span:
            assert isinstance(span, _NoopSpan)

    def test_span_basic(self):
        mt = MemoryTracer()
        with mt.span("test.operation") as span:
            span.set_attribute("key", "value")
        traces = mt.recent()
        assert len(traces) == 1
        assert traces[0].root.name == "test.operation"
        d = traces[0].to_dict()
        assert d["root"]["attributes"]["key"] == "value"

    def test_span_nested(self):
        mt = MemoryTracer()
        with mt.span("outer") as outer:
            outer.set_attribute("level", "outer")
            with mt.span("inner") as inner:
                inner.set_attribute("level", "inner")
        traces = mt.recent()
        assert len(traces) == 1
        d = traces[0].to_dict()
        assert d["root"]["name"] == "outer"
        # Nested spans share the same trace; inner span is NOT auto-added
        # to outer.children (spans are stack-based, not tree-linked)
        assert d["root"]["status"] == "ok"

    def test_span_error(self):
        mt = MemoryTracer()
        try:
            with mt.span("failing.op"):
                raise ValueError("test error")
        except ValueError:
            pass
        traces = mt.recent()
        assert traces[0].root.status == "error"
        assert "error" in traces[0].root.attributes

    def test_record(self):
        mt = MemoryTracer()
        span = mt.record("one-shot", {"key": "val"}, status="ok",
                         duration_ms=100.0)
        assert span.name == "one-shot"
        assert span.attributes["key"] == "val"
        assert len(mt.recent()) == 1

    def test_trace_decorator(self):
        mt = MemoryTracer()

        @mt.trace("my_func")
        def my_function(x):
            return x * 2

        result = my_function(21)
        assert result == 42
        traces = mt.recent()
        assert len(traces) == 1
        d = traces[0].to_dict()
        assert d["root"]["name"] == "my_func"
        assert d["root"]["attributes"]["function"] == "my_function"

    def test_event(self):
        mt = MemoryTracer()
        with mt.span("test"):
            mt.event("cache.miss", {"key": "x"})
        traces = mt.recent()
        events = traces[0].root.events
        assert len(events) == 1
        assert events[0]["name"] == "cache.miss"

    def test_event_no_span(self):
        mt = MemoryTracer()
        # Should not raise
        mt.event("noop")

    def test_set_attribute(self):
        mt = MemoryTracer()
        with mt.span("test"):
            mt.set_attribute("dynamic", 123)
        traces = mt.recent()
        assert traces[0].root.attributes["dynamic"] == 123

    def test_set_attribute_no_span(self):
        mt = MemoryTracer()
        # Should not raise
        mt.set_attribute("noop", 1)

    def test_recent_empty(self):
        mt = MemoryTracer()
        assert mt.recent() == []

    def test_recent_with_n(self):
        mt = MemoryTracer()
        for i in range(10):
            mt.record(f"span{i}")
        assert len(mt.recent(3)) == 3

    def test_clear(self):
        mt = MemoryTracer()
        mt.record("span1")
        mt.record("span2")
        mt.clear()
        assert mt.recent() == []

    def test_on_trace_callback(self):
        mt = MemoryTracer()
        calls = []

        def cb(trace):
            calls.append(trace.trace_id)

        mt.on_trace(cb)
        # Callbacks fire on span() completion, not record()
        with mt.span("span1"):
            pass
        assert len(calls) == 1

    def test_on_trace_callback_error(self):
        mt = MemoryTracer()

        def bad_cb(trace):
            raise RuntimeError("boom")

        mt.on_trace(bad_cb)
        # Should not propagate
        mt.record("span1")
        assert len(mt.recent()) == 1

    def test_summary(self):
        mt = MemoryTracer()
        mt.record("span1")
        mt.record("span2")
        s = mt.summary
        assert s.total_traces == 2
        assert len(s.recent_traces) == 2
        assert s.avg_duration_ms >= 0.0

    def test_summary_empty(self):
        mt = MemoryTracer()
        s = mt.summary
        assert s.total_traces == 0
        assert s.avg_duration_ms == 0.0

    def test_max_traces_eviction(self):
        mt = MemoryTracer(max_traces=3)
        for i in range(5):
            mt.record(f"span{i}")
        traces = mt.recent()
        assert len(traces) == 3
        # Oldest should be evicted
        names = [t.root.name for t in traces]
        assert "span0" not in names
        assert "span1" not in names

    def test_span_multiple_roots(self):
        mt = MemoryTracer()
        with mt.span("first"):
            pass
        with mt.span("second"):
            pass
        traces = mt.recent()
        assert len(traces) == 2
        assert traces[0].root.name == "first"
        assert traces[1].root.name == "second"


class TestNoopSpan:
    def test_set_attribute(self):
        ns = _NoopSpan()
        ns.set_attribute("k", "v")  # no-op

    def test_add_event(self):
        ns = _NoopSpan()
        ns.add_event("e")  # no-op

    def test_set_status(self):
        ns = _NoopSpan()
        ns.set_status("error")  # no-op

    def test_finish(self):
        ns = _NoopSpan()
        ns.finish()  # no-op

    def test_context_manager(self):
        with _NoopSpan() as span:
            assert isinstance(span, _NoopSpan)


class TestGlobalTracer:
    def setup_method(self):
        reset_tracer()

    def teardown_method(self):
        reset_tracer()

    def test_get_tracer_creates_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_get_tracer_disabled(self):
        t = get_tracer(enabled=False)
        with t.span("test") as span:
            assert isinstance(span, _NoopSpan)

    def test_reset_tracer(self):
        t1 = get_tracer()
        reset_tracer()
        t2 = get_tracer()
        assert t1 is not t2


class TestTraceRecall:
    def test_basic(self):
        mt = MemoryTracer()
        ctx_out = {}
        with trace_recall(mt, "Redis timeout query", max_items=5) as ctx:
            ctx["exact_cache"] = []
            ctx["graph"] = ["a1", "a2"]
            ctx["raw"] = ["r1"]
            ctx["final"] = []
            ctx["layers_visited"] = ["graph"]
            ctx_out = ctx
        traces = mt.recent()
        assert len(traces) == 1
        d = traces[0].to_dict()
        attrs = d["root"]["attributes"]
        assert attrs["query"] == "Redis timeout query"
        assert attrs["max_items"] == 5

    def test_with_results(self):
        mt = MemoryTracer()
        with trace_recall(mt, "test query") as ctx:
            from star_graph.graph import StarGraph
            from star_graph.anchor import Anchor
            g = StarGraph()
            a1 = Anchor(id="a1", text="result 1")
            a1.relevance_score = 0.95
            a2 = Anchor(id="a2", text="result 2")
            a2.relevance_score = 0.80
            ctx["exact_cache"] = [a1]
            ctx["graph"] = [a1, a2]
            ctx["raw"] = []
            ctx["final"] = [a1, a2]
            ctx["layers_visited"] = ["exact_cache", "graph"]
        traces = mt.recent()
        d = traces[0].to_dict()
        attrs = d["root"]["attributes"]
        assert attrs["exact_hits"] == 1
        assert attrs["graph_items"] == 2
        assert attrs["final_count"] == 2
        assert "top3_scores" in attrs

    def test_exception(self):
        mt = MemoryTracer()
        try:
            with trace_recall(mt, "failing query"):
                raise ValueError("retrieval failed")
        except ValueError:
            pass
        traces = mt.recent()
        assert traces[0].root.status == "error"


class TestNewId:
    def test_returns_string(self):
        id_ = _new_id()
        assert isinstance(id_, str)
        assert len(id_) == 16

    def test_unique(self):
        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100
