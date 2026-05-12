"""Tests for v0.8 new modules: exact_cache, micro_sleep, cost_estimator,
snapshot, async_manager, tracing, benchmark."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from star_graph.exact_cache import (
    ExactMatchCache, ExactMatchEntry, extract_entity_keys,
)
from star_graph.micro_sleep import (
    MicroSleepScheduler, MicroSleepProgress, MicroSleepResult, MicroPhase,
)
from star_graph.cost_estimator import (
    SleepCostEstimator, CostEstimate,
)
from star_graph.snapshot import (
    SnapshotManager, SnapshotMeta,
)
from star_graph.tracing import (
    MemoryTracer, TraceSpan, Trace, get_tracer, reset_tracer,
)
from star_graph.benchmark import (
    BenchmarkSuite, BenchmarkScenario, Category, BenchmarkResult,
    run_benchmark,
)


# ═══════════════════════════════════════════════════════════════
# Exact Cache
# ═══════════════════════════════════════════════════════════════

class TestExactCache:
    def test_put_and_get(self):
        cache = ExactMatchCache()
        cache.put("alice-birthday", "a1", "Alice's birthday is May 10th", 0.9)
        results = cache.get("alice-birthday")
        assert len(results) == 1
        assert results[0].anchor_id == "a1"
        assert results[0].text == "Alice's birthday is May 10th"

    def test_get_miss_returns_empty(self):
        cache = ExactMatchCache()
        assert cache.get("nonexistent") == []
        assert cache.misses == 1

    def test_has_membership_test(self):
        cache = ExactMatchCache()
        cache.put("key1", "a1", "test")
        assert cache.has("key1")
        assert not cache.has("key2")

    def test_hit_count_increments(self):
        cache = ExactMatchCache()
        cache.put("key1", "a1", "test")
        cache.get("key1")
        cache.get("key1")
        assert cache.hits == 2

    def test_deduplicate_same_anchor_id(self):
        cache = ExactMatchCache()
        e1 = cache.put("key1", "a1", "text v1", 0.5)
        e2 = cache.put("key1", "a1", "text v2", 0.8)
        assert len(cache._store["key1"]) == 1
        assert e2.text == "text v2"
        assert e2.confidence == 0.8

    def test_eviction_when_bucket_full(self):
        cache = ExactMatchCache(max_entries_per_key=2)
        cache.put("key1", "a1", "low", 0.3)
        cache.put("key1", "a2", "mid", 0.5)
        cache.put("key1", "a3", "high", 0.9)
        assert len(cache._store["key1"]) == 2
        # lowest confidence (0.3) should be evicted
        ids = {e.anchor_id for e in cache._store["key1"]}
        assert "a1" not in ids

    def test_remove_anchor(self):
        cache = ExactMatchCache()
        cache.put("key1", "a1", "test1")
        cache.put("key2", "a1", "test2")
        cache.put("key3", "a2", "test3")
        cache.remove_anchor("a1")
        assert not cache.has("key1")
        assert not cache.has("key2")
        assert cache.has("key3")

    def test_harvest_from_anchor(self):
        from star_graph.anchor import Anchor
        anchor = Anchor.create("Alice's birthday is May 10th", tags=["person", "birthday"],
                              importance=0.8)
        cache = ExactMatchCache()
        keys = cache.harvest_from_anchor(anchor)
        assert len(keys) > 0
        assert "alice-birthday" in keys
        assert cache.has("alice-birthday")

    def test_stats(self):
        cache = ExactMatchCache()
        cache.put("k1", "a1", "test")
        cache.get("k1")
        cache.get("k2")
        s = cache.stats
        assert s["entries"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert 0.4 < s["hit_rate"] < 0.6

    def test_clear(self):
        cache = ExactMatchCache()
        cache.put("k1", "a1", "test")
        cache.clear()
        assert cache.size == 0
        assert not cache.has("k1")


class TestExtractEntityKeys:
    def test_extract_entity_attribute(self):
        keys = extract_entity_keys("Alice's birthday is May 10th")
        assert "alice-birthday" in keys

    def test_extract_key_value(self):
        keys = extract_entity_keys("The port is 6379 for Redis")
        assert any("port" in k and "6379" in k for k in keys)

    def test_extract_preference(self):
        keys = extract_entity_keys("User prefers dark mode", ["preference", "ui"])
        assert any("dark_mode" in k for k in keys)

    def test_extract_port(self):
        keys = extract_entity_keys("Server listens on port 8080")
        assert any("port-8080" in k for k in keys)

    def test_extract_version(self):
        keys = extract_entity_keys("Python version 3.11.9 is used")
        assert any("3.11.9" in k or "version" in k for k in keys)

    def test_extract_tags_as_keys(self):
        keys = extract_entity_keys("Some random text", ["redis", "config"])
        assert "tag:redis" in keys
        assert "tag:config" in keys

    def test_max_five_keys(self):
        keys = extract_entity_keys(
            "Alice's birthday is May 10th and Bob's age is 30. "
            "Redis port is 6379. Python version 3.11.9. "
            "User prefers dark mode.",
            ["person", "config", "preference", "ui", "python", "version"]
        )
        assert len(keys) <= 5


# ═══════════════════════════════════════════════════════════════
# Micro-Sleep Scheduler
# ═══════════════════════════════════════════════════════════════

class TestMicroSleep:
    def test_progress_initial_state(self):
        p = MicroSleepProgress()
        assert not p.is_complete
        assert p.phase_index == 0
        assert p.progress_pct == 0.0

    def test_resume_from(self):
        scheduler = MicroSleepScheduler()
        scheduler.resume_from(3)
        assert scheduler.progress.phase_index == 3
        assert len(scheduler.progress.phases_completed) == 3

    def test_reset(self):
        scheduler = MicroSleepScheduler()
        scheduler.resume_from(5)
        scheduler.reset()
        assert scheduler.progress.phase_index == 0

    def test_is_complete_after_all_phases(self):
        scheduler = MicroSleepScheduler()
        scheduler.resume_from(10)
        assert scheduler.is_complete

    def test_get_summary(self):
        scheduler = MicroSleepScheduler()
        summary = scheduler.get_summary()
        assert "0/10" in summary
        assert "0%" in summary


# ═══════════════════════════════════════════════════════════════
# Cost Estimator
# ═══════════════════════════════════════════════════════════════

class TestCostEstimator:
    def test_empty_graph_estimate(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        estimator = SleepCostEstimator()
        est = estimator.estimate(mgr)
        assert est.total_anchors == 0
        assert est.llm_cost_usd == 0.0
        assert est.is_free

    def test_estimate_with_anchors(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        mgr.remember("Test memory 1", tags=["test"])
        mgr.remember("Test memory 2", tags=["test"])
        estimator = SleepCostEstimator()
        est = estimator.estimate(mgr)
        assert est.total_anchors >= 2
        assert est.total_edges >= 0

    def test_template_provider_is_free(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        for i in range(5):
            mgr.remember(f"Memory {i}", tags=["test"], source_session="test")
        estimator = SleepCostEstimator()
        est = estimator.estimate(mgr)
        assert est.provider == "template"
        assert est.llm_cost_usd == 0.0
        assert est.is_free

    def test_dry_run_flag(self):
        est = CostEstimate(dry_run=True)
        assert est.dry_run
        assert "DRY RUN" in est.detailed()

    def test_summary_string(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        mgr.remember("Test", tags=["test"])
        estimator = SleepCostEstimator()
        est = estimator.estimate(mgr)
        summary = est.summary()
        assert "anchor" in summary.lower() or "1" in summary

    def test_detailed_report(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        mgr.remember("Test", tags=["test"])
        estimator = SleepCostEstimator()
        est = estimator.estimate(mgr)
        detailed = est.detailed()
        assert "Anchors" in detailed
        assert "LLM" in detailed


# ═══════════════════════════════════════════════════════════════
# Snapshot Manager
# ═══════════════════════════════════════════════════════════════

class TestSnapshotManager:
    def test_snapshot_and_load(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        mgr.remember("Snapshot test memory", tags=["test"])
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=3, compress=False)

        meta = snap.snapshot(mgr.graph, description="test", force=True)
        assert meta.anchor_count == 1
        assert meta.version == 1
        assert "test" in meta.description

        graph2, meta2 = snap.load_latest()
        assert len(graph2.anchors) == 1
        assert meta2.checksum == meta.checksum

    def test_snapshot_creates_file(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        mgr.remember("File test", tags=["test"])
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=3, compress=False)

        snap.snapshot(mgr.graph, force=True)
        assert snap._snapshot_path(1, False).exists()

    def test_versions_list(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=5, compress=False)

        for i in range(3):
            mgr.remember(f"Memory {i}", tags=["test"])
            snap.snapshot(mgr.graph, force=True)

        assert snap.versions == [1, 2, 3]

    def test_rollback_deletes_newer(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=5, compress=False)

        for i in range(3):
            mgr.remember(f"Memory {i}", tags=["test"])
            snap.snapshot(mgr.graph, force=True)

        snap.rollback(version=1)
        assert snap.versions == [1]

    def test_cleanup_keeps_last_n(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=2, compress=False)

        for i in range(5):
            mgr.remember(f"Memory {i}", tags=["test"])
            snap.snapshot(mgr.graph, force=True)

        assert len(snap.versions) == 2  # kept only last 2

    def test_wal_append_and_recover(self):
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=3)

        snap.wal_append("add_anchor", {
            "id": "test_wal", "text": "WAL recovery test",
            "tags": ["test"], "source_session": "test",
        })

        graph, log = snap.recover()
        assert len(log) > 0
        assert "test_wal" in graph.anchors

    def test_wal_cleared_after_flush(self):
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=3)

        snap.wal_append("add_anchor", {"id": "test", "text": "test"})
        snap._flush_wal()
        assert len(snap._wal_entries) == 0

    def test_stats(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        mgr.remember("Stats test", tags=["test"])
        tmpdir = tempfile.mkdtemp()
        snap = SnapshotManager(base_dir=tmpdir, keep=3, compress=False)

        snap.snapshot(mgr.graph, force=True)
        s = snap.stats
        assert s["snapshots"] == 1
        assert s["latest_version"] == 1


# ═══════════════════════════════════════════════════════════════
# Tracing
# ═══════════════════════════════════════════════════════════════

class TestTracing:
    def setup_method(self):
        reset_tracer()

    def test_span_with_attributes(self):
        tracer = MemoryTracer()
        with tracer.span("test_op") as span:
            span.set_attribute("key", "value")
        traces = tracer.recent()
        assert len(traces) == 1
        assert traces[0].root.attributes.get("key") == "value"

    def test_span_timing(self):
        tracer = MemoryTracer()
        with tracer.span("test_op") as span:
            time.sleep(0.01)
        assert span.duration_ms > 0

    def test_span_error_status(self):
        tracer = MemoryTracer()
        try:
            with tracer.span("failing_op"):
                raise ValueError("test error")
        except ValueError:
            pass
        traces = tracer.recent()
        assert len(traces) == 1
        assert traces[0].root.status == "error"

    def test_record_method(self):
        tracer = MemoryTracer()
        span = tracer.record("one_shot", {
            "query": "test", "results": 5,
        }, duration_ms=100)
        traces = tracer.recent()
        assert len(traces) == 1
        assert "query" in traces[0].root.attributes

    def test_max_traces_eviction(self):
        tracer = MemoryTracer(max_traces=3)
        for i in range(5):
            tracer.record(f"op_{i}", {"i": i})
        assert len(tracer.recent()) == 3

    def test_disabled_tracer(self):
        tracer = MemoryTracer(enabled=False)
        with tracer.span("noop") as span:
            span.set_attribute("x", 1)
        assert len(tracer.recent()) == 0

    def test_summary(self):
        tracer = MemoryTracer()
        tracer.record("op1", duration_ms=10)
        tracer.record("op2", duration_ms=20)
        s = tracer.summary
        assert s.total_traces == 2
        assert s.avg_duration_ms == 15.0

    def test_get_tracer_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_clear(self):
        tracer = MemoryTracer()
        tracer.record("op", {})
        tracer.clear()
        assert len(tracer.recent()) == 0

    def test_event_recording(self):
        tracer = MemoryTracer()
        with tracer.span("parent") as span:
            tracer.event("checkpoint", {"step": 1})
        traces = tracer.recent()
        events = traces[0].root.events
        assert len(events) == 1
        assert events[0]["name"] == "checkpoint"

    def test_span_to_dict(self):
        tracer = MemoryTracer()
        tracer.record("dict_test", {"a": 1}, duration_ms=50)
        d = tracer.recent()[0].to_dict()
        assert d["trace_id"]
        assert d["root"]["name"] == "dict_test"


# ═══════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════

class TestBenchmark:
    def test_exact_fact_scenarios(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        result = run_benchmark(mgr, scenarios="exact_fact")
        assert result.total_scenarios == 5
        assert result.total_passed == 5
        assert result.overall_exact_match == 1.0
        assert result.overall_has_answer == 1.0

    def test_associative_scenarios(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        result = run_benchmark(mgr, scenarios="associative")
        assert result.total_scenarios == 3
        assert result.total_passed == 3

    def test_temporal_scenarios(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        result = run_benchmark(mgr, scenarios="temporal")
        assert result.total_scenarios == 3
        assert result.total_passed == 3

    def test_noise_scenarios(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        result = run_benchmark(mgr, scenarios="noise")
        assert result.total_scenarios == 2
        assert result.total_passed == 2

    def test_standard_suite(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        result = run_benchmark(mgr, scenarios="standard")
        assert result.total_scenarios == 14
        assert result.total_passed == 14

    def test_report_contains_categories(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        result = run_benchmark(mgr, scenarios="exact_fact")
        report = result.report()
        assert "exact_fact" in report
        assert "100.0%" in report or "100%" in report

    def test_benchmark_result_properties(self):
        from star_graph import MemoryManager
        mgr = MemoryManager()
        result = run_benchmark(mgr, scenarios="exact_fact")
        assert result.total_scenarios == 5
        assert result.total_passed == 5
        assert result.overall_exact_match == 1.0
        assert 0 < result.total_duration_ms

    def test_category_enum(self):
        assert Category.EXACT_FACT.value == "exact_fact"
        assert Category.ASSOCIATIVE.value == "associative"
        assert Category.TEMPORAL.value == "temporal"
        assert Category.NOISE.value == "noise"
        assert Category.COMPRESSION.value == "compression"


# ═══════════════════════════════════════════════════════════════
# Async Manager
# ═══════════════════════════════════════════════════════════════

class TestAsyncManager:
    def test_async_remember_recall(self):
        async def _test():
            from star_graph.async_manager import AsyncMemoryManager
            async with AsyncMemoryManager() as amgr:
                await amgr.remember("Async test fact", tags=["test"])
                ctx = await amgr.recall("Async test")
                assert len(ctx.items) >= 1
        asyncio.run(_test())

    def test_async_working_memory(self):
        async def _test():
            from star_graph.async_manager import AsyncMemoryManager
            async with AsyncMemoryManager() as amgr:
                await amgr.remember_working("WM async test", tags=["test"])
                wm = await amgr.get_working()
                assert len(wm) >= 1
        asyncio.run(_test())

    def test_async_health(self):
        async def _test():
            from star_graph.async_manager import AsyncMemoryManager
            async with AsyncMemoryManager() as amgr:
                await amgr.remember("Health test", tags=["test"])
                h = amgr.health
                assert h["anchors"] >= 1
                assert h["connections"] >= 1
        asyncio.run(_test())

    def test_async_estimate(self):
        async def _test():
            from star_graph.async_manager import AsyncMemoryManager
            async with AsyncMemoryManager() as amgr:
                est = await amgr.estimate_sleep_cost()
                assert hasattr(est, 'total_anchors')
        asyncio.run(_test())

    def test_connection_context_manager(self):
        async def _test():
            from star_graph.async_manager import AsyncMemoryManager
            pool = AsyncMemoryManager(max_connections=2)
            async with pool.connection() as conn:
                await conn.remember("Pool test", tags=["test"])
                assert conn.health["anchors"] >= 1
        asyncio.run(_test())
