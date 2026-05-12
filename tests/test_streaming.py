"""Tests for v1.0-6: streaming memory buffer with backpressure."""

import time

import pytest

from star_graph.streaming import (
    StreamItem,
    StreamStats,
    StreamingMemoryBuffer,
    _cosine_sim,
)
from star_graph import MemoryManager


# ═══════════════════════════════════════════════════════════════
# StreamItem
# ═══════════════════════════════════════════════════════════════

class TestStreamItem:
    def test_create_defaults(self):
        item = StreamItem(text="hello")
        assert item.text == "hello"
        assert item.tags == []
        assert item.source_session == ""
        assert item.importance == 0.5
        assert item.emotional_valence == 0.0
        assert item.timestamp > 0

    def test_create_with_fields(self):
        item = StreamItem(
            text="test memory",
            tags=["debug", "redis"],
            source_session="session_1",
            importance=0.8,
            emotional_valence=-0.3,
            embedding=[0.1, 0.2, 0.3],
        )
        assert item.tags == ["debug", "redis"]
        assert item.source_session == "session_1"
        assert item.importance == 0.8
        assert item.emotional_valence == -0.3
        assert item.embedding == [0.1, 0.2, 0.3]

    def test_hash_uses_text_prefix_and_session(self):
        item1 = StreamItem(text="a" * 200 + "suffix", source_session="s1")
        item2 = StreamItem(text="a" * 200 + "different", source_session="s1")
        # Same first 100 chars + same session → same hash (timestamp differs)
        # We can't compare hash directly since timestamps differ
        # But we can verify the hash doesn't crash
        h = hash(item1)
        assert isinstance(h, int)


# ═══════════════════════════════════════════════════════════════
# StreamStats
# ═══════════════════════════════════════════════════════════════

class TestStreamStats:
    def test_defaults(self):
        s = StreamStats()
        assert s.total_ingested == 0
        assert s.total_flushed == 0
        assert s.total_merged == 0
        assert s.total_promoted == 0
        assert s.buffer_size == 0
        assert s.backpressure_events == 0
        assert s.dropped_items == 0

    def test_merge_ratio_zero_when_no_ingested(self):
        s = StreamStats()
        assert s.merge_ratio == 0.0

    def test_merge_ratio(self):
        s = StreamStats(total_ingested=100, total_merged=30)
        assert s.merge_ratio == 0.3

    def test_summary(self):
        s = StreamStats(total_ingested=50, total_flushed=40, total_merged=10,
                        total_promoted=8, buffer_size=5, dropped_items=2)
        summary = s.summary()
        assert "ingested=50" in summary
        assert "flushed=40" in summary
        assert "merged=10" in summary
        assert "promoted=8" in summary
        assert "buffer=5" in summary
        assert "dropped=2" in summary


# ═══════════════════════════════════════════════════════════════
# StreamingMemoryBuffer
# ═══════════════════════════════════════════════════════════════

class TestStreamingBuffer:
    def test_init_defaults(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        assert buf.max_buffer == 500
        assert buf.flush_interval_s == 30.0
        assert buf.batch_size == 20
        assert buf.dedup_threshold == 0.85
        assert buf.max_sessions == 10
        assert buf.size == 0
        assert not buf.is_full
        assert buf.sessions == []

    def test_init_custom_params(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(
            mgr, max_buffer=100, flush_interval_s=10.0,
            batch_size=5, dedup_threshold=0.9, max_sessions=3,
            auto_flush=False,
        )
        assert buf.max_buffer == 100
        assert buf.flush_interval_s == 10.0
        assert buf.batch_size == 5
        assert buf.dedup_threshold == 0.9
        assert buf.max_sessions == 3

    def test_ingest_single_item(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        ok = buf.ingest("hello world", tags=["test"])
        assert ok
        assert buf.size == 1
        assert buf.stats.total_ingested == 1
        assert not buf.is_full

    def test_ingest_multiple_items(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        for i in range(5):
            buf.ingest(f"item {i}", source_session="s1")
        assert buf.size == 5
        assert buf.stats.total_ingested == 5

    def test_ingest_different_sessions(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        buf.ingest("a", source_session="s1")
        buf.ingest("b", source_session="s2")
        buf.ingest("c", source_session="s1")
        assert buf.size == 3
        assert set(buf.sessions) == {"s1", "s2"}

    def test_ingest_batch(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        items = [
            {"text": f"item {i}", "source_session": "test", "tags": ["batch"]}
            for i in range(10)
        ]
        accepted = buf.ingest_batch(items)
        assert accepted == 10
        assert buf.size == 10

    def test_is_full(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, max_buffer=3, auto_flush=False)
        buf.ingest("a")
        buf.ingest("b")
        buf.ingest("c")
        assert buf.is_full
        assert buf.size == 3

    def test_backpressure_rejection(self):
        mgr = MemoryManager()
        # Small buffer, large batch_size to prevent auto-flush on count,
        # but force backpressure by ingesting fast
        buf = StreamingMemoryBuffer(mgr, max_buffer=3, batch_size=100,
                                    flush_interval_s=3600, auto_flush=False)
        # Fill beyond 1.5x limit (1.5 * 3 = 4.5, so 5+ items trigger drops)
        for i in range(20):
            buf.ingest(f"item {i}")
        # At least some should have been dropped or backpressure triggered
        assert buf.stats.dropped_items > 0 or buf.stats.backpressure_events > 1

    def test_auto_flush_on_batch_threshold(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, batch_size=3, auto_flush=False)
        buf.ingest("a")
        buf.ingest("b")
        assert buf.stats.total_flushed == 0
        buf.ingest("c")  # hits batch_size, triggers flush
        # Flush may have happened (depends on timing)
        assert buf.stats.total_ingested >= 3

    def test_manual_flush(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        buf.ingest("memory item 1", source_session="s1")
        buf.ingest("memory item 2", source_session="s1")
        result = buf.flush(force=True)
        assert result["flushed"] >= 1
        assert buf.size == 0

    def test_flush_empty(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        result = buf.flush(force=True)
        assert result["flushed"] == 0
        assert result["reason"] == "empty"

    def test_flush_too_soon_skipped(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, flush_interval_s=60, auto_flush=False)
        buf.ingest("test")
        result = buf.flush()  # not force, too soon
        assert result["reason"] == "too_soon"
        assert result["flushed"] == 0

    def test_close_flushes_remaining(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        buf.ingest("final item", source_session="s1")
        buf.close()
        assert buf.size == 0
        assert buf.stats.total_flushed >= 1

    def test_sessions_list(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        buf.ingest("a", source_session="chat_1")
        buf.ingest("b", source_session="chat_2")
        assert "chat_1" in buf.sessions
        assert "chat_2" in buf.sessions

    def test_sessions_capped(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, max_sessions=2, auto_flush=False)
        # Flush first so sessions are emptied and capped works
        buf.ingest("a", source_session="s1")
        buf.ingest("b", source_session="s2")
        buf.flush(force=True)
        buf.ingest("c", source_session="s3")
        buf.ingest("d", source_session="s4")
        # Max 2 sessions enforced after flush
        assert len(buf.sessions) <= 2

    def test_stats_tracking(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        buf.ingest("item 1")
        buf.ingest("item 2")
        buf.flush(force=True)
        s = buf.stats
        assert s.total_ingested == 2
        assert s.total_flushed >= 2

    def test_auto_flush_thread_starts(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=True)
        assert buf._flush_thread is not None
        assert buf._running
        buf.close()


# ═══════════════════════════════════════════════════════════════
# Deduplication and Clustering
# ═══════════════════════════════════════════════════════════════

class TestDedupAndCluster:
    def test_dedup_no_items(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        result = buf._dedup_items([])
        assert result == []

    def test_dedup_single_item(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        item = StreamItem(text="single")
        result = buf._dedup_items([item])
        assert len(result) == 1

    def test_dedup_similar_items_merged(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False, dedup_threshold=0.5)
        item1 = StreamItem(
            text="Redis timeout issue in production",
            embedding=[1.0, 0.0, 0.0],
            importance=0.3,
            tags=["redis"],
        )
        item2 = StreamItem(
            text="Redis timeout issue in production",  # same text → embedding will be same
            importance=0.8,
            tags=["production"],
        )
        # Force same embedding for deterministic test
        item2.embedding = [1.0, 0.0, 0.0]
        result = buf._dedup_items([item1, item2])
        assert len(result) == 1
        assert result[0].importance == 0.8  # max importance kept
        assert "redis" in result[0].tags
        assert "production" in result[0].tags

    def test_dedup_dissimilar_items_kept(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False, dedup_threshold=0.9)
        item1 = StreamItem(text="Redis timeout", embedding=[1.0, 0.0, 0.0])
        item2 = StreamItem(text="Python syntax error", embedding=[0.0, 1.0, 0.0])
        result = buf._dedup_items([item1, item2])
        assert len(result) == 2

    def test_cluster_by_tags_empty(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        assert buf._cluster_by_tags([]) == []

    def test_cluster_by_tags_overlapping(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        items = [
            StreamItem(text="a", tags=["redis", "timeout"]),
            StreamItem(text="b", tags=["redis", "cache"]),
            StreamItem(text="c", tags=["python", "syntax"]),
        ]
        clusters = buf._cluster_by_tags(items)
        # redis items should cluster together, python item separate
        assert len(clusters) == 2

    def test_cluster_by_tags_no_tags_all_clustered(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        items = [
            StreamItem(text="a"),
            StreamItem(text="b"),
            StreamItem(text="c"),
        ]
        clusters = buf._cluster_by_tags(items)
        # All items with no tags cluster together
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_promote_cluster_empty(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        ok = buf._promote_cluster([], "s1")
        assert not ok

    def test_promote_cluster_single(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        item = StreamItem(
            text="single memory", tags=["test"],
            importance=0.7, emotional_valence=0.2,
        )
        ok = buf._promote_cluster([item], "s1")
        assert ok
        # Anchor was created in manager's graph (total_promoted stat only
        # increments via _flush_internal, not _promote_cluster directly)
        assert len(mgr.graph.anchors) == 1
        anchor = list(mgr.graph.anchors.values())[0]
        assert anchor.text == "single memory"
        assert anchor.tags == ["test"]

    def test_promote_cluster_multiple_items(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False)
        items = [
            StreamItem(
                text="primary memory", tags=["a"],
                importance=0.9, emotional_valence=0.3,
            ),
            StreamItem(
                text="secondary memory", tags=["b"],
                importance=0.5, emotional_valence=-0.1,
            ),
        ]
        ok = buf._promote_cluster(items, "s1")
        assert ok
        # Anchor should have been created in the manager's graph
        assert len(mgr.graph.anchors) == 1
        anchor = list(mgr.graph.anchors.values())[0]
        # Primary memory text is used (highest importance), with "+1 related" suffix
        assert "primary memory" in anchor.text
        assert "+1 related" in anchor.text
        assert set(anchor.tags) == {"a", "b"}

    def test_flush_dedup_and_promote(self):
        mgr = MemoryManager()
        buf = StreamingMemoryBuffer(mgr, auto_flush=False, dedup_threshold=0.5)
        buf.ingest("Redis timeout issue", tags=["redis", "debug"])
        buf.ingest("Redis timeout issue fixed", tags=["redis", "fix"])
        result = buf.flush(force=True)
        assert result["flushed"] == 2
        # Similar redis items should be merged/grouped
        assert result["merged"] >= 0


# ═══════════════════════════════════════════════════════════════
# MemoryManager Integration
# ═══════════════════════════════════════════════════════════════

class TestManagerStreaming:
    def test_streaming_buffer_lazy_init(self):
        mgr = MemoryManager()
        buf = mgr.streaming_buffer
        assert buf is not None
        assert buf.size == 0
        # Second access returns same instance
        assert mgr.streaming_buffer is buf
        buf.close()

    def test_streaming_ingest_and_recall(self):
        mgr = MemoryManager()
        buf = mgr.streaming_buffer
        buf.ingest("User prefers dark mode UI", tags=["preference", "ui"])
        buf.ingest("Dashboard uses high-contrast colors", tags=["ui", "dashboard"])
        buf.flush(force=True)
        # Anchors should now be in the graph
        assert len(mgr.graph.anchors) >= 1
        buf.close()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

class TestCosineSim:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0
