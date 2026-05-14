"""Test Multi-Level Cognitive Cache — query/session/topic/activation caches."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
from star_graph import (
    StarGraph, Anchor,
    QueryCache, SessionCache, TopicCache, ActivationCache,
    CognitiveCacheManager, QueryCacheEntry,
)


class TestQueryCache:
    """Verify LRU query→result cache with TTL."""

    def test_set_and_get(self):
        qc = QueryCache(max_entries=10, ttl_seconds=300)
        qc.set("what is python", ["id1", "id2"], [0.9, 0.8])
        entry = qc.get("what is python")
        assert entry is not None
        assert entry.result_ids == ["id1", "id2"]

    def test_query_normalization(self):
        """Queries should be case-insensitive and whitespace-insensitive."""
        qc = QueryCache(ttl_seconds=300)
        qc.set("  What IS Python  ", ["id1"], [0.9])
        assert qc.get("what is python") is not None
        assert qc.get("WHAT IS PYTHON") is not None

    def test_expired_ttl(self):
        qc = QueryCache(max_entries=10, ttl_seconds=0.0)
        qc.set("query", ["id1"], [0.9])
        # Should expire immediately with ttl=0
        assert qc.get("query") is None

    def test_lru_eviction(self):
        qc = QueryCache(max_entries=2, ttl_seconds=999)
        qc.set("q1", ["a"], [1.0])
        qc.set("q2", ["b"], [1.0])
        qc.set("q3", ["c"], [1.0])  # should evict q1
        assert qc.get("q1") is None
        assert qc.get("q2") is not None
        assert qc.get("q3") is not None

    def test_invalidate_single(self):
        qc = QueryCache(ttl_seconds=300)
        qc.set("q1", ["a"], [1.0])
        qc.set("q2", ["b"], [1.0])
        assert qc.invalidate("q1") == 1
        assert qc.get("q1") is None
        assert qc.get("q2") is not None

    def test_invalidate_all(self):
        qc = QueryCache(ttl_seconds=300)
        qc.set("q1", ["a"], [1.0])
        qc.set("q2", ["b"], [1.0])
        assert qc.invalidate() == 2
        assert len(qc) == 0

    def test_evict_expired(self):
        qc = QueryCache(max_entries=10, ttl_seconds=0.0)
        qc.set("q1", ["a"], [1.0])
        qc.set("q2", ["b"], [1.0])
        assert qc.evict_expired() == 2
        assert len(qc) == 0


class TestSessionCache:
    """Verify per-session working set tracking."""

    def test_record_access_and_promote(self):
        sc = SessionCache(max_entries=50, promote_threshold=3)
        for _ in range(3):
            sc.record_access("anchor_1")
        assert sc.is_hot("anchor_1")
        assert "anchor_1" in sc.get_hot(10)

    def test_not_hot_below_threshold(self):
        sc = SessionCache(promote_threshold=3)
        sc.record_access("anchor_1")
        sc.record_access("anchor_1")
        assert not sc.is_hot("anchor_1")

    def test_start_session_resets(self):
        sc = SessionCache(promote_threshold=2)
        sc.record_access("anchor_1")
        sc.record_access("anchor_1")
        assert sc.is_hot("anchor_1")
        sc.start_session("new_session")
        assert not sc.is_hot("anchor_1")

    def test_max_entries_trim(self):
        sc = SessionCache(max_entries=3, promote_threshold=2)
        for i in range(5):
            for _ in range(2):
                sc.record_access(f"anchor_{i}")
        assert len(sc) == 3  # trimmed to max


class TestTopicCache:
    """Verify tag→anchor pre-computed index."""

    def test_rebuild_and_lookup(self):
        g = StarGraph()
        a1 = Anchor.create("python async programming", tags=["python", "async"])
        a2 = Anchor.create("javascript promises", tags=["javascript", "async"])
        a3 = Anchor.create("python decorators", tags=["python", "decorators"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_anchor(a3)

        tc = TopicCache(max_per_tag=10)
        tags = tc.rebuild(g)
        assert tags >= 1

        # Lookup "python" tag
        results = tc.lookup(["python"], top_k=5)
        assert a1.id in results
        assert a3.id in results

        # Lookup "async" tag
        results = tc.lookup(["async"], top_k=5)
        assert a1.id in results
        assert a2.id in results

    def test_lookup_text(self):
        g = StarGraph()
        a1 = Anchor.create("python async", tags=["python", "async"])
        g.add_anchor(a1)

        tc = TopicCache(max_per_tag=10)
        tc.rebuild(g)

        results = tc.lookup_text("tell me about python development", top_k=5)
        assert a1.id in results

    def test_empty_cache(self):
        tc = TopicCache()
        assert tc.lookup(["nonexistent"]) == []
        assert tc.lookup_text("nonexistent query") == []


class TestActivationCache:
    """Verify activation result cache with TTL."""

    def test_set_and_get(self):
        ac = ActivationCache(max_entries=50, ttl_seconds=60)
        ac.set("seed_1", ["a", "b"], [0.9, 0.7])
        result = ac.get("seed_1")
        assert result is not None
        assert result[0] == ["a", "b"]

    def test_expired(self):
        ac = ActivationCache(max_entries=10, ttl_seconds=0.0)
        ac.set("seed_1", ["a"], [0.9])
        assert ac.get("seed_1") is None

    def test_invalidate(self):
        ac = ActivationCache(ttl_seconds=60)
        ac.set("s1", ["a"], [0.9])
        ac.set("s2", ["b"], [0.8])
        ac.invalidate("s1")
        assert ac.get("s1") is None
        assert ac.get("s2") is not None
        ac.invalidate()
        assert len(ac) == 0

    def test_lru_eviction(self):
        ac = ActivationCache(max_entries=2, ttl_seconds=999)
        ac.set("s1", ["a"], [0.9])
        ac.set("s2", ["b"], [0.8])
        ac.set("s3", ["c"], [0.7])
        assert ac.get("s1") is None
        assert ac.get("s2") is not None


class TestCognitiveCacheManager:
    """Verify multi-level cache orchestration."""

    def test_lookup_integrates_all_caches(self):
        mgr = CognitiveCacheManager()
        # Build a graph and populate topic cache
        g = StarGraph()
        a = Anchor.create("async python development", tags=["python", "async"])
        g.add_anchor(a)
        mgr.rebuild_on_sleep(g)

        # Add query cache entry
        mgr.record_query("python async", [a.id], [0.95])

        # Record session access
        for _ in range(3):
            mgr.record_access(a.id)

        # Lookup
        result = mgr.lookup(query="python async", tags=["python"])
        assert "query_cache" in result
        assert "session_cache" in result or "topic_cache" in result

    def test_rebuild_on_sleep(self):
        g = StarGraph()
        g.add_anchor(Anchor.create("python memory", tags=["python", "test"]))
        mgr = CognitiveCacheManager()
        stats = mgr.rebuild_on_sleep(g)
        assert stats["tags_indexed"] >= 1
        assert "query_evicted" in stats
        assert "activation_evicted" in stats

    def test_start_session(self):
        mgr = CognitiveCacheManager()
        mgr.start_session("test_session")
        assert mgr.session_cache.stats["session_id"] == "test_session"

    def test_all_stats_available(self):
        mgr = CognitiveCacheManager()
        s = mgr.stats
        assert "query_cache" in s
        assert "session_cache" in s
        assert "topic_cache" in s
        assert "activation_cache" in s
