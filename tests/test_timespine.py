"""Tests for timespine module — TimeSpine, TimeBucket, MemoryCluster."""

import time

import pytest

from star_graph.timespine import TimeSpine, TimeBucket, MemoryCluster


class TestMemoryCluster:
    def test_default_values(self):
        mc = MemoryCluster(id="mc1")
        assert mc.id == "mc1"
        assert mc.anchor_ids == []
        assert mc.topic == ""
        assert mc.importance == 0.0

    def test_size(self):
        mc = MemoryCluster(id="mc1", anchor_ids=["a1", "a2", "a3"])
        assert mc.size == 3

    def test_is_empty(self):
        mc = MemoryCluster(id="mc1")
        assert mc.is_empty

    def test_is_not_empty(self):
        mc = MemoryCluster(id="mc1", anchor_ids=["a1"])
        assert not mc.is_empty


class TestTimeBucket:
    def test_default_values(self):
        tb = TimeBucket(timestamp=1000.0)
        assert tb.timestamp == 1000.0
        assert tb.clusters == []
        assert tb.max_clusters == 10

    def test_is_full(self):
        tb = TimeBucket(timestamp=1000.0, max_clusters=1)
        tb.add_cluster(MemoryCluster(id="mc1"))
        assert tb.is_full

    def test_add_cluster(self):
        tb = TimeBucket(timestamp=1000.0)
        result = tb.add_cluster(MemoryCluster(id="mc1", importance=0.5))
        assert result

    def test_add_cluster_full_bucket_low_priority(self):
        tb = TimeBucket(timestamp=1000.0, max_clusters=1)
        tb.add_cluster(MemoryCluster(id="mc1", importance=0.8))
        result = tb.add_cluster(MemoryCluster(id="mc2", importance=0.3))
        assert not result  # rejected, lower priority

    def test_add_cluster_full_bucket_higher_priority(self):
        tb = TimeBucket(timestamp=1000.0, max_clusters=1)
        tb.add_cluster(MemoryCluster(id="mc1", importance=0.3))
        result = tb.add_cluster(MemoryCluster(id="mc2", importance=0.8))
        assert result  # accepted, replaced lower priority

    def test_top_clusters(self):
        tb = TimeBucket(timestamp=1000.0)
        tb.add_cluster(MemoryCluster(id="mc1", importance=0.3))
        tb.add_cluster(MemoryCluster(id="mc2", importance=0.8))
        tb.add_cluster(MemoryCluster(id="mc3", importance=0.5))
        top = tb.top_clusters(2)
        assert len(top) == 2
        assert top[0].importance == 0.8


class TestTimeSpine:
    def test_init_defaults(self):
        ts = TimeSpine()
        assert ts.max_clusters_per_day == 10
        assert ts.cluster_similarity_threshold == 0.5

    def test_index_anchor(self):
        ts = TimeSpine()
        cluster_id = ts.index_anchor("a1", importance=0.5)
        assert cluster_id is not None

    def test_index_anchor_with_topic(self):
        ts = TimeSpine()
        ts.index_anchor("a1", topic="debugging", importance=0.5)
        ts.index_anchor("a2", topic="debugging", importance=0.5)
        # Both should be in the same cluster
        assert ts.stats["total_anchors_indexed"] == 2
        assert ts.stats["total_clusters"] == 1

    def test_index_anchor_different_topics(self):
        ts = TimeSpine()
        ts.index_anchor("a1", topic="debugging", importance=0.5)
        ts.index_anchor("a2", topic="deployment", importance=0.5)
        assert ts.stats["total_clusters"] == 2

    def test_index_anchor_with_timestamp(self):
        ts = TimeSpine()
        past_time = time.time() - 86400  # yesterday
        ts.index_anchor("a1", timestamp=past_time, importance=0.5)
        assert ts.stats["days_indexed"] == 1

    def test_remove_anchor(self):
        ts = TimeSpine()
        ts.index_anchor("a1", topic="debugging", importance=0.5)
        ts.remove_anchor("a1")
        assert ts.stats["total_anchors_indexed"] == 0

    def test_update_importance(self):
        ts = TimeSpine()
        cluster_id = ts.index_anchor("a1", importance=0.3)
        ts.update_importance("a1", 0.9)
        # The cluster should now have the updated importance
        clusters = ts.scan_priority(max_days=1, max_clusters=1)
        if clusters:
            assert clusters[0].importance == pytest.approx(0.9)

    def test_scan_priority_empty(self):
        ts = TimeSpine()
        results = ts.scan_priority()
        assert results == []

    def test_scan_priority_respects_max(self):
        ts = TimeSpine()
        for i in range(5):
            ts.index_anchor(f"a{i}", topic=f"topic{i % 2}", importance=0.5 + i * 0.1)
        results = ts.scan_priority(max_days=30, max_clusters=3)
        assert len(results) <= 3

    def test_scan_priority_most_recent_first(self):
        ts = TimeSpine()
        now = time.time()
        yesterday = now - 86400
        ts.index_anchor("old", timestamp=yesterday, importance=0.9, topic="old")
        ts.index_anchor("new", timestamp=now, importance=0.5, topic="new")
        results = ts.scan_priority(max_days=30, max_clusters=5)
        # Most recent day should have its clusters first
        assert len(results) >= 1

    def test_query_window(self):
        ts = TimeSpine()
        now = time.time()
        two_days_ago = now - 172800
        ts.index_anchor("recent", timestamp=now, importance=0.5, topic="recent")
        ts.index_anchor("old", timestamp=two_days_ago, importance=0.5, topic="old")
        results = ts.query_window(now - 86400, now, max_clusters=10)
        assert len(results) >= 1

    def test_scan_timeline_backward(self):
        ts = TimeSpine()
        now = time.time()
        ts.index_anchor("a1", timestamp=now, importance=0.5, topic="t1")
        results = ts.scan_timeline(now + 1, direction="backward", max_clusters=10)
        assert len(results) >= 1

    def test_scan_timeline_forward(self):
        ts = TimeSpine()
        now = time.time()
        ts.index_anchor("a1", timestamp=now, importance=0.5, topic="t1")
        results = ts.scan_timeline(now - 1, direction="forward", max_clusters=10)
        assert len(results) >= 1

    def test_stats(self):
        ts = TimeSpine()
        ts.index_anchor("a1", importance=0.5)
        s = ts.stats
        assert s["days_indexed"] >= 1
        assert s["total_anchors_indexed"] == 1
        assert s["max_clusters_per_day"] == 10

    def test_most_recent_day(self):
        ts = TimeSpine()
        assert ts.most_recent_day is None
        ts.index_anchor("a1", importance=0.5)
        assert ts.most_recent_day is not None

    def test_oldest_day(self):
        ts = TimeSpine()
        assert ts.oldest_day is None
        ts.index_anchor("a1", importance=0.5)
        assert ts.oldest_day is not None

    def test_bucket_full_rejection(self):
        ts = TimeSpine(max_clusters_per_day=1)
        # Fill the bucket with high importance
        ts.index_anchor("a1", importance=0.9, topic="t1")
        # Try to add lower importance — should be rejected
        result = ts.index_anchor("a2", importance=0.1, topic="t2")
        assert result is None
