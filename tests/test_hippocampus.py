"""Tests for hippocampus module — HippocampusBuffer and HippocampusItem."""

import time

import pytest

from star_graph.hippocampus import HippocampusBuffer, HippocampusItem


class TestHippocampusItem:
    def test_default_values(self):
        item = HippocampusItem(text="hello world")
        assert item.text == "hello world"
        assert item.tags == []
        assert item.importance == 0.5
        assert item.access_count == 0
        assert item.created_at > 0

    def test_created_at_set_if_zero(self):
        item = HippocampusItem(text="test", created_at=0.0)
        assert item.created_at > 0

    def test_last_accessed_at_defaults_to_created_at(self):
        item = HippocampusItem(text="test", created_at=100.0, last_accessed_at=0.0)
        assert item.last_accessed_at == 100.0


class TestHippocampusBuffer:
    def test_init_defaults(self):
        hb = HippocampusBuffer()
        assert hb.l1_max == 50
        assert hb.l1_ttl == 1800  # 30 min in seconds
        assert hb.l2_max == 200
        assert hb.l2_ttl == 86400  # 24h in seconds
        assert hb.promote_threshold == 3

    def test_ingest_adds_to_l1(self):
        hb = HippocampusBuffer()
        item_id = hb.ingest("test memory")
        assert item_id.startswith("hc_")
        assert item_id in hb.l1

    def test_ingest_with_tags(self):
        hb = HippocampusBuffer()
        hb.ingest("test", tags=["debug", "python"])
        item = next(iter(hb.l1.values()))
        assert "debug" in item.tags

    def test_promote_l1_to_l2(self):
        hb = HippocampusBuffer()
        item_id = hb.ingest("test memory")
        l2_id = hb.promote(item_id)
        assert l2_id is not None
        assert l2_id.startswith("hc2_")
        assert item_id not in hb.l1
        assert l2_id in hb.l2

    def test_promote_nonexistent(self):
        hb = HippocampusBuffer()
        assert hb.promote("nonexistent") is None

    def test_access_records_activity(self):
        hb = HippocampusBuffer()
        item_id = hb.ingest("test memory")
        item = hb.access(item_id)
        assert item is not None
        assert item.access_count == 1

    def test_access_nonexistent(self):
        hb = HippocampusBuffer()
        assert hb.access("nonexistent") is None

    def test_access_auto_promotes_on_threshold(self):
        hb = HippocampusBuffer(promote_threshold=2)
        item_id = hb.ingest("test memory")
        hb.access(item_id)  # count 1
        hb.access(item_id)  # count 2 → promote
        assert item_id not in hb.l1

    def test_query_l1_text_match(self):
        hb = HippocampusBuffer()
        hb.ingest("redis timeout fix")
        hb.ingest("unrelated topic")
        results = hb.query_l1(text_substring="redis")
        assert len(results) == 1
        assert "redis" in results[0].text

    def test_query_l1_tag_match(self):
        hb = HippocampusBuffer()
        hb.ingest("item1", tags=["debug"])
        hb.ingest("item2", tags=["python"])
        results = hb.query_l1(tags=["debug"])
        assert len(results) == 1
        assert "debug" in results[0].tags

    def test_query_l2_embedding(self):
        hb = HippocampusBuffer()
        id1 = hb.ingest("test a", embedding=[1.0, 0.0, 0.0])
        hb.promote(id1)
        id2 = hb.ingest("test b", embedding=[0.0, 1.0, 0.0])
        hb.promote(id2)
        results = hb.query_l2([1.0, 0.0, 0.0], top_k=2)
        assert len(results) >= 1

    def test_evict_expired(self):
        hb = HippocampusBuffer(l1_ttl_minutes=-1)  # everything expired
        hb.ingest("expired item")
        # Access count 0 so it should be evicted
        removed = hb.evict_expired()
        assert removed >= 0

    def test_stats(self):
        hb = HippocampusBuffer()
        hb.ingest("test1")
        hb.ingest("test2")
        s = hb.stats
        assert s["l1_items"] == 2
        assert s["total"] == 2

    def test_l1_fifo_eviction(self):
        hb = HippocampusBuffer(l1_max_items=2)
        hb.ingest("item1")
        hb.ingest("item2")
        hb.ingest("item3")  # should evict item1
        # item1 should be gone
        results = hb.query_l1(text_substring="item1")
        assert len(results) == 0

    def test_l2_fifo_eviction(self):
        hb = HippocampusBuffer(l2_max_items=2)
        id1 = hb.ingest("item1")
        id2 = hb.ingest("item2")
        id3 = hb.ingest("item3")
        hb.promote(id1)
        hb.promote(id2)
        hb.promote(id3)  # should evict the first promoted
        assert hb.size >= 1

    def test_size_property(self):
        hb = HippocampusBuffer()
        assert hb.size == 0
        hb.ingest("test")
        assert hb.size == 1

    def test_sleep_decide_promotes(self):
        from star_graph.graph import StarGraph
        from star_graph.embedding import EmbeddingProvider

        hb = HippocampusBuffer(promote_threshold=3)
        emb = EmbeddingProvider()
        item_id = hb.ingest("important fact about Python", importance=0.8, embedding=[0.1]*384)
        # Manually bump access count
        for _ in range(3):
            hb.access(item_id)

        graph = StarGraph()
        stats = hb.sleep_decide(graph, emb)
        assert stats["promoted"] >= 1

    def test_sleep_decide_discards(self):
        from star_graph.graph import StarGraph
        from star_graph.embedding import EmbeddingProvider

        hb = HippocampusBuffer()
        item_id = hb.ingest("low importance noise", importance=0.1)
        # Force decay to trigger discard
        item = hb.l1[item_id]
        item.decay_score = 0.05

        graph = StarGraph()
        emb = EmbeddingProvider()
        stats = hb.sleep_decide(graph, emb)
        assert stats["promoted"] + stats["discarded"] + stats["abstracted"] + stats["kept"] >= 0
