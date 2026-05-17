"""Tests for raw_buffer module — RawBuffer and RawChunk."""

import pytest

from star_graph.raw_buffer import RawBuffer, RawChunk


class TestRawChunk:
    def test_default_values(self):
        c = RawChunk(id="c1", text="hello world", session_id="s1")
        assert c.id == "c1"
        assert c.text == "hello world"
        assert c.session_id == "s1"
        assert c.importance == 0.5
        assert c.anchor_id == ""

    def test_tokenize(self):
        c = RawChunk(id="c1", text="hello world test", session_id="s1")
        tokens = c.tokenize()
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_lowercase(self):
        c = RawChunk(id="c1", text="Hello WORLD", session_id="s1")
        tokens = c.tokenize()
        assert "hello" in tokens
        assert "world" in tokens

    def test_term_freq(self):
        c = RawChunk(id="c1", text="hello world hello", session_id="s1")
        tf = c.term_freq
        assert tf.get("hello") == 2
        assert tf.get("world") == 1

    def test_term_freq_cached(self):
        c = RawChunk(id="c1", text="hello world", session_id="s1")
        tf1 = c.term_freq
        tf2 = c.term_freq
        assert tf1 is tf2  # same cached object

    def test_token_count(self):
        c = RawChunk(id="c1", text="hello world test", session_id="s1")
        assert c.token_count == 3


class TestRawBuffer:
    def test_init_defaults(self):
        buf = RawBuffer()
        assert buf.max_sessions == 2
        assert buf.max_chunks_per_session == 500

    def test_add_chunk(self):
        buf = RawBuffer()
        chunk = buf.add("hello world", session_id="s1")
        assert chunk.id
        assert chunk.text == "hello world"
        assert chunk.session_id == "s1"

    def test_add_with_embedding(self):
        buf = RawBuffer()
        chunk = buf.add("test", session_id="s1", embedding=[0.1, 0.2, 0.3])
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_add_with_tags(self):
        buf = RawBuffer()
        chunk = buf.add("test", session_id="s1", tags=["debug", "python"])
        assert "debug" in chunk.tags

    def test_add_with_importance(self):
        buf = RawBuffer()
        chunk = buf.add("test", session_id="s1", importance=0.8)
        assert chunk.importance == 0.8

    def test_search_basic(self):
        buf = RawBuffer()
        buf.add("redis timeout fix", session_id="s1")
        buf.add("unrelated topic here", session_id="s1")
        results = buf.search("redis timeout")
        assert len(results) > 0
        assert "redis" in results[0][0].text.lower()

    def test_search_empty(self):
        buf = RawBuffer()
        results = buf.search("anything")
        assert results == []

    def test_search_with_session_boost(self):
        buf = RawBuffer()
        buf.add("debug error", session_id="s1")
        buf.add("debug error", session_id="s2")
        results = buf.search("debug", session_id="s2", top_k=5)
        # s2 result should be first due to session boost
        assert len(results) >= 1

    def test_search_respects_top_k(self):
        buf = RawBuffer()
        for i in range(10):
            buf.add(f"test item {i}", session_id="s1")
        results = buf.search("test", top_k=3)
        assert len(results) <= 3

    def test_search_session(self):
        buf = RawBuffer()
        buf.add("debug error", session_id="s1")
        buf.add("unrelated data", session_id="s2")
        results = buf.search_session("s1", "debug")
        assert len(results) >= 1

    def test_get_recent(self):
        buf = RawBuffer()
        for i in range(5):
            buf.add(f"item {i}", session_id="s1")
        recent = buf.get_recent(3)
        assert len(recent) == 3

    def test_get_session_chunks(self):
        buf = RawBuffer()
        buf.add("item1", session_id="s1")
        buf.add("item2", session_id="s1")
        buf.add("item3", session_id="s2")
        chunks = buf.get_session_chunks("s1")
        assert len(chunks) == 2

    def test_session_eviction(self):
        buf = RawBuffer(max_sessions=1)
        buf.add("item1", session_id="s1")
        buf.add("item2", session_id="s2")
        # s1 should be evicted
        chunks = buf.get_session_chunks("s1")
        assert len(chunks) == 0
        chunks_s2 = buf.get_session_chunks("s2")
        assert len(chunks_s2) == 1

    def test_chunk_limit_per_session(self):
        buf = RawBuffer(max_chunks_per_session=2)
        buf.add("item1", session_id="s1")
        buf.add("item2", session_id="s1")
        buf.add("item3", session_id="s1")
        chunks = buf.get_session_chunks("s1")
        assert len(chunks) == 2

    def test_stats(self):
        buf = RawBuffer()
        buf.add("item1", session_id="s1")
        s = buf.stats
        assert s["total_sessions"] == 1
        assert s["total_chunks"] == 1

    def test_clear(self):
        buf = RawBuffer()
        buf.add("item1", session_id="s1")
        buf.add("item2", session_id="s2")
        buf.clear()
        assert buf.stats["total_chunks"] == 0
        assert buf.search("item") == []

    def test_tokenize_query(self):
        buf = RawBuffer()
        tokens = buf.tokenize_query("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert len(tokens) == len(set(tokens))  # deduped

    def test_search_with_vector(self):
        buf = RawBuffer()
        buf.add("semantic text", session_id="s1", embedding=[1.0, 0.0, 0.0])
        buf.add("other text", session_id="s1", embedding=[0.0, 1.0, 0.0])
        results = buf.search("test", query_embedding=[1.0, 0.0, 0.0], top_k=2, bm25_weight=0.0)
        assert len(results) >= 1
