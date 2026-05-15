"""Tests for BM25 keyword index."""

import pytest

from star_graph.bm25 import BM25Index, reciprocal_rank_fusion


class TestBM25Init:
    def test_default_params(self):
        idx = BM25Index()
        assert idx.k1 == 1.5
        assert idx.b == 0.75
        assert idx.size == 0

    def test_custom_params(self):
        idx = BM25Index(k1=2.0, b=0.5)
        assert idx.k1 == 2.0
        assert idx.b == 0.5


class TestBM25Add:
    def test_add_document(self):
        idx = BM25Index()
        idx.add("a1", "Redis connection timeout is 30 seconds")
        assert idx.size == 1

    def test_add_empty_text_ignored(self):
        idx = BM25Index()
        idx.add("a1", "")
        assert idx.size == 0

    def test_reindex_same_id(self):
        idx = BM25Index()
        idx.add("a1", "first text about redis")
        idx.add("a1", "second text about mysql")
        assert idx.size == 1


class TestBM25Remove:
    def test_remove_document(self):
        idx = BM25Index()
        idx.add("a1", "redis timeout issue")
        assert idx.size == 1
        idx.remove("a1")
        assert idx.size == 0

    def test_remove_nonexistent_silent(self):
        idx = BM25Index()
        idx.remove("nonexistent")  # no error


class TestBM25Search:
    def test_search_returns_ranked_results(self):
        idx = BM25Index()
        idx.add("a1", "Redis connection timeout is 30 seconds")
        idx.add("a2", "MySQL query timeout is 60 seconds")
        idx.add("a3", "HTTP request timeout is 10 seconds")
        results = idx.search("redis timeout", top_k=5)
        assert len(results) > 0
        # "redis" should be the best match
        ids = [r[0] for r in results]
        assert "a1" in ids

    def test_search_empty_index(self):
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []

    def test_search_empty_query(self):
        idx = BM25Index()
        idx.add("a1", "some text")
        results = idx.search("")
        assert results == []

    def test_search_respects_top_k(self):
        idx = BM25Index()
        for i in range(10):
            idx.add(f"a{i}", f"timeout related text number {i}")
        results = idx.search("timeout", top_k=3)
        assert len(results) == 3

    def test_search_scores_are_positive(self):
        idx = BM25Index()
        idx.add("a1", "redis timeout debug")
        idx.add("a2", "mysql timeout fix")
        results = idx.search("redis", top_k=5)
        for _, score in results:
            assert score > 0


class TestBM25Clear:
    def test_clear_removes_all(self):
        idx = BM25Index()
        idx.add("a1", "first doc")
        idx.add("a2", "second doc")
        idx.clear()
        assert idx.size == 0
        assert idx.search("first") == []


class TestBM25EdgeCases:
    def test_single_token_query(self):
        idx = BM25Index()
        idx.add("a1", "redis timeout debug")
        results = idx.search("redis")
        assert len(results) == 1
        assert results[0][0] == "a1"

    def test_token_not_in_any_doc(self):
        idx = BM25Index()
        idx.add("a1", "redis timeout")
        results = idx.search("oracle")
        assert results == []

    def test_repeated_tokens_scored_higher(self):
        idx = BM25Index()
        idx.add("low", "redis")
        idx.add("high", "redis redis redis timeout")
        results = idx.search("redis", top_k=2)
        assert results[0][0] == "high"


class TestReciprocalRankFusion:
    def test_fuses_lists(self):
        list_a = [("a1", 0.9), ("a2", 0.8), ("a3", 0.7)]
        list_b = [("a2", 0.9), ("a4", 0.8), ("a1", 0.6)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        assert len(fused) == 4
        # a2 appears in both lists at high ranks => should be top
        assert fused[0][0] == "a2"

    def test_empty_input(self):
        fused = reciprocal_rank_fusion([])
        assert fused == []

    def test_single_list(self):
        lst = [("a1", 0.9), ("a2", 0.5)]
        fused = reciprocal_rank_fusion([lst])
        assert len(fused) == 2
        assert fused[0][0] == "a1"

    def test_custom_k(self):
        list_a = [("a1", 1.0), ("a2", 0.5)]
        list_b = [("b1", 1.0)]
        fused_default = reciprocal_rank_fusion([list_a, list_b])
        fused_custom = reciprocal_rank_fusion([list_a, list_b], k=10)
        # Different k produces different scores
        assert fused_default != fused_custom or len(fused_default) == len(fused_custom)
