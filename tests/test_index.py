"""Tests for ANNIndex — approximate nearest neighbor index."""

import pytest

from star_graph.index import ANNIndex


class TestANNIndex:
    def test_empty_index_has_size_zero(self):
        idx = ANNIndex()
        assert idx.size == 0

    def test_empty_index_query_returns_empty(self):
        idx = ANNIndex()
        assert idx.query([0.1, 0.2]) == []

    def test_add_increases_size(self):
        idx = ANNIndex(dim=4)
        idx.add("a1", [0.1, 0.2, 0.3, 0.4])
        assert idx.size == 1

    def test_add_pads_short_embedding(self):
        idx = ANNIndex(dim=4)
        idx.add("a1", [0.1, 0.2])
        assert idx.size == 1

    def test_add_truncates_long_embedding(self):
        idx = ANNIndex(dim=2)
        idx.add("a1", [0.1, 0.2, 0.3, 0.4])
        assert idx.size == 1

    def test_query_returns_results(self):
        # Use dim > 20 to get brute algorithm (ball_tree doesn't support cosine)
        idx = ANNIndex(dim=25)
        idx.add("a1", [1.0] + [0.0] * 24)
        idx.add("a2", [0.0, 1.0] + [0.0] * 23)
        results = idx.query([1.0] + [0.0] * 24, k=2)
        assert len(results) == 2
        assert results[0][0] == "a1"

    def test_query_respects_k(self):
        idx = ANNIndex(dim=25)
        for i in range(5):
            emb = [float(i)] + [0.0] * 24
            idx.add(f"a{i}", emb)
        results = idx.query([0.0] * 25, k=3)
        assert len(results) == 3

    def test_remove_decreases_size(self):
        idx = ANNIndex(dim=4)
        idx.add("a1", [0.1, 0.2, 0.3, 0.4])
        idx.add("a2", [0.5, 0.6, 0.7, 0.8])
        idx.remove("a1")
        assert idx.size == 1

    def test_remove_nonexistent_does_not_raise(self):
        idx = ANNIndex(dim=4)
        idx.add("a1", [0.1, 0.2, 0.3, 0.4])
        idx.remove("nonexistent")
        assert idx.size == 1

    def test_clear_resets_everything(self):
        idx = ANNIndex(dim=4)
        idx.add("a1", [0.1, 0.2, 0.3, 0.4])
        idx.add("a2", [0.5, 0.6, 0.7, 0.8])
        idx.clear()
        assert idx.size == 0
        assert idx.query([0.1, 0.2, 0.3, 0.4]) == []

    def test_rebuild_empty_does_not_crash(self):
        idx = ANNIndex()
        idx.rebuild()

    def test_query_after_rebuild(self):
        idx = ANNIndex(dim=25)
        idx.add("a1", [1.0] + [0.0] * 24)
        idx.rebuild()
        results = idx.query([1.0] + [0.0] * 24, k=1)
        assert len(results) == 1
        assert results[0][0] == "a1"

    def test_query_with_missing_dimensions(self):
        idx = ANNIndex(dim=25)
        idx.add("a1", [1.0] + [0.0] * 24)
        results = idx.query([1.0, 0.0], k=1)
        assert len(results) == 1

    def test_cosine_similarity_range(self):
        idx = ANNIndex(dim=25)
        idx.add("a1", [1.0] + [0.0] * 24)
        results = idx.query([1.0] + [0.0] * 24, k=1)
        assert results[0][1] == pytest.approx(1.0, abs=0.1)
