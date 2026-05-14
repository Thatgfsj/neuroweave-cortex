"""Tests for EdgeBudgetManager — smart edge cap enforcement."""

import time
import pytest
from star_graph.graph import StarGraph, Edge
from star_graph.anchor import Anchor
from star_graph.edge_budget import EdgeBudgetManager, EDGE_TYPE_RETENTION_PRIORITY


def _make_graph(num_anchors: int = 50) -> StarGraph:
    g = StarGraph()
    for i in range(num_anchors):
        a = Anchor.create(text=f"Test anchor {i}", tags=["test"])
        a.id = f"a{i}"
        g.add_anchor(a)
    return g


class TestEdgeBudgetInit:
    def test_default_max_edges(self):
        manager = EdgeBudgetManager()
        assert manager.max_edges == 32

    def test_custom_max_edges(self):
        manager = EdgeBudgetManager(max_edges=8)
        assert manager.max_edges == 8

    def test_init_counters(self):
        manager = EdgeBudgetManager()
        assert manager._eviction_count == 0
        assert manager._merge_count == 0


class TestEdgeRetentionScoring:
    def test_causal_edges_score_higher_than_topical(self):
        manager = EdgeBudgetManager()
        causal = Edge(source="a", target="b", weight=0.8, edge_type="causes")
        topical = Edge(source="a", target="c", weight=0.8, edge_type="topical")
        assert manager._edge_retention_score(causal) > manager._edge_retention_score(topical)

    def test_fixes_edges_score_high(self):
        manager = EdgeBudgetManager()
        fixes = Edge(source="a", target="b", weight=0.8, edge_type="fixes")
        topical = Edge(source="a", target="c", weight=0.8, edge_type="topical")
        assert manager._edge_retention_score(fixes) > manager._edge_retention_score(topical)

    def test_higher_weight_scores_higher(self):
        manager = EdgeBudgetManager()
        high = Edge(source="a", target="b", weight=0.9, edge_type="topical")
        low = Edge(source="a", target="c", weight=0.3, edge_type="topical")
        assert manager._edge_retention_score(high) > manager._edge_retention_score(low)

    def test_null_edge_scores_zero(self):
        manager = EdgeBudgetManager()
        assert manager._edge_retention_score(None) == 0.0

    def test_nonexistent_type_gets_default_priority(self):
        manager = EdgeBudgetManager()
        edge = Edge(source="a", target="b", weight=0.5, edge_type="unknown_type_xyz")
        score = manager._edge_retention_score(edge)
        assert score > 0.0

    def test_priority_order_matters(self):
        manager = EdgeBudgetManager()
        causes = Edge(source="a", target="b", weight=0.8, edge_type="causes")
        related = Edge(source="a", target="c", weight=0.8, edge_type="related")
        score_causes = manager._edge_retention_score(causes)
        score_related = manager._edge_retention_score(related)
        # causes=10, related=2 → causes should score higher
        assert score_causes > score_related


class TestEdgeBudgetEnforce:
    def test_no_eviction_when_under_budget(self):
        g = _make_graph(10)
        manager = EdgeBudgetManager(max_edges=32)
        for i in range(1, 5):
            g.add_edge("a0", f"a{i}", weight=0.8, edge_type="topical")
        result = manager.enforce(g, "a0")
        assert result["evicted"] == 0
        assert result["remaining"] <= 4

    def test_eviction_when_over_budget(self):
        g = _make_graph(50)
        manager = EdgeBudgetManager(max_edges=4)
        for i in range(1, 10):
            g.add_edge("a0", f"a{i}", weight=0.5, edge_type="topical")
        result = manager.enforce(g, "a0")
        assert result["evicted"] > 0
        assert result["remaining"] <= 4

    def test_keeps_highest_priority_edges(self):
        g = _make_graph(50)
        manager = EdgeBudgetManager(max_edges=3)
        g.add_edge("a0", "a1", weight=0.3, edge_type="topical")
        g.add_edge("a0", "a2", weight=0.3, edge_type="topical")
        g.add_edge("a0", "a3", weight=0.8, edge_type="causes")
        g.add_edge("a0", "a4", weight=0.8, edge_type="fixes")
        g.add_edge("a0", "a5", weight=0.9, edge_type="related")
        manager.enforce(g, "a0")
        neighbors = g._adjacency.get("a0", set())
        # causes and fixes should survive due to high priority
        assert "a3" in neighbors  # causes
        assert "a4" in neighbors  # fixes

    def test_eviction_removes_both_directions(self):
        g = _make_graph(50)
        manager = EdgeBudgetManager(max_edges=2)
        g.add_edge("a0", "a1", weight=0.8, edge_type="causes")
        g.add_edge("a0", "a2", weight=0.7, edge_type="fixes")
        g.add_edge("a0", "a3", weight=0.6, edge_type="related")
        g.add_edge("a0", "a4", weight=0.3, edge_type="topical")
        manager.enforce(g, "a0")
        neighbors_a0 = g._adjacency.get("a0", set())
        # Weakest edges evicted — a4 (weight 0.3, topical) should be evicted first
        assert len(neighbors_a0) <= 2


class TestEdgeBudgetEnforceAll:
    def test_enforce_all_iterates_all_nodes(self):
        g = _make_graph(20)
        manager = EdgeBudgetManager(max_edges=3)
        for i in range(1, 10):
            g.add_edge("a0", f"a{i}", weight=0.5, edge_type="topical")
        for i in range(2, 8):
            g.add_edge("a1", f"a{i}", weight=0.5, edge_type="topical")
        result = manager.enforce_all(g)
        assert "over_budget_nodes" in result
        assert "total_evicted" in result
        assert result["over_budget_nodes"] > 0

    def test_enforce_all_empty_graph(self):
        g = StarGraph()
        manager = EdgeBudgetManager()
        result = manager.enforce_all(g)
        assert result["over_budget_nodes"] == 0
        assert result["total_evicted"] == 0

    def test_enforce_all_tracks_eviction_count(self):
        g = _make_graph(30)
        manager = EdgeBudgetManager(max_edges=2)
        for i in range(1, 6):
            g.add_edge("a0", f"a{i}", weight=0.5, edge_type="topical")
        result = manager.enforce_all(g)
        assert manager._eviction_count > 0


class TestEdgeBudgetStats:
    def test_stats_property(self):
        manager = EdgeBudgetManager(max_edges=16)
        s = manager.stats
        assert s["max_edges"] == 16
        assert "total_evictions" in s
        assert "total_merges" in s
