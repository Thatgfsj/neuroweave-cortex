"""Tests for edge_management module — EdgeBudgetManager, EdgeDecayManager, constants."""

import time

import pytest

from star_graph.edge_management import (
    EDGE_TYPE_RETENTION_PRIORITY,
    DECAY_RATE_PER_HOUR,
    REINFORCEMENT_EXTEND_HOURS,
    DEFAULT_DECAY_RATE,
    DEFAULT_REINFORCEMENT_HOURS,
    EdgeBudgetManager,
    EdgeDecayManager,
)
from star_graph.graph import StarGraph, Edge
from star_graph.anchor import Anchor


def make_anchor(name: str, text: str = "") -> Anchor:
    return Anchor(id=name, text=text or f"Memory {name}")


class TestConstants:
    def test_retention_priority(self):
        assert EDGE_TYPE_RETENTION_PRIORITY["causes"] == 10
        assert EDGE_TYPE_RETENTION_PRIORITY["topical"] == 3
        assert EDGE_TYPE_RETENTION_PRIORITY["contradicts"] == 1

    def test_decay_rates(self):
        assert DECAY_RATE_PER_HOUR["causes"] == 0.0003
        assert DECAY_RATE_PER_HOUR["topical"] == 0.005

    def test_reinforcement_hours(self):
        assert REINFORCEMENT_EXTEND_HOURS["causes"] == 720
        assert REINFORCEMENT_EXTEND_HOURS["related"] == 12


class TestEdgeBudgetManager:
    def test_init_default(self):
        ebm = EdgeBudgetManager()
        assert ebm.max_edges == 32
        assert ebm._eviction_count == 0
        assert ebm._merge_count == 0

    def test_init_custom(self):
        ebm = EdgeBudgetManager(max_edges=10)
        assert ebm.max_edges == 10

    def test_enforce_under_budget(self):
        ebm = EdgeBudgetManager(max_edges=10)
        g = StarGraph()
        a1 = make_anchor("a1")
        a2 = make_anchor("a2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=1.0, edge_type="causes")
        result = ebm.enforce(g, "a1")
        assert result["evicted"] == 0
        assert result["remaining"] == 1

    def test_enforce_over_budget(self):
        ebm = EdgeBudgetManager(max_edges=2)
        g = StarGraph()
        center = make_anchor("center")
        g.add_anchor(center)
        for i in range(5):
            a = make_anchor(f"a{i}")
            g.add_anchor(a)
            g.add_edge("center", f"a{i}", weight=0.5, edge_type="topical")
        result = ebm.enforce(g, "center")
        assert result["evicted"] >= 1
        assert result["remaining"] <= 2

    def test_enforce_keeps_high_priority(self):
        ebm = EdgeBudgetManager(max_edges=1)
        g = StarGraph()
        center = make_anchor("center")
        a1 = make_anchor("a1")
        a2 = make_anchor("a2")
        g.add_anchor(center)
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("center", "a1", weight=1.0, edge_type="causes")
        g.add_edge("center", "a2", weight=0.1, edge_type="related")
        ebm.enforce(g, "center")
        # The "causes" edge should survive (priority 10 vs related 2)
        remaining = g._adjacency.get("center", set())
        assert "a1" in remaining

    def test_enforce_all(self):
        ebm = EdgeBudgetManager(max_edges=1)
        g = StarGraph()
        center = make_anchor("center")
        g.add_anchor(center)
        for i in range(3):
            a = make_anchor(f"a{i}")
            g.add_anchor(a)
            g.add_edge("center", f"a{i}", weight=0.5,
                      edge_type="topical")
        result = ebm.enforce_all(g)
        assert result["over_budget_nodes"] == 1
        assert result["total_evicted"] >= 1

    def test_enforce_all_under_budget(self):
        ebm = EdgeBudgetManager(max_edges=10)
        g = StarGraph()
        a1 = make_anchor("a1")
        a2 = make_anchor("a2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=1.0, edge_type="topical")
        result = ebm.enforce_all(g)
        assert result["over_budget_nodes"] == 0

    def test_edge_retention_score_none(self):
        ebm = EdgeBudgetManager()
        score = ebm._edge_retention_score(None)
        assert score == 0.0

    def test_edge_retention_score(self):
        ebm = EdgeBudgetManager()
        edge = Edge(source="a", target="b", weight=0.8,
                    edge_type="causes", co_activation_count=5)
        score = ebm._edge_retention_score(edge)
        assert score > 0.0

    def test_stats(self):
        ebm = EdgeBudgetManager()
        s = ebm.stats
        assert s["max_edges"] == 32
        assert s["total_evictions"] == 0


class TestEdgeDecayManager:
    def test_init_default(self):
        edm = EdgeDecayManager()
        assert edm.base_decay_multiplier == 1.0
        assert edm.min_edge_weight == 0.02

    def test_init_custom(self):
        edm = EdgeDecayManager(base_decay_multiplier=2.0, min_edge_weight=0.05)
        assert edm.base_decay_multiplier == 2.0
        assert edm.min_edge_weight == 0.05

    def test_decay_rate_for(self):
        edm = EdgeDecayManager()
        edge = Edge(weight=0.5, source="x", target="y", edge_type="causes")
        rate = edm.decay_rate_for(edge)
        assert rate == pytest.approx(DECAY_RATE_PER_HOUR["causes"])

    def test_decay_rate_for_high_success(self):
        edm = EdgeDecayManager()
        edge = Edge(source="x", target="y", weight=0.5, edge_type="topical",
                    success_count=8, failure_count=2)
        assert edge.success_rate == 0.8
        rate = edm.decay_rate_for(edge)
        assert rate < DECAY_RATE_PER_HOUR["topical"]  # halved

    def test_decay_rate_for_low_success(self):
        edm = EdgeDecayManager()
        edge = Edge(source="x", target="y", weight=0.5, edge_type="topical",
                    success_count=2, failure_count=8)
        assert edge.success_rate == 0.2
        rate = edm.decay_rate_for(edge)
        assert rate > DECAY_RATE_PER_HOUR["topical"]  # doubled

    def test_decay_rate_for_stale(self):
        edm = EdgeDecayManager()
        edge = Edge(weight=0.5, source="x", target="y", edge_type="topical")
        edge.is_stale = True
        rate = edm.decay_rate_for(edge)
        assert rate > DECAY_RATE_PER_HOUR["topical"]  # doubled

    def test_apply_decay(self):
        edm = EdgeDecayManager()
        edge = Edge(weight=0.5, source="x", target="y", edge_type="topical",
                    created_at=0.0, last_activated_at=0.0)
        new_w = edm.apply_decay(edge, now=time.time())
        # Edge is very old, should have decayed significantly
        assert new_w < 0.5
        assert new_w >= edm.min_edge_weight

    def test_apply_decay_no_idle_time(self):
        edm = EdgeDecayManager()
        edge = Edge(weight=0.5, source="x", target="y", edge_type="topical")
        edge.last_activated_at = time.time()
        new_w = edm.apply_decay(edge, now=edge.last_activated_at)
        assert new_w == 0.5  # no decay

    def test_reinforce(self):
        edm = EdgeDecayManager()
        edge = Edge(weight=0.5, source="x", target="y", edge_type="causes")
        old_weight = edge.weight
        edm.reinforce(edge)
        assert edge.weight > old_weight

    def test_is_viable_true(self):
        edm = EdgeDecayManager()
        edge = Edge(weight=0.5, source="x", target="y", edge_type="topical")
        assert edm.is_viable(edge) is True

    def test_is_viable_expired(self):
        edm = EdgeDecayManager()
        edge = Edge(weight=0.5, source="x", target="y", edge_type="topical")
        # valid_until is on RichEdge; Edge uses getattr default 0
        # Set valid_until into the past via object.__setattr__
        object.__setattr__(edge, 'valid_until', 100.0)  # expired in 1970
        assert edm.is_viable(edge) is False

    def test_is_viable_too_weak(self):
        edm = EdgeDecayManager(min_edge_weight=0.5)
        edge = Edge(weight=0.3, source="x", target="y", edge_type="topical")
        assert edm.is_viable(edge) is False

    def test_decay_all_edges(self):
        edm = EdgeDecayManager()
        g = StarGraph()
        a1 = make_anchor("a1")
        a2 = make_anchor("a2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.01, edge_type="related")
        # Make the edge very old
        for key, edge in g.edges.items():
            edge.created_at = 0.0
        result = edm.decay_all_edges(g)
        assert result["decayed"] >= 0
        assert isinstance(result["evicted"], int)

    def test_stats(self):
        edm = EdgeDecayManager()
        s = edm.stats
        assert s["base_decay_multiplier"] == 1.0
        assert s["total_decayed"] == 0
