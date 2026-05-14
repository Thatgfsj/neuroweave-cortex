"""Tests for EdgeDecayManager — continuous edge time decay."""

import time
import pytest
from star_graph.graph import StarGraph, Edge
from star_graph.anchor import Anchor
from star_graph.edge_management import EdgeDecayManager, DECAY_RATE_PER_HOUR


class TestEdgeDecayInit:
    def test_default_init(self):
        mgr = EdgeDecayManager()
        assert mgr.base_decay_multiplier == 1.0
        assert mgr.min_edge_weight == 0.02

    def test_custom_init(self):
        mgr = EdgeDecayManager(base_decay_multiplier=2.0, min_edge_weight=0.05)
        assert mgr.base_decay_multiplier == 2.0
        assert mgr.min_edge_weight == 0.05

    def test_stats(self):
        mgr = EdgeDecayManager()
        s = mgr.stats
        assert "total_decayed" in s
        assert "total_evicted" in s


class TestDecayRate:
    def test_causal_decay_rate_low(self):
        mgr = EdgeDecayManager()
        causal = Edge(source="a", target="b", weight=0.8, edge_type="causes")
        rate = mgr.decay_rate_for(causal)
        # Causal edges should decay slowly (0.0003 base)
        assert rate < 0.001

    def test_topical_decay_rate_fast(self):
        mgr = EdgeDecayManager()
        topical = Edge(source="a", target="b", weight=0.8, edge_type="topical")
        rate = mgr.decay_rate_for(topical)
        # Topical edges decay faster (0.005 base)
        assert rate > 0.001

    def test_unknown_type_uses_default(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.8, edge_type="custom_type")
        rate = mgr.decay_rate_for(edge)
        assert rate > 0

    def test_high_success_rate_slows_decay(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.8, edge_type="topical")
        edge.success_count = 100
        edge.failure_count = 0  # 100% success
        rate_high = mgr.decay_rate_for(edge)
        edge.success_count = 0
        edge.failure_count = 100  # 0% success
        rate_low = mgr.decay_rate_for(edge)
        assert rate_high < rate_low  # high success = slower decay


class TestApplyDecay:
    def test_apply_decay_reduces_weight(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.8, edge_type="topical",
                    last_activated_at=time.time() - 100 * 3600)  # 100 hours idle
        old_weight = edge.weight
        mgr.apply_decay(edge)
        assert edge.weight < old_weight

    def test_apply_decay_floor(self):
        mgr = EdgeDecayManager(min_edge_weight=0.02)
        edge = Edge(source="a", target="b", weight=0.1, edge_type="topical",
                    last_activated_at=0.0)  # extremely old
        mgr.apply_decay(edge)
        assert edge.weight >= 0.02

    def test_no_decay_recent_edge(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.8, edge_type="causes",
                    last_activated_at=time.time())  # just activated
        old_weight = edge.weight
        mgr.apply_decay(edge)
        assert edge.weight == old_weight  # no time passed

    def test_causal_decays_slower_than_topical(self):
        mgr = EdgeDecayManager()
        old_time = time.time() - 50 * 3600  # 50 hours ago
        causal = Edge(source="a", target="b", weight=0.8, edge_type="causes",
                     last_activated_at=old_time)
        topical = Edge(source="a", target="c", weight=0.8, edge_type="topical",
                      last_activated_at=old_time)
        mgr.apply_decay(causal)
        mgr.apply_decay(topical)
        assert causal.weight > topical.weight


class TestReinforce:
    def test_reinforce_strengthens(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.5, edge_type="causes")
        old_weight = edge.weight
        mgr.reinforce(edge)
        assert edge.weight > old_weight

    def test_reinforce_extends_valid_until(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.5, edge_type="causes")
        edge.valid_until = time.time() + 100
        old_valid = edge.valid_until
        mgr.reinforce(edge)
        assert edge.valid_until > old_valid


class TestIsViable:
    def test_viable_edge(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.8, edge_type="causes")
        assert mgr.is_viable(edge)

    def test_expired_edge_not_viable(self):
        mgr = EdgeDecayManager()
        edge = Edge(source="a", target="b", weight=0.8, edge_type="topical")
        edge.valid_until = time.time() - 100  # expired
        assert not mgr.is_viable(edge)

    def test_below_min_weight_not_viable(self):
        mgr = EdgeDecayManager(min_edge_weight=0.05)
        edge = Edge(source="a", target="b", weight=0.01, edge_type="topical")
        assert not mgr.is_viable(edge)


class TestDecayAllEdges:
    def test_decay_all_on_graph(self):
        mgr = EdgeDecayManager()
        g = StarGraph()
        for i in range(3):
            a = Anchor.create(text=f"Node {i}")
            a.id = f"n{i}"
            g.add_anchor(a)

        old_time = time.time() - 100 * 3600
        g.add_edge("n0", "n1", weight=0.8, edge_type="topical")
        # Set edge as old
        key = g._key("n0", "n1")
        if key in g.edges:
            g.edges[key].last_activated_at = old_time

        result = mgr.decay_all_edges(g)
        assert result["decayed"] > 0
        assert "evicted" in result

    def test_decay_all_empty_graph(self):
        mgr = EdgeDecayManager()
        g = StarGraph()
        result = mgr.decay_all_edges(g)
        assert result["decayed"] == 0
        assert result["evicted"] == 0
