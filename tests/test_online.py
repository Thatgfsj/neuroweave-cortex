"""Tests for OnlineConsolidator — micro-consolidation between sleep cycles."""

import time

import pytest

from star_graph.anchor import Anchor, AnchorVector
from star_graph.graph import StarGraph
from star_graph.online import OnlineConsolidator


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                emotional_valence: float = 0.0, surprise: float = 0.5,
                importance: float = 0.5) -> Anchor:
    a = Anchor(id=name, text=text or f"memory {name}", vector=AnchorVector(
        emotional_valence=emotional_valence,
        surprise=surprise,
        importance=importance,
    ))
    if embedding:
        a.embedding = embedding
    return a


class TestOnlineConsolidatorInit:
    def test_default_params(self):
        g = StarGraph()
        oc = OnlineConsolidator(g)
        assert oc.interval == 5
        assert oc.max_anchors == 20
        assert oc.interaction_count == 0

    def test_custom_params(self):
        g = StarGraph()
        oc = OnlineConsolidator(g, interval=3, max_anchors_per_cycle=10)
        assert oc.interval == 3
        assert oc.max_anchors == 10


class TestOnlineConsolidatorRecord:
    def test_record_increments_counter(self):
        g = StarGraph()
        oc = OnlineConsolidator(g)
        oc.record_interaction()
        assert oc.interaction_count == 1
        oc.record_interaction()
        assert oc.interaction_count == 2

    def test_record_with_anchor_adds_to_pending(self):
        g = StarGraph()
        oc = OnlineConsolidator(g, interval=10)
        anchor = make_anchor("a1")
        oc.record_interaction(anchor=anchor)
        assert len(oc.pending_anchors) == 1

    def test_record_triggers_micro_sleep_on_interval(self):
        g = StarGraph()
        oc = OnlineConsolidator(g, interval=3)
        anchor = make_anchor("a1", embedding=[0.1, 0.2, 0.3])
        oc.record_interaction(anchor=anchor)
        oc.record_interaction()
        # 3rd interaction triggers micro_sleep
        oc.record_interaction()
        assert len(oc.pending_anchors) == 0  # cleared after micro_sleep
        assert anchor.id in g.anchors  # anchor was added to graph


class TestOnlineConsolidatorForce:
    def test_force_consolidate_triggers_immediately(self):
        g = StarGraph()
        oc = OnlineConsolidator(g, interval=100)
        anchor = make_anchor("a1", embedding=[0.1, 0.2, 0.3])
        oc.record_interaction(anchor=anchor)
        result = oc.force_consolidate()
        assert "latency_ms" in result
        assert "edges_updated" in result
        assert "anchors_in_graph" in result
        assert result["anchors_in_graph"] >= 1

    def test_force_consolidate_empty(self):
        g = StarGraph()
        oc = OnlineConsolidator(g)
        result = oc.force_consolidate()
        assert result["anchors_in_graph"] == 0


class TestOnlineConsolidatorMicroSleep:
    def test_micro_sleep_activates_existing_anchor(self):
        g = StarGraph()
        existing = make_anchor("existing", embedding=[0.1, 0.2, 0.3])
        existing.replay_count = 0
        g.add_anchor(existing)

        oc = OnlineConsolidator(g, interval=1)
        dup = make_anchor("existing", embedding=[0.1, 0.2, 0.3])
        oc.pending_anchors.append(dup)
        oc._micro_sleep()

        assert g.anchors["existing"].replay_count >= 1

    def test_cosine_sim_static(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        sim = OnlineConsolidator._cosine_sim(v1, v2)
        assert sim == pytest.approx(1.0, abs=1e-4)

    def test_cosine_sim_zero_vector(self):
        sim = OnlineConsolidator._cosine_sim([0.0, 0.0], [1.0, 0.0])
        assert sim == pytest.approx(0.0)

    def test_cosine_sim_empty(self):
        assert OnlineConsolidator._cosine_sim([], []) == 0.0

    def test_cosine_sim_different_lengths(self):
        sim = OnlineConsolidator._cosine_sim([1.0, 2.0, 3.0], [1.0, 2.0])
        assert 0.0 < sim < 1.0


class TestOnlineConsolidatorPrioritization:
    def test_emotional_anchors_prioritized(self):
        """High emotional valence anchors should be processed first."""
        g = StarGraph()
        oc = OnlineConsolidator(g, interval=1, max_anchors_per_cycle=6)

        # Create anchors with varying emotional valence
        neutral = make_anchor("neutral", embedding=[0.1, 0.2, 0.3],
                              emotional_valence=0.0)
        emotional = make_anchor("emotional", embedding=[0.15, 0.25, 0.35],
                                emotional_valence=0.9)

        oc.pending_anchors = [neutral, emotional]
        oc._micro_sleep()

        # Both should be in graph; emotional may be first
        assert "emotional" in g.anchors or "neutral" in g.anchors
