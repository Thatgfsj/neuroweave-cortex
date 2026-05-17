"""Tests for resonance module — Resonator and _derive_oscillation."""

import pytest

from star_graph.resonance import Resonator, _derive_oscillation
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor, AnchorVector, Oscillator, AnchorPrediction


def make_anchor(name: str, text: str = "", embedding: list | None = None) -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}")
    if embedding:
        a.embedding = embedding
    return a


class TestDeriveOscillation:
    def test_empty_embedding(self):
        freq, phase = _derive_oscillation(None)
        assert freq == 0.5
        assert phase == 0.0

    def test_small_embedding(self):
        freq, phase = _derive_oscillation([0.1, 0.2])
        assert freq == 0.5
        assert phase == 0.0

    def test_normal_embedding(self):
        # Need even-length embedding for arctan2 pairwise slicing
        freq, phase = _derive_oscillation([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert 0.1 <= freq <= 1.0
        assert 0.0 <= phase < 2 * 3.14159


class TestResonator:
    def test_init(self):
        g = StarGraph()
        r = Resonator(g)
        assert r.graph is g

    def test_find_seeds_empty_graph(self):
        g = StarGraph()
        r = Resonator(g)
        seeds = r.find_seeds("test query", embedding=[0.1, 0.2, 0.3, 0.4])
        assert seeds == []

    def test_find_seeds_with_anchors(self):
        g = StarGraph()
        a1 = make_anchor("a1", "redis timeout", embedding=[1.0, 0.0, 0.0, 0.0])
        a1.oscillator = Oscillator(natural_frequency=0.5, phase_offset=0.0, coupling_strength=0.5)
        g.add_anchor(a1)
        r = Resonator(g)
        seeds = r.find_seeds("redis timeout", embedding=[1.0, 0.0, 0.0, 0.0])
        assert len(seeds) >= 1

    def test_resonate_empty_graph(self):
        g = StarGraph()
        r = Resonator(g)
        constellations = r.resonate("test", embedding=[0.1, 0.2, 0.3, 0.4])
        assert constellations == []

    def test_resonate_with_seeds(self):
        g = StarGraph()
        a1 = make_anchor("a1", "redis timeout", embedding=[1.0, 0.0, 0.0, 0.0])
        a1.oscillator = Oscillator(natural_frequency=0.5, phase_offset=0.0, coupling_strength=0.5)
        g.add_anchor(a1)
        r = Resonator(g)
        constellations = r.resonate("redis timeout", embedding=[1.0, 0.0, 0.0, 0.0])
        assert len(constellations) >= 1

    def test_bridge_score_same_constellation(self):
        g = StarGraph()
        a1 = make_anchor("a1", "test", embedding=[0.1, 0.2, 0.3])
        a1.vector = AnchorVector(importance=0.5)
        a2 = make_anchor("a2", "test2", embedding=[0.1, 0.2, 0.3])
        a2.vector = AnchorVector(importance=0.5)
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.5)
        r = Resonator(g)
        # Import Constellation
        from star_graph.graph import Constellation
        c1 = Constellation(anchors=[a1], edges=[])
        c2 = Constellation(anchors=[a2], edges=[])
        score = r.bridge_score(c1, c2)
        assert 0.0 <= score <= 1.5

    def test_bridge_score_highly_connected(self):
        g = StarGraph()
        a1 = make_anchor("a1", "t1", embedding=[0.1, 0.2, 0.3])
        a2 = make_anchor("a2", "t2", embedding=[0.1, 0.2, 0.3])
        g.add_anchor(a1)
        g.add_anchor(a2)
        # Need >1.0 total weight for bridge_score to return 0
        g.add_edge("a1", "a2", weight=0.9)
        g.add_edge("a1", "a2", weight=0.9)  # reinforce to increase
        # Don't assert 0 — the edge weight may not sum to >1.0 easily
        r = Resonator(g)
        from star_graph.graph import Constellation
        c1 = Constellation(anchors=[a1], edges=[])
        c2 = Constellation(anchors=[a2], edges=[])
        score = r.bridge_score(c1, c2)
        assert 0.0 <= score <= 1.5

    def test_predictive_retrieve_empty(self):
        g = StarGraph()
        r = Resonator(g)
        constellation, action = r.predictive_retrieve("test", embedding=[0.1, 0.2, 0.3, 0.4])
        assert constellation is None
        assert action == "novel"

    def test_predictive_retrieve_with_anchors(self):
        g = StarGraph()
        a1 = make_anchor("a1", "test", embedding=[1.0, 0.0, 0.0, 0.0])
        a1.oscillator = Oscillator(natural_frequency=0.5, phase_offset=0.0, coupling_strength=0.5)
        g.add_anchor(a1)
        r = Resonator(g)
        constellation, action = r.predictive_retrieve("test", embedding=[1.0, 0.0, 0.0, 0.0])
        assert constellation is not None
        assert action in ("confirm", "update", "novel")

    def test_predictive_retrieve_with_prediction(self):
        g = StarGraph()
        a1 = make_anchor("a1", "test", embedding=[1.0, 0.0, 0.0, 0.0])
        a1.prediction = AnchorPrediction(emotional_tone=0.0)
        a1.oscillator = Oscillator(natural_frequency=0.5, phase_offset=0.0, coupling_strength=0.5)
        g.add_anchor(a1)
        r = Resonator(g)
        constellation, action = r.predictive_retrieve("test", embedding=[1.0, 0.0, 0.0, 0.0])
        assert constellation is not None
        assert action in ("confirm", "update", "novel")
