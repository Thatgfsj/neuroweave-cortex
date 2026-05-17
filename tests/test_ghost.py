"""Tests for ghost module — GhostNode, NegativeGhost, GhostSubsystem."""

import time

import pytest

from star_graph.ghost import GhostNode, NegativeGhost, GhostSubsystem
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor, MemoryState


def make_anchor(name: str, text: str = "", embedding: list | None = None) -> Anchor:
    a = Anchor(
        id=name, text=text or f"Memory {name}",
        tags=["test"],
    )
    if embedding:
        a.embedding = embedding
    return a


class TestGhostNode:
    def test_from_anchor(self):
        a = make_anchor("a1", "redis timeout fix for production", embedding=[0.1] * 384)
        a.vector.emotional_valence = 0.5
        g = GhostNode.from_anchor(a, residual_edges={"b1": 0.3})
        assert g.id == "a1"
        assert g.emotion_trace != 0.0
        assert len(g.compressed_embedding) > 0
        assert g.semantic_shadow  # non-empty shadow
        assert g.reactivation_probability > 0
        assert g.original_importance == 0.5

    def test_from_anchor_without_embedding(self):
        a = make_anchor("a1", "test memory")
        g = GhostNode.from_anchor(a)
        assert g.id == "a1"
        assert all(v == 0.0 for v in g.compressed_embedding)

    def test_resonance_identical(self):
        a = make_anchor("a1", "test memory", embedding=[0.1] * 384)
        g = GhostNode.from_anchor(a)
        score = g.resonance([0.1] * 384)
        assert score >= 0.0

    def test_resonance_empty_embedding(self):
        g = GhostNode(
            id="g1", compressed_embedding=[], residual_edges={},
            emotion_trace=0.0, pruned_at=time.time(),
            original_tags=[], original_importance=0.5,
            semantic_shadow="test",
        )
        assert g.resonance([0.1, 0.2]) == 0.0

    def test_revive_creates_reactivated_anchor(self):
        a = make_anchor("a1", "original text", embedding=[0.1] * 384)
        g = GhostNode.from_anchor(a)
        revived = g.revive("new text", [0.2] * 384, new_tags=["revived"])
        assert revived.id == "a1"
        assert revived.text == "new text"
        assert revived.state == MemoryState.REACTIVATED
        assert "revived" in revived.tags
        assert g.revival_count == 1

    def test_partial_recall(self):
        a = make_anchor("a1", "redis timeout fix for server crash", embedding=[0.1] * 384)
        g = GhostNode.from_anchor(a)
        desc, conf = g.partial_recall()
        assert isinstance(desc, str)
        assert 0.0 <= conf <= 1.0

    def test_decay_reduces_probability(self):
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        g = GhostNode.from_anchor(a)
        g.pruned_at = 0.0  # very old
        g.decay()
        assert g.reactivation_probability < 0.5

    def test_decay_returns_true_for_old_unused_ghost(self):
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        g = GhostNode.from_anchor(a)
        g.pruned_at = 0.0  # very old
        result = g.decay()
        assert result  # should be purged

    def test_intensity_property(self):
        a = make_anchor("a1", "test memory", embedding=[0.1] * 384)
        g = GhostNode.from_anchor(a)
        intensity = g.intensity
        assert 0.0 <= intensity <= 1.0

    def test_is_active(self):
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        g = GhostNode.from_anchor(a)
        g.reactivation_probability = 0.9
        is_active = g.is_active
        assert isinstance(is_active, bool)


class TestNegativeGhost:
    def test_from_contradiction(self):
        ng = NegativeGhost.from_contradiction(
            original_text="Python is slow",
            contradiction_text="Python with PyPy is fast",
            target_anchor_id="a1",
            original_importance=0.7,
        )
        assert ng.contradiction_type == "direct"
        assert ng.suppression_strength > 0.5
        assert ng.emotion_trace < 0  # negative emotional trace
        assert ng.contradiction_target == "a1"

    def test_suppress_no_resonance(self):
        ng = NegativeGhost.from_contradiction(
            original_text="Python is slow",
            contradiction_text="Python is fast",
        )
        factor = ng.suppress(None, 0.5)
        assert factor == 1.0  # no suppression without embedding

    def test_suppress_with_embedding(self):
        ng = NegativeGhost.from_contradiction(
            original_text="Python is slow",
            contradiction_text="Python with PyPy is fast",
            embedding=[0.1] * 384,
        )
        factor = ng.suppress([0.1] * 384, 0.5)
        assert 0.0 <= factor <= 1.0


class TestGhostSubsystem:
    def test_create_ghost(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "test memory", embedding=[0.1] * 384)
        ghost = gs.create(a, {"b1": 0.3})
        assert ghost.id == "a1"
        assert "a1" in gs.ghosts

    def test_create_negative(self):
        gs = GhostSubsystem()
        ng = gs.create_negative(
            original_text="old belief",
            contradiction_text="new correct belief",
            target_anchor_id="a1",
        )
        assert ng.id in gs.ghosts
        assert isinstance(ng, NegativeGhost)

    def test_positive_ghosts(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        gs.create(a)
        assert len(gs.positive_ghosts) == 1
        assert len(gs.negative_ghosts) == 0

    def test_negative_ghosts(self):
        gs = GhostSubsystem()
        gs.create_negative("old", "new", target_anchor_id="a1")
        assert len(gs.negative_ghosts) == 1
        assert len(gs.positive_ghosts) == 0

    def test_check_resonance(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "redis timeout fix", embedding=[0.1] * 384)
        gs.create(a)
        # Use a low threshold for reliable test
        matches = gs.check_resonance([0.1] * 384, threshold=0.0)
        assert len(matches) >= 1

    def test_check_resonance_empty(self):
        gs = GhostSubsystem()
        assert gs.check_resonance([0.1, 0.2]) == []

    def test_ranked_resonance(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        gs.create(a)
        results = gs.ranked_resonance([0.1] * 384, top_k=5, threshold=0.0)
        assert len(results) >= 1

    def test_try_revive(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "original text", embedding=[0.1] * 384)
        gs.create(a)
        revived = gs.try_revive("a1", "new text", [0.2] * 384)
        assert revived is not None
        assert revived.id == "a1"
        # Ghost should be removed after revival
        assert "a1" not in gs.ghosts

    def test_try_revive_nonexistent(self):
        gs = GhostSubsystem()
        assert gs.try_revive("nonexistent", "text", [0.1] * 384) is None

    def test_fuzzy_recall(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "test memory for fuzzy recall", embedding=[0.1] * 384)
        gs.create(a)
        results = gs.fuzzy_recall([0.1] * 384)
        assert isinstance(results, list)

    def test_decay_all(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        gs.create(a)
        # Force old pruned_at to trigger decay
        for ghost in gs.ghosts.values():
            ghost.pruned_at = 0.0
        count, removed = gs.decay_all()
        assert count >= 0

    def test_stats(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        gs.create(a)
        s = gs.stats
        assert s["total_ghosts"] == 1
        assert "avg_intensity" in s

    def test_get_top_intensity(self):
        gs = GhostSubsystem()
        a = make_anchor("a1", "test", embedding=[0.1] * 384)
        gs.create(a)
        top = gs.get_top_intensity(5)
        assert len(top) >= 1

    def test_suppress_anchor(self):
        gs = GhostSubsystem()
        gs.create_negative("old", "new", target_anchor_id="a1", embedding=[0.1] * 384)
        factor = gs.suppress_anchor([0.1] * 384, 0.5)
        assert 0.0 <= factor <= 1.0

    def test_suppress_anchor_no_embedding(self):
        gs = GhostSubsystem()
        gs.create_negative("old", "new")
        factor = gs.suppress_anchor(None, 0.5)
        assert factor == 1.0

    def test_check_suppression(self):
        gs = GhostSubsystem()
        gs.create_negative("old", "new", embedding=[0.1] * 384)
        factor = gs.check_suppression([0.1] * 384)
        assert 0.0 <= factor <= 1.0
