"""Tests for MemoryGate — competitive selection with lateral inhibition."""

import pytest

from star_graph.anchor import Anchor, AnchorVector
from star_graph.gate import MemoryGate, GateScore, _cosine_sim
from star_graph.scheduler import MemoryItem, AgentContext, MemoryType


def make_anchor(name: str, text: str = "", retention: float = 0.5,
                embedding: list[float] | None = None,
                emotional_valence: float = 0.0,
                surprise: float = 0.5,
                last_activated: float = 0.0,
                tags: list | None = None) -> Anchor:
    a = Anchor(id=name, text=text, vector=AnchorVector(
        importance=retention,
        frequency=0.5,
        recency=1.0,
        emotional_valence=emotional_valence,
        surprise=surprise,
        success_feedback=0.5,
        confidence=0.5,
    ))
    if embedding:
        a.embedding = embedding
    if last_activated:
        a.last_activated_at = last_activated
    if tags:
        a.tags = list(tags)
    return a


def make_item(anchor: Anchor, score: float = 0.5,
              reasoning_path: list | None = None) -> MemoryItem:
    return MemoryItem(
        anchor=anchor,
        relevance_score=score,
        reasoning_path=reasoning_path or [],
    )


class TestGateScore:
    def test_defaults(self):
        gs = GateScore()
        assert gs.total == 0.0
        assert gs.importance == 0.0

    def test_to_dict(self):
        gs = GateScore(importance=0.5, recency=0.3, total=2.1)
        d = gs.to_dict()
        assert d["importance"] == 0.5
        assert d["recency"] == 0.3
        assert d["total"] == 2.1


class TestCosineSim:
    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-4)

    def test_zero_vectors(self):
        assert _cosine_sim([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_negative(self):
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-4)

    def test_empty(self):
        assert _cosine_sim([], []) == 0.0


class TestMemoryGateInit:
    def test_default_params(self):
        g = MemoryGate()
        assert g.k == 20
        assert g.lateral_inhibition_radius == 0.7
        assert g.inhibition_strength == 0.3

    def test_custom_params(self):
        g = MemoryGate(k=5, lateral_inhibition_radius=0.5,
                       inhibition_strength=0.4)
        assert g.k == 5
        assert g.lateral_inhibition_radius == 0.5
        assert g.inhibition_strength == 0.4

    def test_default_weights(self):
        g = MemoryGate()
        assert g.w_importance == 0.20
        assert g.w_semantic == 0.25
        assert g.w_novelty == 0.05


class TestMemoryGateGate:
    def test_empty_items(self):
        g = MemoryGate(k=5)
        ctx = AgentContext()
        result = g.gate([], ctx)
        assert result == []

    def test_basic_single_item(self):
        g = MemoryGate(k=5)
        anchor = make_anchor("a1", "test memory", retention=0.8)
        item = make_item(anchor, score=0.7)
        ctx = AgentContext()
        result = g.gate([item], ctx)
        assert len(result) == 1
        assert result[0].anchor.id == "a1"

    def test_respects_k(self):
        g = MemoryGate(k=3)
        items = []
        for i in range(10):
            anchor = make_anchor(f"a{i}", f"memory {i}", retention=0.5 + i * 0.02)
            items.append(make_item(anchor, score=0.5))
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert len(result) == 3

    def test_k_larger_than_items(self):
        g = MemoryGate(k=20)
        items = [make_item(make_anchor(f"a{i}"), score=0.5) for i in range(5)]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert len(result) == 5

    def test_higher_retention_wins(self):
        g = MemoryGate(k=1)
        low = make_anchor("low", retention=0.1)
        high = make_anchor("high", retention=0.9)
        items = [make_item(low, score=0.5), make_item(high, score=0.5)]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert result[0].anchor.id == "high"

    def test_higher_semantic_match_wins(self):
        g = MemoryGate(k=1)
        a1 = make_anchor("a1", retention=0.5)
        a2 = make_anchor("a2", retention=0.5)
        items = [make_item(a1, score=0.2), make_item(a2, score=0.9)]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert result[0].anchor.id == "a2"

    def test_causal_bonus(self):
        g = MemoryGate(k=1)
        plain = make_anchor("plain", retention=0.5)
        causal = make_anchor("causal", retention=0.5)
        items = [
            make_item(plain, score=0.5, reasoning_path=["a0"]),
            make_item(causal, score=0.5, reasoning_path=["a0", "a1", "a2"]),
        ]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert result[0].anchor.id == "causal"

    def test_user_focus_with_goals(self):
        g = MemoryGate(k=1)
        a1 = make_anchor("a1", text="debug redis connection timeout", retention=0.5)
        a2 = make_anchor("a2", text="unrelated topic here", retention=0.5)
        items = [make_item(a1, score=0.5), make_item(a2, score=0.5)]
        ctx = AgentContext(active_goals=["debug redis performance"])
        result = g.gate(items, ctx)
        assert result[0].anchor.id == "a1"

    def test_emotional_salience_boosts_negative(self):
        g = MemoryGate(k=1, lateral_inhibition_radius=0.0)
        neutral = make_anchor("neutral", retention=0.5, emotional_valence=0.0)
        emotional = make_anchor("emotional", retention=0.5, emotional_valence=-0.9)
        items = [make_item(neutral, score=0.5), make_item(emotional, score=0.5)]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert result[0].anchor.id == "emotional"

    def test_novelty_boost(self):
        g = MemoryGate(k=1)
        normal = make_anchor("normal", retention=0.5, surprise=0.1)
        surprising = make_anchor("surprising", retention=0.5, surprise=0.9)
        items = [make_item(normal, score=0.5), make_item(surprising, score=0.5)]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert result[0].anchor.id == "surprising"

    def test_lateral_inhibition_suppresses_similar(self):
        # Two very similar items — only one should dominate
        g = MemoryGate(k=2, lateral_inhibition_radius=0.5)
        emb = [0.1 * i for i in range(10)]
        emb_similar = [0.1 * i + 0.01 for i in range(10)]
        a1 = make_anchor("a1", embedding=emb, retention=0.9)
        a2 = make_anchor("a2", embedding=emb_similar, retention=0.85)
        a3 = make_anchor("a3", embedding=[0.9 - 0.1 * i for i in range(10)],
                         retention=0.5, tags=["unique"])
        items = [make_item(a1, score=0.5), make_item(a2, score=0.5),
                 make_item(a3, score=0.5)]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        ids = [r.anchor.id for r in result]
        # a1 and a2 are similar; one should be suppressed
        assert len(result) == 2
        assert "a3" in ids  # unique item survives

    def test_updates_item_relevance_scores(self):
        g = MemoryGate(k=2)
        items = [
            make_item(make_anchor("a1", retention=0.9), score=0.5),
            make_item(make_anchor("a2", retention=0.3), score=0.5),
        ]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        for item in result:
            assert item.relevance_score > 0

    def test_recency_boost(self):
        import time
        g = MemoryGate(k=1)
        old = make_anchor("old", retention=0.5)
        old.last_activated_at = 0  # very old
        recent = make_anchor("recent", retention=0.5)
        recent.last_activated_at = time.time()  # just now
        items = [make_item(old, score=0.5), make_item(recent, score=0.5)]
        ctx = AgentContext()
        result = g.gate(items, ctx)
        assert result[0].anchor.id == "recent"
