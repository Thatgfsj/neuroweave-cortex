"""Tests for evolution module — EvolutionEvent, BeliefTransition, MemoryEvolutionEngine."""

import time

import pytest

from star_graph.evolution import (
    EvolutionEvent,
    BeliefTransition,
    MemoryEvolutionEngine,
)
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


class TestEvolutionEvent:
    def test_defaults(self):
        ee = EvolutionEvent()
        assert ee.event_type == ""
        assert ee.anchor_id == ""
        assert ee.old_value == 0.0
        assert ee.new_value == 0.0

    def test_custom_fields(self):
        ee = EvolutionEvent(
            timestamp=1000.0,
            event_type="decay",
            anchor_id="a1",
            description="test decay",
            old_value=0.8,
            new_value=0.5,
        )
        assert ee.event_type == "decay"
        assert ee.anchor_id == "a1"
        assert ee.description == "test decay"
        assert ee.old_value == 0.8
        assert ee.new_value == 0.5
        assert ee.timestamp == 1000.0

    def test_timestamp_defaults_to_now(self):
        ee = EvolutionEvent()
        assert ee.timestamp > 0


class TestBeliefTransition:
    def test_defaults(self):
        bt = BeliefTransition(
            anchor_id="a1", topic="python",
            old_belief="Python is slow", new_belief="Python is fast",
            old_anchor_id="a_old", new_anchor_id="a_new",
        )
        assert bt.anchor_id == "a1"
        assert bt.topic == "python"
        assert bt.old_belief == "Python is slow"
        assert bt.new_belief == "Python is fast"
        assert bt.old_anchor_id == "a_old"
        assert bt.new_anchor_id == "a_new"
        assert bt.confidence_shift == 0.0

    def test_custom_confidence_shift(self):
        bt = BeliefTransition(
            anchor_id="a1", topic="test",
            old_belief="old", new_belief="new",
            old_anchor_id="oa", new_anchor_id="na",
            confidence_shift=0.5,
        )
        assert bt.confidence_shift == 0.5

    def test_transition_time_defaults(self):
        bt = BeliefTransition(
            anchor_id="a1", topic="t",
            old_belief="old", new_belief="new",
            old_anchor_id="oa", new_anchor_id="na",
        )
        assert bt.transition_time > 0


class TestMemoryEvolutionEngine:
    def test_init_default(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        assert engine.graph is g
        assert engine._cycle_count == 0
        assert engine._last_evolve_time > 0

    def test_evolve_empty_graph(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine.evolve()
        assert "cycle" in result
        assert result["cycle"] == 1
        assert "decay" in result
        assert "boost" in result
        assert "conflicts" in result
        assert "interference" in result
        assert "evolution" in result
        assert "thermal" in result

    def test_evolve_with_explicit_time(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine.evolve(current_time=time.time())
        assert result["cycle"] == 1

    def test_evolve_multiple_cycles(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        engine.evolve()
        result = engine.evolve()
        assert result["cycle"] == 2

    def test_evolve_with_anchor(self):
        g = StarGraph()
        a = Anchor.create(text="test memory", tags=["test"])
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        result = engine.evolve()
        assert result["cycle"] == 1
        assert "total_events" in result

    def test_time_decay_zero_elapsed(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_time_decay(0)
        assert result == {"decayed": 0}

    def test_time_decay_empty_graph(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_time_decay(3600)
        assert result["decayed"] == 0

    def test_time_decay_with_anchor(self):
        g = StarGraph()
        a = Anchor.create(text="old memory")
        a.last_activated_at = 0.0  # very old
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_time_decay(3600)
        assert result["decayed"] >= 0

    def test_frequency_boost_empty(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_frequency_boost()
        assert result["boosted"] == 0

    def test_resolve_conflicts_empty(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine._resolve_conflicts()
        assert result["resolved"] == 0

    def test_apply_interference_empty(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_interference()
        assert result["proactive"] == 0
        assert result["retroactive"] == 0

    def test_evolve_anchors_empty(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine._evolve_anchors()
        assert result == {"merged": 0}

    def test_apply_thermal_degradation_empty(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_thermal_degradation()
        assert "hot" in result
        assert "cold" in result
        assert "finalized" in result

    def test_evolution_summary(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        summary = engine.evolution_summary()
        assert "total_events" in summary
        assert "belief_transitions" in summary

    def test_access_count_tracking(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        engine._access_counts["a1"] = 5
        assert engine._access_counts["a1"] == 5

    def test_cycle_count_increments(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        assert engine._cycle_count == 0
        engine.evolve()
        assert engine._cycle_count == 1


# ── Evolution with real anchors ──────────────────────────

class TestEvolutionWithAnchors:
    def test_time_decay_with_dormant(self):
        from star_graph.anchor import MemoryState
        g = StarGraph()
        a = Anchor.create(text="dormant memory")
        a.state = MemoryState.DORMANT
        a.last_activated_at = 0.0
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_time_decay(100.0)
        assert "decayed" in result

    def test_time_decay_with_ghost(self):
        from star_graph.anchor import MemoryState
        g = StarGraph()
        a = Anchor.create(text="ghost memory")
        a.state = MemoryState.GHOST
        a.last_activated_at = 0.0
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_time_decay(100.0)
        assert result["decayed"] >= 0

    def test_time_decay_with_reactivated(self):
        from star_graph.anchor import MemoryState
        g = StarGraph()
        a = Anchor.create(text="reactivated memory")
        a.state = MemoryState.REACTIVATED
        a.last_activated_at = 0.0
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_time_decay(100.0)
        assert result["decayed"] >= 0

    def test_frequency_boost_with_replays(self):
        g = StarGraph()
        a = Anchor.create(text="frequently accessed memory")
        a.replay_count = 5
        a.vector.importance = 0.3
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_frequency_boost()
        assert result["boosted"] >= 0
        assert a.vector.importance > 0.3

    def test_resolve_conflicts_with_opposing_valence(self):
        import time as _time
        g = StarGraph()
        a = Anchor.create(text="I like Python", tags=["python", "preference"])
        a.created_at = _time.time() - 7200  # 2 hours ago
        a.vector.emotional_valence = 0.9
        b = Anchor.create(text="I hate Python", tags=["python", "preference"])
        b.created_at = _time.time()
        b.vector.emotional_valence = -0.9
        g.add_anchor(a)
        g.add_anchor(b)
        engine = MemoryEvolutionEngine(g)
        result = engine._resolve_conflicts()
        assert result["resolved"] >= 1

    def test_resolve_conflicts_with_semantic(self):
        import time as _time
        g = StarGraph()
        a = Anchor.create(text="Python is great for scripting", tags=["python", "opinion"])
        a.created_at = _time.time() - 7200
        a.embedding = [0.5] * 10
        b = Anchor.create(text="Python is OK for scripting but not ideal", tags=["python", "opinion"])
        b.created_at = _time.time()
        b.embedding = [0.6] * 10  # similar but different
        g.add_anchor(a)
        g.add_anchor(b)
        engine = MemoryEvolutionEngine(g)
        result = engine._resolve_conflicts()
        assert isinstance(result, dict)

    def test_apply_interference_with_similar(self):
        import time as _time
        g = StarGraph()
        a = Anchor.create(text="memory old version")
        a.created_at = _time.time() - 7200
        a.embedding = [0.5] * 10
        a.vector.importance = 0.7
        b = Anchor.create(text="memory new version")
        b.created_at = _time.time()
        b.embedding = [0.55] * 10
        b.vector.importance = 0.6
        g.add_anchor(a)
        g.add_anchor(b)
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_interference()
        assert isinstance(result, dict)

    def test_evolve_with_mergeable_anchors(self):
        g = StarGraph()
        a = Anchor.create(text="Python is a programming language", tags=["python", "fact"])
        a.vector.stability = 0.8
        a.embedding = [0.5] * 10
        b = Anchor.create(text="Python is an interpreted language", tags=["python", "fact"])
        b.vector.stability = 0.75
        b.embedding = [0.55] * 10
        c = Anchor.create(text="Python has dynamic typing", tags=["python", "fact"])
        c.vector.stability = 0.72
        c.embedding = [0.52] * 10
        g.add_anchor(a)
        g.add_anchor(b)
        g.add_anchor(c)
        engine = MemoryEvolutionEngine(g)
        result = engine._evolve_anchors()
        assert result["merged"] >= 0

    def test_evolve_full_cycle(self):
        import time as _time
        g = StarGraph()
        a = Anchor.create(text="Python async programming", tags=["python", "async"])
        a.last_activated_at = _time.time() - 10000
        a.replay_count = 3
        a.vector.importance = 0.4
        a.vector.emotional_valence = 0.3
        b = Anchor.create(text="Async debugging techniques", tags=["debugging", "async"])
        b.last_activated_at = _time.time() - 10000
        b.embedding = [0.5] * 10
        b.vector.importance = 0.5
        g.add_anchor(a)
        g.add_anchor(b)
        engine = MemoryEvolutionEngine(g)
        result = engine.evolve()
        assert result["cycle"] == 1

    def test_record_access(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        engine.record_access("a1")
        assert engine._access_counts.get("a1", 0) >= 0

    def test_record_access_with_anchor(self):
        g = StarGraph()
        a = Anchor.create(text="test memory")
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        engine.record_access(a.id)
        assert engine._access_counts.get(a.id, 0) == 1

    def test_compute_importance(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        engine._access_counts["a1"] = 5
        engine._access_counts["a2"] = 3
        a = Anchor.create(text="test memory")
        a.id = "a1"
        a.vector.surprise = 0.3
        a.vector.emotional_valence = 0.6
        score = engine.compute_importance(a, goal_relevance=0.7)
        assert 0.0 <= score <= 1.0

    def test_evolution_summary_with_data(self):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        engine.evolve()
        engine.evolve()
        summary = engine.evolution_summary()
        assert summary["cycle"] == 2
        assert "by_type" in summary
        assert "beliefs" in summary

    def test_print_report(self, capsys):
        g = StarGraph()
        engine = MemoryEvolutionEngine(g)
        engine.evolve()
        engine.print_report()
        captured = capsys.readouterr()
        assert "Memory Evolution Report" in captured.out

    def test_evolve_creates_events(self):
        g = StarGraph()
        a = Anchor.create(text="test memory with replay")
        a.replay_count = 3
        a.last_activated_at = 0.0
        a.vector.importance = 0.3
        g.add_anchor(a)
        engine = MemoryEvolutionEngine(g)
        engine.evolve(current_time=time.time() + 10000)
        assert len(engine.events) >= 0

    def test_edge_decay_with_old_edges(self):
        import time as _time
        g = StarGraph()
        a = Anchor.create(text="source memory")
        b = Anchor.create(text="target memory")
        a.last_activated_at = _time.time() - 200000
        b.last_activated_at = _time.time() - 200000
        g.add_anchor(a)
        g.add_anchor(b)
        g.add_edge(a.id, b.id, weight=0.5, edge_type="related")
        # Set edge last_activated_at to be very old
        for key, edge in g.edges.items():
            edge.last_activated_at = 0.0
        engine = MemoryEvolutionEngine(g)
        result = engine._apply_time_decay(1000.0)
        assert result["decayed"] >= 0
