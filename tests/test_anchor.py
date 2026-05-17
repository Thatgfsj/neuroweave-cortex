"""Tests for anchor module — Anchor, AnchorVector, Oscillator, MemoryState, etc."""

import math
import time

import pytest

from star_graph.anchor import (
    Anchor,
    AnchorVector,
    Oscillator,
    AnchorPrediction,
    EmbedderRegistry,
    MemoryState,
    ThermalState,
)


# ── AnchorVector ──────────────────────────────────────────

class TestAnchorVector:
    def test_default_values(self):
        v = AnchorVector()
        assert v.importance == 0.5
        assert v.frequency == 0.0
        assert v.recency == 1.0
        assert v.emotional_valence == 0.0
        assert v.stability == 0.0
        assert v.surprise == 0.5
        assert v.hippocampal_dependency == 1.0
        assert v.success_feedback == 0.5
        assert v.confidence == 0.5
        assert v.decay_rate == 0.01

    def test_to_list_returns_10_elements(self):
        v = AnchorVector()
        lst = v.to_list()
        assert len(lst) == 10

    def test_from_list_new_format(self):
        v = AnchorVector.from_list([0.8, 0.3, 0.9, 0.1, 0.4, 0.6, 0.7, 0.5, 0.6, 0.02])
        assert v.importance == 0.8
        assert v.frequency == 0.3
        assert v.decay_rate == 0.02

    def test_from_list_short_pads_defaults(self):
        v = AnchorVector.from_list([0.8, 0.3, 0.9])
        assert v.importance == 0.8
        assert v.frequency == 0.3
        assert v.recency == 0.9
        # Remaining should be defaults
        assert v.stability == 0.0
        assert v.decay_rate == 0.01

    def test_from_list_old_13_element_format(self):
        """Old format had novelty(9), task_relevance(10), future_reusability(11) before decay_rate(12)."""
        v = AnchorVector.from_list([0.8, 0.3, 0.9, 0.1, 0.4, 0.6, 0.7, 0.5, 0.6,
                                      0.5, 0.4, 0.3, 0.02])
        # novelty (0.5) merged with surprise (max(0.6, 0.5) = 0.6)
        assert v.surprise == 0.6
        # task_relevance (0.4) merged with importance (max(0.8, 0.4) = 0.8)
        assert v.importance == 0.8

    def test_custom_values(self):
        v = AnchorVector(importance=0.9, emotional_valence=-0.5, stability=0.7)
        assert v.importance == 0.9
        assert v.emotional_valence == -0.5
        assert v.stability == 0.7


# ── Oscillator ────────────────────────────────────────────

class TestOscillator:
    def test_default_values(self):
        osc = Oscillator()
        assert osc.natural_frequency == 0.5
        assert osc.phase_offset == 0.0
        assert osc.coupling_strength == 0.3
        assert osc.damping == 0.1

    def test_resonance_identical_freq_and_phase(self):
        osc = Oscillator(natural_frequency=0.5, phase_offset=1.0, coupling_strength=0.5)
        r = osc.resonance(0.5, 1.0)
        assert r > 0.0

    def test_resonance_far_frequencies_returns_zero(self):
        osc = Oscillator(natural_frequency=0.2, coupling_strength=0.3)
        r = osc.resonance(0.9, 0.0)
        assert r == 0.0

    def test_resonance_negative_phase_match_returns_zero(self):
        osc = Oscillator(natural_frequency=0.5, phase_offset=0.0, coupling_strength=1.0)
        r = osc.resonance(0.5, math.pi)  # opposite phase
        assert r == 0.0

    def test_resonance_phase_wraps_at_2pi(self):
        osc = Oscillator(natural_frequency=0.5, phase_offset=0.1, coupling_strength=1.0)
        # Phase diff > pi should wrap
        r1 = osc.resonance(0.5, 6.0)  # diff = 5.9, wraps to 2π-5.9 = 0.383...
        assert r1 >= 0.0


# ── AnchorPrediction ──────────────────────────────────────

class TestAnchorPrediction:
    def test_default_values(self):
        p = AnchorPrediction()
        assert p.emotional_tone == 0.0
        assert p.expected_duration == 10.0
        assert p.confidence == 0.5
        assert p.next_topic_embedding is None

    def test_error_identical(self):
        p1 = AnchorPrediction(emotional_tone=0.5, next_topic_embedding=[0.1, 0.2, 0.3])
        p2 = AnchorPrediction(emotional_tone=0.5, next_topic_embedding=[0.1, 0.2, 0.3])
        err = p1.error(p2)
        assert err == pytest.approx(0.0, abs=0.01)

    def test_error_different_emotional(self):
        p1 = AnchorPrediction(emotional_tone=0.5)
        p2 = AnchorPrediction(emotional_tone=0.2)
        err = p1.error(p2)
        assert err == pytest.approx(0.3, abs=0.01)

    def test_error_no_embeddings(self):
        p1 = AnchorPrediction(emotional_tone=0.3)
        p2 = AnchorPrediction(emotional_tone=0.7)
        err = p1.error(p2)
        assert err == pytest.approx(0.4, abs=0.01)


# ── EmbedderRegistry ──────────────────────────────────────

class TestEmbedderRegistry:
    def test_instance_init_with_embedder(self):
        er = EmbedderRegistry(embedder="fake")
        assert er.get_embedder() == "fake"

    def test_instance_init_without_embedder(self):
        er = EmbedderRegistry()
        # Falls back to get_embedder() which may create a real one
        emb = er.get_embedder()
        assert emb is not None

    def test_set_embedder(self):
        er = EmbedderRegistry()
        er.set_embedder("custom")
        assert er.get_embedder() == "custom"

    def test_is_available_when_set(self):
        er = EmbedderRegistry(embedder="fake")
        assert er.is_available

    def test_singleton_set_get(self):
        EmbedderRegistry.set_embedder_singleton("singleton_fake")
        assert EmbedderRegistry.get_embedder_singleton() == "singleton_fake"
        # Cleanup
        EmbedderRegistry._embedder = None

    def test_singleton_availability(self):
        EmbedderRegistry._embedder = "fake"
        assert EmbedderRegistry.is_available_singleton()
        EmbedderRegistry._embedder = None


# ── MemoryState ───────────────────────────────────────────

class TestMemoryState:
    def test_all_states_exist(self):
        states = list(MemoryState)
        assert MemoryState.ACTIVE in states
        assert MemoryState.REHEARSING in states
        assert MemoryState.CONSOLIDATING in states
        assert MemoryState.DORMANT in states
        assert MemoryState.GHOST in states
        assert MemoryState.REACTIVATED in states

    def test_state_values(self):
        assert MemoryState.ACTIVE.value == "active"
        assert MemoryState.GHOST.value == "ghost"


# ── ThermalState ──────────────────────────────────────────

class TestThermalState:
    def test_all_states_exist(self):
        states = list(ThermalState)
        assert ThermalState.HOT in states
        assert ThermalState.WARM in states
        assert ThermalState.COLD in states
        assert ThermalState.FROZEN in states
        assert ThermalState.DEAD in states

    def test_state_values(self):
        assert ThermalState.HOT.value == "hot"
        assert ThermalState.DEAD.value == "dead"


# ── Anchor ────────────────────────────────────────────────

class TestAnchorBasic:
    def test_minimal_anchor(self):
        a = Anchor(id="test1", text="hello world")
        assert a.id == "test1"
        assert a.text == "hello world"
        assert a.state == MemoryState.ACTIVE
        assert a.embedding is None
        assert a.tags == []
        assert a.replay_count == 0

    def test_anchor_with_embedding(self):
        a = Anchor(id="a1", text="test", embedding=[0.1, 0.2, 0.3])
        assert a.embedding == [0.1, 0.2, 0.3]

    def test_anchor_create_basic(self):
        """Anchor.create() generates id and meaningful oscillator params."""
        a = Anchor.create("this is a test memory", source_session="sess1")
        assert a.id  # has an id
        assert len(a.id) == 16  # blake2b hex digest 8 bytes = 16 hex chars
        assert a.text.startswith("this is a test memory")
        assert a.state == MemoryState.ACTIVE
        assert a.source_session == "sess1"

    def test_anchor_is_retrievable_default(self):
        a = Anchor(id="a1", text="test")
        assert a.is_retrievable

    def test_anchor_ghost_not_retrievable(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.GHOST
        assert not a.is_retrievable

    def test_anchor_is_plastic_default(self):
        a = Anchor(id="a1", text="test")
        assert a.is_plastic

    def test_anchor_dormant_not_plastic(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.DORMANT
        assert not a.is_plastic

    def test_relevance_property(self):
        v = AnchorVector(importance=0.8, success_feedback=0.7, emotional_valence=0.5)
        a = Anchor(id="a1", text="test", vector=v)
        r = a.relevance
        # base = importance * success_feedback = 0.8 * 0.7 = 0.56
        # emotional_boost = 1 + |0.5| * 0.2 = 1.1
        # relevance = min(1.0, 0.56 * 1.1) = 0.616
        assert 0.5 < r <= 1.0

    def test_importance_score(self):
        v = AnchorVector(importance=0.8, emotional_valence=0.5, surprise=0.6)
        a = Anchor(id="a1", text="test", vector=v)
        score = a.importance_score
        # 0.8*0.50 + 0.5*0.25 + 0.6*0.25 = 0.4 + 0.125 + 0.15 = 0.675
        assert 0.6 < score < 0.8

    def test_is_cortical(self):
        v = AnchorVector(hippocampal_dependency=0.2)
        a = Anchor(id="a1", text="test", vector=v)
        assert a.is_cortical

    def test_is_not_cortical(self):
        v = AnchorVector(hippocampal_dependency=0.8)
        a = Anchor(id="a1", text="test", vector=v)
        assert not a.is_cortical

    def test_is_labile(self):
        v = AnchorVector(stability=0.2)
        a = Anchor(id="a1", text="test", vector=v)
        assert a.is_labile

    def test_is_not_labile(self):
        v = AnchorVector(stability=0.6)
        a = Anchor(id="a1", text="test", vector=v)
        assert not a.is_labile

    def test_activate_sets_recency_to_1(self):
        a = Anchor(id="a1", text="test")
        a.vector.recency = 0.3
        a.activate()
        assert a.vector.recency == 1.0

    def test_activate_boosts_frequency(self):
        a = Anchor(id="a1", text="test")
        old_freq = a.vector.frequency
        a.activate()
        assert a.vector.frequency > old_freq

    def test_consolidate_strengthen(self):
        a = Anchor(id="a1", text="test")
        old_imp = a.vector.importance
        result = a.consolidate(0.1)  # low error → strengthen
        assert result == "strengthen"
        assert a.vector.importance > old_imp

    def test_consolidate_update(self):
        a = Anchor(id="a1", text="test")
        result = a.consolidate(0.3)
        assert result in ("strengthen", "update")

    def test_consolidate_novel(self):
        a = Anchor(id="a1", text="test")
        result = a.consolidate(0.9)  # high error → novel
        assert result == "novel"

    def test_record_success(self):
        a = Anchor(id="a1", text="test")
        old_sf = a.vector.success_feedback
        a.record_success(0.1)
        assert a.vector.success_feedback > old_sf

    def test_record_failure(self):
        a = Anchor(id="a1", text="test")
        old_sf = a.vector.success_feedback
        a.record_failure(0.1)
        assert a.vector.success_feedback < old_sf

    def test_record_verification(self):
        a = Anchor(id="a1", text="test")
        old_conf = a.vector.confidence
        a.record_verification()
        assert a.vector.confidence > old_conf


class TestAnchorStateMachine:
    def test_initial_state_is_active(self):
        a = Anchor(id="a1", text="test")
        assert a.state == MemoryState.ACTIVE
        assert len(a.state_history) == 0  # no history until transition

    def test_transition_replay(self):
        a = Anchor(id="a1", text="test")
        new_state = a.transition("replay")
        assert new_state == MemoryState.REHEARSING
        assert len(a.state_history) == 1

    def test_transition_consolidate(self):
        a = Anchor(id="a1", text="test")
        new_state = a.transition("consolidate")
        assert new_state == MemoryState.CONSOLIDATING

    def test_transition_prune(self):
        a = Anchor(id="a1", text="test")
        new_state = a.transition("prune")
        assert new_state == MemoryState.GHOST

    def test_transition_invalid_event(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.DORMANT
        new_state = a.transition("prune")
        assert new_state == MemoryState.GHOST

    def test_full_lifecycle(self):
        a = Anchor(id="a1", text="test")
        a.transition("consolidate")  # ACTIVE → CONSOLIDATING
        assert a.state == MemoryState.CONSOLIDATING
        a.transition("stabilize")   # CONSOLIDATING → DORMANT
        assert a.state == MemoryState.DORMANT
        a.transition("retrieve")    # DORMANT → ACTIVE
        assert a.state == MemoryState.ACTIVE


class TestAnchorThermal:
    def test_active_is_hot(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.ACTIVE
        a._thermal_state = None
        a.last_activated_at = time.time()
        assert a.thermal_state == ThermalState.HOT

    def test_ghost_is_cold(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.GHOST
        a._ghost_reactivation_prob = 0.1
        assert a.thermal_state == ThermalState.COLD

    def test_ghost_dead_at_low_prob(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.GHOST
        a._ghost_reactivation_prob = 0.01
        assert a.thermal_state == ThermalState.DEAD

    def test_thermal_priority(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.ACTIVE
        a._thermal_state = ThermalState.HOT
        assert a.thermal_priority == 1.0

    def test_retrieval_cost_hot_is_zero(self):
        a = Anchor(id="a1", text="test")
        a._thermal_state = ThermalState.HOT
        assert a.retrieval_cost == 0.0

    def test_storage_tier_active(self):
        a = Anchor(id="a1", text="test")
        a._thermal_state = ThermalState.HOT
        assert a.storage_tier == "memory"

    def test_thaw_dead_returns_false(self):
        a = Anchor(id="a1", text="test")
        a._thermal_state = ThermalState.DEAD
        assert not a.thaw()

    def test_thaw_cold_ghost_transitions(self):
        a = Anchor(id="a1", text="test")
        a.state = MemoryState.GHOST
        a._ghost_reactivation_prob = 0.1
        a._thermal_state = None
        # thaw should transition ghost to REACTIVATED
        assert a.thaw()
        assert a.state == MemoryState.REACTIVATED

    def test_is_thermally_retrievable(self):
        a = Anchor(id="a1", text="test")
        a._thermal_state = ThermalState.HOT
        assert a.is_thermally_retrievable

    def test_dead_not_thermally_retrievable(self):
        a = Anchor(id="a1", text="test")
        a._thermal_state = ThermalState.DEAD
        assert not a.is_thermally_retrievable

    def test_thermal_summary(self):
        a = Anchor(id="a1", text="test")
        a._thermal_state = ThermalState.HOT
        summary = a.thermal_summary
        assert "hot" in summary


class TestAnchorDecay:
    def test_decay_reduces_recency(self):
        a = Anchor(id="a1", text="test")
        old_recency = a.vector.recency
        a.decay(24.0)  # 24 hours
        assert a.vector.recency < old_recency

    def test_decay_minimum_is_001(self):
        a = Anchor(id="a1", text="test")
        a.decay(100000.0)  # very long time
        assert a.vector.recency >= 0.01

    def test_decay_factor_property(self):
        a = Anchor(id="a1", text="test")
        df = a.decay_factor
        assert 0.0 < df <= 1.0

    def test_retention_score(self):
        a = Anchor(id="a1", text="test")
        score = a.retention_score
        assert 0.0 <= score <= 1.0

    def test_retention_score_cached(self):
        a = Anchor(id="a1", text="test")
        score1 = a.retention_score
        score2 = a.retention_score  # should use cache
        assert score1 == score2

    def test_confidence_score(self):
        a = Anchor(id="a1", text="test")
        cs = a.confidence_score
        assert 0.0 < cs <= 1.0


class TestAnchorSemanticDensity:
    def test_raw_anchor_low_density(self):
        v = AnchorVector(hippocampal_dependency=1.0, stability=0.0)
        a = Anchor(id="a1", text="raw event", vector=v)
        assert a.semantic_density < 0.5

    def test_consolidated_anchor_high_density(self):
        v = AnchorVector(hippocampal_dependency=0.1, stability=0.9)
        a = Anchor(id="a1", text="abstract rule", vector=v, schema_ref="s1")
        assert a.semantic_density > 0.5

    def test_abstract_tags_boost_density(self):
        v = AnchorVector(hippocampal_dependency=0.5, stability=0.5)
        a = Anchor(id="a1", text="rule text", vector=v, tags=["pattern", "lesson"])
        assert a.semantic_density >= 0.5


class TestAnchorActivationPotential:
    def test_activation_potential_range(self):
        a = Anchor(id="a1", text="test")
        ap = a.activation_potential
        assert 0.0 <= ap <= 1.0

    def test_hot_memory_higher_potential(self):
        a_hot = Anchor(id="hot", text="test")
        a_hot._thermal_state = ThermalState.HOT
        a_cold = Anchor(id="cold", text="test")
        a_cold._thermal_state = ThermalState.COLD
        assert a_hot.activation_potential > a_cold.activation_potential


class TestAnchorIsStale:
    def test_new_anchor_not_stale(self):
        a = Anchor(id="a1", text="test")
        assert not a.is_stale

    def test_old_anchor_is_stale(self):
        a = Anchor(id="a1", text="test", last_activated_at=0.0)
        assert a.is_stale
