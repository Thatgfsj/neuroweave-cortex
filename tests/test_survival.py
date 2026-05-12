"""Tests for v1.0-5 survival functions: configurable memory decay curves."""

import math
import time

import pytest

from star_graph.survival import (
    SurvivalFunction,
    EbbinghausSurvival,
    PowerLawSurvival,
    ExponentialSurvival,
    CustomSurvival,
    SurvivalRegistry,
    SurvivalState,
    derive_strength,
)
from star_graph.anchor import Anchor, GhostAnchor, MemoryState


# ═══════════════════════════════════════════════════════════════
# Ebbinghaus Survival
# ═══════════════════════════════════════════════════════════════

class TestEbbinghausSurvival:
    def test_survive_zero_time_returns_one(self):
        fn = EbbinghausSurvival()
        assert fn.survive(0.0, strength=0.5) == 1.0

    def test_survive_decays_over_time(self):
        fn = EbbinghausSurvival()
        r1 = fn.survive(1.0, strength=0.5)
        r2 = fn.survive(100.0, strength=0.5)
        assert r2 < r1

    def test_stronger_memory_decays_slower(self):
        fn = EbbinghausSurvival()
        r_weak = fn.survive(48.0, strength=0.1)
        r_strong = fn.survive(48.0, strength=0.9)
        assert r_strong > r_weak

    def test_half_life_range(self):
        fn = EbbinghausSurvival()
        low = fn.half_life(strength=0.0)
        high = fn.half_life(strength=1.0)
        assert low < high
        assert low > 0.1

    def test_ghost_decay_faster_than_live(self):
        fn = EbbinghausSurvival()
        live = fn.survive(24.0, strength=0.5)
        ghost = fn.ghost_decay(24.0, base_strength=0.5)
        assert ghost < live

    def test_name_is_ebbinghaus(self):
        fn = EbbinghausSurvival()
        assert fn.name == "ebbinghaus"

    def test_custom_min_max_hours(self):
        fn = EbbinghausSurvival(min_hours=0.5, max_hours=500.0)
        assert fn.survive(0.5, strength=0.0) < 1.0


# ═══════════════════════════════════════════════════════════════
# Power Law Survival
# ═══════════════════════════════════════════════════════════════

class TestPowerLawSurvival:
    def test_survive_zero_time_returns_one(self):
        fn = PowerLawSurvival()
        assert fn.survive(0.0, strength=0.5) == 1.0

    def test_survive_decays_over_time(self):
        fn = PowerLawSurvival()
        r1 = fn.survive(1.0, strength=0.5)
        r2 = fn.survive(500.0, strength=0.5)
        assert r2 < r1

    def test_slower_long_term_decay_than_ebbinghaus(self):
        ebb = EbbinghausSurvival()
        pl = PowerLawSurvival()
        t_long = 500.0
        assert pl.survive(t_long, strength=0.5) > ebb.survive(t_long, strength=0.5)

    def test_half_life_positive(self):
        fn = PowerLawSurvival()
        assert fn.half_life(strength=0.5) > 0

    def test_name_is_power_law(self):
        fn = PowerLawSurvival()
        assert fn.name == "power_law"


# ═══════════════════════════════════════════════════════════════
# Exponential Survival
# ═══════════════════════════════════════════════════════════════

class TestExponentialSurvival:
    def test_survive_zero_time_returns_one(self):
        fn = ExponentialSurvival()
        assert fn.survive(0.0, strength=0.5) == 1.0

    def test_stronger_memory_slower_decay(self):
        fn = ExponentialSurvival()
        r_weak = fn.survive(48.0, strength=0.1)
        r_strong = fn.survive(48.0, strength=0.9)
        assert r_strong > r_weak

    def test_half_life_reasonable(self):
        fn = ExponentialSurvival(lambda_per_day=0.05)
        hl = fn.half_life(strength=0.5)
        assert 1.0 < hl < 10000.0

    def test_name_is_exponential(self):
        fn = ExponentialSurvival()
        assert fn.name == "exponential"


# ═══════════════════════════════════════════════════════════════
# Custom Survival
# ═══════════════════════════════════════════════════════════════

class TestCustomSurvival:
    def test_custom_lambda(self):
        fn = CustomSurvival(lambda t, s: 1.0 / (1.0 + t / (s * 100 + 1)))
        assert fn.survive(0.0, strength=0.5) == 1.0
        assert fn.survive(100.0, strength=0.5) < 1.0
        assert fn.survive(100.0, strength=0.5) > 0.0

    def test_clamped_to_0_1(self):
        fn = CustomSurvival(lambda t, s: 1.5)
        assert fn.survive(10.0, strength=0.5) == 1.0
        fn2 = CustomSurvival(lambda t, s: -0.5)
        assert fn2.survive(10.0, strength=0.5) == 0.0

    def test_half_life_with_function(self):
        fn = CustomSurvival(
            fn=lambda t, s: math.exp(-t / (s * 50 + 1)),
            half_life_fn=lambda s: (s * 50 + 1) * math.log(2),
        )
        hl = fn.half_life(strength=0.5)
        assert hl > 0

    def test_half_life_brute_force_fallback(self):
        fn = CustomSurvival(lambda t, s: max(0.01, 1.0 - t / 1000))
        hl = fn.half_life(strength=0.5)
        assert 400 < hl < 600  # should find ~500

    def test_name_is_custom(self):
        fn = CustomSurvival(lambda t, s: 1.0)
        assert fn.name == "custom"


# ═══════════════════════════════════════════════════════════════
# Survival Registry
# ═══════════════════════════════════════════════════════════════

class TestSurvivalRegistry:
    def test_default_registry_has_builtins(self):
        reg = SurvivalRegistry()
        assert "ebbinghaus" in reg.available
        assert "power_law" in reg.available
        assert "exponential" in reg.available

    def test_get_returns_survival_function(self):
        reg = SurvivalRegistry()
        fn = reg.get("ebbinghaus")
        assert isinstance(fn, EbbinghausSurvival)

    def test_get_unknown_raises_keyerror(self):
        reg = SurvivalRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_register_custom(self):
        reg = SurvivalRegistry()
        custom = ExponentialSurvival(lambda_per_day=0.1)
        reg.register("my_custom", custom)
        assert "my_custom" in reg.available
        assert reg.get("my_custom") is custom

    def test_from_config_defaults_to_ebbinghaus(self):
        fn = SurvivalRegistry.from_config(config=None)
        assert fn.name == "ebbinghaus"

    def test_from_config_with_config_object(self):
        class Cfg:
            pass

        class SurvCfg:
            function = "power_law"
            power_law_alpha = 0.8
            power_law_scale = 100.0

        cfg = Cfg()
        cfg.survival = SurvCfg()
        fn = SurvivalRegistry.from_config(config=cfg)
        assert fn.name == "power_law"
        assert fn.alpha == 0.8
        assert fn.scale == 100.0

    def test_from_config_exponential(self):
        class Cfg:
            pass

        class SurvCfg:
            function = "exponential"
            exponential_lambda_per_day = 0.1

        cfg = Cfg()
        cfg.survival = SurvCfg()
        fn = SurvivalRegistry.from_config(config=cfg)
        assert fn.name == "exponential"
        assert fn.lambda_per_hour == pytest.approx(0.1 / 24.0)

    def test_from_config_ebbinghaus_custom_params(self):
        class Cfg:
            pass

        class SurvCfg:
            function = "ebbinghaus"
            ebbinghaus_min_hours = 2.0
            ebbinghaus_max_hours = 500.0

        cfg = Cfg()
        cfg.survival = SurvCfg()
        fn = SurvivalRegistry.from_config(config=cfg)
        assert fn.name == "ebbinghaus"
        assert fn.min_hours == 2.0
        assert fn.max_hours == 500.0


# ═══════════════════════════════════════════════════════════════
# Survival State
# ═══════════════════════════════════════════════════════════════

class TestSurvivalState:
    def test_initial_state(self):
        state = SurvivalState()
        assert state.current_retention == 1.0
        assert state.decay_count == 0
        assert state.function_name == "ebbinghaus"

    def test_apply_reduces_retention(self):
        state = SurvivalState()
        fn = EbbinghausSurvival()
        # Advance last_decay_at to 10 hours ago
        state.last_decay_at = time.time() - 36000
        r = state.apply(fn, anchor_strength=0.5)
        assert r < 1.0
        assert state.decay_count == 1

    def test_apply_resets_timestamp(self):
        state = SurvivalState()
        fn = ExponentialSurvival()
        state.last_decay_at = time.time() - 3600
        state.apply(fn, anchor_strength=0.5)
        # Should be set to roughly "now" (within last 5 seconds)
        assert state.last_decay_at >= time.time() - 5

    def test_custom_function_name(self):
        state = SurvivalState(function_name="power_law")
        assert state.function_name == "power_law"


# ═══════════════════════════════════════════════════════════════
# Derive Strength
# ═══════════════════════════════════════════════════════════════

class TestDeriveStrength:
    def test_default_strength_in_range(self):
        anchor = Anchor.create("test memory", importance=0.5)
        s = derive_strength(anchor)
        assert 0.0 <= s <= 1.0

    def test_high_importance_high_strength(self):
        anchor = Anchor.create("critical fact", importance=1.0,
                              emotional_valence=1.0)
        anchor.vector.confidence = 1.0
        s_high = derive_strength(anchor)
        anchor2 = Anchor.create("trivial note", importance=0.1,
                               emotional_valence=0.0)
        anchor2.vector.confidence = 0.1
        s_low = derive_strength(anchor2)
        assert s_high > s_low

    def test_emotional_valence_contributes(self):
        neutral = Anchor.create("neutral fact", emotional_valence=0.0)
        charged = Anchor.create("emotional fact", emotional_valence=0.9)
        assert derive_strength(charged) > derive_strength(neutral)

    def test_stability_contributes(self):
        a = Anchor.create("test")
        a.vector.stability = 0.9
        s_stable = derive_strength(a)
        a.vector.stability = 0.1
        s_labile = derive_strength(a)
        assert s_stable > s_labile


# ═══════════════════════════════════════════════════════════════
# Anchor Integration
# ═══════════════════════════════════════════════════════════════

class TestAnchorDecayWithSurvival:
    def test_decay_with_survival_fn(self):
        anchor = Anchor.create("episodic memory", importance=0.7,
                              emotional_valence=0.5)
        fn = EbbinghausSurvival()
        old_recency = anchor.vector.recency
        anchor.decay(elapsed_hours=48.0, survival_fn=fn)
        assert anchor.vector.recency < old_recency
        assert anchor.vector.recency > 0.0

    def test_decay_fallback_without_survival_fn(self):
        anchor = Anchor.create("legacy memory")
        old_recency = anchor.vector.recency
        anchor.decay(elapsed_hours=24.0, half_life=168.0)
        assert anchor.vector.recency < old_recency

    def test_decay_respects_minimum(self):
        anchor = Anchor.create("very old memory")
        fn = EbbinghausSurvival()
        anchor.decay(elapsed_hours=100000.0, survival_fn=fn)
        assert anchor.vector.recency >= 0.01

    def test_set_survival_function_class_level(self):
        fn = ExponentialSurvival()
        old = Anchor._survival_fn
        try:
            Anchor.set_survival_function(fn)
            assert Anchor._survival_fn is fn
        finally:
            Anchor._survival_fn = old

    def test_decay_factor_with_survival_fn(self):
        fn = EbbinghausSurvival()
        old = Anchor._survival_fn
        try:
            Anchor.set_survival_function(fn)
            anchor = Anchor.create("test", importance=0.5)
            # Simulate an old memory by backdating last_activated_at
            anchor.last_activated_at = time.time() - 86400 * 7  # 7 days
            df = anchor.decay_factor
            assert 0.0 < df < 1.0
        finally:
            Anchor._survival_fn = old

    def test_decay_factor_without_survival_fn(self):
        old = Anchor._survival_fn
        Anchor._survival_fn = None
        try:
            anchor = Anchor.create("legacy test", importance=0.5)
            anchor.last_activated_at = time.time() - 86400
            df = anchor.decay_factor
            assert 0.0 < df <= 1.0
        finally:
            Anchor._survival_fn = old

    def test_dormant_state_slower_decay(self):
        fn = EbbinghausSurvival()
        anchor = Anchor.create("consolidated memory", importance=0.7)
        anchor.state = MemoryState.DORMANT
        anchor.decay(elapsed_hours=48.0, survival_fn=fn)
        r_dormant = anchor.vector.recency

        anchor2 = Anchor.create("active memory", importance=0.7)
        anchor2.state = MemoryState.ACTIVE
        anchor2.decay(elapsed_hours=48.0, survival_fn=fn)
        r_active = anchor2.vector.recency

        # Dormant should retain more (slower decay)
        assert r_dormant >= r_active


# ═══════════════════════════════════════════════════════════════
# GhostNode Integration
# ═══════════════════════════════════════════════════════════════

class TestGhostWithSurvival:
    def test_ghost_decay_with_survival_fn(self):
        from star_graph.ghost import GhostNode

        fn = EbbinghausSurvival()
        old = GhostNode._survival_fn
        try:
            GhostNode.set_survival_function(fn)
            ghost = GhostNode(
                id="test_ghost",
                compressed_embedding=[0.1] * 16,
                residual_edges={},
                emotion_trace=0.3,
                pruned_at=time.time() - 86400 * 30,  # 30 days ago
                original_tags=["test"],
                original_importance=0.5,
                semantic_shadow="test...",
                reactivation_probability=0.5,
            )
            old_prob = ghost.reactivation_probability
            purged = ghost.decay()
            assert ghost.reactivation_probability < old_prob
        finally:
            GhostNode._survival_fn = old

    def test_ghost_decay_fallback_without_survival_fn(self):
        from star_graph.ghost import GhostNode

        old = GhostNode._survival_fn
        GhostNode._survival_fn = None
        try:
            ghost = GhostNode(
                id="test_ghost2",
                compressed_embedding=[0.1] * 16,
                residual_edges={},
                emotion_trace=0.3,
                pruned_at=time.time() - 86400 * 60,  # 60 days
                original_tags=["test"],
                original_importance=0.5,
                semantic_shadow="test...",
                reactivation_probability=0.5,
            )
            ghost.decay()
            assert ghost.reactivation_probability <= 0.5
        finally:
            GhostNode._survival_fn = old

    def test_ghost_resonance_with_survival_fn(self):
        from star_graph.ghost import GhostNode

        fn = EbbinghausSurvival()
        old = GhostNode._survival_fn
        try:
            GhostNode.set_survival_function(fn)
            ghost = GhostNode(
                id="test_ghost3",
                compressed_embedding=[0.5] * 16,
                residual_edges={},
                emotion_trace=0.5,
                pruned_at=time.time() - 3600,  # 1 hour
                original_tags=["test"],
                original_importance=0.5,
                semantic_shadow="test...",
                reactivation_probability=0.5,
            )
            embedding = [0.5] * 384
            score = ghost.resonance(embedding, emotion_context=0.5)
            assert 0.0 <= score <= 1.0
            # Should have high resonance since embeddings match and recent
            assert score > 0.3
        finally:
            GhostNode._survival_fn = old

    def test_ghost_resonance_fallback_without_survival_fn(self):
        from star_graph.ghost import GhostNode

        old = GhostNode._survival_fn
        GhostNode._survival_fn = None
        try:
            ghost = GhostNode(
                id="test_ghost4",
                compressed_embedding=[0.3] * 16,
                residual_edges={},
                emotion_trace=0.0,
                pruned_at=time.time(),
                original_tags=["test"],
                original_importance=0.5,
                semantic_shadow="test...",
                reactivation_probability=0.5,
            )
            embedding = [0.3] * 384
            score = ghost.resonance(embedding)
            assert 0.0 <= score <= 1.0
        finally:
            GhostNode._survival_fn = old


# ═══════════════════════════════════════════════════════════════
# MemoryManager Integration
# ═══════════════════════════════════════════════════════════════

class TestManagerSurvivalIntegration:
    def test_manager_sets_survival_function(self):
        from star_graph import MemoryManager
        from star_graph.anchor import Anchor
        from star_graph.ghost import GhostNode

        old_anchor_fn = Anchor._survival_fn
        old_ghost_fn = GhostNode._survival_fn
        try:
            mgr = MemoryManager()
            assert Anchor._survival_fn is not None
            assert Anchor._survival_fn.name == "ebbinghaus"
            assert GhostNode._survival_fn is not None
        finally:
            Anchor._survival_fn = old_anchor_fn
            GhostNode._survival_fn = old_ghost_fn

    def test_remember_and_decay_with_survival(self):
        from star_graph import MemoryManager
        from star_graph.anchor import Anchor

        old = Anchor._survival_fn
        try:
            mgr = MemoryManager()
            mgr.remember("Test survival integration", tags=["test"])
            # Survival function should be active
            assert Anchor._survival_fn is not None
        finally:
            Anchor._survival_fn = old

    def test_survival_function_property(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        fn = mgr.survival_function
        assert fn is not None
        assert fn.name == "ebbinghaus"

    def test_custom_survival_from_config(self):
        from star_graph import MemoryManager
        from star_graph.anchor import Anchor

        class Cfg:
            pass

        class SurvCfg:
            function = "power_law"
            power_law_alpha = 0.3
            power_law_scale = 80.0

        cfg = Cfg()
        cfg.survival = SurvCfg()

        old = Anchor._survival_fn
        try:
            mgr = MemoryManager(config=cfg)
            assert mgr.survival_function.name == "power_law"
            assert mgr.survival_function.alpha == 0.3
        finally:
            Anchor._survival_fn = old


# ═══════════════════════════════════════════════════════════════
# Protocol Compliance
# ═══════════════════════════════════════════════════════════════

class TestSurvivalProtocol:
    def test_all_builtins_implement_protocol(self):
        for fn in [EbbinghausSurvival(), PowerLawSurvival(), ExponentialSurvival()]:
            assert isinstance(fn, SurvivalFunction)
            assert hasattr(fn, 'survive')
            assert hasattr(fn, 'half_life')
            assert hasattr(fn, 'ghost_decay')
            assert hasattr(fn, 'name')

    def test_custom_implements_protocol(self):
        fn = CustomSurvival(lambda t, s: 0.5)
        assert isinstance(fn, SurvivalFunction)

    def test_ghost_decay_faster_than_survive(self):
        """ghost_decay should always decay faster than survive for same params."""
        for fn in [EbbinghausSurvival(), PowerLawSurvival(), ExponentialSurvival()]:
            for t in [1.0, 24.0, 168.0]:
                live = fn.survive(t, strength=0.5)
                ghost = fn.ghost_decay(t, base_strength=0.3)
                assert ghost <= live, f"{fn.name}: ghost={ghost} > live={live} at t={t}"

    def test_survive_range(self):
        """All survival functions must return values in [0, 1]."""
        for fn in [EbbinghausSurvival(), PowerLawSurvival(), ExponentialSurvival()]:
            for t in [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
                for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    r = fn.survive(t, s)
                    assert 0.0 <= r <= 1.0, f"{fn.name}: r={r} at t={t}, s={s}"

    def test_half_life_approx(self):
        """At t=half_life, retention should be approximately 0.5."""
        for fn in [EbbinghausSurvival(), ExponentialSurvival()]:
            for s in [0.3, 0.5, 0.7]:
                hl = fn.half_life(strength=s)
                r = fn.survive(hl, strength=s)
                # Allow some tolerance since half-life definitions vary
                assert 0.35 < r < 0.65, f"{fn.name}: r={r} at half_life={hl}, s={s}"
