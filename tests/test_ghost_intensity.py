"""Tests for v1.0-5b: ghost intensity scoring + negative ghosts."""

import math
import time

import pytest

from star_graph.ghost import GhostNode, NegativeGhost, GhostSubsystem
from star_graph.anchor import Anchor, MemoryState


# ═══════════════════════════════════════════════════════════════
# Ghost Intensity
# ═══════════════════════════════════════════════════════════════

class TestGhostIntensity:
    def test_intensity_is_between_0_and_1(self):
        ghost = GhostNode(
            id="test",
            compressed_embedding=[0.1] * 16,
            residual_edges={},
            emotion_trace=0.5,
            pruned_at=time.time(),
            original_tags=["test"],
            original_importance=0.5,
            semantic_shadow="test shadow...",
            reactivation_probability=0.5,
        )
        intensity = ghost.intensity
        assert 0.0 <= intensity <= 1.0

    def test_high_importance_ghost_has_higher_intensity(self):
        ghost_low = GhostNode(
            id="low", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.0,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.1, semantic_shadow="low...",
            reactivation_probability=0.5,
        )
        ghost_high = GhostNode(
            id="high", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.0,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.9, semantic_shadow="high...",
            reactivation_probability=0.5,
        )
        assert ghost_high.intensity > ghost_low.intensity

    def test_emotional_charge_boosts_intensity(self):
        ghost_neutral = GhostNode(
            id="neutral", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.0,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.5, semantic_shadow="...",
            reactivation_probability=0.5,
        )
        ghost_charged = GhostNode(
            id="charged", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.9,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.5, semantic_shadow="...",
            reactivation_probability=0.5,
        )
        assert ghost_charged.intensity > ghost_neutral.intensity

    def test_old_ghost_has_lower_intensity(self):
        ghost_recent = GhostNode(
            id="recent", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.5,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.5, semantic_shadow="...",
            reactivation_probability=0.5,
        )
        ghost_old = GhostNode(
            id="old", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.5,
            pruned_at=time.time() - 86400 * 365,  # 1 year ago
            original_tags=["test"], original_importance=0.5,
            semantic_shadow="...", reactivation_probability=0.5,
        )
        assert ghost_recent.intensity > ghost_old.intensity

    def test_revived_ghost_gets_intensity_bonus(self):
        ghost = GhostNode(
            id="revived", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.3,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.5, semantic_shadow="...",
            reactivation_probability=0.5, revival_count=5,
        )
        ghost2 = GhostNode(
            id="never_revived", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.3,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.5, semantic_shadow="...",
            reactivation_probability=0.5, revival_count=0,
        )
        assert ghost.intensity > ghost2.intensity

    def test_is_active_returns_bool(self):
        ghost = GhostNode(
            id="active_test", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.8,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.9, semantic_shadow="...",
            reactivation_probability=0.8,
        )
        assert isinstance(ghost.is_active, bool)

    def test_zero_probability_ghost_has_low_intensity(self):
        ghost = GhostNode(
            id="dead", compressed_embedding=[0.1] * 16,
            residual_edges={}, emotion_trace=0.0,
            pruned_at=time.time(), original_tags=["test"],
            original_importance=0.0, semantic_shadow="...",
            reactivation_probability=0.0,
        )
        assert ghost.intensity < 0.1


# ═══════════════════════════════════════════════════════════════
# Negative Ghost
# ═══════════════════════════════════════════════════════════════

class TestNegativeGhost:
    def test_from_contradiction_creates_valid_ghost(self):
        ghost = NegativeGhost.from_contradiction(
            original_text="Redis runs on port 6379",
            contradiction_text="Redis was moved to port 6380",
            target_anchor_id="abc123",
            original_importance=0.7,
            contradiction_type="update",
        )
        assert ghost.id
        assert ghost.contradiction_target == "abc123"
        assert ghost.contradiction_type == "update"
        assert ghost.emotion_trace == -0.5  # negative emotional signature
        assert "contradiction" in ghost.original_tags
        assert ghost.suppression_strength > 0.5

    def test_suppress_with_matching_embedding(self):
        ghost = NegativeGhost.from_contradiction(
            original_text="User prefers tabs over spaces",
            contradiction_text="User now prefers spaces",
            original_importance=0.6,
        )
        # Create a matching embedding
        embedding = [0.5] * 384
        ghost.compressed_embedding = [0.5] * 16
        factor = ghost.suppress(embedding, anchor_importance=0.3)
        assert factor < 1.0  # Should suppress

    def test_suppress_with_unrelated_embedding(self):
        ghost = NegativeGhost.from_contradiction(
            original_text="User prefers tabs",
            contradiction_text="User now prefers spaces",
        )
        ghost.compressed_embedding = [0.5] * 16
        unrelated = [0.0] * 384
        factor = ghost.suppress(unrelated, anchor_importance=0.3)
        assert factor > 0.9  # Little to no suppression

    def test_important_anchor_resists_suppression(self):
        ghost = NegativeGhost.from_contradiction(
            original_text="Some fact",
            contradiction_text="Corrected fact",
        )
        ghost.compressed_embedding = [0.7] * 16
        embedding = [0.7] * 384

        factor_low_imp = ghost.suppress(embedding, anchor_importance=0.2)
        factor_high_imp = ghost.suppress(embedding, anchor_importance=0.9)
        assert factor_high_imp > factor_low_imp  # Important anchors resist more

    def test_direct_contradiction_higher_suppression(self):
        direct = NegativeGhost.from_contradiction(
            original_text="X is true",
            contradiction_text="X is NOT true",
            contradiction_type="direct",
        )
        correction = NegativeGhost.from_contradiction(
            original_text="X is true",
            contradiction_text="Actually, X is partly true but Y is more accurate",
            contradiction_type="correction",
        )
        # direct contradictions should have stronger suppression
        assert direct.suppression_strength >= correction.suppression_strength * 0.8

    def test_intensity_affects_suppression(self):
        ghost_strong = NegativeGhost.from_contradiction(
            original_text="Important fact",
            contradiction_text="Corrected important fact",
            original_importance=0.9,
        )
        ghost_weak = NegativeGhost.from_contradiction(
            original_text="Trivial fact",
            contradiction_text="Corrected trivial fact",
            original_importance=0.1,
        )
        ghost_strong.compressed_embedding = [0.5] * 16
        ghost_weak.compressed_embedding = [0.5] * 16
        embedding = [0.5] * 384

        strong_suppression = 1.0 - ghost_strong.suppress(embedding, 0.3)
        weak_suppression = 1.0 - ghost_weak.suppress(embedding, 0.3)
        assert strong_suppression > weak_suppression


# ═══════════════════════════════════════════════════════════════
# GhostSubsystem Intensity + Negative
# ═══════════════════════════════════════════════════════════════

class TestGhostSubsystemIntensity:
    def test_ranked_resonance_sorts_by_intensity(self):
        sub = GhostSubsystem()
        # Create ghosts with different importances
        for i, imp in enumerate([0.2, 0.5, 0.9]):
            ghost = GhostNode(
                id=f"g{i}", compressed_embedding=[0.5] * 16,
                residual_edges={}, emotion_trace=0.3,
                pruned_at=time.time(), original_tags=["test"],
                original_importance=imp, semantic_shadow=f"g{i}...",
                reactivation_probability=0.5,
            )
            sub.ghosts[ghost.id] = ghost

        embedding = [0.5] * 384
        ranked = sub.ranked_resonance(embedding, threshold=0.0, top_k=5)
        assert len(ranked) == 3
        # Highest importance should be first
        intensities = [r[1] for r in ranked]
        assert intensities == sorted(intensities, reverse=True)

    def test_get_top_intensity(self):
        sub = GhostSubsystem()
        for i in range(5):
            ghost = GhostNode(
                id=f"g{i}", compressed_embedding=[0.1] * 16,
                residual_edges={}, emotion_trace=0.2 * i,
                pruned_at=time.time(), original_tags=["test"],
                original_importance=0.2 * i, semantic_shadow=f"g{i}...",
                reactivation_probability=0.3 + 0.1 * i,
            )
            sub.ghosts[ghost.id] = ghost

        top = sub.get_top_intensity(top_k=3)
        assert len(top) == 3
        # Should be sorted descending
        for i in range(len(top) - 1):
            assert top[i][1] >= top[i + 1][1]

    def test_positive_ghosts_excludes_negatives(self):
        sub = GhostSubsystem()
        g1 = GhostNode(id="p1", compressed_embedding=[0.1] * 16,
                      residual_edges={}, emotion_trace=0.0,
                      pruned_at=time.time(), original_tags=["test"],
                      original_importance=0.5, semantic_shadow="...",
                      reactivation_probability=0.5)
        ng = NegativeGhost(id="n1", compressed_embedding=[0.1] * 16,
                          residual_edges={}, emotion_trace=-0.5,
                          pruned_at=time.time(), original_tags=["contradiction"],
                          original_importance=0.5, semantic_shadow="no...",
                          reactivation_probability=0.5,
                          contradiction_target="abc", contradiction_text="X is wrong",
                          contradiction_type="direct")
        sub.ghosts[g1.id] = g1
        sub.ghosts[ng.id] = ng
        assert len(sub.positive_ghosts) == 1
        assert len(sub.negative_ghosts) == 1

    def test_ranked_resonance_excludes_negatives(self):
        sub = GhostSubsystem()
        g1 = GhostNode(id="p1", compressed_embedding=[0.5] * 16,
                      residual_edges={}, emotion_trace=0.3,
                      pruned_at=time.time(), original_tags=["test"],
                      original_importance=0.5, semantic_shadow="...",
                      reactivation_probability=0.5)
        ng = NegativeGhost(id="n1", compressed_embedding=[0.5] * 16,
                          residual_edges={}, emotion_trace=-0.5,
                          pruned_at=time.time(), original_tags=["contradiction"],
                          original_importance=0.5, semantic_shadow="no...",
                          reactivation_probability=0.5,
                          contradiction_target="abc", contradiction_text="X is wrong",
                          contradiction_type="direct")
        sub.ghosts[g1.id] = g1
        sub.ghosts[ng.id] = ng
        embedding = [0.5] * 384
        ranked = sub.ranked_resonance(embedding, threshold=0.0, top_k=10)
        assert len(ranked) == 1
        assert ranked[0][0].id == "p1"


class TestGhostSubsystemNegative:
    def test_create_negative_adds_to_ghosts(self):
        sub = GhostSubsystem()
        ghost = sub.create_negative(
            original_text="User lives in Beijing",
            contradiction_text="User moved to Shanghai",
            original_importance=0.6,
            contradiction_type="update",
        )
        assert ghost.id in sub.ghosts
        assert isinstance(sub.ghosts[ghost.id], NegativeGhost)

    def test_check_suppression_no_negatives(self):
        sub = GhostSubsystem()
        factor = sub.check_suppression([0.5] * 384)
        assert factor == 1.0  # No suppression

    def test_check_suppression_with_negative(self):
        sub = GhostSubsystem()
        ghost = sub.create_negative(
            original_text="The API endpoint is /v1/users",
            contradiction_text="The API endpoint is now /v2/users",
            original_importance=0.7,
        )
        ghost.compressed_embedding = [0.5] * 16

        # Matching query
        factor = sub.check_suppression([0.5] * 384)
        assert factor < 1.0  # Should be suppressed

        # Unrelated query
        factor2 = sub.check_suppression([0.0] * 384)
        assert factor2 > factor  # Less suppressed

    def test_suppress_anchor_targeted(self):
        sub = GhostSubsystem()
        ghost = sub.create_negative(
            original_text="Redis config: maxmemory 1GB",
            contradiction_text="Redis config corrected: maxmemory 2GB",
            target_anchor_id="redis_config_anchor",
            original_importance=0.8,
        )
        ghost.compressed_embedding = [0.5] * 16
        factor = sub.suppress_anchor([0.5] * 384, anchor_importance=0.3)
        assert factor < 1.0

    def test_suppress_anchor_untargeted(self):
        sub = GhostSubsystem()
        ghost = sub.create_negative(
            original_text="General fact",
            contradiction_text="Updated general fact",
            target_anchor_id="",  # no specific target
        )
        ghost.compressed_embedding = [0.5] * 16
        factor = sub.suppress_anchor([0.5] * 384, anchor_importance=0.3)
        assert factor <= 1.0

    def test_stats_includes_negative_count(self):
        sub = GhostSubsystem()
        sub.create_negative(
            original_text="Old info",
            contradiction_text="New info",
        )
        stats = sub.stats
        assert stats["negative_ghosts"] == 1
        assert "avg_intensity" in stats
        assert "max_intensity" in stats

    def test_multiple_negatives_combine(self):
        sub = GhostSubsystem()
        for i in range(3):
            ghost = sub.create_negative(
                original_text=f"Fact {i}",
                contradiction_text=f"Corrected fact {i}",
                original_importance=0.5,
            )
            ghost.compressed_embedding = [0.5] * 16

        factor = sub.check_suppression([0.5] * 384)
        # Multiple negatives should multiply their suppression
        assert factor < 1.0


# ═══════════════════════════════════════════════════════════════
# MemoryManager Integration
# ═══════════════════════════════════════════════════════════════

class TestManagerGhostIntensity:
    def test_ghost_intensity_recall(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        # Create and prune an anchor to generate a ghost
        anchor = mgr.remember("User's favorite color is blue", tags=["preference"])
        mgr.forget(anchor.id, create_ghost=True)

        results = mgr.ghost_intensity_recall("favorite color", top_k=5)
        assert isinstance(results, list)

    def test_create_negative_ghost(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        ghost_id = mgr.create_negative_ghost(
            original_text="Server port is 8080",
            contradiction_text="Server port changed to 9090",
            original_importance=0.7,
            contradiction_type="update",
        )
        assert ghost_id
        assert ghost_id in mgr.ghosts.ghosts

    def test_check_ghost_suppression(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        mgr.create_negative_ghost(
            original_text="API key is stored in config.yaml",
            contradiction_text="API key moved to .env file",
            original_importance=0.6,
        )

        result = mgr.check_ghost_suppression(query="API key config.yaml")
        assert "suppression_factor" in result
        assert "active_negatives" in result
        assert 0.0 <= result["suppression_factor"] <= 1.0

    def test_check_ghost_suppression_empty_query(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        result = mgr.check_ghost_suppression(query="")
        assert result["suppression_factor"] == 1.0
        assert result["active_negatives"] == []


# ═══════════════════════════════════════════════════════════════
# GhostSubsystem stats
# ═══════════════════════════════════════════════════════════════

class TestGhostStats:
    def test_stats_has_all_keys(self):
        sub = GhostSubsystem()
        s = sub.stats
        expected = {
            "total_ghosts", "active_ghosts", "revived_ghosts",
            "negative_ghosts", "avg_intensity", "max_intensity",
            "avg_reactivation_prob", "total_partial_recalls",
        }
        assert expected == set(s.keys())
