"""Ghost Memory Subsystem — latent residual traces with fuzzy recall.

The most original part of star-graph-memory. Not just "deleted nodes" —
a proper latent memory trace system inspired by savings effects in
human memory.

Ghost nodes retain:
  - compressed_embedding (low-dim projection)
  - residual_edges (key connections survive pruning)
  - emotion_trace (emotional residue)
  - reactivation_probability (decays over time)
  - semantic_shadow (fuzzy gist of the original)
  - intensity (composite retrieval ranking score, v1.0-5b)

Negative ghosts (v1.0-5b): encode contradictions and suppress
matching memories during retrieval. When a belief is contradicted,
a negative ghost is created that resonates with similar queries
and lowers confidence of matching anchors.

v0.4 — proper subsystem, not just a dataclass.
v1.0-5b — intensity scoring + negative ghosts for contradiction tracking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, MemoryState
from .config import Config


@dataclass
class GhostNode:
    """A latent memory trace — more than just a pruned node.

    After pruning, key structural and emotional features persist at low
    resolution. These traces can partially reconstruct if reactivated.
    """

    id: str
    compressed_embedding: list[float]    # low-dim projection
    residual_edges: dict[str, float]     # neighbor_id → residual_weight
    emotion_trace: float                 # -1..+1, preserved emotional signature
    pruned_at: float
    original_tags: list[str]
    original_importance: float
    semantic_shadow: str                 # fuzzy gist (first sentence / key terms)
    reactivation_probability: float = 0.5  # decays over time
    revival_count: int = 0
    partial_recall_count: int = 0        # times fuzzy recall was triggered
    last_resonated_at: float = 0.0

    # Class-level survival function — set by MemoryManager at init
    _survival_fn: object = None

    @classmethod
    def set_survival_function(cls, fn) -> None:
        """Set the survival function for all ghosts (called by MemoryManager)."""
        cls._survival_fn = fn

    @classmethod
    def from_anchor(cls, anchor: Anchor, residual_edges: dict[str, float] | None = None) -> GhostNode:
        """Create a rich ghost from an anchor being pruned.

        Retains structural and emotional signatures at reduced resolution.
        """
        c = Config.get().ghost
        # Compress embedding to lower dimension
        dim = c.compressed_dim
        if anchor.embedding:
            emb = anchor.embedding
            step = max(1, len(emb) // dim)
            compressed = [emb[i] for i in range(0, min(len(emb), dim * step), step)][:dim]
        else:
            compressed = [0.0] * dim

        # Semantic shadow: extract key terms
        words = anchor.text.lower().split()
        important_words = [w for w in words if len(w) > 3][:8]
        shadow = "..." + " ".join(important_words) + "..." if important_words else ""

        return cls(
            id=anchor.id,
            compressed_embedding=compressed,
            residual_edges=residual_edges or {},
            emotion_trace=anchor.vector.emotional_valence * c.emotion_attenuation,
            pruned_at=time.time(),
            original_tags=list(anchor.tags),
            original_importance=anchor.vector.importance,
            semantic_shadow=shadow,
            reactivation_probability=c.initial_reactivation_prob,
        )

    def resonance(self, embedding: list[float], emotion_context: float = 0.0) -> float:
        """How strongly does this ghost resonate with new content?

        Combines:
        - Embedding similarity (via compressed vectors)
        - Emotional congruence (similar emotion → boosts)
        - Reactivation probability (decays over time)
        """
        if not self.compressed_embedding or not embedding:
            return 0.0

        c = Config.get().ghost

        # Compress query embedding the same way
        step = max(1, len(embedding) // len(self.compressed_embedding))
        query_compressed = [embedding[i] for i in range(0, min(len(embedding), len(self.compressed_embedding) * step), step)][:len(self.compressed_embedding)]

        # Cosine similarity on compressed vectors
        dot = sum(g * q for g, q in zip(self.compressed_embedding, query_compressed))
        ng = math.sqrt(sum(g**2 for g in self.compressed_embedding))
        nq = math.sqrt(sum(q**2 for q in query_compressed))
        if ng < 1e-8 or nq < 1e-8:
            return 0.0
        sim = dot / (ng * nq)

        # Emotional congruence bonus
        emotion_bonus = 0.0
        threshold = c.emotion_resonance_threshold
        if abs(self.emotion_trace) > threshold and abs(emotion_context) > threshold:
            if (self.emotion_trace > 0) == (emotion_context > 0):
                emotion_bonus = c.emotion_bonus_factor * min(abs(self.emotion_trace), abs(emotion_context))

        # Reactivation probability decays over time
        hours_since_prune = (time.time() - self.pruned_at) / 3600
        if GhostNode._survival_fn is not None:
            retention = GhostNode._survival_fn.ghost_decay(
                hours_since_prune, self.original_importance)
            effective_prob = self.reactivation_probability * retention
        else:
            half_life_hours = c.decay.half_life_days * 24
            effective_prob = self.reactivation_probability * math.exp(-hours_since_prune / half_life_hours)
        effective_prob = max(c.min_effective_prob, effective_prob)

        resonance_score = sim * effective_prob + emotion_bonus

        if resonance_score > c.resonance_persistence:
            self.last_resonated_at = time.time()

        return resonance_score

    def revive(self, new_text: str, new_embedding: list[float],
               new_tags: list[str] | None = None) -> Anchor:
        """Revive this ghost as a full anchor — faster than new learning."""
        c = Config.get().ghost
        self.revival_count += 1
        self.reactivation_probability = min(1.0, self.reactivation_probability + c.revival_stability_boost)

        # Merge tags
        merged_tags = list(set(self.original_tags + (new_tags or [])))

        return Anchor(
            id=self.id,
            text=new_text[:280],
            vector=Anchor.__dataclass_fields__["vector"].default_factory(),
            embedding=new_embedding,
            source_session="revived",
            tags=merged_tags,
            replay_count=0,
            state=MemoryState.REACTIVATED,
        )

    def partial_recall(self) -> tuple[str, float]:
        """Fuzzy recall: what we can remember without full reactivation.

        Returns (vague_description, confidence).
        This is the "I seem to remember..." feeling.
        """
        c = Config.get().ghost.partial_recall
        self.partial_recall_count += 1
        confidence = self.reactivation_probability * c.confidence_factor
        confidence *= math.exp(-self.partial_recall_count * c.decay_per_call)
        confidence = max(c.min_confidence, confidence)

        description = f"[fuzzy trace] {self.semantic_shadow} (confidence: {confidence:.2f})"
        return description, confidence

    def decay(self) -> bool:
        """Decay reactivation probability. Returns True if ghost should be fully purged."""
        c = Config.get().ghost.decay
        hours_since_prune = (time.time() - self.pruned_at) / 3600

        if GhostNode._survival_fn is not None:
            retention = GhostNode._survival_fn.ghost_decay(
                hours_since_prune, self.original_importance)
            self.reactivation_probability *= retention
        else:
            days = hours_since_prune / 24
            self.reactivation_probability *= 0.5 ** (days / c.half_life_days)

        self.reactivation_probability = max(0.0, self.reactivation_probability)

        if (hours_since_prune / 24 > c.purge_no_revival_days
                and self.revival_count == 0 and self.partial_recall_count == 0):
            return True
        if self.reactivation_probability < c.purge_prob_threshold:
            return True
        return False

    # ── v1.0-5b: Intensity scoring ─────────────────────

    @property
    def intensity(self) -> float:
        """Composite ghost intensity for retrieval ranking.

        Combines:
        - reactivation_probability (base signal strength)
        - original_importance (important memories leave stronger traces)
        - emotion_trace magnitude (emotional resonance boost)
        - recency of last resonance (recently triggered = more active)

        Higher intensity = this ghost trace is more influential in retrieval.
        Used to rank ghost-boosted retrieval results and suppress contradictions.
        """
        hours_since = (time.time() - max(self.pruned_at, self.last_resonated_at)) / 3600

        # Base intensity from reactivation probability
        base = self.reactivation_probability

        # Importance weighting: important memories leave stronger ghosts
        importance_factor = 0.5 + self.original_importance * 0.5

        # Emotional salience: |emotion| gives up to +25% boost
        emotional_boost = 1.0 + abs(self.emotion_trace) * 0.25

        # Recency: exponential decay of intensity over time since last activity
        recency = math.exp(-hours_since / (30 * 24))  # 30-day half-life

        # Revival bonus: revived ghosts are more salient
        revival_bonus = 1.0 + min(0.5, self.revival_count * 0.1)

        intensity = base * importance_factor * emotional_boost * recency * revival_bonus
        return max(0.0, min(1.0, intensity))

    @property
    def is_active(self) -> bool:
        """Ghost is 'active' if intensity is above threshold."""
        return self.intensity > Config.get().ghost.active_threshold


@dataclass
class NegativeGhost(GhostNode):
    """A ghost encoding a contradiction — suppresses matching memories.

    Created when a belief is explicitly contradicted. During retrieval,
    a resonant negative ghost reduces confidence of matching anchors.

    Fields:
      contradiction_target: what anchor/text this contradicts
      contradiction_type: "direct" (explicit negation), "update" (new info
                          supersedes old), "correction" (error fix)
      suppression_strength: 0..1 how strongly this suppresses matching items
    """

    contradiction_target: str = ""       # anchor ID or key being contradicted
    contradiction_text: str = ""         # human-readable description of the contradiction
    contradiction_type: str = "direct"   # "direct", "update", "correction"
    suppression_strength: float = 0.5    # how much to suppress matching anchors

    @classmethod
    def from_contradiction(cls, original_text: str,
                          contradiction_text: str,
                          target_anchor_id: str = "",
                          original_importance: float = 0.5,
                          contradiction_type: str = "direct",
                          embedding: list[float] | None = None) -> NegativeGhost:
        """Create a negative ghost from a contradiction.

        Args:
            original_text: The text that was contradicted
            contradiction_text: The correction/contradiction text
            target_anchor_id: ID of the anchor being contradicted
            original_importance: How important the original memory was
            contradiction_type: "direct", "update", or "correction"
            embedding: Optional embedding for resonance matching
        """
        import hashlib
        ghost_id = hashlib.blake2b(
            (original_text + contradiction_text).encode(), digest_size=8
        ).hexdigest()

        c = Config.get().ghost
        dim = c.compressed_dim

        if embedding:
            step = max(1, len(embedding) // dim)
            compressed = [embedding[i] for i in range(0, min(len(embedding), dim * step), step)][:dim]
        else:
            compressed = [0.0] * dim

        return cls(
            id=ghost_id,
            compressed_embedding=compressed,
            residual_edges={},
            emotion_trace=-0.5,  # negative emotional trace
            pruned_at=time.time(),
            original_tags=["contradiction"],
            original_importance=original_importance,
            semantic_shadow=contradiction_text[:120],
            reactivation_probability=c.initial_reactivation_prob,
            contradiction_target=target_anchor_id,
            contradiction_text=contradiction_text[:280],
            contradiction_type=contradiction_type,
            suppression_strength=0.5 + original_importance * 0.3,
        )

    def suppress(self, anchor_embedding: list[float] | None,
                anchor_importance: float = 0.5) -> float:
        """Compute suppression factor for a matching anchor.

        Returns a factor in [0, 1] where:
          0 = fully suppress (anchor should be hidden)
          1 = no suppression (anchor passes through)

        The suppression combines:
        - Resonance strength with target anchor
        - Suppression strength of this negative ghost
        - Importance of the target anchor (important anchors resist suppression)
        """
        if not self.compressed_embedding or not anchor_embedding:
            return 1.0

        resonance = self.resonance(anchor_embedding)
        if resonance < 0.1:
            return 1.0

        # Strong ghosts suppress more, but important anchors resist
        resistance = anchor_importance * 0.6
        effective_suppression = self.suppression_strength * resonance * (1.0 - resistance)

        # Negative ghosts with high intensity are more effective at suppression
        intensity_factor = 0.5 + self.intensity * 0.5
        effective_suppression *= intensity_factor

        return max(0.0, 1.0 - effective_suppression)


class GhostSubsystem:
    """Manages all ghost nodes — creation, resonance, revival, decay, fuzzy recall.

    v1.0-5b: intensity-ranked retrieval and negative ghost suppression.
    """

    def __init__(self):
        self.ghosts: dict[str, GhostNode] = {}
        self._resonance_cache: dict[str, float] = {}

    @property
    def negative_ghosts(self) -> list[NegativeGhost]:
        """Get all negative (contradiction) ghosts."""
        return [g for g in self.ghosts.values() if isinstance(g, NegativeGhost)]

    @property
    def positive_ghosts(self) -> list[GhostNode]:
        """Get all non-negative ghosts."""
        return [g for g in self.ghosts.values() if not isinstance(g, NegativeGhost)]

    def create(self, anchor: Anchor,
               residual_edges: dict[str, float] | None = None) -> GhostNode:
        ghost = GhostNode.from_anchor(anchor, residual_edges)
        self.ghosts[ghost.id] = ghost
        return ghost

    def create_negative(self, original_text: str,
                       contradiction_text: str,
                       target_anchor_id: str = "",
                       original_importance: float = 0.5,
                       contradiction_type: str = "direct",
                       embedding: list[float] | None = None) -> NegativeGhost:
        """Create a negative ghost to track a contradiction.

        During retrieval, negative ghosts that resonate with a query will
        suppress matching anchors that are similar to the original (now-contradicted) memory.
        """
        ghost = NegativeGhost.from_contradiction(
            original_text=original_text,
            contradiction_text=contradiction_text,
            target_anchor_id=target_anchor_id,
            original_importance=original_importance,
            contradiction_type=contradiction_type,
            embedding=embedding,
        )
        self.ghosts[ghost.id] = ghost
        return ghost

    def check_resonance(self, embedding: list[float],
                        emotion_context: float = 0.0,
                        threshold: float | None = None) -> list[tuple[GhostNode, float]]:
        """Find all ghosts that resonate with new content above threshold."""
        if threshold is None:
            threshold = Config.get().ghost.resonance.default_threshold
        matches = []
        for ghost in self.ghosts.values():
            score = ghost.resonance(embedding, emotion_context)
            if score > threshold:
                matches.append((ghost, score))
        matches.sort(key=lambda x: -x[1])
        return matches

    def ranked_resonance(self, embedding: list[float],
                        emotion_context: float = 0.0,
                        threshold: float | None = None,
                        top_k: int = 10) -> list[tuple[GhostNode, float]]:
        """Rank ghosts by intensity score for retrieval boosting.

        Returns top-k ghosts sorted by intensity (descending), which combines
        resonance strength with importance, emotional salience, and recency.
        Use this to decide which ghost traces should boost retrieval results.
        """
        if threshold is None:
            threshold = Config.get().ghost.resonance.default_threshold
        candidates = []
        for ghost in self.ghosts.values():
            if isinstance(ghost, NegativeGhost):
                continue  # negative ghosts handled separately via check_suppression
            score = ghost.resonance(embedding, emotion_context)
            if score > threshold:
                candidates.append((ghost, ghost.intensity))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def check_suppression(self, embedding: list[float],
                         threshold: float = 0.1) -> float:
        """Check negative ghosts for suppression of an embedding.

        Returns a suppression factor in [0, 1]:
          0 = query is fully suppressed (contradicted)
          1 = no suppression (clear)

        All resonant negative ghosts combine multiplicatively.
        """
        factor = 1.0
        for ghost in self.negative_ghosts:
            if ghost.intensity < 0.05:
                continue
            resonance = ghost.resonance(embedding)
            if resonance > threshold:
                factor *= ghost.suppression_strength * (1.0 - resonance * 0.5)
        return max(0.1, factor)

    def suppress_anchor(self, anchor_embedding: list[float] | None,
                       anchor_importance: float = 0.5,
                       threshold: float = 0.1) -> float:
        """Check if any negative ghost suppresses a specific anchor.

        Returns the combined suppression factor from all resonant negative ghosts.
        1.0 = no suppression, lower values = more suppressed.
        """
        if not anchor_embedding:
            return 1.0

        factor = 1.0
        for ghost in self.negative_ghosts:
            if ghost.intensity < 0.05:
                continue
            if ghost.contradiction_target:
                # Targeted contradiction: stronger suppression for specific anchor
                f = ghost.suppress(anchor_embedding, anchor_importance)
                factor *= f
            else:
                # General contradiction: weaker, broad suppression
                resonance = ghost.resonance(anchor_embedding)
                if resonance > threshold:
                    factor *= 1.0 - resonance * ghost.suppression_strength * 0.3

        return max(0.1, factor)

    def get_top_intensity(self, top_k: int = 10) -> list[tuple[GhostNode, float]]:
        """Get top-k ghosts by intensity score (for health monitoring)."""
        scored = [(g, g.intensity) for g in self.ghosts.values()]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def try_revive(self, ghost_id: str, new_text: str, new_embedding: list[float],
                   new_tags: list[str] | None = None) -> Anchor | None:
        """Attempt to revive a specific ghost. Returns Anchor or None."""
        ghost = self.ghosts.get(ghost_id)
        if not ghost:
            return None
        anchor = ghost.revive(new_text, new_embedding, new_tags)
        del self.ghosts[ghost_id]
        return anchor

    def fuzzy_recall(self, embedding: list[float], threshold: float | None = None) -> list[tuple[str, float]]:
        """Fuzzy recall: return vague descriptions of ghosts that weakly resonate.

        This is the "I seem to remember something about..." feeling.
        Low-confidence retrieval from ghost traces.
        """
        c = Config.get().ghost.resonance
        if threshold is None:
            threshold = c.fuzzy_threshold
        results = []
        for ghost in self.ghosts.values():
            score = ghost.resonance(embedding)
            if threshold * c.fuzzy_sub_factor < score < threshold:
                desc, conf = ghost.partial_recall()
                results.append((desc, conf * score))
        results.sort(key=lambda x: -x[1])
        return results

    def decay_all(self) -> tuple[int, list[str]]:
        """Decay all ghosts, remove fully purged ones. Returns (count, removed_ids)."""
        to_remove = []
        for gid, ghost in self.ghosts.items():
            if ghost.decay():
                to_remove.append(gid)
        for gid in to_remove:
            del self.ghosts[gid]
        return len(to_remove), to_remove

    @property
    def stats(self) -> dict:
        active = sum(1 for g in self.ghosts.values() if g.reactivation_probability > Config.get().ghost.active_threshold)
        revived = sum(1 for g in self.ghosts.values() if g.revival_count > 0)
        negative_count = len(self.negative_ghosts)
        intensities = [g.intensity for g in self.ghosts.values()]
        return {
            "total_ghosts": len(self.ghosts),
            "active_ghosts": active,
            "revived_ghosts": revived,
            "negative_ghosts": negative_count,
            "avg_intensity": sum(intensities) / max(1, len(intensities)),
            "max_intensity": max(intensities) if intensities else 0.0,
            "avg_reactivation_prob": sum(
                g.reactivation_probability for g in self.ghosts.values()
            ) / max(1, len(self.ghosts)),
            "total_partial_recalls": sum(
                g.partial_recall_count for g in self.ghosts.values()
            ),
        }
