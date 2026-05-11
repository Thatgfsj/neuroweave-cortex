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

Fuzzy recall: "I seem to remember the user once mentioned something about Redis..."
even when the exact memory is gone.

v0.4 — proper subsystem, not just a dataclass.
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

        # Reactivation probability decays exponentially
        hours_since_prune = (time.time() - self.pruned_at) / 3600
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
        days = hours_since_prune / 24

        self.reactivation_probability *= 0.5 ** (days / c.half_life_days)
        self.reactivation_probability = max(0.0, self.reactivation_probability)

        if (days > c.purge_no_revival_days and self.revival_count == 0 and self.partial_recall_count == 0):
            return True
        if self.reactivation_probability < c.purge_prob_threshold:
            return True
        return False


class GhostSubsystem:
    """Manages all ghost nodes — creation, resonance, revival, decay, fuzzy recall."""

    def __init__(self):
        self.ghosts: dict[str, GhostNode] = {}
        self._resonance_cache: dict[str, float] = {}

    def create(self, anchor: Anchor,
               residual_edges: dict[str, float] | None = None) -> GhostNode:
        ghost = GhostNode.from_anchor(anchor, residual_edges)
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

    def decay_all(self) -> int:
        """Decay all ghosts, remove fully purged ones. Returns count removed."""
        to_remove = []
        for gid, ghost in self.ghosts.items():
            if ghost.decay():
                to_remove.append(gid)
        for gid in to_remove:
            del self.ghosts[gid]
        return len(to_remove)

    @property
    def stats(self) -> dict:
        active = sum(1 for g in self.ghosts.values() if g.reactivation_probability > Config.get().ghost.active_threshold)
        revived = sum(1 for g in self.ghosts.values() if g.revival_count > 0)
        return {
            "total_ghosts": len(self.ghosts),
            "active_ghosts": active,
            "revived_ghosts": revived,
            "avg_reactivation_prob": sum(
                g.reactivation_probability for g in self.ghosts.values()
            ) / max(1, len(self.ghosts)),
            "total_partial_recalls": sum(
                g.partial_recall_count for g in self.ghosts.values()
            ),
        }
