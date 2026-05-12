"""Memory Gate — winner-take-all selection with lateral inhibition.

Phase 3 of sparse activation: from candidate memories (across activated cortices),
select a fixed-size set to enter the agent's context. This is NOT top-k ranking —
it's a competitive, multi-dimensional selection with lateral inhibition.

Key properties:
- Output size is FIXED regardless of how many candidates exist
- Similar memories compete (lateral inhibition) — prevents redundancy
- Multi-dimensional scoring: importance, recency, emotion, semantics, causality, focus
- Biological analogy: prefrontal cortex selecting which items enter working memory
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor
from .scheduler import AgentContext, MemoryItem, MemoryType


@dataclass
class GateScore:
    """Decomposed gating score — each dimension visible for explainability."""
    importance: float = 0.0
    recency: float = 0.0
    emotional_salience: float = 0.0
    semantic_match: float = 0.0
    causal_relevance: float = 0.0
    user_focus: float = 0.0
    novelty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "importance": round(self.importance, 3),
            "recency": round(self.recency, 3),
            "emotional": round(self.emotional_salience, 3),
            "semantic": round(self.semantic_match, 3),
            "causal": round(self.causal_relevance, 3),
            "focus": round(self.user_focus, 3),
            "novelty": round(self.novelty, 3),
            "total": round(self.total, 3),
        }


class MemoryGate:
    """Competitive memory gating with lateral inhibition.

    Only k memories enter the agent's context. The gate score is a weighted
    sum of 7 dimensions, and similar memories suppress each other via
    lateral inhibition (winner-take-all within each similarity cluster).

    Usage:
        gate = MemoryGate(k=20)
        selected = gate.gate(candidates, context, query_emb)
        # selected always has exactly min(k, len(candidates)) items
    """

    def __init__(self, k: int = 20,
                 lateral_inhibition_radius: float = 0.7,
                 inhibition_strength: float = 0.3,
                 config=None):
        self.k = k
        self.lateral_inhibition_radius = lateral_inhibition_radius
        self.inhibition_strength = inhibition_strength

        # Read dimension weights from config, falling back to defaults
        gc = getattr(config, 'gate', None) if config else None
        self.w_importance = getattr(gc, 'w_importance', 0.20) if gc else 0.20
        self.w_recency = getattr(gc, 'w_recency', 0.15) if gc else 0.15
        self.w_emotional = getattr(gc, 'w_emotional', 0.10) if gc else 0.10
        self.w_semantic = getattr(gc, 'w_semantic', 0.25) if gc else 0.25
        self.w_causal = getattr(gc, 'w_causal', 0.10) if gc else 0.10
        self.w_focus = getattr(gc, 'w_focus', 0.15) if gc else 0.15
        self.w_novelty = getattr(gc, 'w_novelty', 0.05) if gc else 0.05

    def gate(self, items: list[MemoryItem],
             context: AgentContext,
             query_embedding: list[float] | None = None,
             query_text: str = "") -> list[MemoryItem]:
        """Select top-k memories via competitive gating.

        1. Score each candidate on 7 dimensions
        2. Apply lateral inhibition (similar memories suppress each other)
        3. Winner-take-all: return top-k

        The returned list always has exactly min(k, len(items)) items.
        """
        if not items:
            return []

        import time
        now = time.time()

        # ── Step 1: Multi-dimensional scoring ────────────
        scored: list[tuple[MemoryItem, GateScore]] = []
        for item in items:
            anchor = item.anchor
            v = anchor.vector

            # Importance: retention_score + importance bonus
            importance = anchor.retention_score

            # Recency: exponential decay from last access
            hours_since = (now - anchor.last_activated_at) / 3600
            recency = math.exp(-hours_since / 168)  # 1 week half-life

            # Emotional salience: absolute valence (negative or positive = memorable)
            emotional = abs(v.emotional_valence)

            # Semantic match: from item's existing relevance score
            semantic = item.relevance_score

            # Causal relevance: boosted for items connected by causal edges
            causal = 0.0
            if item.reasoning_path and len(item.reasoning_path) > 1:
                # Items reached via multi-hop traversal get causal bonus
                causal = 0.3 * (len(item.reasoning_path) - 1) * 0.5

            # User focus: alignment with active goals
            focus = 0.0
            if context.active_goals:
                anchor_words = set(anchor.text.lower().split())
                for goal in context.active_goals:
                    goal_words = set(goal.lower().split())
                    overlap = len(anchor_words & goal_words)
                    if overlap > 0:
                        focus = max(focus, overlap / max(1, len(goal_words)))
                focus = min(1.0, focus)

            # Novelty: how surprising/unexpected this memory is
            novelty = v.surprise

            total = (
                self.w_importance * importance
                + self.w_recency * recency
                + self.w_emotional * emotional
                + self.w_semantic * semantic
                + self.w_causal * causal
                + self.w_focus * focus
                + self.w_novelty * novelty
            )

            gate_score = GateScore(
                importance=importance,
                recency=recency,
                emotional_salience=emotional,
                semantic_match=semantic,
                causal_relevance=causal,
                user_focus=focus,
                novelty=novelty,
                total=total,
            )
            scored.append((item, gate_score))

        # ── Step 2: Lateral inhibition ───────────────────
        # Similar memories suppress each other: within each similarity
        # cluster, only the strongest survives at full strength

        scored.sort(key=lambda x: -x[1].total)
        inhibited: list[tuple[MemoryItem, GateScore]] = []

        for item, score in scored:
            # Check if this item is suppressed by any previously selected item
            suppressed = False
            penalty = 0.0
            for selected_item, _ in inhibited:
                sim = self._memory_similarity(item, selected_item)
                if sim > self.lateral_inhibition_radius:
                    penalty = max(penalty, self.inhibition_strength * sim)
                    if penalty > 0.5:  # strong suppression
                        suppressed = True
                        break

            if not suppressed:
                if penalty > 0:
                    score.total *= (1.0 - penalty)
                inhibited.append((item, score))

        # ── Step 3: Winner-take-all ──────────────────────
        inhibited.sort(key=lambda x: -x[1].total)
        selected = inhibited[:self.k]

        # Update item relevance scores to reflect gate scores
        for item, gate_score in selected:
            item.relevance_score = gate_score.total
            # Attach gate score breakdown for explainability
            item.anchor._gate_score = gate_score

        return [item for item, _ in selected]

    def _memory_similarity(self, a: MemoryItem, b: MemoryItem) -> float:
        """Compute similarity between two memories for lateral inhibition."""
        # Embedding similarity
        if a.anchor.embedding and b.anchor.embedding:
            emb_sim = _cosine_sim(a.anchor.embedding, b.anchor.embedding)
        else:
            emb_sim = 0.0

        # Tag overlap
        a_tags = set(a.anchor.tags)
        b_tags = set(b.anchor.tags)
        if a_tags and b_tags:
            tag_sim = len(a_tags & b_tags) / len(a_tags | b_tags)
        else:
            tag_sim = 0.0

        return 0.7 * emb_sim + 0.3 * tag_sim


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
