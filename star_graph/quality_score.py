"""Memory Quality Score system for the NeuroWeave Cortex cognitive memory runtime.

Provides per-anchor quality scoring across six dimensions (usage frequency,
reasoning contribution, user feedback, task hit rate, freshness, coherence)
and quality-driven cleanup operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .math_utils import clamp, cosine_sim

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph


DEFAULT_WEIGHTS: dict[str, float] = {
    "usage": 0.25,
    "reasoning": 0.25,
    "feedback": 0.15,
    "task_hit": 0.15,
    "freshness": 0.10,
    "coherence": 0.10,
}

_MAX_AGE_SECONDS: float = 30 * 24 * 3600
_EMA_ALPHA: float = 0.2


@dataclass
class MemoryQualityScore:
    """Composite quality metrics for a single memory anchor."""

    anchor_id: str
    usage_frequency: float = 0.0
    reasoning_contribution: float = 0.5
    user_feedback: float = 0.5
    task_hit_rate: float = 0.0
    freshness: float = 0.5
    coherence: float = 0.5
    overall: float = 0.5


class QualityScorer:
    """Computes and tracks per-anchor quality for pruning and retrieval ranking."""

    def __init__(self, graph: StarGraph, weights: dict | None = None) -> None:
        self._graph = graph
        w = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            w.update(weights)
        self._weights = w
        self._recall_counts: dict[str, int] = {}
        self._task_hits: dict[str, int] = {}
        self._feedback_ema: dict[str, float] = {}
        self._reasoning_ema: dict[str, float] = {}

    def _filter_by_layer(self, layer: str) -> dict[str, object]:
        """Return anchors whose tags contain the given layer, or all if layer is empty."""
        if not layer:
            return self._graph.anchors
        return {aid: a for aid, a in self._graph.anchors.items() if layer in a.tags}

    def score_anchor(self, anchor_id: str) -> MemoryQualityScore:
        """Compute the full six-factor quality score for a single anchor."""
        anchor = self._graph.anchors.get(anchor_id)
        if anchor is None:
            return MemoryQualityScore(anchor_id=anchor_id)

        max_recalls = max(self._recall_counts.values()) if self._recall_counts else 1
        usage_frequency = self._recall_counts.get(anchor_id, 0) / max(1, max_recalls)

        reasoning_contribution = self._reasoning_ema.get(anchor_id, 0.5)

        user_feedback = self._feedback_ema.get(anchor_id, 0.5)

        recalls = self._recall_counts.get(anchor_id, 0)
        hits = self._task_hits.get(anchor_id, 0)
        task_hit_rate = hits / recalls if recalls > 0 else 0.0

        now = time.time()
        age = max(0.0, now - anchor.created_at)
        freshness = max(0.0, 1.0 - age / _MAX_AGE_SECONDS)

        coherence = self.compute_coherence(anchor_id)

        overall = clamp(
            self._weights["usage"] * usage_frequency
            + self._weights["reasoning"] * reasoning_contribution
            + self._weights["feedback"] * user_feedback
            + self._weights["task_hit"] * task_hit_rate
            + self._weights["freshness"] * freshness
            + self._weights["coherence"] * coherence,
            0.0,
            1.0,
        )

        return MemoryQualityScore(
            anchor_id=anchor_id,
            usage_frequency=round(usage_frequency, 4),
            reasoning_contribution=round(reasoning_contribution, 4),
            user_feedback=round(user_feedback, 4),
            task_hit_rate=round(task_hit_rate, 4),
            freshness=round(freshness, 4),
            coherence=round(coherence, 4),
            overall=round(overall, 4),
        )

    def score_all(self, layer: str = "") -> dict[str, MemoryQualityScore]:
        """Score all anchors, optionally filtered by layer tag."""
        anchors = self._filter_by_layer(layer)
        return {aid: self.score_anchor(aid) for aid in anchors}

    def top_quality(self, n: int = 10, layer: str = "") -> list[MemoryQualityScore]:
        """Return highest-quality anchors sorted by overall score descending."""
        scores = self.score_all(layer=layer)
        return sorted(scores.values(), key=lambda s: -s.overall)[:n]

    def bottom_quality(self, n: int = 10, layer: str = "") -> list[MemoryQualityScore]:
        """Return lowest-quality anchors, candidates for cleanup."""
        scores = self.score_all(layer=layer)
        return sorted(scores.values(), key=lambda s: s.overall)[:n]

    def record_recall(self, anchor_id: str, task_relevant: bool = True) -> None:
        """Record that an anchor was retrieved; increments usage and task-hit counters."""
        self._recall_counts[anchor_id] = self._recall_counts.get(anchor_id, 0) + 1
        if task_relevant:
            self._task_hits[anchor_id] = self._task_hits.get(anchor_id, 0) + 1

    def record_feedback(self, anchor_id: str, positive: bool = True) -> None:
        """Record implicit or explicit user feedback for an anchor via EMA."""
        new_val = 1.0 if positive else 0.0
        old = self._feedback_ema.get(anchor_id, 0.5)
        self._feedback_ema[anchor_id] = old + _EMA_ALPHA * (new_val - old)

    def record_reasoning_contribution(self, anchor_id: str, contribution: float) -> None:
        """Record how much this anchor contributed to a reasoning step (0-1)."""
        val = clamp(contribution, 0.0, 1.0)
        old = self._reasoning_ema.get(anchor_id, 0.5)
        self._reasoning_ema[anchor_id] = old + _EMA_ALPHA * (val - old)

    def compute_coherence(self, anchor_id: str) -> float:
        """Measure embedding consistency with graph neighbors via average cosine similarity."""
        anchor = self._graph.anchors.get(anchor_id)
        if anchor is None or not anchor.embedding:
            return 0.5

        neighbors = self._graph._adjacency.get(anchor_id, set())
        neighbor_embs: list[list[float]] = []
        for nid in neighbors:
            nb = self._graph.anchors.get(nid)
            if nb is not None and nb.embedding:
                neighbor_embs.append(nb.embedding)

        if not neighbor_embs:
            return 0.5

        sims = [cosine_sim(anchor.embedding, emb) for emb in neighbor_embs]
        return clamp(sum(sims) / len(sims), 0.0, 1.0)

    def snapshot(self) -> dict:
        """Return quality distribution statistics across all scored anchors."""
        scores = self.score_all()
        if not scores:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

        overalls = sorted(s.overall for s in scores.values())
        n = len(overalls)
        mean = sum(overalls) / n
        median = (
            overalls[n // 2]
            if n % 2 == 1
            else (overalls[n // 2 - 1] + overalls[n // 2]) / 2
        )

        return {
            "count": n,
            "min": overalls[0],
            "max": overalls[-1],
            "mean": round(mean, 4),
            "median": round(median, 4),
            "q25": overalls[n // 4],
            "q75": overalls[3 * n // 4],
        }

    def prune_candidates(self, threshold: float = 0.2, max_count: int = 100) -> list[str]:
        """Return anchor IDs with overall quality below the given threshold."""
        scores = self.score_all()
        below = [(aid, s.overall) for aid, s in scores.items() if s.overall < threshold]
        below.sort(key=lambda x: x[1])
        return [aid for aid, _ in below[:max_count]]

    def auto_cleanup(self, target_layer: str = "episodic", max_remove: int = 50) -> list[str]:
        """Remove the lowest-quality anchors from the target layer. Returns removed IDs."""
        scores = self.score_all(layer=target_layer)
        sorted_ids = sorted(scores.keys(), key=lambda aid: scores[aid].overall)
        to_remove = sorted_ids[:max_remove]
        removed: list[str] = []
        for aid in to_remove:
            try:
                self._graph.remove_anchor(aid)
                removed.append(aid)
                self._recall_counts.pop(aid, None)
                self._task_hits.pop(aid, None)
                self._feedback_ema.pop(aid, None)
                self._reasoning_ema.pop(aid, None)
            except Exception:
                pass
        return removed
