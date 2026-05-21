"""Long-Term Stability Control — decay, archive, delete, and drift monitoring."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor

_PROTECTED_TAGS = frozenset({"protected", "pinned", "core"})
_CORE_IDENTITY_HINTS = frozenset({"identity", "core_identity", "core_self", "persona"})


@dataclass
class StabilityConfig:
    """Configuration for long-term stability control."""

    decay_curve: str = "exponential"       # exponential | linear | custom
    exponential_lambda: float = 0.01       # lambda for exponential decay
    linear_rate: float = 0.001             # per-day linear decay
    core_identity_exempt: bool = True      # never decay core identity
    protected_tags_exempt: bool = True     # tags: protected, pinned, core
    importance_threshold_exempt: float = 0.9  # don't decay very important anchors
    decay_report_interval: int = 100       # generate stability report every N decays
    max_decay_per_cycle: float = 0.3       # max fraction of weight to decay in one cycle
    archive_threshold: float = 0.1         # weight below this = archive to cold
    delete_threshold: float = 0.01         # weight below this = delete (with ghost)


@dataclass
class StabilityScore:
    """Per-anchor stability metrics spanning importance, recency, and emotional salience."""

    anchor_id: str
    importance_score: float       # content significance 0..1
    recency_score: float          # temporal freshness 0..1
    repetition_score: float       # access_count normalized 0..1
    emotional_weight: float       # abs(emotional_valence) 0..1
    task_relevance: float         # how often task-relevant 0..1
    composite_weight: float       # combined 0..1 score
    decay_resistance: float       # 0..1 resistance to decay
    archive_candidate: bool       # below archive_threshold
    delete_candidate: bool        # below delete_threshold


class StabilityController:
    """Long-term stability controller — decay, archive, delete, and drift tracking."""

    def __init__(self, graph, config: StabilityConfig | None = None):
        self.graph = graph
        self.config = config or StabilityConfig()
        self._decay_count: int = 0
        self._decayed_this_cycle: int = 0
        self._archived_this_cycle: int = 0
        self._deleted_this_cycle: int = 0
        self._weight_deltas: list[float] = []
        self._max_delta_history: int = 200

    # ── Scoring ──────────────────────────────────────

    def compute_score(self, anchor_id: str) -> StabilityScore:
        """Compute full stability score for one anchor."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return StabilityScore(
                anchor_id=anchor_id, importance_score=0.0, recency_score=0.0,
                repetition_score=0.0, emotional_weight=0.0, task_relevance=0.0,
                composite_weight=0.0, decay_resistance=0.0,
                archive_candidate=True, delete_candidate=True,
            )

        v = anchor.vector
        importance = anchor.importance_score
        recency = v.recency
        repetition = v.frequency
        emotional = abs(v.emotional_valence)
        task_rel = v.frequency * 0.6 + v.recency * 0.4

        composite = (
            0.35 * importance
            + 0.25 * recency
            + 0.20 * repetition
            + 0.10 * emotional
            + 0.10 * task_rel
        )

        decay_resistance = min(1.0, (
            0.4 * importance
            + 0.3 * repetition
            + 0.2 * emotional
            + 0.1 * task_rel
        ))

        return StabilityScore(
            anchor_id=anchor_id,
            importance_score=importance,
            recency_score=recency,
            repetition_score=repetition,
            emotional_weight=emotional,
            task_relevance=task_rel,
            composite_weight=max(0.0, min(1.0, composite)),
            decay_resistance=decay_resistance,
            archive_candidate=composite < self.config.archive_threshold,
            delete_candidate=composite < self.config.delete_threshold,
        )

    def compute_all_scores(self) -> dict[str, StabilityScore]:
        """Compute stability scores for every anchor in the graph."""
        return {aid: self.compute_score(aid) for aid in self.graph.anchors}

    # ── Exemption ────────────────────────────────────

    def is_exempt(self, anchor: Anchor) -> bool:
        """Check if anchor is exempt from decay (core identity, protected tags, high importance)."""
        cfg = self.config

        if cfg.core_identity_exempt:
            tag_set = {t.lower() for t in anchor.tags}
            if tag_set & _CORE_IDENTITY_HINTS:
                return True
            if anchor.cortex_path and any(
                hint in anchor.cortex_path.lower() for hint in _CORE_IDENTITY_HINTS
            ):
                return True

        if cfg.protected_tags_exempt:
            tag_set = {t.lower() for t in anchor.tags}
            if tag_set & _PROTECTED_TAGS:
                return True

        if cfg.importance_threshold_exempt > 0:
            if anchor.importance_score >= cfg.importance_threshold_exempt:
                return True

        return False

    # ── Decay ────────────────────────────────────────

    def decay_weight(self, score: StabilityScore, days_since_creation: float) -> float:
        """Compute new weight after applying temporal decay with resistance boost."""
        cfg = self.config
        weight_old = score.composite_weight

        if cfg.decay_curve == "linear":
            weight_raw = weight_old - (cfg.linear_rate * days_since_creation)
        else:  # exponential (default) or custom
            weight_raw = weight_old * math.exp(-cfg.exponential_lambda * days_since_creation)

        base_loss = weight_old - weight_raw
        effective_loss = base_loss * (1.0 - score.decay_resistance)
        max_loss = weight_old * cfg.max_decay_per_cycle
        effective_loss = min(effective_loss, max_loss)
        return max(0.0, min(1.0, weight_old - effective_loss))

    def _decay_anchor(self, anchor_id: str) -> tuple[bool, bool, bool]:
        """Apply decay to a single anchor. Returns (decayed, archived, deleted)."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return (False, False, False)

        if self.is_exempt(anchor):
            return (False, False, False)

        score = self.compute_score(anchor_id)
        days = (time.time() - anchor.created_at) / 86400.0
        new_weight = self.decay_weight(score, days)

        if new_weight >= score.composite_weight:
            return (False, False, False)

        ratio = new_weight / max(score.composite_weight, 0.001)
        anchor.vector.recency = max(0.01, anchor.vector.recency * ratio)
        anchor._ret_cached = -1.0

        self._decay_count += 1
        delta = score.composite_weight - new_weight
        self._weight_deltas.append(delta)
        if len(self._weight_deltas) > self._max_delta_history:
            self._weight_deltas = self._weight_deltas[-self._max_delta_history:]

        archived = False
        deleted = False

        if new_weight < self.config.delete_threshold:
            deleted = self._delete_anchor(anchor_id, create_ghost=True)
        elif new_weight < self.config.archive_threshold:
            archived = self._archive_anchor(anchor_id)

        return (True, archived, deleted)

    def apply_decay(self, anchor_id: str = "", all: bool = False) -> dict:
        """Apply decay to one or all anchors. Returns {decayed, archived, deleted} counts."""
        cfg = self.config
        decayed = 0
        archived = 0
        deleted = 0

        if anchor_id:
            d, a, dl = self._decay_anchor(anchor_id)
            decayed += d
            archived += a
            deleted += dl
        elif all:
            for aid in list(self.graph.anchors.keys()):
                d, a, dl = self._decay_anchor(aid)
                decayed += d
                archived += a
                deleted += dl

            if self._decay_count > 0 and self._decay_count % cfg.decay_report_interval == 0:
                self.generate_report()

        self._decayed_this_cycle = decayed
        self._archived_this_cycle += archived
        self._deleted_this_cycle += deleted

        return {"decayed": decayed, "archived": archived, "deleted": deleted}

    # ── Archive & Delete ─────────────────────────────

    def _archive_anchor(self, anchor_id: str) -> bool:
        """Move a single anchor to cold tier. Returns True on success."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return False

        from .memory_core.anchor import MemoryState
        anchor.transition("consolidate")
        anchor.transition("stabilize")
        anchor.vector.stability = min(1.0, anchor.vector.stability + 0.3)
        return True

    def auto_archive(self) -> list[str]:
        """Move all anchors below archive_threshold to cold tier. Returns archived IDs."""
        archived = []
        for aid in list(self.graph.anchors.keys()):
            anchor = self.graph.anchors.get(aid)
            if anchor is None or self.is_exempt(anchor):
                continue
            score = self.compute_score(aid)
            if score.composite_weight < self.config.archive_threshold:
                if self._archive_anchor(aid):
                    archived.append(aid)
        self._archived_this_cycle += len(archived)
        return archived

    def _delete_anchor(self, anchor_id: str, create_ghost: bool = True) -> bool:
        """Remove a single anchor from the graph, optionally creating a ghost trace."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return False

        if create_ghost:
            try:
                residual_edges: dict[str, float] = {}
                for neighbor in self.graph._adjacency.get(anchor_id, set()):
                    key = self.graph._key(anchor_id, neighbor)
                    edge = self.graph.edges.get(key)
                    if edge:
                        residual_edges[neighbor] = edge.weight * 0.3
                self.graph._ghost_subsystem.create(anchor, residual_edges)
            except Exception:
                pass

        self.graph.remove_anchor(anchor_id)
        return True

    def auto_delete(self, create_ghost: bool = True) -> list[str]:
        """Remove all anchors below delete_threshold. Returns deleted IDs."""
        deleted: list[str] = []
        for aid in list(self.graph.anchors.keys()):
            anchor = self.graph.anchors.get(aid)
            if anchor is None or self.is_exempt(anchor):
                continue
            score = self.compute_score(aid)
            if score.composite_weight < self.config.delete_threshold:
                if self._delete_anchor(aid, create_ghost=create_ghost):
                    deleted.append(aid)
        self._deleted_this_cycle += len(deleted)
        return deleted

    # ── Drift & Reporting ────────────────────────────

    def drift_score(self) -> float:
        """Measure personality/memory drift (0 = stable, 1 = major shift)."""
        total = len(self.graph.anchors) + self._archived_this_cycle + self._deleted_this_cycle
        if total == 0:
            return 0.0

        churn = (self._archived_this_cycle + self._deleted_this_cycle) / total
        avg_delta = (
            sum(self._weight_deltas) / len(self._weight_deltas)
            if self._weight_deltas else 0.0
        )
        return max(0.0, min(1.0, churn + avg_delta))

    def generate_report(self) -> dict:
        """Generate a stability report with drift metrics and recommendations."""
        scores = self.compute_all_scores()
        total = len(self.graph.anchors)
        avg_weight = (
            sum(s.composite_weight for s in scores.values()) / max(total, 1)
        )

        exempt_count = sum(
            1 for a in self.graph.anchors.values() if self.is_exempt(a)
        )

        topic_weights: dict[str, list[float]] = defaultdict(list)
        for score in scores.values():
            anchor = self.graph.anchors.get(score.anchor_id)
            if anchor:
                for tag in anchor.tags:
                    topic_weights[tag.lower()].append(score.composite_weight)

        topic_avgs = {
            t: sum(w) / len(w) for t, w in topic_weights.items()
        }
        sorted_topics = sorted(topic_avgs.items(), key=lambda x: x[1], reverse=True)
        most_stable = [t for t, _ in sorted_topics[:5]]
        least_stable = [t for t, _ in sorted_topics[-5:]]
        least_stable.reverse()

        drift = self.drift_score()
        if drift < 0.1:
            recommendation = "stable"
        elif drift < 0.3:
            recommendation = "minor_drift"
        elif drift < 0.5:
            recommendation = "significant_drift"
        else:
            recommendation = "critical"

        return {
            "total_anchors": total,
            "decayed_this_cycle": self._decayed_this_cycle,
            "archived_this_cycle": self._archived_this_cycle,
            "deleted_this_cycle": self._deleted_this_cycle,
            "exempt_count": exempt_count,
            "average_weight": round(avg_weight, 3),
            "drift_score": round(drift, 3),
            "most_stable_topics": most_stable,
            "least_stable_topics": least_stable,
            "recommendation": recommendation,
        }
