"""Cognitive Priority Layer — assigns priority levels to memory anchors for budget-aware injection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph


class PriorityLevel(Enum):
    ACTIVE_GOAL = 1
    LONG_TERM_GOAL = 2
    CORE_IDENTITY = 3
    FREQUENT_KNOWLEDGE = 4
    GENERAL_EVENT = 5


DEFAULT_CONFIG: dict[str, Any] = {
    "active_goal_ttl_seconds": 3600,
    "long_term_goal_ttl_seconds": 604800,
    "core_identity_protected": True,
    "core_identity_forced_inject": True,
    "frequent_knowledge_threshold_access": 10,
}

_ACTIVE_GOAL_TAGS = frozenset({"active_goal", "current_task", "goal", "objective"})
_LONG_TERM_TAGS = frozenset({"long_term_goal", "project", "milestone", "ambition"})
_CORE_IDENTITY_TAGS = frozenset({"identity", "personality", "core_value", "belief", "value"})
_KNOWLEDGE_TAGS = frozenset({"knowledge", "skill", "expertise", "frequent", "preference"})


@dataclass
class CognitivePriority:
    """Assigned priority for a single memory anchor."""
    anchor_id: str
    level: PriorityLevel
    priority_score: float
    is_protected: bool
    is_forced_inject: bool


class PriorityEngine:
    """Computes and caches cognitive priority assignments for graph anchors."""

    def __init__(self, graph: StarGraph, config: dict[str, Any] | None = None) -> None:
        self._graph = graph
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self._config = cfg
        self._cache: dict[str, CognitivePriority] = {}

    # ── Priority inference ────────────────────────────

    def assign_priority(self, anchor_id: str) -> CognitivePriority:
        """Compute priority for one anchor using rule-based inference."""
        anchor = self._graph.anchors.get(anchor_id)
        if anchor is None:
            cp = CognitivePriority(
                anchor_id=anchor_id,
                level=PriorityLevel.GENERAL_EVENT,
                priority_score=0.0,
                is_protected=False,
                is_forced_inject=False,
            )
            self._cache[anchor_id] = cp
            return cp

        tags_lower = {t.lower() for t in anchor.tags}
        v = anchor.vector
        importance = v.importance
        valence = v.emotional_valence
        now = time.time()
        age_seconds = now - anchor.created_at

        ttl_active = self._config["active_goal_ttl_seconds"]
        threshold_access = self._config["frequent_knowledge_threshold_access"]

        # ── ACTIVE_GOAL ──
        if (
            tags_lower & _ACTIVE_GOAL_TAGS
            or (importance > 0.8 and age_seconds <= ttl_active)
        ):
            score = _clamp(importance * (1.0 - min(1.0, age_seconds / ttl_active)), 0.01, 1.0)
            cp = CognitivePriority(
                anchor_id=anchor_id,
                level=PriorityLevel.ACTIVE_GOAL,
                priority_score=score,
                is_protected=False,
                is_forced_inject=score > 0.9,
            )
            self._cache[anchor_id] = cp
            return cp

        # ── LONG_TERM_GOAL ──
        if (
            tags_lower & _LONG_TERM_TAGS
            or (importance > 0.7 and "goal" in tags_lower)
            or (valence > 0.5 and importance > 0.6)
        ):
            score = _clamp(importance * 0.7 + max(0.0, valence) * 0.3, 0.01, 1.0)
            cp = CognitivePriority(
                anchor_id=anchor_id,
                level=PriorityLevel.LONG_TERM_GOAL,
                priority_score=score,
                is_protected=False,
                is_forced_inject=score > 0.9,
            )
            self._cache[anchor_id] = cp
            return cp

        # ── CORE_IDENTITY ──
        if (
            tags_lower & _CORE_IDENTITY_TAGS
            or importance > 0.9
            or (abs(valence) > 0.7 and importance > 0.8)
        ):
            score = _clamp(importance * 0.6 + abs(valence) * 0.4, 0.7, 1.0)
            cp = CognitivePriority(
                anchor_id=anchor_id,
                level=PriorityLevel.CORE_IDENTITY,
                priority_score=score,
                is_protected=self._config["core_identity_protected"],
                is_forced_inject=self._config["core_identity_forced_inject"],
            )
            self._cache[anchor_id] = cp
            return cp

        # ── FREQUENT_KNOWLEDGE ──
        access_count = getattr(anchor, "replay_count", 0)
        if access_count > threshold_access or tags_lower & _KNOWLEDGE_TAGS:
            score = _clamp(access_count / max(1, threshold_access * 5) * 0.8 + importance * 0.2, 0.01, 1.0)
            cp = CognitivePriority(
                anchor_id=anchor_id,
                level=PriorityLevel.FREQUENT_KNOWLEDGE,
                priority_score=score,
                is_protected=False,
                is_forced_inject=False,
            )
            self._cache[anchor_id] = cp
            return cp

        # ── GENERAL_EVENT (default) ──
        age_hours = age_seconds / 3600.0
        recency_factor = max(0.01, 1.0 - min(1.0, age_hours / 720))
        score = _clamp(anchor.retention_score * 0.6 + recency_factor * 0.4, 0.0, 1.0)
        cp = CognitivePriority(
            anchor_id=anchor_id,
            level=PriorityLevel.GENERAL_EVENT,
            priority_score=score,
            is_protected=False,
            is_forced_inject=False,
        )
        self._cache[anchor_id] = cp
        return cp

    # ── Bulk operations ───────────────────────────────

    def assign_all(self) -> dict[str, CognitivePriority]:
        """Compute priorities for all anchors in the graph."""
        self._cache.clear()
        for anchor_id in self._graph.anchors:
            self.assign_priority(anchor_id)
        return dict(self._cache)

    # ── Lookup ────────────────────────────────────────

    def get_priority(self, anchor_id: str) -> CognitivePriority:
        """Return cached priority, computing it if missing."""
        if anchor_id not in self._cache:
            return self.assign_priority(anchor_id)
        return self._cache[anchor_id]

    def get_forced_injections(self) -> list[str]:
        """Return all anchor IDs marked is_forced_inject."""
        return [
            aid for aid, cp in self._cache.items()
            if cp.is_forced_inject
        ]

    # ── Sorting / filtering ───────────────────────────

    def _resolve_id(self, item: Any) -> str:
        """Extract anchor ID from an item (string, Anchor, or object with .id)."""
        if isinstance(item, str):
            return item
        if hasattr(item, "id"):
            return item.id
        if hasattr(item, "anchor_id"):
            return item.anchor_id
        return str(item)

    def sort_by_priority(self, items: list[Any]) -> list[Any]:
        """Sort items by cognitive priority, highest first."""
        def _sort_key(item: Any) -> tuple[int, float]:
            aid = self._resolve_id(item)
            cp = self.get_priority(aid)
            return (cp.level.value, -cp.priority_score)
        return sorted(items, key=_sort_key)

    def budget_filter(self, items: list[Any], max_items: int) -> list[Any]:
        """Take top-N by priority, always including forced injections."""
        if max_items <= 0:
            return []
        sorted_items = self.sort_by_priority(items)
        result: list[Any] = []
        seen: set[str] = set()
        for item in sorted_items:
            aid = self._resolve_id(item)
            if aid in seen:
                continue
            cp = self.get_priority(aid)
            if cp.is_forced_inject:
                seen.add(aid)
                result.append(item)
        for item in sorted_items:
            aid = self._resolve_id(item)
            if aid in seen:
                continue
            seen.add(aid)
            result.append(item)
            if len(result) >= max_items:
                break
        return result


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
