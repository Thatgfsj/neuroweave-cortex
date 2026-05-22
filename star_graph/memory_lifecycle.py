"""Memory Lifecycle — unified full-lifecycle management for memory anchors.

Consolidates: tier system (Hot/Warm/Cold), 4-layer pyramid, stability_control,
thermal_store, ghost subsystem, and memory_budget eviction.

Lifecycle stages:
  PERCEPTION → WORKING → SHORT_TERM → LONG_TERM → CONSOLIDATED → ARCHIVED → DORMANT → GHOST → DEAD
"""

from __future__ import annotations

import enum
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


class LifecycleStage(enum.Enum):
    PERCEPTION = "perception"          # initial intake (not yet stored)
    WORKING = "working"                # in cognitive workspace
    SHORT_TERM = "short_term"          # stored, not consolidated
    LONG_TERM = "long_term"            # consolidated, actively retrievable
    CONSOLIDATED = "consolidated"      # deeply integrated, high stability
    ARCHIVED = "archived"              # low access, in cold storage
    DORMANT = "dormant"               # very low access, near ghost
    GHOST = "ghost"                    # forgotten but potentially revivable
    DEAD = "dead"                      # permanently removed


@dataclass
class LifecycleTransition:
    """Record of a lifecycle stage transition."""
    anchor_id: str
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleState:
    """Current lifecycle state of a memory anchor."""
    anchor_id: str
    current_stage: LifecycleStage = LifecycleStage.SHORT_TERM
    entered_at: float = field(default_factory=time.time)
    time_in_stage: float = 0.0
    access_count_in_stage: int = 0
    demotion_risk: float = 0.0           # 0..1 probability of demotion soon


# ── Default stage policies ──────────────────────────────────────

_DEFAULT_STAGE_POLICIES: dict[str, dict] = {
    "working": {
        "max_items": 200, "ttl_seconds": 3600,
        "promote_to": "short_term",
        "promote_condition": "importance > 0.3",
    },
    "short_term": {
        "max_items": 3000, "ttl_seconds": 604800,  # 7 days
        "promote_to": "long_term",
        "promote_condition": "access_count >= 3 OR stability > 0.5",
        "demote_to": "archived",
        "demote_condition": "idle_seconds > 259200 AND access_count < 2",  # 3 days
    },
    "long_term": {
        "max_items": 30000, "ttl_seconds": 0,  # infinite
        "promote_to": "consolidated",
        "promote_condition": "stability > 0.7 AND access_count >= 10",
        "demote_to": "archived",
        "demote_condition": "idle_seconds > 2592000",  # 30 days
    },
    "consolidated": {
        "max_items": 15000, "ttl_seconds": 0,
        "demote_to": "archived",
        "demote_condition": "idle_seconds > 7776000",  # 90 days
    },
    "archived": {
        "max_items": 50000, "ttl_seconds": 0,
        "demote_to": "dormant",
        "demote_condition": "idle_seconds > 15552000",  # 180 days
    },
    "dormant": {
        "max_items": 0,  # unlimited
        "demote_to": "ghost",
        "demote_condition": "retention < 0.05",
    },
    "ghost": {
        "max_items": 0,
        "purge_after_days": 90,
    },
}


class MemoryLifecycleManager:
    """Unified lifecycle management for all memory anchors."""

    def __init__(self, graph=None, *,
                 config: dict[str, Any] | None = None,
                 stability_controller=None,
                 memory_budget=None,
                 thermal_store=None,
                 ghost_subsystem=None):
        self.graph = graph
        self._config = config or {}
        self._stability_controller = stability_controller
        self._memory_budget = memory_budget
        self._thermal_store = thermal_store
        self._ghost_subsystem = ghost_subsystem

        # Stage policies
        cfg_policies = self._config.get("stage_policies", {})
        self._policies: dict[str, dict] = {}
        for stage_name, defaults in _DEFAULT_STAGE_POLICIES.items():
            self._policies[stage_name] = {**defaults, **cfg_policies.get(stage_name, {})}

        self._transitions: list[LifecycleTransition] = []
        self._state_cache: dict[str, LifecycleState] = {}

        self._auto_interval = self._config.get("auto_transition_interval_hours", 6.0)
        self._max_transitions = self._config.get("max_transitions_per_cycle", 1000)

    # ── Stage Management ───────────────────────────────────

    def classify_anchor(self, anchor_id: str, anchor_obj=None) -> LifecycleStage:
        """Determine which lifecycle stage an anchor belongs in."""
        if not anchor_obj and self.graph:
            try:
                anchor_obj = self.graph.get_anchor(anchor_id)
            except Exception:
                pass

        if not anchor_obj:
            return LifecycleStage.SHORT_TERM

        # Use anchor attributes to determine stage
        stability = getattr(anchor_obj, 'stability', 0.5)
        access_count = getattr(anchor_obj, 'access_count', 0)
        importance = getattr(anchor_obj, 'importance', 0.5)
        retention = getattr(anchor_obj, 'retention_score', 0.5)
        state = getattr(anchor_obj, 'state', None)

        # Check ghost state first
        if state and hasattr(state, 'value') and 'ghost' in str(state.value).lower():
            return LifecycleStage.GHOST

        if retention < 0.05:
            return LifecycleStage.DORMANT
        if retention < 0.15:
            return LifecycleStage.ARCHIVED
        if stability > 0.7 and access_count >= 10:
            return LifecycleStage.CONSOLIDATED
        if stability > 0.5 or access_count >= 3:
            return LifecycleStage.LONG_TERM
        return LifecycleStage.SHORT_TERM

    def transition(self, anchor_id: str, to_stage: LifecycleStage,
                   reason: str = "") -> LifecycleTransition | None:
        """Transition an anchor to a new lifecycle stage."""
        current = self._state_cache.get(anchor_id)
        from_stage = current.current_stage if current else self.classify_anchor(anchor_id)

        if from_stage == to_stage:
            return None

        trans = LifecycleTransition(
            anchor_id=anchor_id,
            from_stage=from_stage,
            to_stage=to_stage,
            reason=reason,
        )
        self._transitions.append(trans)

        # Update cache
        self._state_cache[anchor_id] = LifecycleState(
            anchor_id=anchor_id,
            current_stage=to_stage,
        )

        return trans

    def auto_transition_all(self) -> list[LifecycleTransition]:
        """Scan all anchors and auto-transition based on policy rules."""
        transitions: list[LifecycleTransition] = []

        # Check promotions
        promotions = self._check_promotions()
        transitions.extend(promotions)

        # Check demotions
        if len(transitions) < self._max_transitions:
            demotions = self._check_demotions()
            remaining = self._max_transitions - len(transitions)
            transitions.extend(demotions[:remaining])

        return transitions

    # ── Promotion / Demotion ───────────────────────────────

    def _check_promotions(self) -> list[LifecycleTransition]:
        """Find anchors eligible for promotion."""
        transitions: list[LifecycleTransition] = []

        for anchor_id, state in list(self._state_cache.items()):
            stage_name = state.current_stage.value
            policy = self._policies.get(stage_name, {})
            promote_to = policy.get("promote_to")
            if not promote_to:
                continue

            condition = policy.get("promote_condition", "")
            if self._evaluate_condition(condition, anchor_id, state):
                try:
                    to_stage = LifecycleStage(promote_to)
                    transitions.append(self.transition(
                        anchor_id, to_stage,
                        reason=f"auto-promote: {condition}"
                    ))
                except ValueError:
                    pass

        return transitions

    def _check_demotions(self) -> list[LifecycleTransition]:
        """Find anchors that should be demoted."""
        transitions: list[LifecycleTransition] = []

        for anchor_id, state in list(self._state_cache.items()):
            stage_name = state.current_stage.value
            policy = self._policies.get(stage_name, {})
            demote_to = policy.get("demote_to")
            if not demote_to:
                continue

            condition = policy.get("demote_condition", "")
            if self._evaluate_condition(condition, anchor_id, state):
                try:
                    to_stage = LifecycleStage(demote_to)
                    transitions.append(self.transition(
                        anchor_id, to_stage,
                        reason=f"auto-demote: {condition}"
                    ))
                except ValueError:
                    pass

        return transitions

    def _evaluate_condition(self, condition: str, anchor_id: str,
                            state: LifecycleState) -> bool:
        """Evaluate a promotion/demotion condition string."""
        if not condition:
            return False

        # Simple condition evaluator
        try:
            # Handle "importance > 0.3"
            if "importance >" in condition:
                threshold = float(condition.split(">")[-1].strip())
                anchor = self._get_anchor(anchor_id)
                return getattr(anchor, 'importance', 0.5) > threshold

            if "stability >" in condition:
                threshold = float(condition.split(">")[-1].strip())
                anchor = self._get_anchor(anchor_id)
                return getattr(anchor, 'stability', 0.5) > threshold

            if "access_count >=" in condition:
                threshold = int(condition.split(">=")[-1].strip())
                return state.access_count_in_stage >= threshold

            if "idle_seconds >" in condition:
                threshold = float(condition.split(">")[-1].strip())
                idle = time.time() - state.entered_at
                return idle > threshold

            if "retention <" in condition:
                threshold = float(condition.split("<")[-1].strip())
                anchor = self._get_anchor(anchor_id)
                return getattr(anchor, 'retention_score', 0.5) < threshold
        except (ValueError, IndexError):
            pass

        return False

    def _get_anchor(self, anchor_id: str) -> Any:
        """Get anchor object."""
        if self.graph and hasattr(self.graph, 'get_anchor'):
            try:
                return self.graph.get_anchor(anchor_id)
            except Exception:
                pass
        return None

    # ── End of Life ────────────────────────────────────────

    def archive(self, anchor_id: str):
        self.transition(anchor_id, LifecycleStage.ARCHIVED, "explicit archive")

    def ghost(self, anchor_id: str):
        self.transition(anchor_id, LifecycleStage.GHOST, "explicit forget")

    def purge(self, anchor_id: str):
        """Permanently delete."""
        self.transition(anchor_id, LifecycleStage.DEAD, "explicit purge")
        self._state_cache.pop(anchor_id, None)

    def revive(self, anchor_id: str):
        """Revive from ghost/dormant back to short_term."""
        if anchor_id in self._state_cache:
            stage = self._state_cache[anchor_id].current_stage
            if stage in (LifecycleStage.GHOST, LifecycleStage.DORMANT, LifecycleStage.ARCHIVED):
                self.transition(anchor_id, LifecycleStage.SHORT_TERM, "revival")

    # ── Lifecycle Mapping ──────────────────────────────────

    @staticmethod
    def map_to_tier(stage: LifecycleStage) -> str:
        """Map lifecycle stage to existing tier system (Hot/Warm/Cold)."""
        mapping = {
            LifecycleStage.WORKING: "hot",
            LifecycleStage.SHORT_TERM: "hot",
            LifecycleStage.LONG_TERM: "warm",
            LifecycleStage.CONSOLIDATED: "warm",
            LifecycleStage.ARCHIVED: "cold",
            LifecycleStage.DORMANT: "cold",
            LifecycleStage.GHOST: "cold",
            LifecycleStage.DEAD: "dead",
        }
        return mapping.get(stage, "warm")

    @staticmethod
    def map_to_layer(stage: LifecycleStage) -> str:
        """Map lifecycle stage to 4-layer pyramid."""
        mapping = {
            LifecycleStage.PERCEPTION: "working",
            LifecycleStage.WORKING: "working",
            LifecycleStage.SHORT_TERM: "episodic",
            LifecycleStage.LONG_TERM: "episodic",
            LifecycleStage.CONSOLIDATED: "semantic",
            LifecycleStage.ARCHIVED: "semantic",
            LifecycleStage.DORMANT: "core_identity",
            LifecycleStage.GHOST: "core_identity",
            LifecycleStage.DEAD: "dead",
        }
        return mapping.get(stage, "episodic")

    # ── Stats ──────────────────────────────────────────────

    def get_stage_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for state in self._state_cache.values():
            counts[state.current_stage.value] += 1
        return dict(counts)

    def get_lifecycle_report(self) -> dict:
        return {
            "stage_distribution": self.get_stage_distribution(),
            "total_transitions": len(self._transitions),
            "cached_anchors": len(self._state_cache),
            "recent_transitions": [
                {"anchor_id": t.anchor_id[:12],
                 "from": t.from_stage.value,
                 "to": t.to_stage.value,
                 "reason": t.reason}
                for t in self._transitions[-20:]
            ],
        }

    def get_health_score(self) -> float:
        """0..1 overall lifecycle health."""
        dist = self.get_stage_distribution()
        total = sum(dist.values()) or 1
        # Healthy: most in long_term + consolidated, less in archived/ghost
        healthy = dist.get('long_term', 0) + dist.get('consolidated', 0) + dist.get('short_term', 0)
        return min(1.0, healthy / total)

    def record_access(self, anchor_id: str):
        """Record that an anchor was accessed."""
        if anchor_id in self._state_cache:
            self._state_cache[anchor_id].access_count_in_stage += 1
