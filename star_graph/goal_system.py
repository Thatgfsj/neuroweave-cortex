"""Goal System — goal hierarchy, conflict detection, goal-driven inference.

Goals are NOT just stored objectives. They actively shape what the system
pays attention to, what it remembers, and how it reasons.

Extends agent_state.py GoalNode with:
- Goal hierarchy (decomposition, progress tracking)
- Conflict detection (resource, priority, contradictory)
- Goal-driven inference (what information is needed to achieve a goal)
- Emotional drive weighting
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ── Data Structures ──────────────────────────────────────────────


@dataclass
class GoalFrame:
    """A goal in the cognitive goal system."""
    id: str
    description: str
    parent_id: str = ""
    status: str = "pending"            # pending | active | blocked | achieved | abandoned
    priority: float = 0.5              # 0..1
    goal_type: str = "explicit"        # explicit | implicit | inferred | emotional
    source: str = ""                   # user_stated | inferred | perception | reasoning
    deadline: float = 0.0              # optional timestamp
    success_criteria: list[str] = field(default_factory=list)
    progress: float = 0.0              # 0..1 estimated completion
    blockers: list[str] = field(default_factory=list)  # goal IDs blocking this
    sub_goals: list[str] = field(default_factory=list)  # child goal IDs
    related_memories: list[str] = field(default_factory=list)
    emotional_drive: float = 0.0       # emotional urgency
    cognitive_priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.status == "active"

    def is_blocked(self) -> bool:
        return self.status == "blocked"

    def is_complete(self) -> bool:
        return self.status in ("achieved", "abandoned")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description,
            "parent_id": self.parent_id, "status": self.status,
            "priority": round(self.priority, 3),
            "goal_type": self.goal_type, "source": self.source,
            "progress": round(self.progress, 3),
            "sub_goals_count": len(self.sub_goals),
            "blockers_count": len(self.blockers),
            "emotional_drive": round(self.emotional_drive, 3),
            "cognitive_priority": round(self.cognitive_priority, 3),
        }


@dataclass
class GoalConflict:
    """A detected conflict between two goals."""
    id: str
    goal_a_id: str
    goal_b_id: str
    conflict_type: str = "resource"    # resource | priority | contradictory | temporal
    severity: float = 0.5              # 0..1
    description: str = ""
    suggested_resolution: str = ""
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal_a": self.goal_a_id,
            "goal_b": self.goal_b_id,
            "type": self.conflict_type,
            "severity": round(self.severity, 3),
            "description": self.description,
            "resolved": self.resolved,
        }


@dataclass
class GoalDrivenInference:
    """An inference triggered by a goal — what needs to be remembered/recalled."""
    goal_id: str
    inference_type: str = "need_info"  # need_info | need_action | need_verification
    query: str = ""                    # what to search for in memory
    expected_answer_type: str = "fact" # fact | procedure | concept | preference
    priority: float = 0.5
    timestamp: float = field(default_factory=time.time)


# ── Goal System ─────────────────────────────────────────────────


class GoalSystem:
    """Goal-driven cognitive system.

    Goals actively shape attention, memory retrieval, and reasoning.
    """

    def __init__(self, graph=None, *,
                 config: dict[str, Any] | None = None,
                 embedder=None):
        self.graph = graph
        self._config = config or {}
        self._embedder = embedder

        self._goals: dict[str, GoalFrame] = {}
        self._conflicts: dict[str, GoalConflict] = {}

        self._max_goals = self._config.get("max_goals", 100)
        self._max_active = self._config.get("max_active_goals", 10)
        self._max_sub_goals = self._config.get("max_sub_goals_per_goal", 10)
        self._abandon_stale_days = self._config.get("abandon_stale_days", 90)
        self._archive_achieved_days = self._config.get("archive_achieved_days", 30)

    # ── Goal Lifecycle ────────────────────────────────────

    def add_goal(self, description: str, *,
                 goal_type: str = "explicit",
                 parent_id: str = "",
                 priority: float = 0.5,
                 source: str = "user_stated",
                 success_criteria: list[str] | None = None,
                 deadline: float = 0.0,
                 emotional_drive: float = 0.0) -> GoalFrame:
        """Add a new goal."""
        if len(self._goals) >= self._max_goals:
            self._prune_stale()

        goal = GoalFrame(
            id=str(uuid.uuid4()),
            description=description,
            parent_id=parent_id,
            status="pending",
            priority=priority,
            goal_type=goal_type,
            source=source,
            deadline=deadline,
            success_criteria=success_criteria or [],
            emotional_drive=emotional_drive,
            cognitive_priority=priority,
        )

        # Link to parent
        if parent_id and parent_id in self._goals:
            parent = self._goals[parent_id]
            if len(parent.sub_goals) < self._max_sub_goals:
                parent.sub_goals.append(goal.id)

        self._goals[goal.id] = goal
        return goal

    def activate_goal(self, goal_id: str) -> GoalFrame | None:
        """Set a goal as active."""
        if goal_id not in self._goals:
            return None

        # Enforce max active
        active = [g for g in self._goals.values() if g.status == "active"]
        if len(active) >= self._max_active:
            # Deactivate lowest priority active goal
            lowest = min(active, key=lambda g: g.cognitive_priority)
            lowest.status = "pending"

        goal = self._goals[goal_id]
        goal.status = "active"
        goal.updated_at = time.time()
        return goal

    def update_progress(self, goal_id: str, progress: float):
        """Update goal progress."""
        if goal_id not in self._goals:
            return
        goal = self._goals[goal_id]
        goal.progress = max(0.0, min(1.0, progress))
        goal.updated_at = time.time()

        if goal.progress >= 1.0:
            self.complete_goal(goal_id)

        # Propagate to parent
        if goal.parent_id and goal.parent_id in self._goals:
            parent = self._goals[goal.parent_id]
            if parent.sub_goals:
                child_progress = sum(
                    self._goals[cid].progress for cid in parent.sub_goals
                    if cid in self._goals
                ) / max(len(parent.sub_goals), 1)
                parent.progress = min(1.0, child_progress)

    def complete_goal(self, goal_id: str, outcome: str = ""):
        """Mark a goal as achieved."""
        if goal_id not in self._goals:
            return
        goal = self._goals[goal_id]
        goal.status = "achieved"
        goal.progress = 1.0
        goal.completed_at = time.time()
        goal.updated_at = time.time()
        if outcome:
            goal.metadata["outcome"] = outcome

    def abandon_goal(self, goal_id: str, reason: str = ""):
        """Abandon a goal."""
        if goal_id not in self._goals:
            return
        goal = self._goals[goal_id]
        goal.status = "abandoned"
        goal.updated_at = time.time()
        if reason:
            goal.metadata["abandon_reason"] = reason

    def block_goal(self, goal_id: str, blocker_goal_id: str):
        """Mark a goal as blocked by another goal."""
        if goal_id not in self._goals:
            return
        goal = self._goals[goal_id]
        goal.status = "blocked"
        if blocker_goal_id not in goal.blockers:
            goal.blockers.append(blocker_goal_id)
        goal.updated_at = time.time()

    def decompose_goal(self, goal_id: str,
                       sub_goal_descriptions: list[str]) -> list[GoalFrame]:
        """Break a goal into sub-goals."""
        if goal_id not in self._goals:
            return []
        parent = self._goals[goal_id]

        created: list[GoalFrame] = []
        for desc in sub_goal_descriptions:
            if len(parent.sub_goals) >= self._max_sub_goals:
                break
            sub = self.add_goal(
                desc,
                goal_type=parent.goal_type,
                parent_id=goal_id,
                priority=parent.priority * 0.9,
                source="decomposed",
            )
            parent.sub_goals.append(sub.id)
            created.append(sub)

        return created

    # ── Queries ────────────────────────────────────────────

    def get_active_goals(self) -> list[GoalFrame]:
        return [g for g in self._goals.values() if g.status == "active"]

    def get_pending_goals(self) -> list[GoalFrame]:
        return [g for g in self._goals.values() if g.status == "pending"]

    def get_blocked_goals(self) -> list[GoalFrame]:
        return [g for g in self._goals.values() if g.status == "blocked"]

    def get_goals_by_priority(self, min_priority: float = 0.5) -> list[GoalFrame]:
        goals = [g for g in self._goals.values() if g.cognitive_priority >= min_priority]
        goals.sort(key=lambda g: g.cognitive_priority, reverse=True)
        return goals

    def get_goal_tree(self, root_goal_id: str | None = None) -> dict:
        """Get full goal hierarchy as a tree dict."""
        if root_goal_id and root_goal_id in self._goals:
            return self._build_tree(root_goal_id)

        # Find roots (goals with no parent)
        roots = [g for g in self._goals.values() if not g.parent_id]
        return {"roots": [self._build_tree(r.id) for r in roots]}

    def _build_tree(self, goal_id: str) -> dict:
        goal = self._goals[goal_id]
        node = goal.to_dict()
        node["children"] = [
            self._build_tree(cid) for cid in goal.sub_goals if cid in self._goals
        ]
        return node

    # ── Conflict Detection ─────────────────────────────────

    def detect_conflicts(self) -> list[GoalConflict]:
        """Detect conflicts between active goals."""
        active = self.get_active_goals()
        conflicts: list[GoalConflict] = []

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                conflict = self._check_pair(a, b)
                if conflict:
                    conflicts.append(conflict)
                    if conflict.id not in self._conflicts:
                        self._conflicts[conflict.id] = conflict

        return conflicts

    def _check_pair(self, a: GoalFrame, b: GoalFrame) -> GoalConflict | None:
        """Check if two goals conflict."""
        # Contradictory: descriptions are opposites
        a_words = set(a.description.lower().split())
        b_words = set(b.description.lower().split())
        shared = a_words & b_words

        # Priority conflict: both high priority, limited attention
        if a.cognitive_priority > 0.7 and b.cognitive_priority > 0.7:
            if len(shared) == 0:  # unrelated topics
                return GoalConflict(
                    id=str(uuid.uuid4()),
                    goal_a_id=a.id, goal_b_id=b.id,
                    conflict_type="priority",
                    severity=0.6,
                    description=f"Both goals have high priority: '{a.description[:60]}' vs '{b.description[:60]}'",
                    suggested_resolution="Decompose one goal or adjust priority of the less urgent one",
                )

        # Resource conflict: similar domain, both active
        if len(shared) / max(len(a_words | b_words), 1) > 0.3:
            return GoalConflict(
                id=str(uuid.uuid4()),
                goal_a_id=a.id, goal_b_id=b.id,
                conflict_type="resource",
                severity=0.4,
                description=f"Goals share resources: '{a.description[:60]}' vs '{b.description[:60]}'",
                suggested_resolution="Sequence the goals or decompose into non-overlapping sub-goals",
            )

        return None

    def resolve_conflict(self, conflict_id: str, resolution: str):
        """Apply a conflict resolution."""
        if conflict_id in self._conflicts:
            self._conflicts[conflict_id].resolved = True
            self._conflicts[conflict_id].resolution = resolution

    # ── Goal-Driven Inference ──────────────────────────────

    def infer_needs(self, goal_id: str) -> list[GoalDrivenInference]:
        """Infer what information/actions are needed to achieve a goal."""
        if goal_id not in self._goals:
            return []
        goal = self._goals[goal_id]
        inferences: list[GoalDrivenInference] = []

        # Need info about the topic
        inferences.append(GoalDrivenInference(
            goal_id=goal_id,
            inference_type="need_info",
            query=f"information about: {goal.description}",
            expected_answer_type="fact",
            priority=goal.cognitive_priority,
        ))

        # Need procedures if this is a task
        if goal.goal_type in ("explicit", "inferred"):
            inferences.append(GoalDrivenInference(
                goal_id=goal_id,
                inference_type="need_action",
                query=f"how to accomplish: {goal.description}",
                expected_answer_type="procedure",
                priority=goal.cognitive_priority * 0.9,
            ))

        # If blocked, need to understand blockers
        if goal.blockers:
            for blocker_id in goal.blockers:
                if blocker_id in self._goals:
                    inferences.append(GoalDrivenInference(
                        goal_id=goal_id,
                        inference_type="need_info",
                        query=f"status of blocker: {self._goals[blocker_id].description}",
                        expected_answer_type="fact",
                        priority=goal.cognitive_priority,
                    ))

        return inferences

    def get_retrieval_context(self) -> dict:
        """Get retrieval context shaped by active goals."""
        active = self.get_active_goals()
        if not active:
            return {"goal_ids": [], "goal_descriptions": [], "priority_sum": 0.0}

        return {
            "goal_ids": [g.id for g in active],
            "goal_descriptions": [g.description for g in active],
            "priority_sum": sum(g.cognitive_priority for g in active),
            "top_goal": max(active, key=lambda g: g.cognitive_priority).description,
        }

    # ── Maintenance ────────────────────────────────────────

    def _prune_stale(self):
        """Remove long-abandoned or long-achieved goals."""
        now = time.time()
        for gid, goal in list(self._goals.items()):
            if goal.status == "abandoned":
                age_days = (now - goal.updated_at) / 86400
                if age_days > self._abandon_stale_days:
                    del self._goals[gid]
            elif goal.status == "achieved":
                age_days = (now - goal.completed_at) / 86400
                if age_days > self._archive_achieved_days:
                    del self._goals[gid]

    def auto_maintain(self):
        """Auto-abandon stale pending goals."""
        now = time.time()
        for goal in self._goals.values():
            if goal.status in ("pending", "active"):
                idle_days = (now - goal.updated_at) / 86400
                if idle_days > self._abandon_stale_days and goal.progress < 0.1:
                    goal.status = "abandoned"
                    goal.metadata["abandon_reason"] = "auto: stale"

    # ── Summary ────────────────────────────────────────────

    def summarize_goals(self) -> str:
        """Compressed goal overview for LLM injection."""
        active = self.get_active_goals()
        blocked = self.get_blocked_goals()
        pending = self.get_pending_goals()[:5]

        parts = []
        if active:
            parts.append("Active goals:")
            for g in active:
                parts.append(f"  - {g.description} (progress: {g.progress:.0%}, priority: {g.cognitive_priority:.2f})")
        if blocked:
            parts.append("Blocked goals:")
            for g in blocked:
                blockers = [self._goals.get(bid) for bid in g.blockers if bid in self._goals]
                blocker_desc = ", ".join(b.description for b in blockers if b)
                parts.append(f"  - {g.description} (blocked by: {blocker_desc})")
        if pending and not active:
            parts.append(f"Pending goals: {len(pending)}")

        conflicts = self.detect_conflicts()
        if conflicts:
            parts.append(f"Goal conflicts: {len(conflicts)}")

        return "\n".join(parts)

    # ── Properties ─────────────────────────────────────────

    @property
    def goal_count(self) -> int:
        return len(self._goals)

    def get_goal(self, goal_id: str) -> GoalFrame | None:
        return self._goals.get(goal_id)
