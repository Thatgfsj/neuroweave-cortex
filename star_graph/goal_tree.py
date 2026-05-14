"""Goal Tree — hierarchical goal decomposition with progress tracking.

Features:
  - Goal nodes with parent/child relationships
  - Auto-detection of goals from memory content
  - Progress tracking via sub-goal completion
  - Goal lifecycle: active → achieved/abandoned → archived
  - Temporal metrics: active duration, last progress, staleness

Wired into runtime.sleep() for progress propagation and stale goal detection.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


class GoalStatus:
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"
    BLOCKED = "blocked"
    ARCHIVED = "archived"


@dataclass
class GoalNode:
    """A goal in the tree."""
    id: str
    description: str
    status: str = GoalStatus.ACTIVE
    parent_id: str = ""
    children: list[str] = field(default_factory=list)
    progress: float = 0.0                 # 0.0 - 1.0
    priority: float = 0.5                 # higher = more important
    confidence: float = 0.5               # detection confidence
    evidence_anchor_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_progress_at: float = field(default_factory=time.time)
    achieved_at: float = 0.0
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent_id == ""

    @property
    def staleness_hours(self) -> float:
        """Hours since last progress."""
        return (time.time() - self.last_progress_at) / 3600


class GoalTree:
    """Hierarchical goal manager with auto-detection and progress propagation.

    Usage:
        gt = GoalTree()
        gt.detect_from_graph(graph)       # scan memories for goals
        gt.mark_progress("goal_1", 0.3)   # advance progress
        gt.propagate_progress()            # sub-goal → parent update
        gt.archive_stale(hours=168)        # archive goals idle for a week
    """

    # --- Goal detection patterns ---
    _GOAL_PATTERNS = [
        # Explicit goal statements
        (r'(?:need to|have to|must|should|gotta|gonna|will|plan to|going to)\s+(.+?)(?:[\.\n]|$)', 0.7),
        (r'(?:需要|必须|要|将会|打算|计划)\s*(.+?)(?:[。\n]|$)', 0.7),
        # Task tracking
        (r'(?:working on|implementing|building|fixing|debugging)\s+(.+?)(?:[\.\n]|$)', 0.6),
        (r'(?:在做|正在做|开发|修复|调试)\s*(.+?)(?:[。\n]|$)', 0.6),
        # TODO / task markers
        (r'(?:TODO|FIXME|TASK)[: ]\s*(.+?)(?:[\.\n]|$)', 0.8),
        # Explicit goal markers
        (r'(?:目标|任务|goal|task|objective)[:：]\s*(.+?)(?:[\.\n]|$)', 0.85),
    ]

    def __init__(self):
        self._goals: dict[str, GoalNode] = {}
        self._detection_cache: set[str] = set()  # hash of detected goal texts
        self._total_detected = 0
        self._total_achieved = 0
        self._total_abandoned = 0

    # ── Goal detection from memory ────────────────────────────

    def detect_from_graph(self, graph) -> list[GoalNode]:
        """Scan graph anchors for goal statements. Returns newly detected goals."""
        new_goals = []
        for anchor in graph.anchors.values():
            if not anchor.is_retrievable:
                continue
            detected = self._detect_goals_in_text(anchor.text, anchor)
            for goal in detected:
                if goal.id not in self._goals:
                    self._goals[goal.id] = goal
                    new_goals.append(goal)
                    self._total_detected += 1
        return new_goals

    def _detect_goals_in_text(self, text: str, anchor) -> list[GoalNode]:
        """Extract goal nodes from text."""
        goals = []
        for pattern, confidence in self._GOAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                goal_text = match.group(1).strip()[:120]
                goal_hash = hashlib.md5(goal_text.lower().encode()).hexdigest()[:10]

                # Deduplicate
                if goal_hash in self._detection_cache:
                    continue
                self._detection_cache.add(goal_hash)

                # Auto-categorize goal
                tags = self._infer_goal_tags(goal_text)
                priority = self._infer_priority(goal_text, tags)

                goal = GoalNode(
                    id=f"goal_{goal_hash}",
                    description=goal_text,
                    confidence=confidence,
                    evidence_anchor_ids=[anchor.id],
                    tags=tags,
                    priority=priority,
                )
                goals.append(goal)
        return goals

    def _infer_goal_tags(self, text: str) -> list[str]:
        """Infer tags from goal text."""
        text_lower = text.lower()
        tags = []
        if any(kw in text_lower for kw in {'fix', 'bug', 'debug', 'error', '修复', 'bug', '错误'}):
            tags.append('bugfix')
        if any(kw in text_lower for kw in {'build', 'create', 'implement', '开发', '构建', '实现'}):
            tags.append('development')
        if any(kw in text_lower for kw in {'learn', 'study', 'read', '学习', '阅读', '了解'}):
            tags.append('learning')
        if any(kw in text_lower for kw in {'deploy', 'release', 'publish', '部署', '发布'}):
            tags.append('deployment')
        if any(kw in text_lower for kw in {'test', 'verify', '测试', '验证'}):
            tags.append('testing')
        if any(kw in text_lower for kw in {'refactor', 'clean', '重构', '清理'}):
            tags.append('refactoring')
        if any(kw in text_lower for kw in {'config', 'setup', 'install', '配置', '安装'}):
            tags.append('setup')
        return tags or ['general']

    @staticmethod
    def _infer_priority(text: str, tags: list[str]) -> float:
        """Infer goal priority from urgency signals."""
        text_lower = text.lower()
        priority = 0.5

        # Urgency signals
        if any(kw in text_lower for kw in {'urgent', 'asap', 'critical', '紧急', '关键', '重要'}):
            priority += 0.3
        if any(kw in text_lower for kw in {'maybe', 'someday', 'later', '也许', '以后'}):
            priority -= 0.2
        if 'bugfix' in tags or 'deployment' in tags:
            priority += 0.1
        if 'learning' in tags:
            priority -= 0.1

        return max(0.1, min(1.0, priority))

    # ── Goal management ──────────────────────────────────────

    def add_subgoal(self, parent_id: str, description: str,
                    priority: float = 0.5) -> GoalNode | None:
        """Add a sub-goal to an existing goal."""
        if parent_id not in self._goals:
            return None
        goal_hash = hashlib.md5(description.lower().encode()).hexdigest()[:10]
        goal_id = f"goal_{goal_hash}"

        parent = self._goals[parent_id]
        goal = GoalNode(
            id=goal_id,
            description=description,
            parent_id=parent_id,
            depth=parent.depth + 1,
            priority=priority,
        )
        self._goals[goal_id] = goal
        parent.children.append(goal_id)
        self._total_detected += 1
        return goal

    def mark_progress(self, goal_id: str, amount: float) -> bool:
        """Advance progress on a goal by the given amount (0.0-1.0)."""
        if goal_id not in self._goals:
            return False
        goal = self._goals[goal_id]
        goal.progress = min(1.0, goal.progress + amount)
        goal.last_progress_at = time.time()

        if goal.progress >= 1.0:
            self._mark_achieved(goal_id)
        return True

    def mark_achieved(self, goal_id: str) -> bool:
        """Mark a goal as achieved."""
        return self._mark_achieved(goal_id)

    def _mark_achieved(self, goal_id: str) -> bool:
        if goal_id not in self._goals:
            return False
        goal = self._goals[goal_id]
        goal.status = GoalStatus.ACHIEVED
        goal.progress = 1.0
        goal.achieved_at = time.time()
        self._total_achieved += 1
        # Propagate to parent
        self._propagate_to_parent(goal_id)
        return True

    def mark_abandoned(self, goal_id: str) -> bool:
        """Mark a goal as abandoned."""
        if goal_id not in self._goals:
            return False
        goal = self._goals[goal_id]
        goal.status = GoalStatus.ABANDONED
        self._total_abandoned += 1
        return True

    def mark_blocked(self, goal_id: str, reason: str = "") -> bool:
        """Mark a goal as blocked."""
        if goal_id not in self._goals:
            return False
        self._goals[goal_id].status = GoalStatus.BLOCKED
        return True

    # ── Progress propagation ─────────────────────────────────

    def propagate_progress(self) -> int:
        """Propagate sub-goal progress upward to parents. Returns updated count."""
        updated = 0
        # Process from deepest to shallowest (leaves first)
        sorted_goals = sorted(
            self._goals.values(),
            key=lambda g: -g.depth,
        )
        for goal in sorted_goals:
            if goal.parent_id and goal.parent_id in self._goals:
                self._update_parent_progress(goal.parent_id)
                updated += 1
        return updated

    def _update_parent_progress(self, parent_id: str):
        """Recalculate parent progress from children."""
        parent = self._goals.get(parent_id)
        if not parent or not parent.children:
            return

        child_progresses = []
        for child_id in parent.children:
            child = self._goals.get(child_id)
            if child:
                child_progresses.append(child.progress)

        if child_progresses:
            parent.progress = sum(child_progresses) / len(child_progresses)
            parent.last_progress_at = time.time()

    def _propagate_to_parent(self, goal_id: str):
        """After a goal is achieved, update parent progress."""
        goal = self._goals.get(goal_id)
        if goal and goal.parent_id:
            self._update_parent_progress(goal.parent_id)

    # ── Stale goal archival ──────────────────────────────────

    def archive_stale(self, hours: float = 168.0) -> int:
        """Archive goals that have had no progress for N hours. Returns count."""
        count = 0
        for goal in list(self._goals.values()):
            if goal.status == GoalStatus.ACTIVE and goal.staleness_hours >= hours:
                goal.status = GoalStatus.ARCHIVED
                count += 1
        return count

    def archive_achieved(self, age_hours: float = 720.0) -> int:
        """Archive achieved goals older than N hours."""
        count = 0
        cutoff = time.time() - age_hours * 3600
        for goal in list(self._goals.values()):
            if goal.status == GoalStatus.ACHIEVED and goal.achieved_at < cutoff:
                goal.status = GoalStatus.ARCHIVED
                count += 1
        return count

    # ── Query API ────────────────────────────────────────────

    def get_goal(self, goal_id: str) -> GoalNode | None:
        return self._goals.get(goal_id)

    def get_active_goals(self) -> list[GoalNode]:
        """Get all currently active goals, sorted by priority."""
        active = [g for g in self._goals.values()
                 if g.status == GoalStatus.ACTIVE]
        active.sort(key=lambda g: (-g.priority, -g.progress))
        return active

    def get_blocked_goals(self) -> list[GoalNode]:
        return [g for g in self._goals.values()
                if g.status == GoalStatus.BLOCKED]

    def get_recently_achieved(self, hours: float = 168.0) -> list[GoalNode]:
        """Goals achieved within the last N hours."""
        cutoff = time.time() - hours * 3600
        return [g for g in self._goals.values()
                if g.status == GoalStatus.ACHIEVED and g.achieved_at >= cutoff]

    def get_root_goals(self) -> list[GoalNode]:
        """Root-level goals (no parent)."""
        return [g for g in self._goals.values() if g.is_root]

    def get_children(self, goal_id: str) -> list[GoalNode]:
        """Get direct children of a goal."""
        goal = self._goals.get(goal_id)
        if not goal:
            return []
        return [self._goals[cid] for cid in goal.children
                if cid in self._goals]

    def get_stale_goals(self, hours: float = 168.0) -> list[GoalNode]:
        """Active goals with no progress for N hours."""
        return [g for g in self._goals.values()
                if g.status == GoalStatus.ACTIVE and g.staleness_hours >= hours]

    # ── Stats ────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        status_counts = defaultdict(int)
        for g in self._goals.values():
            status_counts[g.status] += 1
        return {
            "total_goals": len(self._goals),
            "active": status_counts.get(GoalStatus.ACTIVE, 0),
            "achieved": status_counts.get(GoalStatus.ACHIEVED, 0),
            "abandoned": status_counts.get(GoalStatus.ABANDONED, 0),
            "blocked": status_counts.get(GoalStatus.BLOCKED, 0),
            "archived": status_counts.get(GoalStatus.ARCHIVED, 0),
            "total_detected": self._total_detected,
            "total_achieved": self._total_achieved,
            "total_abandoned": self._total_abandoned,
            "active_goals": [
                {"id": g.id, "description": g.description[:80],
                 "progress": round(g.progress, 2),
                 "priority": round(g.priority, 2)}
                for g in self.get_active_goals()[:5]
            ],
        }
