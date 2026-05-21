"""Agent State Memory — session-level goals, tool calls, checkpoints, and reasoning phases."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph

from .memory_core.anchor import Anchor

try:
    from .memory_core.anchor import AnchorCreate
except ImportError:
    AnchorCreate = None  # type: ignore[assignment]


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class GoalNode:
    """A node in the agent's hierarchical goal tree."""
    id: str
    description: str
    parent_id: str = ""
    status: str = "pending"
    priority: int = 5
    children: list[str] = field(default_factory=list)
    created_at: float = 0.0
    completed_at: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description,
            "parent_id": self.parent_id, "status": self.status,
            "priority": self.priority, "children": self.children,
            "created_at": self.created_at, "completed_at": self.completed_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GoalNode:
        return cls(
            id=d["id"], description=d["description"],
            parent_id=d.get("parent_id", ""), status=d.get("status", "pending"),
            priority=d.get("priority", 5), children=d.get("children", []),
            created_at=d.get("created_at", 0.0), completed_at=d.get("completed_at", 0.0),
            notes=d.get("notes", ""),
        )


@dataclass
class ToolCallRecord:
    """A record of a single tool invocation."""
    id: str
    tool_name: str
    arguments: dict = field(default_factory=dict)
    result_summary: str = ""
    success: bool = True
    duration_ms: float = 0.0
    timestamp: float = 0.0
    related_goal_id: str = ""

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "tool_name": self.tool_name,
            "arguments": self.arguments, "result_summary": self.result_summary,
            "success": self.success, "duration_ms": self.duration_ms,
            "timestamp": self.timestamp, "related_goal_id": self.related_goal_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ToolCallRecord:
        return cls(
            id=d["id"], tool_name=d["tool_name"],
            arguments=d.get("arguments", {}), result_summary=d.get("result_summary", ""),
            success=d.get("success", True), duration_ms=d.get("duration_ms", 0.0),
            timestamp=d.get("timestamp", 0.0), related_goal_id=d.get("related_goal_id", ""),
        )


@dataclass
class Checkpoint:
    """A snapshot of agent state for later restoration."""
    id: str
    label: str
    goal_tree_snapshot: dict
    tool_history_snapshot: list
    created_at: float
    context_summary: str

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label,
            "goal_tree_snapshot": self.goal_tree_snapshot,
            "tool_history_snapshot": self.tool_history_snapshot,
            "created_at": self.created_at,
            "context_summary": self.context_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Checkpoint:
        return cls(
            id=d["id"], label=d.get("label", ""),
            goal_tree_snapshot=d.get("goal_tree_snapshot", {}),
            tool_history_snapshot=d.get("tool_history_snapshot", []),
            created_at=d.get("created_at", 0.0),
            context_summary=d.get("context_summary", ""),
        )


@dataclass
class AgentState:
    """Current cognitive state for an agent session."""
    session_id: str
    current_goals: list[GoalNode] = field(default_factory=list)
    tool_history: list[ToolCallRecord] = field(default_factory=list)
    reasoning_phase: str = "idle"
    active_checkpoints: list[Checkpoint] = field(default_factory=list)
    last_updated: float = 0.0

    def __post_init__(self) -> None:
        if self.last_updated == 0.0:
            self.last_updated = time.time()


# ── Manager ───────────────────────────────────────────────────────

_VALID_PHASES: set[str] = {"exploration", "planning", "execution", "validation", "idle"}


class AgentStateManager:
    """Manages agent state backed by a StarGraph for optional persistence."""

    def __init__(self, graph: StarGraph) -> None:
        self._graph = graph
        self._state: AgentState | None = None
        self._session_id: str = ""

    def set_session(self, session_id: str) -> None:
        """Start or switch to a new session."""
        self._session_id = session_id
        self._state = AgentState(session_id=session_id)

    def get_state(self) -> AgentState:
        """Return the current agent state, initializing a default if needed."""
        if self._state is None:
            self._state = AgentState(session_id=self._session_id or "default")
        return self._state

    # ── Goal management ──────────────────────────────────────────

    def add_goal(self, description: str, parent_id: str = "",
                 priority: int = 5) -> GoalNode:
        """Add a new goal to the tree, linking to parent if specified."""
        state = self.get_state()
        goal = GoalNode(
            id=f"goal_{uuid.uuid4().hex[:12]}",
            description=description,
            parent_id=parent_id,
            priority=max(1, min(10, priority)),
        )
        if parent_id:
            parent = self._find_goal(parent_id)
            if parent:
                parent.children.append(goal.id)
        state.current_goals.append(goal)
        state.last_updated = time.time()
        return goal

    def update_goal_status(self, goal_id: str, status: str) -> None:
        """Update the status of a goal."""
        goal = self._find_goal(goal_id)
        if goal:
            goal.status = status
            self.get_state().last_updated = time.time()

    def complete_goal(self, goal_id: str) -> None:
        """Mark a goal as completed."""
        goal = self._find_goal(goal_id)
        if goal:
            goal.status = "completed"
            goal.completed_at = time.time()
            self.get_state().last_updated = time.time()

    def get_goal_tree(self) -> list[GoalNode]:
        """Return all goals in the tree."""
        return list(self.get_state().current_goals)

    def get_active_goals(self) -> list[GoalNode]:
        """Return goals that are pending or in_progress."""
        return [g for g in self.get_state().current_goals
                if g.status in ("pending", "in_progress")]

    def get_blocked_goals(self) -> list[GoalNode]:
        """Return goals with status 'blocked'."""
        return [g for g in self.get_state().current_goals
                if g.status == "blocked"]

    # ── Tool call recording ──────────────────────────────────────

    def record_tool_call(self, tool_name: str, arguments: dict,
                         result_summary: str, success: bool,
                         duration_ms: float,
                         related_goal_id: str = "") -> ToolCallRecord:
        """Record a tool invocation and return the record."""
        state = self.get_state()
        record = ToolCallRecord(
            id=f"tool_{uuid.uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
            result_summary=result_summary[:200],
            success=success,
            duration_ms=duration_ms,
            related_goal_id=related_goal_id,
        )
        state.tool_history.append(record)
        state.last_updated = time.time()
        return record

    def get_tool_history(self, n: int = 50) -> list[ToolCallRecord]:
        """Return the most recent n tool call records."""
        history = self.get_state().tool_history
        return history[-n:] if n > 0 else list(history)

    def get_tools_for_goal(self, goal_id: str) -> list[ToolCallRecord]:
        """Return tool calls associated with a specific goal."""
        return [t for t in self.get_state().tool_history
                if t.related_goal_id == goal_id]

    # ── Checkpoints ──────────────────────────────────────────────

    def create_checkpoint(self, label: str = "") -> Checkpoint:
        """Snapshot current goal tree and tool history into a checkpoint."""
        state = self.get_state()
        cp = Checkpoint(
            id=f"ckpt_{uuid.uuid4().hex[:12]}",
            label=label or f"checkpoint_{len(state.active_checkpoints) + 1}",
            goal_tree_snapshot=json.dumps(
                [g.to_dict() for g in state.current_goals]
            ),
            tool_history_snapshot=json.dumps(
                [t.to_dict() for t in state.tool_history]
            ),
            created_at=time.time(),
            context_summary=(
                f"Phase: {state.reasoning_phase}, "
                f"Active goals: {len(self.get_active_goals())}, "
                f"Tool calls: {len(state.tool_history)}"
            ),
        )
        state.active_checkpoints.append(cp)
        state.last_updated = time.time()
        return cp

    def restore_checkpoint(self, checkpoint_id: str) -> AgentState:
        """Restore agent state from a checkpoint snapshot."""
        state = self.get_state()
        cp = self._find_checkpoint(checkpoint_id)
        if cp is None:
            return state
        try:
            goal_snap = (
                json.loads(cp.goal_tree_snapshot)
                if isinstance(cp.goal_tree_snapshot, str)
                else cp.goal_tree_snapshot
            )
            tool_snap = (
                json.loads(cp.tool_history_snapshot)
                if isinstance(cp.tool_history_snapshot, str)
                else cp.tool_history_snapshot
            )
            state.current_goals = [GoalNode.from_dict(g) for g in goal_snap]
            state.tool_history = [ToolCallRecord.from_dict(t) for t in tool_snap]
            state.last_updated = time.time()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return state

    def list_checkpoints(self) -> list[Checkpoint]:
        """Return all active checkpoints."""
        return list(self.get_state().active_checkpoints)

    # ── Reasoning phase ──────────────────────────────────────────

    def set_reasoning_phase(self, phase: str) -> None:
        """Update the current reasoning phase."""
        state = self.get_state()
        if phase in _VALID_PHASES:
            state.reasoning_phase = phase
            state.last_updated = time.time()

    def get_reasoning_phase(self) -> str:
        """Return the current reasoning phase."""
        return self.get_state().reasoning_phase

    # ── State summary ────────────────────────────────────────────

    def summarize_state(self) -> str:
        """Return a human-readable state summary for prompt injection."""
        state = self.get_state()
        active = self.get_active_goals()
        blocked = self.get_blocked_goals()
        recent = self.get_tool_history(5)

        active_str = ", ".join(
            f"{g.description[:60]}" for g in active
        ) if active else "none"

        tools_str = ", ".join(
            f"{t.tool_name}({'OK' if t.success else 'FAIL'})"
            for t in recent
        ) if recent else "none"

        blocked_str = ", ".join(
            f"{g.description[:60]}" for g in blocked
        ) if blocked else "none"

        return (
            f"[Agent State]\n"
            f"Active Goals: {active_str}\n"
            f"Recent Tools: {tools_str}\n"
            f"Current Phase: {state.reasoning_phase}\n"
            f"Blocked: {blocked_str}"
        )

    # ── Graph persistence ────────────────────────────────────────

    def persist_to_graph(self) -> None:
        """Save goals, tool records, and checkpoints as tagged graph anchors."""
        state = self.get_state()
        sid = state.session_id or self._session_id

        for goal in state.current_goals:
            anchor = Anchor.create(
                text=json.dumps(goal.to_dict()),
                source_session=sid,
                tags=["__agent_state__", "__goal__"],
                importance=goal.priority / 10.0,
            )
            self._graph.add_anchor(anchor)

        for tool in state.tool_history:
            anchor = Anchor.create(
                text=json.dumps(tool.to_dict()),
                source_session=sid,
                tags=["__agent_state__", "__tool_call__"],
            )
            self._graph.add_anchor(anchor)

        for cp in state.active_checkpoints:
            anchor = Anchor.create(
                text=json.dumps(cp.to_dict()),
                source_session=sid,
                tags=["__agent_state__", "__checkpoint__"],
            )
            self._graph.add_anchor(anchor)

    def restore_from_graph(self) -> None:
        """Load goals, tool records, and checkpoints from graph anchors."""
        state = self.get_state()
        goals: list[GoalNode] = []
        tools: list[ToolCallRecord] = []
        checkpoints: list[Checkpoint] = []

        for anchor in self._graph.anchors.values():
            tags = set(anchor.tags)
            if "__agent_state__" not in tags:
                continue
            try:
                data = json.loads(anchor.text)
            except (json.JSONDecodeError, TypeError):
                continue

            if "__goal__" in tags:
                goals.append(GoalNode.from_dict(data))
            elif "__tool_call__" in tags:
                tools.append(ToolCallRecord.from_dict(data))
            elif "__checkpoint__" in tags:
                checkpoints.append(Checkpoint.from_dict(data))

        state.current_goals = goals
        state.tool_history = sorted(tools, key=lambda t: t.timestamp)
        state.active_checkpoints = sorted(checkpoints, key=lambda c: c.created_at)
        state.last_updated = time.time()

    # ── Internal helpers ─────────────────────────────────────────

    def _find_goal(self, goal_id: str) -> GoalNode | None:
        for g in self.get_state().current_goals:
            if g.id == goal_id:
                return g
        return None

    def _find_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        for cp in self.get_state().active_checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None
