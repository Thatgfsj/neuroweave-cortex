"""Autonomous Reasoning Loop — contradiction → activation → resolution.

System automatically runs reasoning cycles without LLM involvement at every step.
Uses ActivationEngine for memory retrieval, CognitiveWorkspace for reasoning chains,
and updates beliefs/goals/concepts based on resolution outcomes.

Loop flow:
  1. Detect trigger (contradiction, blocked goal, uncertainty spike)
  2. Activate relevant memories/concepts via ActivationEngine
  3. Form reasoning chain in CognitiveWorkspace
  4. Attempt resolution (evidence search, goal decomposition)
  5. Update beliefs/concepts/goals based on resolution
  6. Record reasoning trace
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ReasoningTrigger:
    """What triggered the autonomous reasoning loop."""
    trigger_type: str = "contradiction"  # contradiction | blocked_goal | high_uncertainty | curiosity
    entity_ids: list[str] = field(default_factory=list)
    description: str = ""
    severity: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveUpdate:
    """A single cognitive change resulting from reasoning."""
    entity_type: str = "belief"        # belief | concept | goal | memory
    entity_id: str = ""
    field: str = ""                    # what was changed
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    confidence: float = 0.5


@dataclass
class ReasoningTrace:
    """Full trace of an autonomous reasoning session."""
    id: str
    trigger: ReasoningTrigger
    steps: list[dict] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    cognitive_updates: list[CognitiveUpdate] = field(default_factory=list)
    duration_ms: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class AutonomousReasoningLoop:
    """Self-directed reasoning cycle — without LLM involvement."""

    def __init__(self, graph=None, *,
                 workspace=None,
                 activation_engine=None,
                 goal_system=None,
                 concept_cortex=None,
                 self_model=None,
                 config: dict[str, Any] | None = None):
        self.graph = graph
        self._workspace = workspace
        self._activation_engine = activation_engine
        self._goal_system = goal_system
        self._concept_cortex = concept_cortex
        self._self_model = self_model
        self._config = config or {}

        self._enabled = self._config.get("enabled", False)
        self._max_iterations = self._config.get("max_iterations_per_run", 5)
        self._max_duration = self._config.get("max_duration_seconds", 30.0)
        self._min_severity = self._config.get("min_trigger_severity", 0.5)
        self._contradiction_threshold = self._config.get("contradiction_threshold", 0.7)
        self._uncertainty_threshold = self._config.get("uncertainty_spike_threshold", 0.3)
        self._evidence_search_depth = self._config.get("evidence_search_depth", 2)
        self._evidence_min = self._config.get("evidence_min_count", 3)
        self._resolution_confidence = self._config.get("resolution_confidence_threshold", 0.6)

        self._traces: list[ReasoningTrace] = []
        self._max_traces = self._config.get("trace_retention", 50)

    # ── Main Loop ──────────────────────────────────────────

    def run(self, *, max_iterations: int | None = None,
            max_duration_seconds: float | None = None) -> ReasoningTrace | None:
        """Run autonomous reasoning until resolution or max iterations."""
        if not self._enabled:
            return None

        t0 = time.perf_counter()
        max_iter = max_iterations or self._max_iterations
        max_dur = max_duration_seconds or self._max_duration

        # Step 1: Detect triggers
        triggers = self.detect_triggers()
        if not triggers:
            return None

        # Take the highest severity trigger
        trigger = max(triggers, key=lambda t: t.severity)
        if trigger.severity < self._min_severity:
            return None

        trace = ReasoningTrace(
            id=str(uuid.uuid4()),
            trigger=trigger,
        )

        # Step 2-4: Reasoning iterations
        for iteration in range(max_iter):
            elapsed = time.perf_counter() - t0
            if elapsed > max_dur:
                trace.conclusion = "timeout"
                break

            # Activate relevant memories
            if self._activation_engine:
                result = self._activation_engine.activate_from_query(
                    trigger.description
                )
                trace.steps.append({
                    "iteration": iteration,
                    "action": "activate",
                    "nodes_found": len(result.activated_nodes),
                    "paths_found": len(result.semantic_paths),
                })

            # Form reasoning in workspace
            if self._workspace:
                chain = self._workspace.start_reasoning(
                    goal_id=trigger.entity_ids[0] if trigger.entity_ids else "",
                    description=trigger.description,
                )
                if chain:
                    self._workspace.add_reasoning_step(
                        chain.id, "deduce",
                        f"Investigating: {trigger.description}",
                        confidence=0.6,
                    )
                    trace.steps.append({
                        "iteration": iteration,
                        "action": "reason",
                        "chain_id": chain.id,
                    })

            # Attempt resolution based on trigger type
            update: CognitiveUpdate | None = None
            if trigger.trigger_type == "contradiction":
                update = self._resolve_contradiction(trigger)
            elif trigger.trigger_type == "blocked_goal":
                update = self._unblock_goal(trigger)
            elif trigger.trigger_type == "high_uncertainty":
                update = self._reduce_uncertainty(trigger)

            if update and update.confidence >= self._resolution_confidence:
                trace.cognitive_updates.append(update)
                trace.conclusion = f"Resolved: {update.new_value}"
                trace.confidence = update.confidence
                break
            elif update:
                trace.cognitive_updates.append(update)
                trace.steps.append({
                    "iteration": iteration,
                    "action": "partial_resolution",
                    "confidence": update.confidence,
                })

        trace.completed_at = time.time()
        trace.duration_ms = (trace.completed_at - trace.started_at) * 1000

        self._traces.append(trace)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

        return trace

    # ── Trigger Detection ──────────────────────────────────

    def detect_triggers(self) -> list[ReasoningTrigger]:
        """Scan for reasoning triggers."""
        triggers: list[ReasoningTrigger] = []
        triggers.extend(self._detect_contradictions())
        triggers.extend(self._detect_blocked_goals())
        triggers.extend(self._detect_uncertainty_spikes())
        return triggers

    def _detect_contradictions(self) -> list[ReasoningTrigger]:
        """Detect contradicting beliefs."""
        triggers: list[ReasoningTrigger] = []
        if self._self_model:
            try:
                state = self._self_model.construct_state()
                for conflict in state.conflicting_beliefs[:3]:
                    severity = conflict.get('severity', 0.5)
                    if severity >= self._contradiction_threshold:
                        triggers.append(ReasoningTrigger(
                            trigger_type="contradiction",
                            entity_ids=[conflict.get('item_a', ''), conflict.get('item_b', '')],
                            description=f"Conflict: {conflict}",
                            severity=severity,
                        ))
            except Exception:
                pass
        return triggers

    def _detect_blocked_goals(self) -> list[ReasoningTrigger]:
        """Detect goals with blockers that need decomposition."""
        triggers: list[ReasoningTrigger] = []
        if self._goal_system:
            try:
                blocked = self._goal_system.get_blocked_goals()
                for goal in blocked[:3]:
                    triggers.append(ReasoningTrigger(
                        trigger_type="blocked_goal",
                        entity_ids=[goal.id],
                        description=f"Blocked goal: {goal.description}",
                        severity=0.6,
                    ))
            except Exception:
                pass
        return triggers

    def _detect_uncertainty_spikes(self) -> list[ReasoningTrigger]:
        """Detect when uncertainty has risen above threshold."""
        triggers: list[ReasoningTrigger] = []
        if self._self_model:
            try:
                history = self._self_model.get_state_history(limit=5)
                if len(history) >= 2:
                    prev = history[-2].get('confidence', 0.5)
                    curr = history[-1].get('confidence', 0.5)
                    drop = prev - curr
                    if drop > self._uncertainty_threshold:
                        triggers.append(ReasoningTrigger(
                            trigger_type="high_uncertainty",
                            description=f"Confidence dropped from {prev:.2f} to {curr:.2f}",
                            severity=drop,
                        ))
            except Exception:
                pass
        return triggers

    # ── Resolution Strategies ──────────────────────────────

    def _resolve_contradiction(self, trigger: ReasoningTrigger) -> CognitiveUpdate | None:
        """Attempt to resolve a contradiction through evidence search."""
        # Simple: flag the contradiction for the user/LLM
        return CognitiveUpdate(
            entity_type="belief",
            entity_id=trigger.entity_ids[0] if trigger.entity_ids else "",
            field="contradiction_detected",
            old_value="conflicting",
            new_value="flagged_for_review",
            reason=trigger.description,
            confidence=0.4,  # low confidence — needs LLM or user
        )

    def _unblock_goal(self, trigger: ReasoningTrigger) -> CognitiveUpdate | None:
        """Attempt to unblock a goal through decomposition."""
        if self._goal_system and trigger.entity_ids:
            goal = self._goal_system.get_goal(trigger.entity_ids[0])
            if goal and goal.blockers:
                # Try: decomposing into sub-goals that bypass the blocker
                return CognitiveUpdate(
                    entity_type="goal",
                    entity_id=goal.id,
                    field="status",
                    old_value="blocked",
                    new_value="decomposed",
                    reason=f"Decomposed to bypass blocker: {goal.blockers}",
                    confidence=0.5,
                )
        return None

    def _reduce_uncertainty(self, trigger: ReasoningTrigger) -> CognitiveUpdate | None:
        """Attempt to fill knowledge gaps through memory search."""
        return CognitiveUpdate(
            entity_type="belief",
            field="uncertainty",
            old_value="high",
            new_value="reduced",
            reason="Targeted memory search recommended",
            confidence=0.3,
        )

    # ── Traces ─────────────────────────────────────────────

    def get_reasoning_history(self, limit: int = 20) -> list[dict]:
        return [
            {
                "id": t.id,
                "trigger": t.trigger.trigger_type,
                "severity": t.trigger.severity,
                "conclusion": t.conclusion,
                "updates": len(t.cognitive_updates),
                "duration_ms": round(t.duration_ms, 1),
            }
            for t in self._traces[-limit:]
        ]
