"""Self Model — maintain system cognitive state for LLM injection.

This is the primary interface between NWC and the LLM. Instead of sending
raw memory lists to the LLM, the SelfModel produces a compressed, structured
CognitiveState that the LLM can efficiently consume.

Key output: get_prompt_injection() — the natural-language prompt context
that replaces raw memory dumps.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ── Data Structures ──────────────────────────────────────────────


@dataclass
class SelfModelConfig:
    update_interval_seconds: float = 30.0
    max_uncertainties: int = 10
    max_bias_detection: int = 5
    emotional_smoothing_factor: float = 0.3
    max_goal_summary_count: int = 5
    prompt_format: str = "compact"     # compact | detailed | structured_json
    max_prompt_tokens: int = 2000


@dataclass
class CognitiveState:
    """The system's self-model at a point in time.

    This is what gets injected into the LLM prompt — NOT raw memories.
    """

    focus: list[str] = field(default_factory=list)
    primary_task: str = ""
    attention_allocation: dict[str, float] = field(default_factory=dict)

    goals: list[dict] = field(default_factory=list)
    active_goal_count: int = 0
    blocked_goals: list[str] = field(default_factory=list)

    uncertainties: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    conflicting_beliefs: list[dict] = field(default_factory=list)

    emotional_tone: float = 0.0
    cognitive_biases: list[str] = field(default_factory=list)
    confidence_level: float = 0.5

    active_concepts: list[str] = field(default_factory=list)
    dominant_concept: str = ""

    active_reasoning_chains: list[dict] = field(default_factory=list)
    reasoning_depth: int = 0
    pending_inferences: list[str] = field(default_factory=list)

    memory_anchors_accessible: int = 0
    recent_memories_quality: float = 0.5
    compression_ratio: float = 0.0

    timestamp: float = field(default_factory=time.time)
    cognitive_load: float = 0.0

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Compress self-model to a natural-language prompt injection."""
        parts = ["# Cognitive State Summary"]

        if self.primary_task:
            parts.append(f"## Current Focus: {self.primary_task}")
        if self.focus:
            parts.append(f"Focus areas: {', '.join(self.focus[:5])}")

        if self.goals:
            parts.append("## Active Goals:")
            for g in self.goals[:5]:
                progress = g.get('progress', 0)
                desc = g.get('description', str(g))[:100]
                parts.append(f"  - {desc} (progress: {progress:.0%})")

        if self.blocked_goals:
            parts.append(f"Blocked goals: {len(self.blocked_goals)}")

        if self.uncertainties:
            parts.append("## Uncertainties:")
            for u in self.uncertainties[:5]:
                parts.append(f"  - {u}")

        if abs(self.emotional_tone) > 0.1:
            tone = "positive" if self.emotional_tone > 0 else "negative"
            parts.append(f"## Emotional Tone: {tone} ({self.emotional_tone:.2f})")

        if self.active_concepts:
            parts.append(f"## Key Active Concepts: {', '.join(self.active_concepts[:8])}")

        if self.active_reasoning_chains:
            parts.append("## Active Reasoning:")
            for chain in self.active_reasoning_chains[:3]:
                desc = chain.get('description', '')[:80]
                steps = chain.get('step_count', 0)
                conf = chain.get('confidence', 0)
                parts.append(f"  - {desc}: {steps} steps, confidence={conf:.2f}")

        if self.cognitive_biases:
            parts.append(f"## Detected Biases: {', '.join(self.cognitive_biases[:3])}")

        parts.append(f"## Cognitive Load: {self.cognitive_load:.2f}")

        result = "\n".join(parts)
        # Rough token estimate: words * 1.3
        estimated_tokens = len(result.split()) * 1.3
        if estimated_tokens > max_tokens:
            # Truncate: keep only the first sections
            lines = result.split("\n")
            truncated = []
            token_count = 0
            for line in lines:
                line_tokens = len(line.split()) * 1.3
                if token_count + line_tokens > max_tokens:
                    break
                truncated.append(line)
                token_count += line_tokens
            result = "\n".join(truncated)

        return result

    def to_dict(self) -> dict:
        return {
            "focus": self.focus[:5],
            "primary_task": self.primary_task,
            "active_goal_count": self.active_goal_count,
            "uncertainties": self.uncertainties[:5],
            "emotional_tone": round(self.emotional_tone, 3),
            "active_concepts": self.active_concepts[:8],
            "dominant_concept": self.dominant_concept,
            "cognitive_load": round(self.cognitive_load, 3),
            "confidence": round(self.confidence_level, 3),
            "biases": self.cognitive_biases[:3],
            "reasoning_depth": self.reasoning_depth,
            "timestamp": self.timestamp,
        }


@dataclass
class BiasDetection:
    """Detected cognitive bias in the system's reasoning."""
    bias_type: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    suggestion: str = ""
    detected_at: float = field(default_factory=time.time)


# ── Self Model ──────────────────────────────────────────────────


class SelfModel:
    """Maintains the system's self-model — what it 'knows' about its own state.

    Reads from all Phase 6 subsystems to construct a coherent CognitiveState.
    """

    def __init__(self, *,
                 config: SelfModelConfig | None = None,
                 workspace=None,
                 goal_system=None,
                 concept_cortex=None,
                 salience_engine=None,
                 graph=None):
        self._config = config or SelfModelConfig()
        self._workspace = workspace
        self._goal_system = goal_system
        self._concept_cortex = concept_cortex
        self._salience_engine = salience_engine
        self.graph = graph

        self._emotional_tone_ema: float = 0.0
        self._state_history: list[CognitiveState] = []
        self._max_history = getattr(config, 'max_state_history', 100) if config else 100

    # ── State Construction ─────────────────────────────────

    def construct_state(self) -> CognitiveState:
        """Build complete cognitive state from all subsystems."""
        state = CognitiveState()

        # From workspace
        if self._workspace:
            try:
                snap = self._workspace.snapshot()
                state.cognitive_load = snap.cognitive_load

                # Extract focus items
                for item in snap.active_items[:5]:
                    if isinstance(item, dict):
                        thought = item.get('thought', {})
                        if isinstance(thought, dict):
                            state.focus.append(thought.get('content', '')[:120])

                # Reasoning chains
                for chain in snap.reasoning_chains[:3]:
                    if isinstance(chain, dict):
                        state.active_reasoning_chains.append(chain)
                        state.reasoning_depth = max(
                            state.reasoning_depth,
                            chain.get('step_count', 0)
                        )

                state.emotional_tone = snap.emotional_tone
            except Exception:
                pass

        # From goal system
        if self._goal_system:
            try:
                active_goals = self._goal_system.get_active_goals()
                state.active_goal_count = len(active_goals)
                state.goals = [g.to_dict() for g in active_goals[:5]]
                state.blocked_goals = [
                    g.description for g in self._goal_system.get_blocked_goals()
                ]
                summary = self._goal_system.summarize_goals()
            except Exception:
                pass

        # From concept cortex
        if self._concept_cortex:
            try:
                active = self._concept_cortex.get_active_concepts()
                state.active_concepts = [c.label for c in active[:8]]
                if active:
                    state.dominant_concept = max(active, key=lambda c: c.activation).label
            except Exception:
                pass

        # From salience engine
        if self._salience_engine:
            try:
                focus = self._salience_engine.get_attention_focus()
                state.attention_allocation = focus.attention_distribution
                if focus.primary_focus_id:
                    state.primary_task = focus.primary_focus_id
            except Exception:
                pass

        # From graph
        if self.graph:
            try:
                state.memory_anchors_accessible = getattr(
                    self.graph, 'anchor_count', 0
                )
            except Exception:
                pass

        # EMA smooth emotional tone
        alpha = self._config.emotional_smoothing_factor
        self._emotional_tone_ema = (alpha * state.emotional_tone +
                                    (1 - alpha) * self._emotional_tone_ema)
        state.emotional_tone = self._emotional_tone_ema

        # Detect biases
        biases = self._detect_biases()
        state.cognitive_biases = [b.bias_type for b in biases if b.confidence > 0.5]

        # Store history
        self._state_history.append(state)
        if len(self._state_history) > self._max_history:
            self._state_history = self._state_history[-self._max_history:]

        return state

    # ── Prompt Injection ───────────────────────────────────

    def get_prompt_injection(self, max_tokens: int = 2000) -> str:
        """Get natural-language prompt injection for the LLM.

        This is the KEY output of NWC Phase 6 — what the LLM actually sees.
        """
        state = self.construct_state()
        return state.to_prompt_context(max_tokens=max_tokens)

    def get_structured_state(self) -> dict:
        """Get full structured state for API/MCP consumption."""
        state = self.construct_state()
        return state.to_dict()

    # ── Bias Detection ─────────────────────────────────────

    def _detect_biases(self) -> list[BiasDetection]:
        """Detect cognitive biases in current reasoning patterns."""
        biases: list[BiasDetection] = []

        # Recency bias: check if recent items dominate attention
        if self._salience_engine:
            focus = self._salience_engine.get_attention_focus()
            if focus.primary_focus_id:
                # If attention hasn't shifted in a while, possible anchoring
                age = time.time() - focus.shift_timestamp
                if age > 600:  # 10 min
                    biases.append(BiasDetection(
                        bias_type="anchoring",
                        confidence=min(0.8, age / 1800),
                        evidence=[f"Attention unchanged for {age:.0f}s"],
                        suggestion="Consider exploring alternative perspectives",
                    ))

        # Confirmation bias: check if workspace has conflicting items being ignored
        if self._workspace:
            try:
                snap = self._workspace.snapshot()
                if len(snap.pending_conflicts) > 2:
                    biases.append(BiasDetection(
                        bias_type="confirmation_bias",
                        confidence=min(0.7, len(snap.pending_conflicts) * 0.15),
                        evidence=[f"{len(snap.pending_conflicts)} unresolved conflicts"],
                        suggestion="Actively seek disconfirming evidence",
                    ))
            except Exception:
                pass

        return biases[:self._config.max_bias_detection]

    # ── History ────────────────────────────────────────────

    def get_state_history(self, limit: int = 20) -> list[dict]:
        """Retrieve historical cognitive states for drift analysis."""
        return [s.to_dict() for s in self._state_history[-limit:]]

    def get_emotional_drift(self) -> float:
        """How much emotional tone has shifted over recent history."""
        if len(self._state_history) < 5:
            return 0.0
        recent = [s.emotional_tone for s in self._state_history[-10:]]
        return max(recent) - min(recent)
