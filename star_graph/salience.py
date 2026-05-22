"""Salience Engine — attention competition for cognitive workspace.

Decides what enters the cognitive workspace through attention competition.
Implements: multi-component salience computation, winner-take-all competition,
cognitive load management, attention shift detection.

Core principle: 少激活 / 强关联 / 高权重 / 动态变化
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SalienceComponents:
    """Decomposed salience factors."""
    task_relevance: float = 0.0
    goal_alignment: float = 0.0
    emotional_intensity: float = 0.0
    novelty: float = 0.0
    recency: float = 0.0
    urgency: float = 0.0
    frequency: float = 0.0
    cognitive_priority: float = 0.0
    social_relevance: float = 0.0
    contradiction_flag: float = 0.0


_DEFAULT_COMPONENT_WEIGHTS = {
    "task_relevance": 0.25, "goal_alignment": 0.20,
    "emotional_intensity": 0.10, "novelty": 0.10,
    "recency": 0.10, "urgency": 0.10,
    "frequency": 0.05, "cognitive_priority": 0.05,
    "social_relevance": 0.03, "contradiction_flag": 0.02,
}


@dataclass
class SalienceSignal:
    """Computed salience for a cognitive entity."""
    entity_id: str
    entity_type: str = "memory"        # memory | concept | goal | thought | perception
    salience: float = 0.0              # 0..1 overall
    attention_weight: float = 0.0      # current attention allocation
    components: SalienceComponents = field(default_factory=SalienceComponents)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "salience": round(self.salience, 3),
            "attention_weight": round(self.attention_weight, 3),
        }


@dataclass
class AttentionFocus:
    """The current focus of attention."""
    primary_focus_id: str = ""
    secondary_foci: list[str] = field(default_factory=list)
    attention_distribution: dict[str, float] = field(default_factory=dict)
    cognitive_load: float = 0.0
    shift_timestamp: float = field(default_factory=time.time)


class SalienceEngine:
    """Decides what enters the cognitive workspace through attention competition."""

    def __init__(self, *,
                 config: dict[str, Any] | None = None,
                 priority_engine=None,
                 goal_system=None,
                 concept_cortex=None):
        self._config = config or {}
        self._priority_engine = priority_engine
        self._goal_system = goal_system
        self._concept_cortex = concept_cortex

        # Component weights
        cfg_weights = self._config.get("component_weights", {})
        self._weights = {**_DEFAULT_COMPONENT_WEIGHTS, **cfg_weights}

        self._max_slots = self._config.get("max_attention_slots", 10)
        self._decay_rate = self._config.get("attention_decay_rate", 0.05)
        self._focus_boost = self._config.get("focus_boost_factor", 2.0)
        self._load_warning = self._config.get("cognitive_load_warning", 0.8)
        self._load_critical = self._config.get("cognitive_load_critical", 0.95)
        self._inhibition_radius = self._config.get("inhibition_radius", 0.7)
        self._inhibition_strength = self._config.get("inhibition_strength", 0.3)
        self._shift_hysteresis = self._config.get("shift_hysteresis", 0.15)
        self._min_salience = self._config.get("min_salience_for_admission", 0.15)

        self._focus = AttentionFocus()
        self._signals: dict[str, SalienceSignal] = {}
        self._access_history: dict[str, int] = defaultdict(int)

    # ── Salience Computation ───────────────────────────────

    def compute_salience(self, entity_id: str, entity_type: str, *,
                         context: dict | None = None) -> SalienceSignal:
        """Compute multi-dimensional salience for one entity."""
        ctx = context or {}
        comp = SalienceComponents(
            task_relevance=ctx.get("task_relevance", 0.5),
            goal_alignment=ctx.get("goal_alignment", 0.0),
            emotional_intensity=ctx.get("emotional_intensity", 0.0),
            novelty=ctx.get("novelty", 0.3),
            recency=ctx.get("recency", 0.5),
            urgency=ctx.get("urgency", 0.0),
            frequency=ctx.get("frequency", 0.1),
            cognitive_priority=ctx.get("cognitive_priority", 0.5),
            social_relevance=ctx.get("social_relevance", 0.0),
            contradiction_flag=ctx.get("contradiction_flag", 0.0),
        )

        # Weighted sum
        salience = (
            comp.task_relevance * self._weights["task_relevance"] +
            comp.goal_alignment * self._weights["goal_alignment"] +
            comp.emotional_intensity * self._weights["emotional_intensity"] +
            comp.novelty * self._weights["novelty"] +
            comp.recency * self._weights["recency"] +
            comp.urgency * self._weights["urgency"] +
            comp.frequency * self._weights["frequency"] +
            comp.cognitive_priority * self._weights["cognitive_priority"] +
            comp.social_relevance * self._weights["social_relevance"] +
            comp.contradiction_flag * self._weights["contradiction_flag"]
        )
        salience = max(0.0, min(1.0, salience))

        # Frequency boost from access history
        access_count = self._access_history.get(entity_id, 0)
        salience += min(0.1, access_count * 0.01)

        signal = SalienceSignal(
            entity_id=entity_id,
            entity_type=entity_type,
            salience=salience,
            components=comp,
        )
        self._signals[entity_id] = signal
        return signal

    def compute_batch(self, entities: list[tuple[str, str, dict]]) -> list[SalienceSignal]:
        """Batch salience computation."""
        return [self.compute_salience(eid, etype, context=ctx)
                for eid, etype, ctx in entities]

    # ── Attention Competition ──────────────────────────────

    def compete(self, candidates: list[SalienceSignal],
                max_winners: int | None = None) -> list[SalienceSignal]:
        """Winner-take-all attention competition."""
        max_w = max_winners or self._max_slots
        if len(candidates) <= max_w:
            return candidates

        # Apply lateral inhibition
        self._apply_inhibition(candidates)

        # Sort by salience, take top
        candidates.sort(key=lambda s: s.salience, reverse=True)
        winners = candidates[:max_w]

        # Allocate attention weights (softmax of salience)
        if winners:
            s_vals = [w.salience for w in winners]
            s_max = max(s_vals)
            s_exp = [math.exp((s - s_max) * 3) for s in s_vals]  # temperature=3
            s_sum = sum(s_exp)
            for w, se in zip(winners, s_exp):
                w.attention_weight = se / s_sum

        return winners

    def get_attention_focus(self) -> AttentionFocus:
        """Current attention distribution."""
        return self._focus

    def shift_attention(self, new_focus_id: str, reason: str = ""):
        """Shift attention to a new primary focus."""
        if new_focus_id not in self._signals:
            return

        current_salience = 0.0
        if self._focus.primary_focus_id in self._signals:
            current_salience = self._signals[self._focus.primary_focus_id].salience

        new_salience = self._signals[new_focus_id].salience

        # Only shift if new salience exceeds old by hysteresis
        if new_salience > current_salience + self._shift_hysteresis:
            old_primary = self._focus.primary_focus_id
            if old_primary and old_primary in self._signals:
                if old_primary not in self._focus.secondary_foci:
                    self._focus.secondary_foci.append(old_primary)
                self._focus.secondary_foci = self._focus.secondary_foci[:3]

            self._focus.primary_focus_id = new_focus_id
            self._focus.shift_timestamp = time.time()

            # Boost new focus
            self._signals[new_focus_id].attention_weight *= self._focus_boost

    # ── Cognitive Load ─────────────────────────────────────

    @property
    def cognitive_load(self) -> float:
        """0..1 current cognitive load."""
        if not self._signals:
            return 0.0
        total_attention = sum(s.attention_weight for s in self._signals.values())
        return min(1.0, total_attention / self._max_slots)

    def can_admit(self, salience: float) -> bool:
        """Whether a new entity can enter attention given current load."""
        load = self.cognitive_load
        if load >= self._load_critical:
            return salience > 0.8  # only very high salience can enter
        if load >= self._load_warning:
            return salience > self._min_salience * 1.5
        return salience >= self._min_salience

    # ── Modulation ─────────────────────────────────────────

    def boost_goal_aligned(self, goal_id: str, multiplier: float = 1.5):
        for signal in self._signals.values():
            if goal_id in getattr(signal, 'goal_relation', []):
                signal.salience = min(1.0, signal.salience * multiplier)

    def decay_all(self, dt: float):
        """Decay salience of all signals."""
        for signal in self._signals.values():
            signal.salience *= math.exp(-self._decay_rate * dt)

        # Remove very low salience signals
        for eid in list(self._signals.keys()):
            if self._signals[eid].salience < 0.01:
                del self._signals[eid]

    # ── Internal ───────────────────────────────────────────

    def _apply_inhibition(self, candidates: list[SalienceSignal]):
        """Lateral inhibition between similar candidates."""
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                # Simple inhibition: if saliences are close, suppress lower
                diff = abs(a.salience - b.salience)
                if diff < self._inhibition_radius:
                    if a.salience >= b.salience:
                        b.salience *= (1.0 - self._inhibition_strength * (1.0 - diff))
                    else:
                        a.salience *= (1.0 - self._inhibition_strength * (1.0 - diff))

    def record_access(self, entity_id: str):
        """Record that an entity was accessed (for frequency tracking)."""
        self._access_history[entity_id] += 1
