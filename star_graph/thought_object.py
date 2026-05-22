"""Thought Object — unified cognitive base class for Phase 6 Cognitive Cortex.

Replaces static memory nodes with activated cognitive entities.
Every Phase 6 module (concepts, goals, workspace items, beliefs, uncertainties)
is a ThoughtObject or contains one.

A ThoughtObject is a cognitive entity that:
- Has activation energy (0..1) — not just stored, but "alive"
- Can be related to goals
- Has a TTL — decays when not accessed
- Carries confidence, emotional valence, and lineage (derived_from)
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtObject:
    """Unified cognitive base class — replaces static memory nodes.

    A ThoughtObject is the fundamental cognitive unit in Phase 6.
    Every piece of information that enters the cognitive workspace,
    every concept, every goal, every belief is a ThoughtObject.

    Key properties:
    - activation_energy: how "lit up" this thought is right now (0=dormant, 1=fully active)
    - confidence: how confident the system is in this thought (0..1)
    - ttl: seconds until auto-decay kicks in
    - goal_relation: which goals this thought is relevant to
    - derived_from: lineage — what thoughts/concepts/memories generated this
    """

    id: str
    type: str = "thought"           # thought | concept | goal | belief | uncertainty | memory
    state: str = "dormant"          # dormant | activating | active | decaying | consolidated
    content: str = ""               # natural language representation
    embedding: list[float] | None = None
    confidence: float = 0.5         # 0..1 belief confidence
    activation_energy: float = 0.0  # 0..1 current activation level
    goal_relation: list[str] = field(default_factory=list)   # related goal IDs
    derived_from: list[str] = field(default_factory=list)    # source thought IDs (lineage)
    ttl: float = 300.0              # seconds until auto-decay
    priority: float = 0.5           # 0..1 cognitive priority
    emotional_valence: float = 0.0  # -1..+1
    emotional_arousal: float = 0.0  # 0..1 intensity
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)
    activation_count: int = 0

    # ── Activation dynamics ──────────────────────────────────

    _ACTIVATION_THRESHOLD: float = 0.1   # above this → "active"
    _DORMANT_THRESHOLD: float = 0.05     # below this → "dormant"

    def activate(self, energy: float = 0.5) -> float:
        """Add activation energy. Returns new activation_energy level.

        Activation resets TTL and updates access metadata.
        Energy is additive but capped at 1.0.
        """
        self.activation_energy = min(1.0, self.activation_energy + energy)
        self.last_activated_at = time.time()
        self.activation_count += 1
        self.ttl = max(self.ttl, 60.0)  # at least 60s of life after activation
        self._update_state()
        return self.activation_energy

    def decay(self, dt: float, *, decay_rate: float | None = None) -> float:
        """Apply exponential decay over dt seconds. Returns remaining energy.

        Decay rate is modulated by priority — high-priority thoughts decay slower.
        """
        if decay_rate is None:
            # Priority-modulated decay: high priority → slow decay
            protection = 1.0 - self.priority * 0.9  # 0.1..1.0
            decay_rate = 0.01 * protection
        self.activation_energy *= math.exp(-decay_rate * dt)
        self.activation_energy = max(0.0, min(1.0, self.activation_energy))
        self.ttl = max(0.0, self.ttl - dt)
        self._update_state()
        return self.activation_energy

    def _update_state(self):
        """Infer state from activation_energy."""
        if self.activation_energy >= self._ACTIVATION_THRESHOLD:
            self.state = "active"
        elif self.activation_energy >= self._DORMANT_THRESHOLD:
            self.state = "activating" if self.activation_count > 0 else "dormant"
        else:
            self.state = "dormant"

    # ── Properties ───────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state == "active"

    @property
    def is_dormant(self) -> bool:
        return self.state in ("dormant", "decaying", "consolidated")

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activated_at

    @property
    def activation_momentum(self) -> float:
        """How much 'momentum' this thought has — activation × recency × frequency."""
        recency = math.exp(-self.idle_seconds / 600)  # decay over 10 min
        frequency = math.log(1 + self.activation_count) * 0.1
        return self.activation_energy * 0.5 + recency * 0.3 + min(frequency, 0.2)

    # ── Relations ────────────────────────────────────────────

    def relate_to_goal(self, goal_id: str):
        """Link this thought to a goal."""
        if goal_id not in self.goal_relation:
            self.goal_relation.append(goal_id)

    def derive_from(self, parent_id: str):
        """Record lineage — this thought was derived from parent."""
        if parent_id not in self.derived_from:
            self.derived_from.append(parent_id)

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "state": self.state,
            "content": self.content[:200],
            "confidence": round(self.confidence, 3),
            "activation_energy": round(self.activation_energy, 3),
            "goal_relation": self.goal_relation[:10],
            "derived_from": self.derived_from[:10],
            "ttl": round(self.ttl, 1),
            "priority": round(self.priority, 3),
            "emotional_valence": round(self.emotional_valence, 3),
            "emotional_arousal": round(self.emotional_arousal, 3),
            "tags": self.tags,
            "activation_count": self.activation_count,
            "created_at": self.created_at,
            "last_activated_at": self.last_activated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThoughtObject:
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            type=d.get("type", "thought"),
            state=d.get("state", "dormant"),
            content=d.get("content", ""),
            confidence=d.get("confidence", 0.5),
            activation_energy=d.get("activation_energy", 0.0),
            goal_relation=d.get("goal_relation", []),
            derived_from=d.get("derived_from", []),
            ttl=d.get("ttl", 300.0),
            priority=d.get("priority", 0.5),
            emotional_valence=d.get("emotional_valence", 0.0),
            emotional_arousal=d.get("emotional_arousal", 0.0),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            last_activated_at=d.get("last_activated_at", time.time()),
            activation_count=d.get("activation_count", 0),
        )

    def __repr__(self) -> str:
        return (f"ThoughtObject(id={self.id[:12]}..., type={self.type}, "
                f"state={self.state}, activation={self.activation_energy:.2f})")


# ── Factory helpers ────────────────────────────────────────────


def new_thought(content: str, *,
                thought_type: str = "thought",
                confidence: float = 0.5,
                priority: float = 0.5,
                activation: float = 0.3,
                ttl: float = 300.0,
                emotional_valence: float = 0.0,
                emotional_arousal: float = 0.0,
                tags: list[str] | None = None,
                goal_ids: list[str] | None = None,
                derived_from: list[str] | None = None,
                metadata: dict[str, Any] | None = None) -> ThoughtObject:
    """Factory: create a new ThoughtObject with sensible defaults."""
    return ThoughtObject(
        id=str(uuid.uuid4()),
        type=thought_type,
        state="activating" if activation > 0.1 else "dormant",
        content=content,
        confidence=confidence,
        activation_energy=activation,
        goal_relation=goal_ids or [],
        derived_from=derived_from or [],
        ttl=ttl,
        priority=priority,
        emotional_valence=emotional_valence,
        emotional_arousal=emotional_arousal,
        tags=tags or [],
        metadata=metadata or {},
    )


def thought_from_anchor(anchor, *, activation: float = 0.3) -> ThoughtObject:
    """Bridge: create a ThoughtObject from an existing memory Anchor."""
    return ThoughtObject(
        id=anchor.id if hasattr(anchor, 'id') else str(uuid.uuid4()),
        type="memory",
        state="activating" if activation > 0.1 else "dormant",
        content=anchor.text if hasattr(anchor, 'text') else str(anchor),
        embedding=anchor.embedding if hasattr(anchor, 'embedding') else None,
        confidence=getattr(anchor, 'stability', 0.5),
        activation_energy=activation,
        priority=getattr(anchor, 'importance', 0.5),
        emotional_valence=getattr(anchor, 'emotional_valence', 0.0),
        tags=list(getattr(anchor, 'tags', [])),
        ttl=300.0,
    )
