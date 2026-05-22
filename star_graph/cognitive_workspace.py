"""Cognitive Workspace — the core of Phase 6 Cognitive Cortex.

NOT a passive buffer. This is where reasoning happens. The workspace:
- Admits PerceptionFrames and activated memories
- Maintains active thought items with priority-based decay
- Runs reasoning chains (step-by-step inference tied to goals)
- Manages cognitive load (capacity-limited, eviction of low-priority items)
- Exports WorkspaceState for SelfModel → LLM prompt injection

Core philosophy: LLM reasoning happens HERE, not in long-term memory.
The workspace is the "conscious" layer of the cognitive cortex.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .thought_object import ThoughtObject, new_thought


# ── Workspace Item ─────────────────────────────────────────────


@dataclass
class WorkspaceItem:
    """A single item in the cognitive workspace — richer than WorkingMemoryEntry.

    Each item wraps a ThoughtObject with workspace-specific metadata:
    relevance to current task, contribution to reasoning, attention weight.
    """

    id: str
    thought: ThoughtObject
    source: str = "perception"           # perception | recall | inference | goal | concept
    relevance_to_current_task: float = 0.5
    attention_weight: float = 0.0        # allocated by SalienceEngine
    related_items: list[str] = field(default_factory=list)  # IDs of related workspace items
    contribution_to_reasoning: float = 0.0  # cumulative contribution score
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    @property
    def activation_energy(self) -> float:
        return self.thought.activation_energy

    @property
    def priority(self) -> float:
        return self.thought.priority

    @property
    def is_active(self) -> bool:
        return self.thought.is_active

    def activate(self, energy: float = 0.3):
        self.thought.activate(energy)
        self.last_accessed_at = time.time()
        self.access_count += 1

    def decay(self, dt: float):
        self.thought.decay(dt)

    def relate(self, other_id: str):
        if other_id not in self.related_items:
            self.related_items.append(other_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thought": self.thought.to_dict(),
            "source": self.source,
            "relevance": round(self.relevance_to_current_task, 3),
            "attention_weight": round(self.attention_weight, 3),
            "reasoning_contribution": round(self.contribution_to_reasoning, 3),
            "related_count": len(self.related_items),
            "access_count": self.access_count,
        }


# ── Reasoning Structures ────────────────────────────────────────


@dataclass
class ReasoningStep:
    """One step in a reasoning chain."""
    step_id: str
    thought_ids: list[str]           # workspace item IDs involved
    operation: str                   # deduce | compare | generalize | question | resolve | observe
    conclusion: str                  # what was concluded in this step
    confidence: float                # 0..1
    evidence_ids: list[str] = field(default_factory=list)  # anchor IDs supporting this step
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "thought_ids": self.thought_ids,
            "operation": self.operation,
            "conclusion": self.conclusion[:300],
            "confidence": round(self.confidence, 3),
            "evidence_count": len(self.evidence_ids),
            "timestamp": self.timestamp,
        }


@dataclass
class ReasoningChain:
    """A chain of reasoning steps, tied to a goal.

    Represents one line of thought — e.g. "debug memory leak" or "plan refactor".
    Multiple chains can coexist; SalienceEngine decides which gets attention.
    """

    id: str
    goal_id: str = ""                    # which goal this reasoning serves
    description: str = ""                # what the chain is trying to figure out
    steps: list[ReasoningStep] = field(default_factory=list)
    confidence: float = 0.5
    state: str = "active"               # active | stalled | resolved | abandoned
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    resolution: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ReasoningStep) -> int:
        self.steps.append(step)
        self.updated_at = time.time()
        return len(self.steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def avg_confidence(self) -> float:
        if not self.steps:
            return self.confidence
        return sum(s.confidence for s in self.steps) / len(self.steps)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "step_count": len(self.steps),
            "confidence": round(self.confidence, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "state": self.state,
            "resolution": self.resolution,
        }


# ── Workspace State ─────────────────────────────────────────────


@dataclass
class WorkspaceState:
    """Snapshot of the entire workspace at a point in time."""
    active_items: list[dict] = field(default_factory=list)
    reasoning_chains: list[dict] = field(default_factory=list)
    current_focus: str = ""              # ID of most attended item
    cognitive_load: float = 0.0          # 0..1
    pending_conflicts: list[dict] = field(default_factory=list)
    emotional_tone: float = 0.0          # -1..+1 aggregated valence
    item_count: int = 0
    active_chain_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "active_items": self.active_items,
            "reasoning_chains": self.reasoning_chains,
            "current_focus": self.current_focus,
            "cognitive_load": round(self.cognitive_load, 3),
            "pending_conflicts": self.pending_conflicts,
            "emotional_tone": round(self.emotional_tone, 3),
            "item_count": self.item_count,
            "active_chain_count": self.active_chain_count,
            "timestamp": self.timestamp,
        }


# ── Cognitive Workspace ─────────────────────────────────────────


class CognitiveWorkspace:
    """Active cognitive workspace — the core of NWC Phase 6.

    This is where reasoning happens. The workspace admits perception frames,
    activates long-term memories, runs reasoning chains tied to goals, and
    exports cognitive state for LLM prompt injection.

    Capacity is limited (default 20 items). On overflow, lowest-priority
    items are evicted. Items decay over time; high-priority items decay slower.
    """

    def __init__(self,
                 max_items: int = 20,
                 default_ttl: float = 300.0,
                 config: dict[str, Any] | None = None,
                 graph=None):
        self.max_items = max_items
        self.default_ttl = default_ttl
        self._config = config or {}

        self._items: dict[str, WorkspaceItem] = {}
        self._chains: dict[str, ReasoningChain] = {}
        self._focus_id: str = ""
        self._emotional_tone: float = 0.0   # EMA of admitted item valences

        # Decay parameters from config
        self._decay_high = self._config.get("decay_lambda_high_priority", 0.001)
        self._decay_med = self._config.get("decay_lambda_medium_priority", 0.01)
        self._decay_low = self._config.get("decay_lambda_low_priority", 0.05)
        self._focus_boost = self._config.get("focus_boost_factor", 1.5)
        self._diffusion_decay = self._config.get("diffusion_decay_per_hop", 0.3)
        self._max_chains = self._config.get("max_reasoning_chains", 5)
        self._max_steps_per_chain = self._config.get("max_steps_per_chain", 20)

        # Optional graph reference for LTM activation
        self.graph = graph

    # ── Core Operations ────────────────────────────────────

    def admit_thought(self, thought: ThoughtObject, *,
                      source: str = "perception",
                      relevance: float = 0.5) -> WorkspaceItem:
        """Admit a ThoughtObject into the workspace."""
        self._decay_all(time.time())

        item = WorkspaceItem(
            id=str(uuid.uuid4()),
            thought=thought,
            source=source,
            relevance_to_current_task=relevance,
        )

        # Enforce capacity
        if len(self._items) >= self.max_items:
            self._evict_one()

        self._items[item.id] = item

        # Update emotional tone (EMA)
        if thought.emotional_valence != 0.0:
            alpha = 0.3
            self._emotional_tone = (alpha * thought.emotional_valence +
                                    (1 - alpha) * self._emotional_tone)

        if not self._focus_id:
            self._focus_id = item.id

        return item

    def admit_text(self, text: str, *,
                   source: str = "perception",
                   relevance: float = 0.5,
                   priority: float = 0.5,
                   emotional_valence: float = 0.0) -> WorkspaceItem:
        """Convenience: admit raw text as a ThoughtObject."""
        thought = new_thought(
            text, thought_type="thought",
            activation=0.3, priority=priority,
            emotional_valence=emotional_valence,
            ttl=self.default_ttl,
        )
        return self.admit_thought(thought, source=source, relevance=relevance)

    def focus(self, item_id: str) -> WorkspaceItem | None:
        """Set attention focus on a specific item."""
        if item_id not in self._items:
            return None
        self._focus_id = item_id
        item = self._items[item_id]
        item.activate(0.3)
        item.attention_weight = min(1.0, item.attention_weight + 0.2)
        return item

    def diffuse_attention(self):
        """Spread activation from focus item to related items."""
        if not self._focus_id or self._focus_id not in self._items:
            return

        focus_item = self._items[self._focus_id]
        focus_energy = focus_item.thought.activation_energy

        # Find related items via shared tags or explicit relations
        for item_id, item in self._items.items():
            if item_id == self._focus_id:
                continue
            # Compute relatedness
            related_score = self._compute_relatedness(focus_item, item)
            if related_score > 0.1:
                boost = focus_energy * self._diffusion_decay * related_score
                item.activate(boost)

    def _compute_relatedness(self, a: WorkspaceItem, b: WorkspaceItem) -> float:
        """Compute how related two workspace items are (shared tags, relations)."""
        score = 0.0
        # Explicit relations
        if b.id in a.related_items or a.id in b.related_items:
            score += 0.5
        # Shared tags
        a_tags = set(a.thought.tags)
        b_tags = set(b.thought.tags)
        if a_tags and b_tags:
            overlap = len(a_tags & b_tags) / max(len(a_tags | b_tags), 1)
            score += overlap * 0.3
        # Shared goal relations
        a_goals = set(a.thought.goal_relation)
        b_goals = set(b.thought.goal_relation)
        if a_goals and b_goals:
            overlap = len(a_goals & b_goals) / max(len(a_goals | b_goals), 1)
            score += overlap * 0.2
        return min(1.0, score)

    # ── Reasoning ──────────────────────────────────────────

    def start_reasoning(self, goal_id: str = "",
                        description: str = "",
                        initial_thoughts: list[str] | None = None) -> ReasoningChain:
        """Begin a new reasoning chain."""
        # Enforce chain limit
        active_chains = [c for c in self._chains.values() if c.state == "active"]
        if len(active_chains) >= self._max_chains:
            # Abandon oldest stalled chain
            oldest = min(active_chains, key=lambda c: c.updated_at)
            oldest.state = "abandoned"

        chain = ReasoningChain(
            id=str(uuid.uuid4()),
            goal_id=goal_id,
            description=description,
        )
        if initial_thoughts:
            for text in initial_thoughts:
                thought = new_thought(text, thought_type="reasoning_step",
                                      activation=0.4, ttl=self.default_ttl)
                item = self.admit_thought(thought, source="inference")
                step = ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    thought_ids=[item.id],
                    operation="observe",
                    conclusion=text,
                    confidence=0.5,
                )
                chain.add_step(step)
                chain.confidence = chain.avg_confidence

        self._chains[chain.id] = chain
        return chain

    def add_reasoning_step(self, chain_id: str,
                           operation: str,
                           conclusion: str,
                           thought_ids: list[str] | None = None,
                           confidence: float = 0.5,
                           evidence_ids: list[str] | None = None) -> ReasoningStep | None:
        """Add a step to an existing reasoning chain."""
        if chain_id not in self._chains:
            return None

        chain = self._chains[chain_id]
        if chain.step_count >= self._max_steps_per_chain:
            chain.state = "stalled"
            return None

        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            thought_ids=thought_ids or [],
            operation=operation,
            conclusion=conclusion,
            confidence=confidence,
            evidence_ids=evidence_ids or [],
        )
        chain.add_step(step)
        chain.confidence = chain.avg_confidence

        # Mark involved items as contributing to reasoning
        for tid in step.thought_ids:
            if tid in self._items:
                self._items[tid].contribution_to_reasoning += confidence

        return step

    def resolve_chain(self, chain_id: str, resolution: str = "",
                      success: bool = True):
        """Mark a reasoning chain as resolved."""
        if chain_id not in self._chains:
            return
        chain = self._chains[chain_id]
        chain.state = "resolved" if success else "abandoned"
        chain.resolution = resolution
        chain.updated_at = time.time()

    # ── Eviction & Decay ───────────────────────────────────

    def _decay_all(self, now: float):
        """Apply temporal decay to all items."""
        for item_id, item in list(self._items.items()):
            # Choose decay rate based on priority
            p = item.priority
            if p > 0.7:
                rate = self._decay_high
            elif p > 0.3:
                rate = self._decay_med
            else:
                rate = self._decay_low

            # Focus boost: if this item is the focus, decay slower
            if item_id == self._focus_id:
                rate /= self._focus_boost

            # Compute dt since last access
            dt = now - item.last_accessed_at
            item.decay(dt)

            # Apply rate-based additional decay
            item.thought.decay(dt, decay_rate=rate)

            # Remove fully decayed items
            if item.thought.activation_energy < 0.01 or item.thought.ttl <= 0:
                del self._items[item_id]
                if self._focus_id == item_id:
                    self._focus_id = ""

    def _evict_one(self):
        """Evict the lowest-priority item."""
        if not self._items:
            return

        # Score for eviction: lower = more evictable
        def eviction_score(item: WorkspaceItem) -> float:
            return (
                item.thought.activation_energy * 0.3 +
                item.priority * 0.25 +
                item.relevance_to_current_task * 0.2 +
                item.attention_weight * 0.15 +
                (1.0 / (1.0 + item.access_count)) * 0.1
            )

        target = min(self._items.values(), key=eviction_score)
        del self._items[target.id]
        if self._focus_id == target.id:
            self._focus_id = ""

    def remove(self, item_id: str):
        """Explicitly remove an item."""
        self._items.pop(item_id, None)
        if self._focus_id == item_id:
            self._focus_id = ""

    # ── State Export ───────────────────────────────────────

    def snapshot(self) -> WorkspaceState:
        """Full workspace snapshot for SelfModel."""
        now = time.time()
        self._decay_all(now)

        active_items = []
        for item in self._items.values():
            d = item.to_dict()
            active_items.append(d)

        chains_data = [c.to_dict() for c in self._chains.values()
                       if c.state == "active"]

        conflict_items = self._detect_conflicts()

        return WorkspaceState(
            active_items=active_items,
            reasoning_chains=chains_data,
            current_focus=self._focus_id,
            cognitive_load=self.cognitive_load,
            pending_conflicts=conflict_items,
            emotional_tone=round(self._emotional_tone, 3),
            item_count=len(self._items),
            active_chain_count=sum(1 for c in self._chains.values() if c.state == "active"),
        )

    def _detect_conflicts(self) -> list[dict]:
        """Detect conflicting items in the workspace (simple cosine tag overlap)."""
        conflicts = []
        items_list = list(self._items.values())
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                a, b = items_list[i], items_list[j]
                # Conflict: opposing emotional valence with overlapping content
                if (abs(a.thought.emotional_valence - b.thought.emotional_valence) > 1.0 and
                        self._compute_relatedness(a, b) > 0.3):
                    conflicts.append({
                        "item_a": a.id,
                        "item_b": b.id,
                        "a_text": a.thought.content[:80],
                        "b_text": b.thought.content[:80],
                        "valence_diff": round(abs(a.thought.emotional_valence - b.thought.emotional_valence), 2),
                    })
        return conflicts[:5]

    @property
    def cognitive_load(self) -> float:
        """0..1 how full the workspace is."""
        if self.max_items == 0:
            return 1.0
        return min(1.0, len(self._items) / self.max_items)

    # ── Summary for LLM ────────────────────────────────────

    def get_reasoning_summary(self, max_items: int = 8) -> str:
        """Compressed summary of current workspace state for LLM prompt injection."""
        self._decay_all(time.time())

        parts = []
        # Active focus
        if self._focus_id and self._focus_id in self._items:
            focus = self._items[self._focus_id]
            parts.append(f"Current focus: {focus.thought.content[:200]}")

        # Top items by attention
        sorted_items = sorted(self._items.values(),
                              key=lambda i: i.attention_weight, reverse=True)
        if sorted_items:
            parts.append("Active thoughts:")
            for item in sorted_items[:max_items]:
                parts.append(f"  [{item.source}] {item.thought.content[:150]}")

        # Active reasoning chains
        active_chains = [c for c in self._chains.values() if c.state == "active"]
        if active_chains:
            parts.append("Active reasoning:")
            for chain in active_chains[:3]:
                parts.append(f"  {chain.description}: {chain.step_count} steps, "
                             f"confidence={chain.avg_confidence:.2f}")

        # Emotional tone
        if abs(self._emotional_tone) > 0.1:
            tone = "positive" if self._emotional_tone > 0 else "negative"
            parts.append(f"Emotional tone: {tone} ({self._emotional_tone:.2f})")

        parts.append(f"Cognitive load: {self.cognitive_load:.2f}")
        return "\n".join(parts)

    def get_active_facts(self, max_items: int = 10) -> list[str]:
        """List most relevant active facts currently in workspace."""
        sorted_items = sorted(self._items.values(),
                              key=lambda i: i.relevance_to_current_task + i.attention_weight,
                              reverse=True)
        return [item.thought.content for item in sorted_items[:max_items]]

    # ── Integration Hooks ──────────────────────────────────

    def on_goal_updated(self, goal_id: str, new_status: str):
        """Called when GoalSystem updates a goal."""
        for item in self._items.values():
            if goal_id in item.thought.goal_relation:
                if new_status == "active":
                    item.activate(0.2)
                elif new_status in ("achieved", "abandoned"):
                    item.decay(60.0)  # fast decay when goal resolved

    def on_salience_change(self, item_id: str, new_salience: float):
        """Called when SalienceEngine changes salience of an item."""
        if item_id in self._items:
            self._items[item_id].attention_weight = new_salience

    def on_concept_activated(self, concept_id: str, activation: float):
        """Called when ConceptCortex activates a concept node."""
        for item in self._items.values():
            if concept_id in item.thought.tags:
                item.activate(activation * 0.3)

    def on_perception(self, frame) -> list[WorkspaceItem]:
        """Admit a PerceptionFrame (from 6.1) into the workspace."""
        items = []
        for concept in getattr(frame, 'extracted_concepts', []):
            item = self.admit_text(concept, source="perception",
                                   relevance=0.7, priority=0.6)
            items.append(item)
        for goal in getattr(frame, 'explicit_goals', []):
            item = self.admit_text(goal, source="perception",
                                   relevance=0.8, priority=0.8)
            items.append(item)
        for need in getattr(frame, 'implicit_needs', []):
            item = self.admit_text(need, source="perception",
                                   relevance=0.5, priority=0.4)
            items.append(item)
        return items

    # ── Persistence Helpers ────────────────────────────────

    def get_items(self) -> list[WorkspaceItem]:
        return list(self._items.values())

    def get_chains(self) -> list[ReasoningChain]:
        return list(self._chains.values())

    def get_item(self, item_id: str) -> WorkspaceItem | None:
        return self._items.get(item_id)

    def get_chain(self, chain_id: str) -> ReasoningChain | None:
        return self._chains.get(chain_id)

    def clear(self):
        self._items.clear()
        self._chains.clear()
        self._focus_id = ""
        self._emotional_tone = 0.0

    def __len__(self) -> int:
        return len(self._items)
