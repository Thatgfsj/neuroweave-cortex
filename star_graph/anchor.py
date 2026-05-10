"""Anchor point — the fundamental unit of star-graph memory.

Each anchor is a summary of a conversation turn or session, augmented with
a dynamic importance vector and a 6-state memory lifecycle.

v0.4 adds: MemoryState machine (ACTIVE→REHEARSING→CONSOLIDATING→DORMANT→GHOST→REACTIVATED),
meaningful oscillator derivation from embeddings, and interference-aware retention.
"""

from __future__ import annotations

import enum
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional


class MemoryState(enum.Enum):
    """Six-state memory lifecycle — not just 'stored' or 'deleted'.

    Each state governs retrieval behavior, update plasticity, and decay rate.
    """
    ACTIVE = "active"              # Just created or recently retrieved — full plasticity
    REHEARSING = "rehearsing"      # Being replayed during sleep SWR — temporarily elevated
    CONSOLIDATING = "consolidating"  # Undergoing hippocampal→cortical transfer
    DORMANT = "dormant"            # Stable, low-activity, cortical retrieval only
    GHOST = "ghost"                # Pruned but with residual trace (savings effect)
    REACTIVATED = "reactivated"    # Ghost revived — lower stability, elevated surprise


# State transition rules: (current, event) → next
# Events: 'create', 'replay', 'consolidate', 'retrieve', 'prune', 'revive', 'stabilize'
_TRANSITIONS = {
    (MemoryState.ACTIVE, 'replay'): MemoryState.REHEARSING,
    (MemoryState.ACTIVE, 'consolidate'): MemoryState.CONSOLIDATING,
    (MemoryState.ACTIVE, 'prune'): MemoryState.GHOST,
    (MemoryState.REHEARSING, 'consolidate'): MemoryState.CONSOLIDATING,
    (MemoryState.REHEARSING, 'retrieve'): MemoryState.ACTIVE,
    (MemoryState.CONSOLIDATING, 'stabilize'): MemoryState.DORMANT,
    (MemoryState.CONSOLIDATING, 'retrieve'): MemoryState.ACTIVE,
    (MemoryState.DORMANT, 'retrieve'): MemoryState.ACTIVE,
    (MemoryState.DORMANT, 'prune'): MemoryState.GHOST,
    (MemoryState.DORMANT, 'replay'): MemoryState.REHEARSING,
    (MemoryState.GHOST, 'revive'): MemoryState.REACTIVATED,
    (MemoryState.REACTIVATED, 'consolidate'): MemoryState.CONSOLIDATING,
    (MemoryState.REACTIVATED, 'stabilize'): MemoryState.DORMANT,
}


@dataclass
class AnchorVector:
    """Dynamic importance/state vector attached to each anchor.

    Governs memory dynamics: what gets kept, what fades, what connects to what.
    """

    importance: float = 0.5        # 0..1  content significance
    frequency: float = 0.0         # normalized activation count
    recency: float = 1.0           # 1.0 = just created, decays over time
    emotional_valence: float = 0.0 # -1..+1  negative/neutral/positive
    stability: float = 0.0         # 0..1  resistance to change (consolidated)
    surprise: float = 0.5          # 0..1  how unexpected this memory was
    hippocampal_dependency: float = 1.0  # 1.0 = fully hippocampal, 0 = cortical

    def to_list(self) -> list[float]:
        return [
            self.importance, self.frequency, self.recency,
            self.emotional_valence, self.stability, self.surprise,
            self.hippocampal_dependency,
        ]

    @classmethod
    def from_list(cls, values: list[float]) -> AnchorVector:
        defaults = [0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0]
        merged = values + defaults[len(values):]
        return cls(
            importance=merged[0], frequency=merged[1], recency=merged[2],
            emotional_valence=merged[3], stability=merged[4], surprise=merged[5],
            hippocampal_dependency=merged[6],
        )


@dataclass
class AnchorPrediction:
    """What this anchor predicts about future states."""

    next_topic_embedding: Optional[list[float]] = None
    emotional_tone: float = 0.0
    expected_duration: float = 10.0
    confidence: float = 0.5

    def error(self, actual: AnchorPrediction) -> float:
        errors = []
        if self.next_topic_embedding and actual.next_topic_embedding:
            dot = sum(a * b for a, b in zip(
                self.next_topic_embedding, actual.next_topic_embedding))
            na = math.sqrt(sum(x**2 for x in self.next_topic_embedding))
            nb = math.sqrt(sum(x**2 for x in actual.next_topic_embedding))
            topic_err = 1.0 - (dot / (na * nb + 1e-8))
            errors.append(topic_err)
        errors.append(abs(self.emotional_tone - actual.emotional_tone))
        return sum(errors) / len(errors) if errors else 0.0


@dataclass
class Oscillator:
    """Oscillatory properties for phase-locked resonance retrieval."""

    natural_frequency: float = 0.5   # 0.1..1.0
    phase_offset: float = 0.0        # 0..2π
    coupling_strength: float = 0.3   # 0..1
    damping: float = 0.1

    def resonance(self, driving_freq: float, driving_phase: float) -> float:
        freq_diff = abs(self.natural_frequency - driving_freq)
        phase_diff = abs(self.phase_offset - driving_phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        critical_width = 0.15 * self.coupling_strength
        if freq_diff > critical_width:
            return 0.0
        freq_match = math.exp(-(freq_diff ** 2) / (2 * critical_width ** 2))
        phase_match = math.cos(phase_diff)
        return self.coupling_strength * freq_match * max(0.0, phase_match)


@dataclass
class Anchor:
    """A single memory anchor point with state machine and oscillatory properties."""

    id: str
    text: str
    vector: AnchorVector = field(default_factory=AnchorVector)
    embedding: Optional[list[float]] = None
    prediction: Optional[AnchorPrediction] = None
    oscillator: Oscillator = field(default_factory=Oscillator)
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)
    source_session: str = ""
    tags: list[str] = field(default_factory=list)
    schema_ref: Optional[str] = None
    replay_count: int = 0
    # v0.4: state machine
    state: MemoryState = MemoryState.ACTIVE
    state_history: list[tuple[MemoryState, float]] = field(default_factory=list)

    @classmethod
    def create(cls, text: str, source_session: str = "",
               embedding: list[float] | None = None,
               emotional_valence: float = 0.0,
               surprise: float = 0.5,
               tags: list[str] | None = None,
               importance: float = 0.5,
               **vec_kw) -> Anchor:
        anchor_id = hashlib.blake2b(
            (text + source_session).encode(), digest_size=8
        ).hexdigest()
        vec_fields = {"importance", "frequency", "recency",
                      "emotional_valence", "stability", "surprise",
                      "hippocampal_dependency"}
        vec_kw.setdefault("importance", importance)
        vec_kw.setdefault("emotional_valence", emotional_valence)
        vec_kw.setdefault("surprise", surprise)
        vector_kw = {k: v for k, v in vec_kw.items() if k in vec_fields}
        anchor_kw = {k: v for k, v in vec_kw.items() if k not in vec_fields}

        # Derive meaningful oscillator params from real signal
        try:
            from .embedding import get_embedder
            embedder = get_embedder()
            freq = embedder.derive_frequency(
                importance=vector_kw.get("importance", importance),
                emotional_valence=vector_kw.get("emotional_valence", emotional_valence),
                text_length=len(text),
            )
            if embedding is None:
                embedding = embedder.encode(text)
            phase = embedder.derive_phase(
                text, embedding,
                importance=vector_kw.get("importance", importance),
                emotional_valence=vector_kw.get("emotional_valence", emotional_valence),
                timestamp=vec_kw.pop("_timestamp", None),
            )
            coupling = 0.25 + 0.15 * abs(vector_kw.get("emotional_valence", emotional_valence))
        except Exception:
            import math as _math
            freq = 0.5
            phase = _math.fmod(abs(hash(text)) * 0.001, 2 * _math.pi)
            coupling = 0.3

        return cls(
            id=anchor_id,
            text=text[:280],
            vector=AnchorVector(**vector_kw),
            embedding=embedding,
            oscillator=Oscillator(
                natural_frequency=freq,
                phase_offset=phase,
                coupling_strength=min(1.0, coupling),
            ),
            source_session=source_session,
            tags=tags or [],
            state=MemoryState.ACTIVE,
            state_history=[(MemoryState.ACTIVE, time.time())],
            **anchor_kw,
        )

    # ── State machine ──────────────────────────────────

    def transition(self, event: str) -> MemoryState:
        """Attempt a state transition. Returns new state (no-op if invalid)."""
        key = (self.state, event)
        new_state = _TRANSITIONS.get(key)
        if new_state is not None:
            self.state = new_state
            self.state_history.append((new_state, time.time()))
            self._on_enter_state(new_state)
        return self.state

    def _on_enter_state(self, state: MemoryState) -> None:
        """State-entry side effects."""
        now = time.time()
        if state == MemoryState.REHEARSING:
            self.replay_count += 1
            self.last_activated_at = now
        elif state == MemoryState.CONSOLIDATING:
            self.vector.stability = min(1.0, self.vector.stability + 0.1)
        elif state == MemoryState.DORMANT:
            self.vector.stability = max(self.vector.stability, 0.7)
        elif state == MemoryState.GHOST:
            pass  # GhostAnchor handles this
        elif state == MemoryState.REACTIVATED:
            self.vector.stability = 0.2
            self.vector.surprise = 0.8
            self.vector.recency = 1.0
            self.last_activated_at = now

    @property
    def is_retrievable(self) -> bool:
        """Can this anchor be returned in retrieval results?"""
        return self.state not in (MemoryState.GHOST,)

    @property
    def is_plastic(self) -> bool:
        """Can this anchor be modified/updated?"""
        return self.state in (MemoryState.ACTIVE, MemoryState.REHEARSING, MemoryState.REACTIVATED)

    # ── Dynamics ──────────────────────────────────────

    def decay(self, elapsed_hours: float, half_life: float = 168.0) -> None:
        # State-dependent decay: ghost decays faster, dormant slower
        if self.state == MemoryState.GHOST:
            half_life *= 0.3
        elif self.state == MemoryState.DORMANT:
            half_life *= 1.5
        elif self.state == MemoryState.REACTIVATED:
            half_life *= 0.7
        self.vector.recency *= 0.5 ** (elapsed_hours / half_life)
        self.vector.recency = max(0.01, self.vector.recency)

    def activate(self) -> None:
        """Called when this anchor is retrieved/used — triggers reconsolidation."""
        self.vector.frequency = min(1.0, self.vector.frequency + 0.05)
        self.vector.recency = 1.0
        self.vector.stability *= 0.7  # labile during reconsolidation window
        self.last_activated_at = time.time()
        # Retrieve from dormant → active
        if self.state == MemoryState.DORMANT:
            self.transition('retrieve')

    def consolidate(self, prediction_error: float) -> str:
        STRENGTHEN = 0.15
        UPDATE = 0.50

        if prediction_error < STRENGTHEN:
            self.vector.importance = min(1.0, self.vector.importance + 0.10)
            self.vector.stability = min(1.0, self.vector.stability + 0.30)
            return "strengthen"
        elif prediction_error < UPDATE:
            self.vector.stability = max(0.0, self.vector.stability - 0.15)
            self.vector.surprise = (self.vector.surprise + prediction_error) / 2
            return "update"
        else:
            return "novel"

    # ── Properties ────────────────────────────────────

    @property
    def retention_score(self) -> float:
        v = self.vector
        base = (
            0.25 * v.importance
            + 0.20 * v.frequency
            + 0.20 * v.recency
            + 0.15 * v.stability
            + 0.10 * v.surprise
            + 0.10 * (1.0 - v.hippocampal_dependency)
        )
        # State-dependent modifier
        if self.state == MemoryState.REHEARSING:
            base *= 1.3  # boosted during replay
        elif self.state == MemoryState.GHOST:
            base *= 0.3  # ghost = weak
        elif self.state == MemoryState.REACTIVATED:
            base *= 0.7
        return max(0.0, min(1.0, base))

    @property
    def is_cortical(self) -> bool:
        return self.vector.hippocampal_dependency < 0.3

    @property
    def is_labile(self) -> bool:
        return self.vector.stability < 0.4


@dataclass
class GhostAnchor:
    """A pruned anchor's residual trace — enables the savings effect.

    When an anchor enters GHOST state, it leaves this residual. If similar
    content appears later, the ghost resonates and enables faster relearning.
    """

    id: str
    residue: list[float]
    original_tags: list[str]
    pruned_at: float
    revival_count: int = 0
    original_importance: float = 0.5

    @classmethod
    def from_anchor(cls, anchor: Anchor) -> GhostAnchor:
        residue = anchor.embedding[:16] if anchor.embedding else []
        anchor.transition('prune')  # ACTIVE/DORMANT → GHOST
        return cls(
            id=anchor.id,
            residue=residue,
            original_tags=list(anchor.tags),
            pruned_at=time.time(),
            original_importance=anchor.vector.importance,
        )

    def resonance(self, new_embedding: list[float] | None) -> float:
        if not self.residue or not new_embedding:
            return 0.0
        residue = self.residue[:min(len(self.residue), len(new_embedding))]
        new = new_embedding[:len(residue)]
        dot = sum(r * n for r, n in zip(residue, new))
        nr = math.sqrt(sum(r**2 for r in residue))
        nn = math.sqrt(sum(n**2 for n in new))
        return dot / (nr * nn + 1e-8)

    def revive(self, new_text: str, new_embedding: list[float] | None = None) -> Anchor:
        self.revival_count += 1
        anchor = Anchor(
            id=self.id,
            text=new_text[:280],
            vector=AnchorVector(
                importance=self.original_importance * 0.6 + 0.1 * self.revival_count,
                frequency=0.1,
                recency=1.0,
                stability=0.2,
                surprise=0.8,
            ),
            embedding=new_embedding,
            state=MemoryState.REACTIVATED,
        )
        anchor.transition('revive')  # Will try GHOST→REACTIVATED
        return anchor
