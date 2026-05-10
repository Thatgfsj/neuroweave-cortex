"""Anchor point — the fundamental unit of star-graph memory.

Each anchor is a ≤200-char summary of a conversation turn or session,
augmented with a dynamic importance vector that evolves over time.

v0.2 adds: oscillatory resonance properties, predictive coding fields,
reconsolidation support, ghost anchors for savings effect.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional


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
    """What this anchor predicts about future states.

    The predictive coding framework: every memory encodes not just the past
    but an implicit prediction about what comes next.
    """

    next_topic_embedding: Optional[list[float]] = None  # expected continuation
    emotional_tone: float = 0.0      # -1..+1 expected emotional context
    expected_duration: float = 10.0  # expected conversation duration (minutes)
    confidence: float = 0.5          # how confident is this prediction

    def error(self, actual: AnchorPrediction) -> float:
        """Compute prediction error against actual outcome."""
        errors = []
        # Topic error
        if self.next_topic_embedding and actual.next_topic_embedding:
            dot = sum(a * b for a, b in zip(
                self.next_topic_embedding, actual.next_topic_embedding))
            na = math.sqrt(sum(x**2 for x in self.next_topic_embedding))
            nb = math.sqrt(sum(x**2 for x in actual.next_topic_embedding))
            topic_err = 1.0 - (dot / (na * nb + 1e-8))
            errors.append(topic_err)
        # Tone error
        errors.append(abs(self.emotional_tone - actual.emotional_tone))
        return sum(errors) / len(errors) if errors else 0.0


@dataclass
class Oscillator:
    """Oscillatory properties for phase-locked resonance retrieval.

    Each anchor has a natural frequency and phase. When context provides
    a driving oscillation, anchors with matching frequencies phase-lock
    and fire together — forming a constellation through synchrony.
    """

    natural_frequency: float = 0.5   # 0.1..1.0  preferred oscillation frequency
    phase_offset: float = 0.0        # 0..2π     characteristic phase
    coupling_strength: float = 0.3   # 0..1      how strongly it couples to drivers
    damping: float = 0.1             # oscillation decay rate

    def resonance(self, driving_freq: float, driving_phase: float) -> float:
        """Compute resonance strength with a driving oscillation.

        Uses the Arnold tongue model: resonance when frequencies are close
        and phases align within a critical window.
        """
        freq_diff = abs(self.natural_frequency - driving_freq)
        phase_diff = abs(self.phase_offset - driving_phase)

        # Normalize phase difference to [0, π]
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff

        # Arnold tongue: frequency window narrows as coupling weakens
        critical_width = 0.15 * self.coupling_strength

        if freq_diff > critical_width:
            return 0.0

        # Within the tongue: strength depends on both frequency match and phase alignment
        freq_match = math.exp(-(freq_diff ** 2) / (2 * critical_width ** 2))
        phase_match = math.cos(phase_diff)

        return self.coupling_strength * freq_match * max(0.0, phase_match)


@dataclass
class Anchor:
    """A single memory anchor point with predictive and oscillatory properties."""

    id: str
    text: str                         # suggested ≤200 char summary (soft limit)
    vector: AnchorVector = field(default_factory=AnchorVector)
    embedding: Optional[list[float]] = None
    prediction: Optional[AnchorPrediction] = None
    oscillator: Oscillator = field(default_factory=Oscillator)
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)
    source_session: str = ""
    tags: list[str] = field(default_factory=list)
    schema_ref: Optional[str] = None  # ID of schema this instantiates
    replay_count: int = 0             # times replayed during sleep

    @classmethod
    def create(cls, text: str, source_session: str = "",
               embedding: list[float] | None = None,
               emotional_valence: float = 0.0,
               surprise: float = 0.5,
               tags: list[str] | None = None,
               importance: float = 0.5,
               **vec_kw) -> Anchor:
        """Create a new anchor from a summary."""
        anchor_id = hashlib.blake2b(
            (text + source_session).encode(), digest_size=8
        ).hexdigest()
        # Separate AnchorVector fields from Anchor fields
        vec_fields = {"importance", "frequency", "recency",
                      "emotional_valence", "stability", "surprise",
                      "hippocampal_dependency"}
        vec_kw.setdefault("importance", importance)
        vec_kw.setdefault("emotional_valence", emotional_valence)
        vec_kw.setdefault("surprise", surprise)
        vector_kw = {k: v for k, v in vec_kw.items() if k in vec_fields}
        anchor_kw = {k: v for k, v in vec_kw.items() if k not in vec_fields}
        # Derive meaningful oscillator params (not hash-based noise)
        try:
            from .embedding import get_embedder
            embedder = get_embedder()
            freq = embedder.derive_frequency(
                importance=vector_kw.get("importance", importance),
                emotional_valence=vector_kw.get("emotional_valence", emotional_valence),
                text_length=len(text),
            )
            # Auto-encode text if no embedding provided
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
            **anchor_kw,
        )

    # ── Dynamics ──────────────────────────────────────

    def decay(self, elapsed_hours: float, half_life: float = 168.0) -> None:
        self.vector.recency *= 0.5 ** (elapsed_hours / half_life)
        self.vector.recency = max(0.01, self.vector.recency)

    def activate(self) -> None:
        """Called when this anchor is retrieved/used.

        Activation triggers reconsolidation: stability temporarily drops
        (memory becomes labile), allowing update on next prediction check.
        """
        self.vector.frequency = min(1.0, self.vector.frequency + 0.05)
        self.vector.recency = 1.0
        self.vector.stability *= 0.7  # labile during reconsolidation window
        self.last_activated_at = time.time()

    def consolidate(self, prediction_error: float) -> str:
        """Reconsolidation based on prediction error.

        Returns: 'strengthen', 'update', or 'novel'
        """
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
        return (
            0.25 * v.importance
            + 0.20 * v.frequency
            + 0.20 * v.recency
            + 0.15 * v.stability
            + 0.10 * v.surprise
            + 0.10 * (1.0 - v.hippocampal_dependency)  # cortical = more retained
        )

    @property
    def is_cortical(self) -> bool:
        """Has this memory been consolidated to cortex?"""
        return self.vector.hippocampal_dependency < 0.3

    @property
    def is_labile(self) -> bool:
        """Is this memory currently in a reconsolidation window?"""
        return self.vector.stability < 0.4


@dataclass
class GhostAnchor:
    """A pruned anchor's residual trace — enables the savings effect.

    When an anchor is pruned, it leaves a ghost. If similar content appears
    later, the ghost resonates and enables faster relearning (savings).
    """

    id: str
    residue: list[float]     # partial embedding (kept small intentionally)
    original_tags: list[str]
    pruned_at: float
    revival_count: int = 0
    original_importance: float = 0.5

    @classmethod
    def from_anchor(cls, anchor: Anchor) -> GhostAnchor:
        """Create a ghost from a pruned anchor."""
        residue = anchor.embedding[:16] if anchor.embedding else []
        return cls(
            id=anchor.id,
            residue=residue,
            original_tags=list(anchor.tags),
            pruned_at=time.time(),
            original_importance=anchor.vector.importance,
        )

    def resonance(self, new_embedding: list[float] | None) -> float:
        """How strongly does this ghost resonate with new content?"""
        if not self.residue or not new_embedding:
            return 0.0
        residue = self.residue[:min(len(self.residue), len(new_embedding))]
        new = new_embedding[:len(residue)]
        dot = sum(r * n for r, n in zip(residue, new))
        nr = math.sqrt(sum(r**2 for r in residue))
        nn = math.sqrt(sum(n**2 for n in new))
        return dot / (nr * nn + 1e-8)

    def revive(self, new_text: str, new_embedding: list[float] | None = None) -> Anchor:
        """Create a revived anchor — faster than new learning."""
        self.revival_count += 1
        return Anchor(
            id=self.id,
            text=new_text[:280],  # soft limit, prefer ≤200
            vector=AnchorVector(
                importance=self.original_importance * 0.6 + 0.1 * self.revival_count,
                frequency=0.1,
                recency=1.0,
                stability=0.2,
                surprise=0.8,  # surprising to see this again
            ),
            embedding=new_embedding,
        )
