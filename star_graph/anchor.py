"""Anchor point — the fundamental unit of star-graph memory.

Each anchor is a ≤200-char summary of a conversation turn or session,
augmented with a dynamic importance vector that evolves over time.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnchorVector:
    """Dynamic importance/state vector attached to each anchor.

    Not a semantic embedding (that's separate), but a set of scalar
    properties that govern memory dynamics: what gets kept, what fades,
    what connects to what.
    """

    importance: float = 0.5       # 0..1  content significance
    frequency: float = 0.0        # normalized activation count
    recency: float = 1.0          # 1.0 = just created, decays over time
    emotional_valence: float = 0.0  # -1..+1  negative/neutral/positive
    stability: float = 0.0        # 0..1  resistance to change (consolidated = stable)
    surprise: float = 0.5         # 0..1  how unexpected this memory was

    def to_list(self) -> list[float]:
        return [
            self.importance,
            self.frequency,
            self.recency,
            self.emotional_valence,
            self.stability,
            self.surprise,
        ]

    @classmethod
    def from_list(cls, values: list[float]) -> AnchorVector:
        return cls(
            importance=values[0],
            frequency=values[1],
            recency=values[2],
            emotional_valence=values[3],
            stability=values[4],
            surprise=values[5],
        )


@dataclass
class Anchor:
    """A single memory anchor point."""

    id: str
    text: str                        # ≤200 char summary
    vector: AnchorVector = field(default_factory=AnchorVector)
    embedding: Optional[list[float]] = None  # semantic embedding (lazy)
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)
    source_session: str = ""         # session ID this came from
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, text: str, source_session: str = "", **vec_kw) -> Anchor:
        """Create a new anchor from a summary."""
        anchor_id = hashlib.blake2b(
            (text + source_session).encode(), digest_size=8
        ).hexdigest()
        return cls(
            id=anchor_id,
            text=text[:200],
            vector=AnchorVector(**vec_kw),
            source_session=source_session,
        )

    def decay(self, elapsed_hours: float, half_life: float = 168.0) -> None:
        """Decay recency exponentially. Default half-life = 1 week."""
        self.vector.recency *= 0.5 ** (elapsed_hours / half_life)
        if self.vector.recency < 0.01:
            self.vector.recency = 0.01

    def activate(self) -> None:
        """Called when this anchor is retrieved/used."""
        self.vector.frequency = min(1.0, self.vector.frequency + 0.05)
        self.vector.recency = 1.0
        self.last_activated_at = time.time()

    @property
    def retention_score(self) -> float:
        """Composite score: should this memory be kept?

        High importance + high frequency + recent + stable = keep.
        Low on all fronts = candidate for pruning.
        """
        v = self.vector
        return (
            0.30 * v.importance
            + 0.25 * v.frequency
            + 0.25 * v.recency
            + 0.10 * v.stability
            + 0.10 * v.surprise
        )
