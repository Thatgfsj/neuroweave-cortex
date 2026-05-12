"""Anchor point — the fundamental unit of star-graph memory. [Layer 1: Storage]

Each anchor is a summary of a conversation turn or session, augmented with
a dynamic importance vector and a 6-state memory lifecycle.

v0.4 adds: MemoryState machine (ACTIVE→REHEARSING→CONSOLIDATING→DORMANT→GHOST→REACTIVATED),
meaningful oscillator derivation from embeddings, and interference-aware retention.

Layer boundary: this module uses an embedder registry to avoid importing from
Layer 3 (embedding.py). Call embedder_registry.set_embedder() at startup.
"""

from __future__ import annotations

import enum
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional


class EmbedderRegistry:
    """Registry to avoid L1→L3 import. Layer 3 injects the embedder at startup."""

    _embedder = None

    @classmethod
    def set_embedder(cls, embedder) -> None:
        cls._embedder = embedder

    @classmethod
    def get_embedder(cls):
        if cls._embedder is None:
            from .embedding import get_embedder
            cls._embedder = get_embedder()
        return cls._embedder

    @classmethod
    def is_available(cls) -> bool:
        if cls._embedder is not None:
            return True
        try:
            from .embedding import get_embedder
            return True
        except Exception:
            return False


class ThermalState(enum.Enum):
    """Four-level thermal memory lifecycle — governs storage tier and retrieval cost.

    HOT  → high-frequency, full retrieval, highest priority
    WARM → low frequency but important, compressed summary available
    COLD → frozen, index-only, must be thawed for full retrieval
    DEAD → metadata/hash only, not retrievable (audit trail)

    Thermal state is DERIVED from retention_score and access patterns —
    it is not stored directly. This avoids dual-state-machine sync issues.
    """
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DEAD = "dead"


class MemoryState(enum.Enum):
    """Six-state memory lifecycle — not just 'stored' or 'deleted'.

    Each state governs retrieval behavior, update plasticity, and decay rate.

    Thermal overlay:
      HOT  ← ACTIVE, REHEARSING
      WARM ← CONSOLIDATING, DORMANT
      COLD ← DORMANT (low retention), GHOST (with partial recall)
      DEAD ← GHOST (fully decayed, metadata only)
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

    Core decay dimensions (multiplicative):
      retention = relevance × recency × frequency × success_feedback × confidence
    """

    importance: float = 0.5        # 0..1  content significance
    frequency: float = 0.0         # normalized activation count (0..1)
    recency: float = 1.0           # 1.0 = just created, decays over time
    emotional_valence: float = 0.0 # -1..+1  negative/neutral/positive
    stability: float = 0.0         # 0..1  resistance to change (consolidated)
    surprise: float = 0.5          # 0..1  how unexpected this memory was
    hippocampal_dependency: float = 1.0  # 1.0 = fully hippocampal, 0 = cortical
    # v0.5: multiplicative decay fields
    success_feedback: float = 0.5  # 0..1  how often this memory led to good outcomes
    confidence: float = 0.5        # 0..1  how reliable/repeatable this memory is
    novelty: float = 0.5           # 0..1  how new/unique this memory is (high = novel)
    task_relevance: float = 0.5    # 0..1  relevance to current/active tasks
    future_reusability: float = 0.5  # 0..1  estimated long-term value
    decay_rate: float = 0.01       # per-day natural decay coefficient

    def to_list(self) -> list[float]:
        return [
            self.importance, self.frequency, self.recency,
            self.emotional_valence, self.stability, self.surprise,
            self.hippocampal_dependency,
            self.success_feedback, self.confidence,
            self.novelty, self.task_relevance, self.future_reusability,
            self.decay_rate,
        ]

    @classmethod
    def from_list(cls, values: list[float]) -> AnchorVector:
        defaults = [0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.01]
        merged = values + defaults[len(values):]
        return cls(
            importance=merged[0], frequency=merged[1], recency=merged[2],
            emotional_valence=merged[3], stability=merged[4], surprise=merged[5],
            hippocampal_dependency=merged[6],
            success_feedback=merged[7], confidence=merged[8],
            novelty=merged[9], task_relevance=merged[10], future_reusability=merged[11],
            decay_rate=merged[12],
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

        # Derive meaningful oscillator params via embedder registry (no L1→L3 import)
        try:
            embedder = EmbedderRegistry.get_embedder()
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
        from .config import Config
        c = Config.get().anchor.state_entry
        now = time.time()
        if state == MemoryState.REHEARSING:
            self.replay_count += 1
            self.last_activated_at = now
        elif state == MemoryState.CONSOLIDATING:
            self.vector.stability = min(1.0, self.vector.stability + c.consolidating_stability_boost)
        elif state == MemoryState.DORMANT:
            self.vector.stability = max(self.vector.stability, c.dormant_min_stability)
        elif state == MemoryState.GHOST:
            pass  # GhostAnchor handles this
        elif state == MemoryState.REACTIVATED:
            self.vector.stability = c.reactivated_stability
            self.vector.surprise = c.reactivated_surprise
            self.vector.recency = 1.0
            self.last_activated_at = now

    @property
    def is_retrievable(self) -> bool:
        """Can this anchor be returned in retrieval results?

        Ghosts are not retrievable unless revived. DEAD memories are inaccessible.
        """
        if self.state == MemoryState.GHOST:
            return False
        if self.thermal_state == ThermalState.DEAD:
            return False
        return True

    @property
    def is_plastic(self) -> bool:
        """Can this anchor be modified/updated?"""
        return self.state in (MemoryState.ACTIVE, MemoryState.REHEARSING, MemoryState.REACTIVATED)

    # ── Dynamics ──────────────────────────────────────

    def decay(self, elapsed_hours: float, half_life: float | None = None) -> None:
        from .config import Config
        c = Config.get().anchor.decay
        if half_life is None:
            half_life = c.base_half_life_hours
        if self.state == MemoryState.GHOST:
            half_life *= c.ghost_half_life_factor
        elif self.state == MemoryState.DORMANT:
            half_life *= c.dormant_half_life_factor
        elif self.state == MemoryState.REACTIVATED:
            half_life *= c.reactivated_half_life_factor
        self.vector.recency *= 0.5 ** (elapsed_hours / half_life)
        self.vector.recency = max(0.01, self.vector.recency)

    def activate(self) -> None:
        """Called when this anchor is retrieved/used — triggers reconsolidation."""
        from .config import Config
        c = Config.get().anchor
        self.vector.frequency = min(1.0, self.vector.frequency + c.state_entry.activation_frequency_boost)
        self.vector.recency = 1.0
        self.vector.stability *= c.consolidation.reconsolidation_stability_factor
        self.last_activated_at = time.time()
        # Retrieve from dormant → active
        if self.state == MemoryState.DORMANT:
            self.transition('retrieve')

    def consolidate(self, prediction_error: float) -> str:
        from .config import Config
        c = Config.get().anchor.consolidation

        if prediction_error < c.strengthen_threshold:
            self.vector.importance = min(1.0, self.vector.importance + c.strengthen_importance_delta)
            self.vector.stability = min(1.0, self.vector.stability + c.strengthen_stability_delta)
            return "strengthen"
        elif prediction_error < c.update_threshold:
            self.vector.stability = max(0.0, self.vector.stability + c.update_stability_delta)
            self.vector.surprise = (self.vector.surprise + prediction_error) / 2
            return "update"
        else:
            return "novel"

    # ── Properties ────────────────────────────────────

    @property
    def relevance(self) -> float:
        """Composite relevance: importance × task_relevance × (1+|valence|×0.2)."""
        v = self.vector
        base = v.importance * max(v.task_relevance, 0.01)
        # Emotional memories get slight boost (0-20%)
        emotional_boost = 1.0 + abs(v.emotional_valence) * 0.2
        return max(0.01, min(1.0, base * emotional_boost))

    @property
    def decay_factor(self) -> float:
        """Natural decay: exponential based on age and decay_rate."""
        from .config import Config
        c = Config.get().anchor.retention
        hours_since = (time.time() - self.last_activated_at) / 3600
        days_since = hours_since / 24
        # Exponential decay with configurable half-life
        half_life_days = c.decay_half_life_days
        decay = math.exp(-days_since * math.log(2) / half_life_days)
        # Stability slows decay
        decay = decay + (1.0 - decay) * self.vector.stability * 0.5
        return max(0.01, decay)

    @property
    def importance_score(self) -> float:
        """Compute importance from multiple signals.

        importance = novelty × 0.25 + |emotional_weight| × 0.25
                   + task_relevance × 0.25 + future_reusability × 0.25
        """
        v = self.vector
        return max(0.01, min(1.0,
            v.novelty * 0.25
            + abs(v.emotional_valence) * 0.25
            + v.task_relevance * 0.25
            + v.future_reusability * 0.25
        ))

    @property
    def retention_score(self) -> float:
        """Multiplicative memory decay model using geometric mean.

        retention = (relevance × recency × frequency′ × success_feedback × confidence)^(1/5)

        Geometric mean preserves multiplicative interaction (a near-zero factor hurts)
        while keeping scores in a normal range. Compared to raw product, it doesn't
        collapse to near-zero just because one dimension is untested.
        """
        from .config import Config
        c = Config.get().anchor.retention
        v = self.vector

        factors = [
            max(0.01, self.relevance),
            max(0.01, v.recency),
            max(0.01, v.frequency + 0.01),
            max(0.01, v.success_feedback),
            max(0.01, v.confidence),
        ]
        n = len(factors)
        product = 1.0
        for f in factors:
            product *= f
        score = product ** (1.0 / n)

        # Apply state modifiers
        if self.state == MemoryState.REHEARSING:
            score *= c.rehearsing_boost
        elif self.state == MemoryState.GHOST:
            score *= c.ghost_penalty
        elif self.state == MemoryState.REACTIVATED:
            score *= c.reactivated_penalty

        # Apply natural decay
        score *= self.decay_factor

        # Importance bonus: richer memories (novel, task-relevant, reusable) get up to +30%
        importance_bonus = 1.0 + self.importance_score * 0.3
        score *= importance_bonus

        # Confidence penalty: each contradiction reduces confidence
        confidence_penalty = max(0.1, 1.0 - getattr(self, '_contradiction_count', 0) * 0.15)
        score *= confidence_penalty

        return max(0.0, min(1.0, score))

    @property
    def confidence_score(self) -> float:
        """How trustworthy this memory is — factors in source count, verification, contradictions."""
        v = self.vector
        base = v.confidence
        # Source verification: implicit penalty for unverified memories
        source_factor = min(1.0, getattr(self, '_source_count', 1) * 0.33)
        # Age of last verification
        last_verified = getattr(self, '_last_verified', self.created_at)
        hours_since_verify = (time.time() - last_verified) / 3600
        verify_freshness = math.exp(-hours_since_verify / (24 * 7))  # 1-week half-life
        return max(0.01, min(1.0, base * source_factor * verify_freshness))

    @property
    def is_cortical(self) -> bool:
        return self.vector.hippocampal_dependency < 0.3

    @property
    def is_labile(self) -> bool:
        return self.vector.stability < 0.4

    @property
    def is_stale(self) -> bool:
        """Memory that hasn't been accessed/verified recently."""
        hours_since = (time.time() - self.last_activated_at) / 3600
        return hours_since > 168  # 1 week

    # ── Thermal lifecycle ───────────────────────────────

    @property
    def thermal_state(self) -> ThermalState:
        """Derived thermal level governing storage tier and retrieval cost.

        HOT  — recent access (< 24h idle) OR high retention (> 0.4) OR high frequency
        WARM — retention > 0.15, not recently accessed but still relevant
        COLD — retention <= 0.15, or state GHOST with some reactivation probability
        DEAD — state GHOST with near-zero reactivation probability (metadata only)

        The state is derived from retention_score + recency + frequency,
        not stored. This avoids dual-state-machine sync issues.
        """
        r = self.retention_score
        v = self.vector
        hours_idle = (time.time() - self.last_activated_at) / 3600

        if self.state == MemoryState.GHOST:
            react_prob = getattr(self, '_ghost_reactivation_prob', 0.0)
            if react_prob <= 0.05:
                return ThermalState.DEAD
            return ThermalState.COLD

        # Recently active memories are HOT regardless of computed retention
        is_recent = hours_idle < 24
        is_high_plasticity = self.state in (MemoryState.ACTIVE, MemoryState.REHEARSING, MemoryState.REACTIVATED)

        # Only treat as HOT if genuinely recent AND in a high-plasticity state
        if is_high_plasticity and is_recent:
            return ThermalState.HOT

        # High retention or frequency → HOT
        if r > 0.4 or v.frequency > 0.1:
            return ThermalState.HOT

        # Medium retention → WARM
        if r > 0.15:
            return ThermalState.WARM

        # Low retention but not quite dead → COLD
        if r > 0.03:
            return ThermalState.COLD

        # Near-zero retention → DEAD (offload to metadata)
        return ThermalState.DEAD

    @property
    def thermal_priority(self) -> float:
        """Retrieval priority weight based on thermal state.

        HOT:  1.0 (full priority)
        WARM: 0.6 (medium priority)
        COLD: 0.2 (low priority, must be explicitly sought)
        DEAD: 0.0 (not retrievable)
        """
        return {
            ThermalState.HOT: 1.0,
            ThermalState.WARM: 0.6,
            ThermalState.COLD: 0.2,
            ThermalState.DEAD: 0.0,
        }[self.thermal_state]

    @property
    def retrieval_cost(self) -> float:
        """Relative cost to retrieve this memory (0=free, 1=expensive).

        HOT memories are cheap (active in working set).
        COLD memories are expensive (must be thawed from index).
        DEAD memories are inaccessible.
        """
        return {
            ThermalState.HOT: 0.0,
            ThermalState.WARM: 0.3,
            ThermalState.COLD: 0.8,
            ThermalState.DEAD: 1.0,
        }[self.thermal_state]

    @property
    def storage_tier(self) -> str:
        """Which storage backend tier this memory belongs in.

        HOT  → in-memory graph (fast, plastic)
        WARM → disk-backed graph (indexed, retrievable)
        COLD → index-only (metadata + embedding, text offloaded)
        DEAD → audit log (hash only, retention for compliance)
        """
        return {
            ThermalState.HOT: "memory",
            ThermalState.WARM: "disk",
            ThermalState.COLD: "index",
            ThermalState.DEAD: "audit",
        }[self.thermal_state]

    def thaw(self) -> bool:
        """Attempt to revive from COLD → WARM/HOT.

        Success depends on having enough residual information
        (embedding or ghost trace) to reconstruct the memory.
        """
        if self.thermal_state == ThermalState.DEAD:
            return False
        if self.thermal_state == ThermalState.COLD:
            if self.state == MemoryState.GHOST:
                self.transition('revive')
            else:
                # Boost retention enough to reach WARM
                self.vector.recency = 1.0
                self.last_activated_at = time.time()
                self.vector.frequency = max(self.vector.frequency, 0.05)
            return True
        return True  # already HOT or WARM

    @property
    def is_thermally_retrievable(self) -> bool:
        """Can this memory be retrieved at its current thermal state?

        HOT/WARM: yes
        COLD: partial (metadata + summary only)
        DEAD: no
        """
        return self.thermal_state != ThermalState.DEAD

    @property
    def thermal_summary(self) -> str:
        """One-line thermal status for health reports."""
        return f"{self.thermal_state.value} (ret={self.retention_score:.2f}, freq={self.vector.frequency:.2f})"

    def record_success(self, benefit: float = 0.1):
        """Call when this memory contributed to a successful outcome."""
        self.vector.success_feedback = min(1.0, self.vector.success_feedback + benefit)
        self.vector.confidence = min(1.0, self.vector.confidence + benefit * 0.5)

    def record_failure(self, penalty: float = 0.1):
        """Call when this memory was involved in a failure."""
        self.vector.success_feedback = max(0.01, self.vector.success_feedback - penalty)
        self.vector.confidence = max(0.01, self.vector.confidence - penalty * 0.5)
        self._contradiction_count = getattr(self, '_contradiction_count', 0) + 1

    def record_verification(self):
        """Call when this memory's accuracy is verified."""
        self._last_verified = time.time()
        self._source_count = getattr(self, '_source_count', 1) + 1
        self.vector.confidence = min(1.0, self.vector.confidence + 0.05)


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
