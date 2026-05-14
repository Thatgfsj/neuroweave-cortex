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
    """Per-instance embedder store — avoids multi-Manager singleton pollution.

    Class-level: backward-compatible singleton for single-instance callers.
        EmbedderRegistry.set_embedder_singleton(embedder)
        emb = EmbedderRegistry.get_embedder_singleton()

    Instance-level: each MemoryRuntime gets its own registry.
        registry = EmbedderRegistry(embedder)
        emb = registry.get_embedder()
    """

    _embedder = None  # class-level fallback

    def __init__(self, embedder=None):
        self._embedder = embedder  # instance-level (shadows class attr on self)

    def set_embedder(self, embedder) -> None:
        self._embedder = embedder

    def get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        # Fall back to class-level singleton (backward compat)
        if EmbedderRegistry._embedder is not None:
            return EmbedderRegistry._embedder
        from .embedding import get_embedder
        emb = get_embedder()
        self._embedder = emb
        return emb

    @property
    def is_available(self) -> bool:
        if self._embedder is not None or EmbedderRegistry._embedder is not None:
            return True
        try:
            from .embedding import get_embedder
            return True
        except Exception:
            return False

    # ── Class-level (singleton) methods ──

    @classmethod
    def set_embedder_singleton(cls, embedder) -> None:
        cls._embedder = embedder

    @classmethod
    def get_embedder_singleton(cls):
        if cls._embedder is None:
            from .embedding import get_embedder
            cls._embedder = get_embedder()
        return cls._embedder

    @classmethod
    def is_available_singleton(cls) -> bool:
        if cls._embedder is not None:
            return True
        try:
            from .embedding import get_embedder
            return True
        except Exception:
            return False


class ThermalState(enum.Enum):
    """Five-level thermal memory lifecycle — governs storage tier and retrieval cost.

    HOT    → high-frequency, full retrieval, highest priority
    WARM   → low frequency but important, compressed summary available
    COLD   → index-only, must be thawed for full retrieval
    FROZEN → disk-only, excluded from ANN index, archive tier
    DEAD   → metadata/hash only, not retrievable (audit trail)

    Thermal state is DERIVED from retention_score and access patterns —
    it is not stored directly. This avoids dual-state-machine sync issues.
    """
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    FROZEN = "frozen"
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


# Explicit MemoryState → ThermalState mapping.
# Updated during transition() to avoid dual-state-machine drift.
_STATE_THERMAL_MAP = {
    MemoryState.ACTIVE: ThermalState.HOT,
    MemoryState.REHEARSING: ThermalState.HOT,
    MemoryState.CONSOLIDATING: ThermalState.WARM,
    MemoryState.DORMANT: ThermalState.WARM,
    MemoryState.GHOST: ThermalState.COLD,
    MemoryState.REACTIVATED: ThermalState.HOT,
}
# DORMANT with very low retention maps to FROZEN (done in thermal_state property logic)

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
    decay_rate: float = 0.01       # per-day natural decay coefficient

    def to_list(self) -> list[float]:
        """Serialize to 10-element list (v1.0.8 compact format)."""
        return [
            self.importance, self.frequency, self.recency,
            self.emotional_valence, self.stability, self.surprise,
            self.hippocampal_dependency,
            self.success_feedback, self.confidence,
            self.decay_rate,
        ]

    @classmethod
    def from_list(cls, values: list[float]) -> AnchorVector:
        """Deserialize with backward compat for old 13-element vectors.

        Old format (13): importance, frequency, recency, emotional_valence,
            stability, surprise, hippocampal_dependency, success_feedback,
            confidence, novelty, task_relevance, future_reusability, decay_rate.

        New format (10): same minus novelty, task_relevance, future_reusability.
        """
        v = list(values)
        # Pad with defaults if short
        defaults = [0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0,
                    0.5, 0.5, 0.01]
        while len(v) < len(defaults):
            v.append(defaults[len(v)])

        # If old 13-element format, merge removed dimensions into survivors:
        # novelty→surprise(max), task_relevance→importance(max), future_reusability→drop
        if len(v) >= 13:
            v[5] = max(v[5], v[9])       # surprise = max(surprise, novelty)
            v[0] = max(v[0], v[10])      # importance = max(importance, task_relevance)
            # future_reusability (v[11]) is dropped

        return cls(
            importance=v[0], frequency=v[1], recency=v[2],
            emotional_valence=v[3], stability=v[4], surprise=v[5],
            hippocampal_dependency=v[6],
            success_feedback=v[7], confidence=v[8],
            decay_rate=v[9],
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
    """A single memory anchor point with state machine and oscillatory properties.

    Fields added in v0.6:
    - cortex_path: hierarchical path within a cortex (e.g. "dev.python.gui")
    - segment_id: which Segment this node belongs to (for hub bridging)
    - semantic_density (property): 0=raw event → 1=abstract rule
    - activation_potential (property): dynamic energy for gating competition
    """

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
    _thermal_state: ThermalState | None = None  # set during transition(), avoids dual-state drift
    _ret_cached: float = -1.0   # cached retention_score, -1 = dirty
    _ret_ts: float = 0.0        # timestamp of last cache write
    # v0.6: cortex integration
    cortex_path: str = ""          # hierarchical path e.g. "dev.python.gui"
    segment_id: str = ""           # which Segment this node belongs to
    community_id: str = ""                       # primary community
    secondary_community_ids: list[str] = field(default_factory=list)  # bridge
    # v0.8: exact match cache fields
    exact_match_keys: list[str] = field(default_factory=list)  # deterministic lookup keys
    salience: float = 0.0          # 0..1 how salient/easily-recallable this memory is

    @classmethod
    def create(cls, text: str, source_session: str = "",
               embedding: list[float] | None = None,
               emotional_valence: float = 0.0,
               surprise: float = 0.5,
               tags: list[str] | None = None,
               importance: float = 0.5,
               salience: float | None = None,
               exact_match_keys: list[str] | None = None,
               **vec_kw) -> Anchor:
        anchor_id = hashlib.blake2b(
            (text + source_session).encode(), digest_size=8
        ).hexdigest()
        vec_fields = {"importance", "frequency", "recency",
                      "emotional_valence", "stability", "surprise",
                      "hippocampal_dependency", "success_feedback", "confidence",
                      "decay_rate"}
        vec_kw.setdefault("importance", importance)
        vec_kw.setdefault("emotional_valence", emotional_valence)
        vec_kw.setdefault("surprise", surprise)
        vector_kw = {k: v for k, v in vec_kw.items() if k in vec_fields}
        anchor_kw = {k: v for k, v in vec_kw.items() if k not in vec_fields}

        # Derive meaningful oscillator params via embedder registry (no L1→L3 import)
        try:
            embedder = EmbedderRegistry.get_embedder_singleton()
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

        # Derive salience: weighted combination of importance, emotional salience, and specificity
        if salience is None:
            _imp = vector_kw.get("importance", importance)
            _emo = abs(vector_kw.get("emotional_valence", emotional_valence))
            _len = min(1.0, len(text) / 200)
            salience = 0.4 * _imp + 0.35 * _len + 0.25 * _emo

        # Auto-extract exact match keys from text if not provided
        if exact_match_keys is None:
            from .exact_cache import extract_entity_keys
            exact_match_keys = extract_entity_keys(text, tags)

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
            salience=salience,
            exact_match_keys=exact_match_keys,
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
        """State-entry side effects, including thermal state synchronization."""
        from .config import Config
        c = Config.get().anchor.state_entry
        now = time.time()

        # Synchronize thermal state and invalidate retention cache
        self._thermal_state = _STATE_THERMAL_MAP.get(state, ThermalState.WARM)
        self._ret_cached = -1.0

        if state == MemoryState.REHEARSING:
            self.replay_count += 1
            self.last_activated_at = now
        elif state == MemoryState.CONSOLIDATING:
            self.vector.stability = min(1.0, self.vector.stability + c.consolidating_stability_boost)
        elif state == MemoryState.DORMANT:
            self.vector.stability = max(self.vector.stability, c.dormant_min_stability)
        elif state == MemoryState.GHOST:
            pass  # GhostNode handles this
        elif state == MemoryState.REACTIVATED:
            self.vector.stability = c.reactivated_stability
            self.vector.surprise = c.reactivated_surprise
            self.vector.recency = 1.0
            self.last_activated_at = now

    @property
    def is_retrievable(self) -> bool:
        """Can this anchor be returned in retrieval results?

        Ghosts and FROZEN/DEAD memories are not retrievable unless revived.
        """
        if self.state == MemoryState.GHOST:
            return False
        if self.thermal_state in (ThermalState.FROZEN, ThermalState.DEAD):
            return False
        return True

    @property
    def is_plastic(self) -> bool:
        """Can this anchor be modified/updated?"""
        return self.state in (MemoryState.ACTIVE, MemoryState.REHEARSING, MemoryState.REACTIVATED)

    # ── Dynamics ──────────────────────────────────────

    # Class-level survival function — set by MemoryManager at init
    _survival_fn: object = None

    @classmethod
    def set_survival_function(cls, fn) -> None:
        """Set the survival function for all anchors (called by MemoryManager)."""
        cls._survival_fn = fn

    def decay(self, elapsed_hours: float, half_life: float | None = None,
              survival_fn=None) -> None:
        """Apply time-based decay to recency.

        If a survival function is available (class-level or passed), uses the
        configurable curve. Otherwise falls back to simple exponential decay.
        """
        sf = survival_fn or Anchor._survival_fn
        if sf is not None:
            from .survival import derive_strength
            strength = derive_strength(self)
            retention = sf.survive(elapsed_hours, strength)

            # State modifiers adjust retention
            from .config import Config
            c = Config.get().anchor.decay
            if self.state == MemoryState.GHOST:
                retention *= c.ghost_half_life_factor
            elif self.state == MemoryState.DORMANT:
                retention = retention + (1.0 - retention) * 0.3  # slower decay
            elif self.state == MemoryState.REACTIVATED:
                retention = retention + (1.0 - retention) * 0.15  # moderate decay

            self.vector.recency *= retention
        else:
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
        self._ret_cached = -1.0  # invalidate cache
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
        """Composite relevance: importance × success_feedback × (1+|valence|×0.2)."""
        v = self.vector
        base = v.importance * max(v.success_feedback, 0.01)
        emotional_boost = 1.0 + abs(v.emotional_valence) * 0.2
        return max(0.01, min(1.0, base * emotional_boost))

    @property
    def decay_factor(self) -> float:
        """Natural decay factor based on age and memory strength.

        If a survival function is configured (class-level), uses the full
        configurable curve. Otherwise falls back to simple exponential decay.
        """
        hours_since = (time.time() - self.last_activated_at) / 3600

        if Anchor._survival_fn is not None:
            from .survival import derive_strength
            strength = derive_strength(self)
            decay = Anchor._survival_fn.survive(hours_since, strength)
        else:
            from .config import Config
            c = Config.get().anchor.retention
            half_life_days = c.decay_half_life_days
            decay = math.exp(-hours_since * math.log(2) / (half_life_days * 24))

        # Stability slows decay (keeps retention higher for stable memories)
        decay = decay + (1.0 - decay) * self.vector.stability * 0.5
        return max(0.01, decay)

    @property
    def importance_score(self) -> float:
        """Compute importance from surviving signals.

        importance = base × 0.50 + |emotional_valence| × 0.25 + surprise × 0.25
        """
        v = self.vector
        return max(0.01, min(1.0,
            v.importance * 0.50
            + abs(v.emotional_valence) * 0.25
            + v.surprise * 0.25
        ))

    @property
    def retention_score(self) -> float:
        """Multiplicative memory decay model using geometric mean.

        retention = (relevance × recency × frequency′ × success × confidence)^(1/5)

        Cached for 0.5s to avoid recomputation in hot retrieval loops.
        """
        # Return cached value if fresh (< 0.5s since last compute)
        if self._ret_cached >= 0.0 and (time.time() - self._ret_ts) < 0.5:
            return self._ret_cached

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

        # Importance bonus
        importance_bonus = 1.0 + self.importance_score * 0.3
        score *= importance_bonus

        # Confidence penalty
        confidence_penalty = max(0.1, 1.0 - getattr(self, '_contradiction_count', 0) * 0.15)
        score *= confidence_penalty

        score = max(0.0, min(1.0, score))
        self._ret_cached = score
        self._ret_ts = time.time()
        return score

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

    # ── v0.6: Semantic density & activation potential ───

    @property
    def semantic_density(self) -> float:
        """How abstracted this memory is: 0=raw episodic event, 1=abstract rule.

        Derived from:
        - hippocampal_dependency: high hipp → episodic → low density
        - stability: high stability → consolidated → high density
        - schema_ref: has schema → abstracted → high density
        - procedural tags: preference/style/rule → high density
        """
        v = self.vector
        # Base: inverse of hippocampal dependency (cortical = abstract)
        density = 1.0 - v.hippocampal_dependency

        # Stability boosts density (consolidated memories are more abstract)
        density = 0.6 * density + 0.4 * v.stability

        # Schema association strongly indicates abstraction
        if self.schema_ref:
            density = max(density, 0.7)

        # Check for procedural/rule-like tags
        abstract_tags = {'preference', 'style', 'rule', 'pattern', 'habit',
                         'convention', 'principle', 'lesson', 'knowledge',
                         'architecture', 'design'}
        tag_overlap = abstract_tags & {t.lower() for t in self.tags}
        if tag_overlap:
            density = max(density, 0.5 + 0.1 * len(tag_overlap))

        return max(0.0, min(1.0, density))

    @property
    def activation_potential(self) -> float:
        """Dynamic energy level for gating competition.

        Combines:
        - retention_score (stability + importance)
        - recency (recently active = higher potential)
        - frequency (often accessed = easier to activate)
        - thermal_priority (HOT > WARM > COLD > DEAD)
        - emotional salience (|valence| gives slight boost)

        This is the main signal used in MemoryGate winner-take-all competition.
        Higher = more likely to "fire" and enter the agent's context.
        """
        v = self.vector
        hours_idle = (time.time() - self.last_activated_at) / 3600

        # Base activation from retention
        base = self.retention_score

        # Recency boost: exponential decay over 7 days
        recency_boost = math.exp(-hours_idle / 168)

        # Frequency bonus: up to +20% for frequently accessed memories
        freq_bonus = 1.0 + min(0.2, v.frequency * 0.2)

        # Emotional salience: |valence| gives up to +15%
        emotional_boost = 1.0 + abs(v.emotional_valence) * 0.15

        # Thermal weighting
        thermal_weight = self.thermal_priority

        potential = base * recency_boost * freq_bonus * emotional_boost * thermal_weight
        return max(0.0, min(1.0, potential))

    # ── Thermal lifecycle ───────────────────────────────

    @property
    def thermal_state(self) -> ThermalState:
        """Thermal level governing storage tier and retrieval cost.

        Primary: derived from MemoryState via _STATE_THERMAL_MAP (set during transition()).
        Fallback: computed from retention_score + recency + frequency for edge cases.

        HOT  — ACTIVE, REHEARSING, REACTIVATED
        WARM — CONSOLIDATING, DORMANT
        COLD — GHOST with partial recall
        DEAD — GHOST with near-zero reactivation probability
        """
        # Use synchronized thermal state from the state machine if available
        if self._thermal_state is not None and self.state != MemoryState.GHOST:
            return self._thermal_state

        # GHOST requires nuanced handling: check reactivation probability
        if self.state == MemoryState.GHOST:
            react_prob = getattr(self, '_ghost_reactivation_prob', 0.0)
            if react_prob <= 0.05:
                return ThermalState.DEAD
            return ThermalState.COLD

        # Fallback: compute from retention for anchors created before _thermal_state sync
        r = self.retention_score
        v = self.vector
        hours_idle = (time.time() - self.last_activated_at) / 3600

        is_recent = hours_idle < 24
        is_high_plasticity = self.state in (MemoryState.ACTIVE, MemoryState.REHEARSING, MemoryState.REACTIVATED)

        if is_high_plasticity and is_recent:
            return ThermalState.HOT
        if r > 0.4 or v.frequency > 0.1:
            return ThermalState.HOT
        if r > 0.15:
            return ThermalState.WARM
        if r > 0.06:
            return ThermalState.COLD
        if r > 0.01:
            return ThermalState.FROZEN
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
            ThermalState.FROZEN: 0.05,
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
            ThermalState.FROZEN: 0.95,
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
            ThermalState.FROZEN: "archive",
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


