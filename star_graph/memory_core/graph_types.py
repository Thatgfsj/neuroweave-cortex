"""Graph data types — edges, reflections, constellations, schemas. [Layer 1: Storage]

Extracted verbatim from `graph.py` in Issue #8 (pure move, no behaviour change).
`star_graph.memory_core.graph` re-exports every public symbol defined here, so
existing import paths keep working.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .anchor import Anchor, AnchorVector


@dataclass
class Edge:
    """A typed, weighted connection between two anchors.

    Simple edge — kept for backward compatibility. For rich temporal edges
    with versioning, confidence, and lifecycle, use RichEdge.

    Edge type taxonomy:
      causes, fixes, depends_on, caused_by, derived_from, supports,
      contradicts, superseded_by, invalidated_by, before, after,
      related_workflow, summarizes, resolves, causal, temporal, topical.
    """

    source: str
    target: str
    weight: float = 0.5
    edge_type: str = "topical"  # causal, temporal, topical, personal, revision, bridge, contradiction
    co_activation_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)
    # Success/failure tracking for RL-based rewiring
    success_count: int = 0       # times this edge was part of a successful reasoning chain
    failure_count: int = 0       # times this edge was part of a failed reasoning chain
    last_success_at: float = 0.0
    last_failure_at: float = 0.0
    # v1.3: activation-graph decay tracking
    decay_rate: float = 0.001    # per-day natural decay coefficient for this edge

    def strengthen(self, delta: float = 0.05) -> None:
        self.weight = min(1.0, self.weight + delta)
        self.co_activation_count += 1
        self.last_activated_at = time.time()

    def weaken(self, delta: float = 0.02) -> None:
        self.weight = max(0.0, self.weight - delta)

    @property
    def success_rate(self) -> float:
        """Ratio of successful uses to total uses. Returns 0.5 if untested."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # neutral prior
        return self.success_count / total

    def record_success(self) -> None:
        self.success_count += 1
        self.last_success_at = time.time()
        self.strengthen(0.03)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_at = time.time()
        self.weaken(0.02)

    @property
    def traversal_weight(self) -> float:
        """Effective traversal weight = edge weight × type multiplier."""
        multiplier = EDGE_TRAVERSAL_WEIGHTS.get(self.edge_type, 1.0)
        return self.weight * multiplier

    @property
    def is_active(self) -> bool:
        return self.weight > 0.1


# Valid temporal ordering values
TEMPORAL_ORDER_BEFORE = "before"
TEMPORAL_ORDER_AFTER = "after"
TEMPORAL_ORDER_SIMULTANEOUS = "simultaneous"

# State-transition edge types — track knowledge evolution, not just association
EDGE_SUPERSEDED_BY = "superseded_by"    # new info replaces old (A is obsolete, B is current)
EDGE_INVALIDATED_BY = "invalidated_by"  # evidence contradicts this (A is disproven by B)
EDGE_CONTRADICTS = "contradicts"        # mutual contradiction (A and B cannot both be true)
EDGE_CAUSED_BY = "caused_by"            # reverse causal (A was caused by B)
EDGE_DERIVED_FROM = "derived_from"      # inference chain (A was deduced from B)
EDGE_SUPPORTS = "supports"              # positive reinforcement / agreement (A supports B)

# Explicable edge relation types — edges MUST use one of these (or a domain
# subtype) to pass the sparsification gate. Topical/cosine-only edges are rejected.
#
# Taxonomy — five categories:
#   Causal/Logical     causes, fixes, depends_on, caused_by, derived_from, supports, resolves
#   Contradiction      contradicts, superseded_by, invalidated_by
#   Temporal           before, after
#   Domain/Grouping    related_workflow
#   Semantic (legacy)  causal, temporal, topical  — allowed but discouraged; prefer above
#
EXPLICABLE_RELATIONS: set[str] = {
    # --- Causal / Logical (explicit) ---
    "causes", "fixes", "depends_on", "caused_by", "derived_from",
    "supports", "resolves",
    # --- Contradiction / Evolution ---
    "contradicts", "superseded_by", "invalidated_by",
    # --- Temporal ---
    "before", "after",
    # --- Domain / Grouping ---
    "related_workflow", "summarizes",
    # --- Legacy (allowed but discouraged — prefer a typed alternative) ---
    "causal", "temporal", "topical",
}
# Legacy edge types that are NOT explicable — auto-rejected when implicit
LEGACY_EDGE_TYPES: set[str] = {"topical", "semantic", "similar", "related"}
# Relations that are "strong" — carry causal/explicit weight, not just semantic
STRONG_RELATIONS: set[str] = {
    "causes", "fixes", "depends_on", "contradicts",
    "superseded_by", "invalidated_by", "caused_by", "derived_from",
    "supports", "resolves",
}

# Edge traversal weight multipliers — different edge types contribute
# differently to graph traversal and spreading activation.
EDGE_TRAVERSAL_WEIGHTS: dict[str, float] = {
    # --- Causal / Logical ---
    "causes": 1.5,           # causal = most important for reasoning
    "fixes": 1.4,            # fix relationships = high utility
    "depends_on": 1.3,       # dependency = structural
    "caused_by": 1.4,        # reverse causal
    "derived_from": 1.2,     # inference chain
    "supports": 1.2,         # positive reinforcement / agreement
    "resolves": 1.3,         # resolution
    # --- Contradiction / Evolution ---
    "contradicts": 0.5,      # contradiction = useful but downweighted
    "superseded_by": 0.4,    # outdated = low traversal
    "invalidated_by": 0.3,   # disproven = very low traversal
    # --- Temporal ---
    "before": 1.0,           # temporal before
    "after": 1.0,            # temporal after
    # --- Domain / Grouping ---
    "related_workflow": 1.2, # workflow = process continuity
    "summarizes": 1.1,       # summary = compression link
    # --- Legacy ---
    "causal": 1.5,           # causal (legacy, prefer "causes"/"caused_by")
    "temporal": 1.2,         # temporal (legacy, prefer "before"/"after")
    "topical": 1.0,          # baseline semantic (legacy)
    # --- Structural / Infrastructure ---
    "instance_of": 1.0,      # abstraction link
    "bridge": 0.8,           # cross-constellation bridge
    "transitive_closure": 0.7, # inferred transitive
    "cross_domain": 1.3,     # cross-cortex link
    "preference": 1.3,       # user preference signal
    "task_flow": 1.4,        # task workflow continuity
    "ownership": 1.1,        # belonging relationship
}


@dataclass
class RichEdge:
    """A temporal, versioned edge with confidence and lifecycle management.

    Unlike simple Edge, a RichEdge tracks:
    - How the relationship was established (explicit vs inferred)
    - How confident we are in it
    - How many times it's been reinforced across sessions
    - When it decays and at what rate
    - Version history for contradiction resolution
    - Whether it's been superseded by a newer belief
    - Temporal ordering (which happened first)
    - Causal strength (how strongly A implies/causes B)
    - State-transition types for knowledge evolution

    Edge type taxonomy (canonical set):
      Causal/Logical:
        causes           — A causes B
        fixes            — A fixes/resolves an issue in B
        depends_on       — A depends on B
        caused_by        — A was caused by B (reverse causal)
        derived_from     — A was inferred/deduced from B
        supports         — A supports/confirms B (positive evidence)
        resolves         — A resolves/answers B
      Contradiction/Evolution:
        contradicts      — A and B cannot both be true (mutual)
        superseded_by    — A is replaced by newer B
        invalidated_by   — A is disproven by B
      Temporal:
        before           — A happened before B
        after            — A happened after B
      Domain/Grouping:
        related_workflow — A and B belong to the same workflow
        summarizes       — A is a summary/compression of B
      Legacy (discouraged — use a typed alternative):
        causal           — generic causal (prefer causes/caused_by)
        temporal         — generic temporal (prefer before/after)
        topical          — shared subject matter

    Retrieval scoring:
      score = weight + 0.2*confidence + 0.1*recency_bonus + 0.15*reinforcement_bonus
              + 0.1*causal_strength (if causal/temporal)

    This is the foundation for temporal memory, contradiction resolution,
    and memory lifecycle management — what separates a graph DB from a
    cognitive memory system.
    """

    source: str
    target: str
    weight: float = 0.5
    edge_type: str = "topical"
    relation: str = ""            # human-readable: "likes", "uses", "built", "prefers"
    confidence: float = 0.5       # 0..1: explicit=0.95, implicit=0.42, inferred=0.3
    source_type: str = "implicit" # "explicit" | "implicit" | "inferred" | "reinforced"
    reinforcement_count: int = 0  # times confirmed across sessions
    co_activation_count: int = 0  # times co-activated (Hebbian)
    decay_rate: float = 0.01      # per-hour decay factor
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)
    last_reinforced_at: float = 0.0
    is_stale: bool = False        # marked when newer belief contradicts
    stale_since: float = 0.0
    replaced_by: str = ""         # edge_key that supersedes this one
    version_history: list[dict] = field(default_factory=list)  # [{old_w, new_w, ts, reason}]
    source_session: str = ""      # which session created this edge
    # v0.5: temporal and causal fields
    temporal_order: str = ""      # "before" | "after" | "simultaneous" | ""
    causal_strength: float = 0.0  # 0..1: how strongly A implies/causes B
    stability: float = 0.0        # 0..1: resistance to change (grows with reinforcement)
    valid_until: float = 0.0      # optional expiration timestamp (0 = never)
    used_at: float = 0.0          # last time this edge contributed to a retrieval
    # Success/failure tracking for RL-based rewiring
    success_count: int = 0
    failure_count: int = 0
    last_success_at: float = 0.0
    last_failure_at: float = 0.0

    def strengthen(self, delta: float = 0.05) -> None:
        old_w = self.weight
        self.weight = min(1.0, self.weight + delta)
        self.co_activation_count += 1
        self.last_activated_at = time.time()
        self._log_version(old_w, self.weight, "strengthen")

    def weaken(self, delta: float = 0.02) -> None:
        old_w = self.weight
        self.weight = max(0.0, self.weight - delta)
        self._log_version(old_w, self.weight, "weaken")

    def reinforce(self, source_session: str = "") -> None:
        """Called when the same relationship is confirmed in another session."""
        self.reinforcement_count += 1
        self.last_reinforced_at = time.time()
        self.source_type = "reinforced"
        # Each reinforcement boosts confidence toward 1.0 with diminishing returns
        boost = 0.1 * (1.0 - self.confidence)
        old_c = self.confidence
        self.confidence = min(1.0, self.confidence + boost)
        # Stability grows with each reinforcement
        self.stability = min(1.0, self.stability + 0.05)
        self._log_version(old_c, self.confidence, f"reinforced (session={source_session})")

    def mark_stale(self, replaced_by_edge_key: str = "") -> None:
        """Mark this edge as stale — a newer belief contradicts it."""
        self.is_stale = True
        self.stale_since = time.time()
        self.replaced_by = replaced_by_edge_key
        self._log_version(self.weight, self.weight * 0.3, "stale")

    def apply_decay(self, elapsed_hours: float) -> None:
        """Apply temporal decay: w_t = w_0 * e^(-λt * (1 - stability*0.7)).

        Stability slows decay — well-reinforced edges resist forgetting.
        Stale edges decay 2x faster.
        """
        stability_damping = 1.0 - self.stability * 0.7  # 1.0 → 0.3 as stability grows
        effective_rate = self.decay_rate * stability_damping
        if self.is_stale:
            effective_rate *= 2.0
        old_w = self.weight
        self.weight *= math.exp(-effective_rate * elapsed_hours)
        self.weight = max(0.0, self.weight)
        if abs(old_w - self.weight) > 0.001:
            self._log_version(old_w, self.weight, f"decay ({elapsed_hours:.1f}h)")

    def _log_version(self, old_value: float, new_value: float, reason: str) -> None:
        self.version_history.append({
            "timestamp": time.time(),
            "old_value": round(old_value, 4),
            "new_value": round(new_value, 4),
            "reason": reason,
        })
        # Keep only last 20 versions
        if len(self.version_history) > 20:
            self.version_history = self.version_history[-20:]

    @property
    def is_active(self) -> bool:
        return self.weight > 0.1 and not self.is_stale

    def mark_used(self) -> None:
        """Record that this edge contributed to a retrieval."""
        self.used_at = time.time()
        self.last_activated_at = time.time()

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def record_success(self) -> None:
        self.success_count += 1
        self.last_success_at = time.time()
        self.strengthen(0.03)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_at = time.time()
        self.weaken(0.02)

    @property
    def traversal_weight(self) -> float:
        """Effective traversal weight = edge weight × type multiplier."""
        multiplier = EDGE_TRAVERSAL_WEIGHTS.get(self.edge_type, 1.0)
        return self.weight * multiplier

    @property
    def is_expired(self) -> bool:
        """Check if this temporal edge has passed its validity window."""
        return self.valid_until > 0 and time.time() > self.valid_until

    @property
    def retrieval_score(self) -> float:
        """Composite retrieval score for ranking during graph traversal.

        score = weight + 0.2*confidence + 0.1*recency_bonus + 0.15*reinforcement_bonus
                + 0.1*causal_strength

        State-transition edges (superseded_by, invalidated_by) get a penalty
        since they represent outdated knowledge paths.
        """
        if self.is_expired:
            return 0.0

        hours_since = (time.time() - self.last_activated_at) / 3600
        recency_bonus = math.exp(-hours_since / 168)  # decays over 1 week
        reinforcement_bonus = min(1.0, self.reinforcement_count * 0.2)
        staleness_penalty = 0.3 if self.is_stale else 1.0

        # State-transition edges: useful for tracing evolution, but not for traversal
        state_transition_types = {EDGE_SUPERSEDED_BY, EDGE_INVALIDATED_BY, EDGE_CONTRADICTS}
        state_penalty = 0.4 if self.edge_type in state_transition_types else 1.0

        # Causal/temporal edges get a bonus from causal_strength
        causal_bonus = 0.1 * self.causal_strength if self.causal_strength > 0 else 0.0

        # Temporal ordering bonus: ordered edges are more informative
        temporal_bonus = 0.05 if self.temporal_order else 0.0

        score = staleness_penalty * state_penalty * (
            self.weight
            + 0.2 * self.confidence
            + 0.1 * recency_bonus
            + 0.15 * reinforcement_bonus
            + causal_bonus
            + temporal_bonus
        )
        return min(1.5, max(0.0, score))

    @classmethod
    def explicit(cls, source: str, target: str, relation: str = "",
                 weight: float = 0.7, session: str = "") -> "RichEdge":
        """Create an edge from explicit user statement — high confidence."""
        return cls(
            source=source, target=target, weight=weight,
            relation=relation, confidence=0.95, source_type="explicit",
            source_session=session, edge_type="topical",
        )

    @classmethod
    def implicit(cls, source: str, target: str, relation: str = "",
                 weight: float = 0.4) -> "RichEdge":
        """Create an edge from co-occurrence — basic implicit association."""
        return cls(
            source=source, target=target, weight=weight,
            relation=relation, confidence=0.42, source_type="implicit",
            edge_type="topical",
        )

    @classmethod
    def inferred(cls, source: str, target: str, relation: str = "",
                 weight: float = 0.4) -> "RichEdge":
        """Create an edge from implicit inference — lower confidence."""
        return cls(
            source=source, target=target, weight=weight,
            relation=relation, confidence=0.42, source_type="inferred",
            edge_type="topical",
        )

    @classmethod
    def temporal(cls, source: str, target: str, order: str,
                 weight: float = 0.6, valid_days: float = 0.0,
                 causal_strength: float = 0.0, session: str = "") -> "RichEdge":
        """Create a temporal edge — A happened `order` B.

        Args:
            order: "before" | "after" | "simultaneous"
            valid_days: how many days until this temporal link expires (0 = never)
        """
        valid_until = time.time() + valid_days * 86400 if valid_days > 0 else 0.0
        return cls(
            source=source, target=target, weight=weight,
            edge_type="temporal", relation=order,
            confidence=0.85, source_type="explicit",
            temporal_order=order, causal_strength=causal_strength,
            valid_until=valid_until,
            source_session=session,
        )

    @classmethod
    def causal(cls, source: str, target: str, strength: float = 0.7,
               session: str = "") -> "RichEdge":
        """Create a causal edge — A causes/implies B."""
        return cls(
            source=source, target=target, weight=strength,
            edge_type="causal", relation="causes",
            confidence=0.8, source_type="inferred",
            causal_strength=strength,
            source_session=session,
        )

    @classmethod
    def supersedes(cls, old_edge_key: str, new_source: str, new_target: str,
                   session: str = "") -> "RichEdge":
        """Create an edge showing new knowledge replaces old."""
        return cls(
            source=new_source, target=new_target, weight=0.6,
            edge_type=EDGE_SUPERSEDED_BY, relation="supersedes",
            confidence=0.7, source_type="explicit",
            source_session=session,
        )

    @classmethod
    def contradicts(cls, source: str, target: str, confidence: float = 0.6,
                    session: str = "") -> "RichEdge":
        """Create a contradiction edge — A and B cannot both be true."""
        return cls(
            source=source, target=target, weight=0.5,
            edge_type=EDGE_CONTRADICTS, relation="contradicts",
            confidence=confidence, source_type="inferred",
            source_session=session,
        )

    @classmethod
    def derived_from(cls, source: str, target: str, weight: float = 0.4,
                     session: str = "") -> "RichEdge":
        """Create a derivation edge — A was inferred/deduced from B."""
        return cls(
            source=source, target=target, weight=weight,
            edge_type=EDGE_DERIVED_FROM, relation="derived_from",
            confidence=0.6, source_type="inferred",
            source_session=session,
        )

    @classmethod
    def from_edge(cls, edge: Edge) -> "RichEdge":
        """Upgrade a simple Edge to RichEdge."""
        return cls(
            source=edge.source, target=edge.target,
            weight=edge.weight, edge_type=edge.edge_type,
            confidence=0.5, source_type="implicit",
            created_at=edge.created_at,
            last_activated_at=edge.last_activated_at,
        )


@dataclass
class ReflectionNode:
    """Meta-cognitive insight about past memories — "what did I learn?"

    Reflection nodes sit above the memory graph, connecting to source anchors
    via state-transition edges (caused_by, derived_from, invalidated_by).

    Types:
      - failure_analysis: why something failed, what to avoid
      - success_pattern: what worked well, what to repeat
      - root_cause: underlying reason for an outcome
      - lesson_learned: actionable takeaway for future decisions

    During retrieval, reflection nodes connected to highly-scored anchors
    surface as context-enriching insights.
    """

    id: str
    text: str                            # the insight / lesson
    reflection_type: str = "lesson_learned"  # failure_analysis | success_pattern | root_cause | lesson_learned
    source_anchor_ids: list[str] = field(default_factory=list)  # anchors this insight is based on
    confidence: float = 0.5              # how confident we are in this insight
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    strength: float = 0.5                # grows with confirmatory evidence, decays with neglect

    @classmethod
    def from_failure(cls, text: str, source_anchor_ids: list[str],
                     confidence: float = 0.7) -> "ReflectionNode":
        import hashlib
        rid = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
        return cls(id=rid, text=text, reflection_type="failure_analysis",
                   source_anchor_ids=source_anchor_ids, confidence=confidence,
                   strength=0.6)

    @classmethod
    def from_success(cls, text: str, source_anchor_ids: list[str],
                     confidence: float = 0.8) -> "ReflectionNode":
        import hashlib
        rid = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
        return cls(id=rid, text=text, reflection_type="success_pattern",
                   source_anchor_ids=source_anchor_ids, confidence=confidence,
                   strength=0.7)

    @classmethod
    def from_lesson(cls, text: str, source_anchor_ids: list[str],
                    confidence: float = 0.6) -> "ReflectionNode":
        import hashlib
        rid = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
        return cls(id=rid, text=text, reflection_type="lesson_learned",
                   source_anchor_ids=source_anchor_ids, confidence=confidence,
                   strength=0.5)

    @property
    def is_relevant(self) -> bool:
        """Is this reflection still fresh enough to surface?"""
        days_since = (time.time() - self.last_accessed_at) / 86400
        return days_since < 30 and self.strength > 0.1

    def reinforce(self) -> None:
        """Boost strength when confirmed by new evidence."""
        self.strength = min(1.0, self.strength + 0.1)
        self.last_accessed_at = time.time()

    def weaken(self) -> None:
        """Reduce strength when contradicted."""
        self.strength = max(0.0, self.strength - 0.15)


@dataclass
class Constellation:
    """A connected subgraph — a "star string" of related anchors."""

    anchors: list[Anchor]
    edges: list[Edge]
    label: str = ""

    @property
    def centroid_vector(self) -> AnchorVector:
        if not self.anchors:
            return AnchorVector()
        vecs = [a.vector for a in self.anchors]
        n = len(vecs)
        return AnchorVector(
            importance=sum(v.importance for v in vecs) / n,
            frequency=sum(v.frequency for v in vecs) / n,
            recency=sum(v.recency for v in vecs) / n,
            emotional_valence=sum(v.emotional_valence for v in vecs) / n,
            stability=sum(v.stability for v in vecs) / n,
            surprise=sum(v.surprise for v in vecs) / n,
        )

    @property
    def total_weight(self) -> float:
        return sum(e.weight for e in self.edges)

    @property
    def dominant_oscillation(self) -> tuple[float, float]:
        """The dominant frequency and phase of this constellation."""
        if not self.anchors:
            return (0.5, 0.0)
        freqs = [a.oscillator.natural_frequency for a in self.anchors]
        phases = [a.oscillator.phase_offset for a in self.anchors]
        return (
            sum(freqs) / len(freqs),
            sum(phases) / len(phases),
        )


@dataclass
class Schema:
    """Abstract pattern extracted from multiple related anchors.

    After repeated sleep cycles, common structures across episodes
    are extracted as schemas. Schemas guide encoding of new memories.
    """

    id: str
    template: str                  # abstract pattern description
    slots: dict[str, str]          # slot_name -> description
    instance_ids: list[str]        # anchor IDs instantiating this schema
    confidence: float = 0.0        # 0..1
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def match(self, text: str, embedding: list[float] | None = None) -> tuple[float, dict]:
        """Try to match text against this schema. Returns (confidence, slot_values).

        Matching combines keyword overlap with embedding cosine similarity
        when an embedding is available. Slot values are extracted via keyword
        presence in the input text.
        """
        template_words = set(self.template.lower().split())
        text_words = set(text.lower().split())

        # Keyword overlap score
        overlap = len(template_words & text_words) / max(1, len(template_words))

        # Embedding similarity (when available)
        emb_score = 0.0
        if embedding is not None and hasattr(self, '_embedding') and self._embedding:
            import numpy as np
            a = np.array(embedding, dtype=np.float32)
            b = np.array(self._embedding, dtype=np.float32)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm > 0:
                emb_score = float(np.dot(a, b) / norm)

        # Combined: 60% keyword, 40% embedding (when both available)
        if embedding is not None and emb_score > 0:
            score = overlap * 0.6 + emb_score * 0.4
        else:
            score = overlap

        # Slot value extraction: find which slot keywords appear in text
        slot_values: dict[str, str] = {}
        for slot_name, slot_desc in self.slots.items():
            slot_keywords = set(slot_desc.lower().split())
            matched = slot_keywords & text_words
            if matched:
                slot_values[slot_name] = ", ".join(matched)

        return min(1.0, score), slot_values


def _cosine_sim_simple(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity for graph-internal use."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-8)
