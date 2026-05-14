"""Star graph — nodes (anchors) connected by weighted, typed edges. [Layer 1: Storage]

Provides the core data structure: anchors, edges, adjacency, ghosts, schemas.
CRUD operations are Layer 1. Cognitive convenience methods (spread_activation,
oscillatory_resonance, etc.) are marked with "Cognitive" comments — they belong
to Layer 2 but operate on the graph's public API.

Layer boundary: this module imports only from anchor, index, and config (all L1).
Cognitive modules (L2) depend on this module, never the reverse.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, AnchorVector
from .config import Config


@dataclass
class Edge:
    """A typed, weighted connection between two anchors.

    Simple edge — kept for backward compatibility. For rich temporal edges
    with versioning, confidence, and lifecycle, use RichEdge.

    Edge types: topical, semantic, causal, temporal, contradiction,
    superseded_by, invalidated_by, caused_by, derived_from.
    """

    source: str
    target: str
    weight: float = 0.5
    edge_type: str = "topical"  # causal, temporal, topical, personal, revision, bridge, contradiction
    co_activation_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)

    def strengthen(self, delta: float = 0.05) -> None:
        self.weight = min(1.0, self.weight + delta)
        self.co_activation_count += 1
        self.last_activated_at = time.time()

    def weaken(self, delta: float = 0.02) -> None:
        self.weight = max(0.0, self.weight - delta)

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

    Edge types:
      topical — shared subject matter (default)
      semantic — embedding similarity
      causal — A causes B
      temporal — A happened before/after B
      contradiction — A and B conflict
      superseded_by — A is replaced by newer B
      invalidated_by — A is disproven by B
      caused_by — A was caused by B (reverse causal)
      derived_from — A was inferred from B

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
        """Try to match text against this schema. Returns (confidence, slot_values)."""
        # Simple implementation: check if schema keywords appear
        template_words = set(self.template.lower().split())
        text_words = set(text.lower().split())
        overlap = len(template_words & text_words) / max(1, len(template_words))
        return overlap, {}


class StarGraph:
    """The core star-graph memory structure with cortical index and ghosts."""

    def __init__(self):
        self.anchors: dict[str, Anchor] = {}
        self.edges: dict[tuple[str, str], Edge] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self.cortical_index: list[tuple[list[float], str]] = []  # (embedding, anchor_id)
        self.schemas: dict[str, Schema] = {}
        # v0.4: ANN index for sub-linear retrieval
        self._ann_index = None  # lazy init on first use
        self.abstracts: dict[str, any] = {}  # AbstractNode dict, lazy import
        from .ghost import GhostSubsystem
        self._ghost_subsystem = GhostSubsystem()  # always initialized
        self.reflections: dict[str, ReflectionNode] = {}  # v0.5: meta-cognitive insights

    def _key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def node_degree(self, node_id: str) -> int:
        """Count of edges incident to this node."""
        return len(self._adjacency.get(node_id, set()))

    def _evict_weakest_edge(self, node_id: str):
        """Remove the weakest edge incident to node_id to stay within edge budget."""
        neighbors = self._adjacency.get(node_id, set())
        if not neighbors:
            return
        weakest_key = None
        weakest_weight = float('inf')
        for neighbor in neighbors:
            key = self._key(node_id, neighbor)
            edge = self.edges.get(key)
            if edge and edge.weight < weakest_weight:
                weakest_weight = edge.weight
                weakest_key = key
        if weakest_key:
            a, b = weakest_key
            self._adjacency[a].discard(b)
            self._adjacency[b].discard(a)
            self.edges.pop(weakest_key, None)

    def _evict_anchors(self, count: int, policy: str = "lowest_retention") -> list[str]:
        """Evict `count` anchors by policy. Returns list of evicted anchor IDs.

        Policies:
          - lowest_retention: remove anchors with lowest retention_score
          - lru: remove anchors with oldest last_activated_at
          - fifo: remove anchors with oldest created_at

        Skips anchors created in the last 5s to protect active-context items.
        Falls back to full anchor list if all anchors are recent.
        """
        if count <= 0 or not self.anchors:
            return []
        now = time.time()
        recent_cutoff = now - 5  # protect only very recent anchors

        candidates = [
            aid for aid, a in self.anchors.items()
            if a.created_at < recent_cutoff
        ]
        if not candidates:
            # All anchors are recent — still evict from full set
            candidates = list(self.anchors.keys())

        if policy == "lru":
            candidates.sort(key=lambda aid: self.anchors[aid].last_activated_at)
        elif policy == "fifo":
            candidates.sort(key=lambda aid: self.anchors[aid].created_at)
        else:  # lowest_retention (default)
            candidates.sort(key=lambda aid: self.anchors[aid].retention_score)

        evicted = []
        for aid in candidates[:count]:
            if aid in self.anchors:
                # Create ghost trace before removal
                if self._ghost_subsystem:
                    residual_edges = {}
                    for neighbor in self._adjacency.get(aid, set()):
                        key = self._key(aid, neighbor)
                        edge = self.edges.get(key)
                        if edge:
                            residual_edges[neighbor] = edge.weight * 0.3
                    self._ghost_subsystem.create(self.anchors[aid], residual_edges)
                self.remove_anchor(aid)
                evicted.append(aid)

        return evicted

    # ── CRUD ──────────────────────────────────────────────

    def _get_ann_index(self):
        if self._ann_index is None:
            from .index import ANNIndex
            self._ann_index = ANNIndex()
        return self._ann_index

    def add_anchor(self, anchor: Anchor) -> str:
        self.anchors[anchor.id] = anchor
        if anchor.embedding:
            self.cortical_index.append((anchor.embedding, anchor.id))
            if self._ann_index is not None:
                self._ann_index.add(anchor.id, anchor.embedding)
        return anchor.id

    def add_edge(self, src: str, tgt: str, weight: float = 0.5,
                 edge_type: str = "topical", confidence: float | None = None,
                 source_type: str = "implicit",
                 temporal_order: str = "", causal_strength: float = 0.0,
                 valid_until: float = 0.0, relation: str = "",
                 session: str = "") -> Optional[Edge]:
        if src not in self.anchors or tgt not in self.anchors:
            return None
        key = self._key(src, tgt)

        # Already connected — reinforce instead of duplicate
        if key in self.edges:
            existing = self.edges[key]
            existing.strengthen(0.05)
            return existing

        # State-transition edges: mark the old edge as stale
        if edge_type in (EDGE_SUPERSEDED_BY, EDGE_INVALIDATED_BY, EDGE_CONTRADICTS):
            if key in self.edges:
                existing = self.edges[key]
                if isinstance(existing, RichEdge):
                    existing.mark_stale(replaced_by_edge_key=str(key))

        # Edge budget enforcement — prevent node degree explosion
        max_edges = getattr(self, '_max_edges_per_node', 0)
        if max_edges > 0:
            for node_id in (src, tgt):
                if self.node_degree(node_id) >= max_edges:
                    self._evict_weakest_edge(node_id)

        # Use RichEdge when confidence, explicit source_type, temporal, or causal fields are set
        is_rich = (
            confidence is not None
            or source_type != "implicit"
            or temporal_order
            or causal_strength > 0
            or valid_until > 0
            or relation
            or edge_type in (EDGE_SUPERSEDED_BY, EDGE_INVALIDATED_BY, EDGE_CONTRADICTS,
                             EDGE_CAUSED_BY, EDGE_DERIVED_FROM, "causal", "temporal")
        )

        if is_rich:
            edge = RichEdge(
                source=key[0], target=key[1], weight=weight,
                edge_type=edge_type,
                confidence=confidence or (0.85 if source_type == "explicit" else 0.5),
                source_type=source_type,
                temporal_order=temporal_order,
                causal_strength=causal_strength,
                valid_until=valid_until,
                relation=relation or edge_type,
                source_session=session,
            )
        else:
            edge = Edge(source=key[0], target=key[1], weight=weight, edge_type=edge_type)

        self.edges[key] = edge
        self._adjacency[src].add(tgt)
        self._adjacency[tgt].add(src)
        return edge

    def add_reflection(self, reflection: ReflectionNode) -> str:
        """Store a meta-cognitive reflection and link it to source anchors.

        Reflections are NOT graph nodes — they don't participate in traversal.
        They are discovered via find_reflections() which checks source_anchor_ids.
        """
        self.reflections[reflection.id] = reflection
        return reflection.id

    def find_reflections(self, anchor_ids: list[str],
                         types: list[str] | None = None) -> list[ReflectionNode]:
        """Find reflection nodes connected to any of the given anchor IDs."""
        result: list[ReflectionNode] = []
        seen: set[str] = set()
        for rid, r in self.reflections.items():
            if rid in seen:
                continue
            if types and r.reflection_type not in types:
                continue
            if any(aid in r.source_anchor_ids for aid in anchor_ids):
                if r.is_relevant:
                    result.append(r)
                    seen.add(rid)
                    r.last_accessed_at = time.time()
        return sorted(result, key=lambda r: -(r.strength * r.confidence))

    def remove_anchor(self, anchor_id: str) -> None:
        if anchor_id in self.anchors:
            self.anchors.pop(anchor_id)
        to_remove = [(a, b) for (a, b) in self.edges if a == anchor_id or b == anchor_id]
        for key in to_remove:
            self.edges.pop(key, None)
        self._adjacency.pop(anchor_id, None)
        for adj in self._adjacency.values():
            adj.discard(anchor_id)
        self.cortical_index = [(e, aid) for e, aid in self.cortical_index if aid != anchor_id]
        if self._ann_index is not None:
            self._ann_index.remove(anchor_id)

    # ── Navigation / Cognitive convenience methods ──────
    # These operate on the graph's public API but belong to Layer 2 conceptually.
    # They're kept here for convenience — the graph is the natural namespace
    # for graph-traversal operations.

    def neighbors(self, anchor_id: str, min_weight: float = 0.0) -> list[tuple[str, float]]:
        result = []
        for other in self._adjacency.get(anchor_id, set()):
            edge = self.edges.get(self._key(anchor_id, other))
            if edge and edge.weight >= min_weight:
                result.append((other, edge.weight))
        return sorted(result, key=lambda x: -x[1])

    def spread_activation(self, seed_ids: list[str], steps: int | None = None,
                          decay: float | None = None) -> dict[str, float]:
        c = Config.get().graph.spreading
        if steps is None:
            steps = c.default_steps
        if decay is None:
            decay = c.default_decay
        activation: dict[str, float] = {aid: 1.0 for aid in seed_ids if aid in self.anchors}
        current = dict(activation)

        for _ in range(steps):
            next_wave: dict[str, float] = defaultdict(float)
            for node_id, level in current.items():
                for neighbor, weight in self.neighbors(node_id, min_weight=Config.get().graph.spreading.min_weight):
                    if neighbor not in activation:
                        next_wave[neighbor] += level * weight * decay
            for nid, val in next_wave.items():
                activation[nid] = val
            current = next_wave
            if not current:
                break

        # Normalize to [0,1] — summed contributions from multiple neighbors
        # can explode past 1.0 in dense graphs with multiple steps.
        if activation:
            max_val = max(activation.values())
            if max_val > 1.0:
                activation = {k: v / max_val for k, v in activation.items()}

        return activation

    def find_constellation(self, seed_id: str, max_size: int = 20) -> Constellation:
        if seed_id not in self.anchors:
            return Constellation(anchors=[], edges=[])

        visited: set[str] = set()
        anchors: list[Anchor] = []
        edges: list[Edge] = []
        queue = [seed_id]
        visited.add(seed_id)

        while queue and len(visited) < max_size:
            node = queue.pop(0)
            anchors.append(self.anchors[node])

            for neighbor, weight in self.neighbors(node, min_weight=Config.get().retrieval.constellation.min_edge_weight):
                key = self._key(node, neighbor)
                edge = self.edges.get(key)
                if edge and key not in {(e.source, e.target) for e in edges}:
                    edges.append(edge)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return Constellation(anchors=anchors, edges=edges)

    # ── Phase-locked resonance ───────────────────────────

    def oscillatory_resonance(self, driving_freq: float, driving_phase: float,
                              min_strength: float = 0.1) -> dict[str, float]:
        """Find anchors that phase-lock with a driving oscillation.

        This is the core of resonance-based retrieval: instead of keyword
        search, find anchors whose natural frequency matches the context's
        driving oscillation. Phase-locked anchors fire synchronously.
        """
        resonance_map: dict[str, float] = {}
        for anchor in self.anchors.values():
            strength = anchor.oscillator.resonance(driving_freq, driving_phase)
            if strength >= min_strength:
                resonance_map[anchor.id] = strength
        return resonance_map

    def cortical_lookup(self, embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Direct embedding-based lookup — the 'cortical' retrieval pathway.

        Uses ANNIndex for sub-linear retrieval when embeddings are available.
        Falls back to linear scan for small graphs or when index is cold.
        """
        if not self.cortical_index and self._ann_index is None:
            return []

        # Use ANNIndex for sub-linear lookup
        # add_anchor/remove_anchor maintain the index incrementally;
        # full rebuild only happens during sleep Index Rebuild phase
        ann = self._get_ann_index()
        if ann.size > 0:
            results = ann.query(embedding, k=top_k * 2)
            # Weight by retention_score and filter to existing anchors
            weighted = []
            for aid, sim in results:
                if aid in self.anchors:
                    weighted.append((aid, sim * self.anchors[aid].retention_score))
            weighted.sort(key=lambda x: -x[1])
            return weighted[:top_k]

        # Fallback: linear scan (for cold start / small graphs)
        scores = []
        for cort_emb, aid in self.cortical_index:
            if aid not in self.anchors:
                continue
            min_len = min(len(cort_emb), len(embedding))
            dot = sum(cort_emb[i] * embedding[i] for i in range(min_len))
            na = math.sqrt(sum(x**2 for x in cort_emb))
            nb = math.sqrt(sum(x**2 for x in embedding))
            sim = dot / (na * nb + 1e-8)
            scores.append((sim * self.anchors[aid].retention_score, aid))
        scores.sort(key=lambda x: -x[0])
        return [(aid, s) for s, aid in scores[:top_k]]

    def _ids_in_ann_sync(self) -> bool:
        """Check if ANN index matches current anchors with embeddings."""
        ann = self._get_ann_index()
        embed_count = sum(1 for a in self.anchors.values() if a.embedding)
        return ann.size == embed_count and ann.size > 0

    # ── Contradiction detection ──────────────────────────

    def find_contradictions(self, threshold: float | None = None,
                            k: int = 10) -> list[tuple[str, str, float]]:
        """Find anchor pairs that may contradict each other.

        Contradiction = high semantic similarity but opposite emotional valence
        or mutually exclusive tags.

        Uses ANN index for O(n*k) pre-filtering when available; falls back
        to O(n²) full scan only when ANN is not populated.
        """
        if threshold is None:
            threshold = Config.get().graph.contradiction_threshold

        contradictions = []
        seen_pairs: set[tuple[str, str]] = set()

        ann = self._get_ann_index()
        if ann.size > 0:
            # ANN-accelerated: O(n * k) — check only near neighbors
            for aid_a, anchor_a in self.anchors.items():
                if not anchor_a.embedding:
                    continue
                neighbors = ann.query(anchor_a.embedding, k=k + 1)  # +1 for self
                anchor_valence = anchor_a.vector.emotional_valence
                for aid_b, sim in neighbors:
                    if aid_b == aid_a:
                        continue
                    pair_key = (aid_a, aid_b) if aid_a < aid_b else (aid_b, aid_a)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    if sim > threshold:
                        anchor_b = self.anchors.get(aid_b)
                        if anchor_b and abs(anchor_valence - anchor_b.vector.emotional_valence) > 1.0:
                            contradictions.append((aid_a, aid_b, sim))
        else:
            # Fallback O(n²) for graphs without ANN index
            from .math_utils import cosine_sim
            ids = list(self.anchors.keys())
            for i, aid_a in enumerate(ids):
                a = self.anchors[aid_a]
                if not a.embedding:
                    continue
                for aid_b in ids[i + 1:]:
                    b = self.anchors[aid_b]
                    if not b.embedding:
                        continue
                    sim = cosine_sim(a.embedding, b.embedding)
                    valence_opposed = abs(a.vector.emotional_valence - b.vector.emotional_valence) > 1.0
                    if sim > threshold and valence_opposed:
                        contradictions.append((aid_a, aid_b, sim))
        return contradictions

    # ── Ghost operations ─────────────────────────────────

    def check_ghosts(self, embedding: list[float] | None,
                     revival_threshold: float | None = None) -> Optional[Anchor]:
        """Check if new content resonates with any ghosts (savings effect).

        Delegates to GhostSubsystem for resonance detection and revival.
        """
        if not embedding or not self._ghost_subsystem:
            return None
        if revival_threshold is None:
            revival_threshold = Config.get().graph.ghost_revival_threshold
        resonances = self._ghost_subsystem.check_resonance(embedding, threshold=revival_threshold)
        if resonances:
            ghost, _ = resonances[0]
            return self._ghost_subsystem.try_revive(
                ghost.id, ghost.semantic_shadow or "", embedding,
            )
        return None

    # ── Analysis ─────────────────────────────────────────

    def get_prune_candidates(self, threshold: float = 0.15) -> list[str]:
        return [aid for aid, a in self.anchors.items()
                if a.retention_score < threshold]

    def get_dormant_edges(self, threshold: float = 0.1) -> list[tuple[str, str]]:
        return [key for key, e in self.edges.items() if e.weight < threshold]

    def stats(self) -> dict:
        return {
            "anchors": len(self.anchors),
            "edges": len(self.edges),
            "ghosts": len(self._ghost_subsystem.ghosts) if self._ghost_subsystem else 0,
            "schemas": len(self.schemas),
            "cortical_index": len(self.cortical_index),
            "avg_retention": sum(a.retention_score for a in self.anchors.values())
            / max(1, len(self.anchors)),
            "avg_edge_weight": sum(e.weight for e in self.edges.values())
            / max(1, len(self.edges)),
            "constellations": self.count_constellations(),
            "avg_hippocampal_dep": sum(
                a.vector.hippocampal_dependency for a in self.anchors.values()
            ) / max(1, len(self.anchors)),
        }

    def count_constellations(self) -> int:
        visited: set[str] = set()
        count = 0
        for aid in self.anchors:
            if aid not in visited:
                count += 1
                stack = [aid]
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    for neighbor in self._adjacency.get(node, set()):
                        if neighbor not in visited:
                            stack.append(neighbor)
        return count

    # ── Community helpers ───────────────────────────────────

    def get_community_anchors(self, community_id: str) -> list[Anchor]:
        """Return all anchors belonging to a specific community."""
        return [a for a in self.anchors.values()
                if a.community_id == community_id]

    def anchors_by_community(self) -> dict[str, list[str]]:
        """Return mapping of community_id -> list of anchor_ids."""
        result: dict[str, list[str]] = defaultdict(list)
        for aid, anchor in self.anchors.items():
            if anchor.community_id:
                result[anchor.community_id].append(aid)
        return dict(result)

    def get_bridge_neighbors(self, anchor_id: str) -> list[tuple[str, float]]:
        """Return neighbors of an anchor that belong to a different community.

        Bridge neighbors connect across community boundaries and are
        essential for community-aware retrieval expansion.

        Returns list of (neighbor_id, edge_weight) sorted by weight descending.
        """
        anchor = self.anchors.get(anchor_id)
        if not anchor or not anchor.community_id:
            return []

        result = []
        for neighbor_id in self._adjacency.get(anchor_id, set()):
            neighbor = self.anchors.get(neighbor_id)
            if not neighbor:
                continue
            if (neighbor.community_id
                    and neighbor.community_id != anchor.community_id):
                edge_key = self._key(anchor_id, neighbor_id)
                edge = self.edges.get(edge_key)
                weight = edge.weight if edge else 0.5
                result.append((neighbor_id, weight))

        return sorted(result, key=lambda x: -x[1])
