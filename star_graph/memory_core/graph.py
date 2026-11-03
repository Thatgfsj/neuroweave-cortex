"""Star graph — nodes (anchors) connected by weighted, typed edges. [Layer 1: Storage]

Provides the core data structure: anchors, edges, adjacency, ghosts, schemas.
CRUD operations are Layer 1. Cognitive convenience methods (spread_activation,
oscillatory_resonance, etc.) are marked with "Cognitive" comments — they belong
to Layer 2 but operate on the graph's public API.

Data types (`Edge`, `RichEdge`, `ReflectionNode`, `Constellation`, `Schema`) and
the edge-taxonomy constants live in `graph_types`; they are re-exported here so
that `from star_graph.memory_core.graph import Edge` keeps working.

Layer boundary: this module imports only from anchor, index, and config (all L1).
Cognitive modules (L2) depend on this module, never the reverse.
"""

from __future__ import annotations

import math
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Optional

from .anchor import Anchor
from .graph_types import (  # noqa: F401 — re-exported for backward compatibility
    EDGE_CAUSED_BY,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_INVALIDATED_BY,
    EDGE_SUPERSEDED_BY,
    EDGE_SUPPORTS,
    EDGE_TRAVERSAL_WEIGHTS,
    EXPLICABLE_RELATIONS,
    LEGACY_EDGE_TYPES,
    STRONG_RELATIONS,
    TEMPORAL_ORDER_AFTER,
    TEMPORAL_ORDER_BEFORE,
    TEMPORAL_ORDER_SIMULTANEOUS,
    Constellation,
    Edge,
    ReflectionNode,
    RichEdge,
    Schema,
    _cosine_sim_simple,
)
from ..config import Config


class StarGraph:
    """The core star-graph memory structure with cortical index and ghosts."""

    def __init__(self):
        self._lock = threading.RLock()
        self.anchors: dict[str, Anchor] = {}
        self.edges: dict[tuple[str, str], Edge] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self.cortical_index: list[tuple[list[float], str]] = []  # (embedding, anchor_id)
        self.schemas: dict[str, Schema] = {}
        # v0.4: ANN index for sub-linear retrieval
        self._ann_index = None  # lazy init on first use
        self.abstracts: dict[str, Any] = {}  # AbstractNode dict, lazy import
        from ..ghost import GhostSubsystem
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

    # ── Thread safety ─────────────────────────────────────

    @contextmanager
    def lock(self):
        """Context manager for thread-safe compound operations on the graph."""
        with self._lock:
            yield

    def _get_ann_index(self):
        if self._ann_index is None:
            from .index import ANNIndex
            self._ann_index = ANNIndex()
        return self._ann_index

    def add_anchor(self, anchor: Anchor) -> str:
        with self._lock:
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
        with self._lock:
            if src not in self.anchors or tgt not in self.anchors:
                return None

            # ── Edge Sparsification Gate ───────────────────────
            # Reject or downgrade edges based on explicability.
            # Only edges with approved relation types pass at full weight.
            relation_lower = relation.lower() if relation else ""
            effective_relation = relation_lower or edge_type.lower()

            if effective_relation not in EXPLICABLE_RELATIONS or effective_relation in LEGACY_EDGE_TYPES:
                if source_type == "implicit" and confidence is None and causal_strength == 0.0:
                    # Pure cosine-similarity edge with no explicable reason — downgrade severely
                    weight = max(0.02, weight * 0.3)  # heavy penalty for legacy implicit edges

            # Non-strong relations get a weight penalty to favor explicable edges
            if effective_relation in STRONG_RELATIONS:
                weight = min(1.0, weight * 1.1)  # slight boost for strong relations
            elif effective_relation not in EXPLICABLE_RELATIONS:
                weight = max(0.05, weight * 0.5)  # penalty for weak/implicit relations

            # Default edge TTL: 14 days for weak, 90 days for strong
            if valid_until <= 0:
                import time as _time
                days = 90 if effective_relation in STRONG_RELATIONS else 14
                valid_until = _time.time() + days * 86400

            key = self._key(src, tgt)

            # Already connected — reinforce instead of duplicate.
            # State-transition edges (superseded_by / invalidated_by / contradicts)
            # mark the existing edge as stale first: the newer belief supersedes
            # or contradicts whatever the old edge encoded. (This stale-marking
            # used to sit *after* the early return below, where it was dead code
            # and never executed.)
            if key in self.edges:
                existing = self.edges[key]
                if edge_type in (EDGE_SUPERSEDED_BY, EDGE_INVALIDATED_BY, EDGE_CONTRADICTS) \
                        and isinstance(existing, RichEdge):
                    existing.mark_stale(replaced_by_edge_key=str(key))
                existing.strengthen(0.05)
                return existing

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
                                 EDGE_CAUSED_BY, EDGE_DERIVED_FROM, EDGE_SUPPORTS,
                                 "resolves", "causal", "temporal")
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
        """Find reflection nodes connected to any of the given anchor IDs.

        Results are deduplicated by reflection id, and by text so the same
        insight registered under two different ids surfaces only once.
        """
        result: list[ReflectionNode] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        for rid, r in self.reflections.items():
            if rid in seen_ids or r.text in seen_texts:
                continue
            if types and r.reflection_type not in types:
                continue
            if any(aid in r.source_anchor_ids for aid in anchor_ids):
                if r.is_relevant:
                    result.append(r)
                    seen_ids.add(rid)
                    seen_texts.add(r.text)
                    r.last_accessed_at = time.time()
        return sorted(result, key=lambda r: -(r.strength * r.confidence))

    def remove_anchor(self, anchor_id: str) -> None:
        with self._lock:
            self.anchors.pop(anchor_id, None)
            # O(degree) removal via the adjacency index (every edge is
            # registered under both endpoints), instead of the old O(E) scan
            # over self.edges. Discard from each neighbor's set in the same
            # pass so edges and _adjacency stay consistent.
            for neighbor in self._adjacency.get(anchor_id, set()):
                self.edges.pop(self._key(anchor_id, neighbor), None)
                self._adjacency[neighbor].discard(anchor_id)
            self._adjacency.pop(anchor_id, None)
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
                tw = edge.traversal_weight if hasattr(edge, 'traversal_weight') else edge.weight
                result.append((other, tw))
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
            from ..math_utils import cosine_sim
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

    def record_chain_success(self, anchor_ids: list[str]) -> int:
        """Record that a chain of anchors led to a successful outcome.

        Strengthens edges along the chain and boosts the anchors' success_feedback.
        Returns number of edges updated.
        """
        updated = 0
        for i in range(len(anchor_ids)):
            aid = anchor_ids[i]
            if aid in self.anchors:
                self.anchors[aid].record_success(benefit=0.05)
            if i + 1 < len(anchor_ids):
                key = self._key(aid, anchor_ids[i + 1])
                if key in self.edges:
                    self.edges[key].record_success()
                    updated += 1
                # Also check reverse
                rev_key = self._key(anchor_ids[i + 1], aid)
                if rev_key in self.edges:
                    self.edges[rev_key].record_success()
                    updated += 1
        return updated

    def record_chain_failure(self, anchor_ids: list[str]) -> int:
        """Record that a chain of anchors led to a failed outcome.

        Weakens edges along the chain and reduces the anchors' success_feedback.
        Returns number of edges updated.
        """
        updated = 0
        for i in range(len(anchor_ids)):
            aid = anchor_ids[i]
            if aid in self.anchors:
                self.anchors[aid].record_failure(penalty=0.05)
            if i + 1 < len(anchor_ids):
                key = self._key(aid, anchor_ids[i + 1])
                if key in self.edges:
                    self.edges[key].record_failure()
                    updated += 1
                rev_key = self._key(anchor_ids[i + 1], aid)
                if rev_key in self.edges:
                    self.edges[rev_key].record_failure()
                    updated += 1
        return updated

    def evict_expired_edges(self) -> int:
        """Remove edges that have passed their valid_until date. Returns count removed."""
        now = time.time()
        expired = []
        for key, edge in self.edges.items():
            valid_until = getattr(edge, 'valid_until', 0.0)
            if 0 < valid_until < now:
                expired.append(key)
        for key in expired:
            a, b = key
            self._adjacency[a].discard(b)
            self._adjacency[b].discard(a)
            self.edges.pop(key, None)
        return len(expired)

    def temporal_slice(self, max_core: int = 7,
                       max_active: int = 20,
                       query_embedding: list[float] | None = None,
                       current_time: float | None = None) -> dict:
        """Project a limited active memory surface — only the most relevant
        memories participate in recall.

        Three tiers:
        - Core (max 7): highest scoring, always in context
        - Active (max 20): good candidates, participate in recall
        - Background: the rest, summarized but not individually accessed
        - Noise: excluded from recall entirely

        Returns dict with {core_ids, active_ids, background_count, noise_count}.
        """
        now = current_time or time.time()

        # Score all anchors by composite relevance
        scored: list[tuple[str, float]] = []
        for aid, a in self.anchors.items():
            if not a.is_retrievable:
                continue

            v = a.vector
            # Composite score: retention + recency + success feedback + emotional
            hours_idle = (now - a.last_activated_at) / 3600
            recency = max(0.0, 1.0 - hours_idle / 720)  # decay over 30 days

            score = (
                a.retention_score * 0.35
                + recency * 0.25
                + v.success_feedback * 0.15
                + abs(v.emotional_valence) * 0.10
                + v.frequency * 0.10
                + v.stability * 0.05
            )

            # Bonus for matching query embedding
            if query_embedding and a.embedding:
                sim = _cosine_sim_simple(query_embedding, a.embedding)
                score *= (1.0 + sim * 0.5)

            scored.append((aid, score))

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Partition into tiers
        core_ids = [aid for aid, _ in scored[:max_core]]
        active_ids = [aid for aid, _ in scored[max_core:max_core + max_active]]
        background_count = max(0, len(scored) - max_core - max_active)
        noise_count = sum(1 for a in self.anchors.values() if not a.is_retrievable)

        return {
            "core_ids": core_ids,
            "active_ids": active_ids,
            "background_count": background_count,
            "noise_count": noise_count,
            "total_anchors": len(self.anchors),
            "active_surface": len(core_ids) + len(active_ids),
        }

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
