"""Star graph — nodes (anchors) connected by weighted, typed edges.

v0.2 adds: ghost anchors for savings effect, cortical index for direct
retrieval, contradiction detection, schema references.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, AnchorVector, GhostAnchor


@dataclass
class Edge:
    """A typed, weighted connection between two anchors."""

    source: str
    target: str
    weight: float = 0.5
    edge_type: str = "topical"  # causal, temporal, topical, personal, revision, bridge
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
        # v0.2 additions
        self.ghosts: dict[str, GhostAnchor] = {}
        self.cortical_index: list[tuple[list[float], str]] = []  # (embedding, anchor_id)
        self.schemas: dict[str, Schema] = {}
        # v0.4: ANN index for sub-linear retrieval
        self._ann_index = None  # lazy init on first use
        self.abstracts: dict[str, any] = {}  # AbstractNode dict, lazy import
        self._ghost_subsystem = None  # GhostSubsystem, lazy init

    def _key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

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
                 edge_type: str = "topical") -> Optional[Edge]:
        if src not in self.anchors or tgt not in self.anchors:
            return None
        key = self._key(src, tgt)
        edge = Edge(source=key[0], target=key[1], weight=weight, edge_type=edge_type)
        self.edges[key] = edge
        self._adjacency[src].add(tgt)
        self._adjacency[tgt].add(src)
        return edge

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

    # ── Navigation / Resonance ────────────────────────────

    def neighbors(self, anchor_id: str, min_weight: float = 0.0) -> list[tuple[str, float]]:
        result = []
        for other in self._adjacency.get(anchor_id, set()):
            edge = self.edges.get(self._key(anchor_id, other))
            if edge and edge.weight >= min_weight:
                result.append((other, edge.weight))
        return sorted(result, key=lambda x: -x[1])

    def spread_activation(self, seed_ids: list[str], steps: int = 3,
                          decay: float = 0.6) -> dict[str, float]:
        activation: dict[str, float] = {aid: 1.0 for aid in seed_ids if aid in self.anchors}
        current = dict(activation)

        for _ in range(steps):
            next_wave: dict[str, float] = defaultdict(float)
            for node_id, level in current.items():
                for neighbor, weight in self.neighbors(node_id, min_weight=0.05):
                    if neighbor not in activation:
                        next_wave[neighbor] += level * weight * decay
            for nid, val in next_wave.items():
                activation[nid] = val
            current = next_wave
            if not current:
                break

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

            for neighbor, weight in self.neighbors(node, min_weight=0.1):
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
        ann = self._get_ann_index()
        if ann.size > 0:
            # Ensure index is built
            if not self._ids_in_ann_sync():
                ann.clear()
                for aid, a in self.anchors.items():
                    if a.embedding:
                        ann.add(aid, a.embedding)
                ann.rebuild()
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
        """Check if ANN index matches current anchors (cheap heuristic)."""
        ann = self._get_ann_index()
        return ann.size == len(self.anchors) and ann.size > 0

    # ── Contradiction detection ──────────────────────────

    def find_contradictions(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Find anchor pairs that may contradict each other.

        Contradiction = high semantic similarity but opposite emotional valence
        or mutually exclusive tags.
        """
        contradictions = []
        ids = list(self.anchors.keys())
        for i, aid_a in enumerate(ids):
            for aid_b in ids[i + 1:]:
                a = self.anchors[aid_a]
                b = self.anchors[aid_b]
                # Opposite valence on similar topics
                if a.embedding and b.embedding:
                    dot = sum(x * y for x, y in zip(a.embedding, b.embedding))
                    na = math.sqrt(sum(x**2 for x in a.embedding))
                    nb = math.sqrt(sum(x**2 for x in b.embedding))
                    sim = dot / (na * nb + 1e-8)
                    valence_opposed = abs(a.vector.emotional_valence - b.vector.emotional_valence) > 1.0
                    if sim > threshold and valence_opposed:
                        contradictions.append((aid_a, aid_b, sim))
        return contradictions

    # ── Ghost operations ─────────────────────────────────

    def add_ghost(self, anchor: Anchor) -> None:
        """Create a ghost from an anchor being pruned."""
        self.ghosts[anchor.id] = GhostAnchor.from_anchor(anchor)

    def check_ghosts(self, embedding: list[float] | None,
                     revival_threshold: float = 0.75) -> Optional[Anchor]:
        """Check if new content resonates with any ghosts (savings effect)."""
        if not embedding:
            return None
        for ghost_id, ghost in list(self.ghosts.items()):
            resonance = ghost.resonance(embedding)
            if resonance > revival_threshold:
                del self.ghosts[ghost_id]
                return ghost.revive("", embedding)
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
            "ghosts": len(self.ghosts),
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
