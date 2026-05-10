"""Star graph — nodes (anchors) connected by weighted, typed edges.

The graph supports:
- Spreading activation for constellation discovery
- Weighted, typed edges (causal, temporal, topical, personal)
- Merge operations for similar anchors
- Prune operations for weak/dormant connections
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, AnchorVector


@dataclass
class Edge:
    """A typed, weighted connection between two anchors."""

    source: str          # anchor id
    target: str          # anchor id
    weight: float = 0.5  # 0..1
    edge_type: str = "topical"  # causal, temporal, topical, personal
    co_activation_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = field(default_factory=time.time)

    def strengthen(self, delta: float = 0.05) -> None:
        """Hebbian-like strengthening on co-activation."""
        self.weight = min(1.0, self.weight + delta)
        self.co_activation_count += 1
        self.last_activated_at = time.time()

    def weaken(self, delta: float = 0.02) -> None:
        """Decay when not used."""
        self.weight = max(0.0, self.weight - delta)

    @property
    def is_active(self) -> bool:
        return self.weight > 0.1


@dataclass
class Constellation:
    """A connected subgraph — a "star string" of related anchors."""

    anchors: list[Anchor]
    edges: list[Edge]
    label: str = ""  # auto-generated label for the constellation

    @property
    def centroid_vector(self) -> AnchorVector:
        """Average vector of all anchors in this constellation."""
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


class StarGraph:
    """The core star-graph memory structure."""

    def __init__(self):
        self.anchors: dict[str, Anchor] = {}
        self.edges: dict[tuple[str, str], Edge] = {}  # (src, tgt) sorted
        self._adjacency: dict[str, set[str]] = defaultdict(set)

    def _key(self, a: str, b: str) -> tuple[str, str]:
        """Canonical edge key (sorted)."""
        return (a, b) if a < b else (b, a)

    # ── CRUD ──────────────────────────────────────────────

    def add_anchor(self, anchor: Anchor) -> str:
        self.anchors[anchor.id] = anchor
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
            del self.anchors[anchor_id]
        # Remove all edges involving this anchor
        to_remove = []
        for (a, b) in self.edges:
            if a == anchor_id or b == anchor_id:
                to_remove.append((a, b))
        for key in to_remove:
            del self.edges[key]
        if anchor_id in self._adjacency:
            del self._adjacency[anchor_id]
        for adj in self._adjacency.values():
            adj.discard(anchor_id)

    # ── Navigation / Resonance ────────────────────────────

    def neighbors(self, anchor_id: str, min_weight: float = 0.0) -> list[tuple[str, float]]:
        """Get neighbors with edge weights."""
        result = []
        for other in self._adjacency.get(anchor_id, set()):
            key = self._key(anchor_id, other)
            edge = self.edges.get(key)
            if edge and edge.weight >= min_weight:
                result.append((other, edge.weight))
        return sorted(result, key=lambda x: -x[1])

    def spread_activation(self, seed_ids: list[str], steps: int = 3,
                          decay: float = 0.6) -> dict[str, float]:
        """Spreading activation from seed anchors.

        Returns a dict of anchor_id -> activation value.
        Similar to how the brain's associative networks fire.
        """
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
        """Find the constellation (connected subgraph) containing the seed."""
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

    # ── Bulk operations ───────────────────────────────────

    def get_prune_candidates(self, threshold: float = 0.15) -> list[str]:
        """Find anchors below retention threshold."""
        return [aid for aid, a in self.anchors.items()
                if a.retention_score < threshold]

    def get_dormant_edges(self, threshold: float = 0.1) -> list[tuple[str, str]]:
        """Find edges that have weakened below threshold."""
        return [key for key, e in self.edges.items() if e.weight < threshold]

    def stats(self) -> dict:
        return {
            "anchors": len(self.anchors),
            "edges": len(self.edges),
            "avg_retention": sum(a.retention_score for a in self.anchors.values())
            / max(1, len(self.anchors)),
            "avg_edge_weight": sum(e.weight for e in self.edges.values())
            / max(1, len(self.edges)),
            "constellations": self.count_constellations(),
        }

    def count_constellations(self) -> int:
        """Count connected components (constellations)."""
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
