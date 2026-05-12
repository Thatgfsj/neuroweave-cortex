"""Brain Sphere — innermost fast-routing cache (L1 cache of the memory system).

Position: center of all spheres. Queried FIRST before any cortex or hub search.

Stores:
1. Common nodes — hot memory copies that are frequently activated (like CPU L1 cache).
   Limited to max size (default 5000). Least-recently-used eviction.
2. Hub center points — entry point to each cortex: its centroid embedding + summary.
   This is the index that enables fast cortex routing without scanning all cortices.

If a query hits the BrainSphere, it returns immediately (fast path).
If not, the BrainSphere provides cortex hub centers so the router can jump
to the right cortex without O(n_cortices) scanning.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HubCenter:
    """Entry point for a cortex — stored in BrainSphere for fast routing.

    Each cortex registers one HubCenter in the BrainSphere. The HubCenter
    holds the cortex's centroid embedding (for similarity routing) and a
    compressed summary of what the cortex contains.
    """
    cortex_name: str
    entry_embedding: list[float]      # centroid of the cortex's memory space
    summary: str = ""                  # compressed description of cortex contents
    node_count: int = 0                # how many nodes in this cortex
    last_updated: float = field(default_factory=time.time)
    access_count: int = 0              # how often this cortex was queried

    def touch(self):
        self.access_count += 1
        self.last_updated = time.time()


class BrainSphere:
    """Innermost fast-routing memory sphere.

    Layer 0 of the 5-layer retrieval pipeline. Queried before any cortex search.
    Provides two things:
    1. Direct cache hits for frequently-accessed memories (fast path)
    2. Hub center index for O(1) cortex routing

    Usage:
        brain = BrainSphere(max_common_nodes=5000)
        brain.register_cortex("dev", centroid_emb, "Python, Docker, debugging")
        brain.cache_node(hot_anchor)  # boost frequently-used memories

        # Query hits BrainSphere first
        hits = brain.query(query_emb, top_k=3)
        if hits: return hits  # fast path

        # Fallback: get hub centers for cortex routing
        centers = brain.get_relevant_centers(query_emb, top_k=3)
    """

    def __init__(self, max_common_nodes: int = 5000):
        self.max_common_nodes = max_common_nodes
        # Common nodes: LRU-ordered cache (most-recently-used at end)
        self._common_nodes: OrderedDict[str, object] = OrderedDict()  # anchor_id → Anchor
        # Hub centers: cortex_name → HubCenter
        self._hub_centers: dict[str, HubCenter] = {}

        # Stats
        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    # ── Hub center management ───────────────────────────

    def register_cortex(self, cortex_name: str,
                        entry_embedding: list[float],
                        summary: str = "",
                        node_count: int = 0):
        """Register or update a cortex's entry point in the brain."""
        if cortex_name in self._hub_centers:
            center = self._hub_centers[cortex_name]
            center.entry_embedding = entry_embedding
            center.summary = summary
            center.node_count = node_count
            center.last_updated = time.time()
        else:
            self._hub_centers[cortex_name] = HubCenter(
                cortex_name=cortex_name,
                entry_embedding=entry_embedding,
                summary=summary,
                node_count=node_count,
            )

    def remove_cortex(self, cortex_name: str):
        """Remove a cortex entry point."""
        self._hub_centers.pop(cortex_name, None)

    def get_center(self, cortex_name: str) -> HubCenter | None:
        """Get a specific cortex's hub center."""
        return self._hub_centers.get(cortex_name)

    def get_relevant_centers(self, query_embedding: list[float],
                             top_k: int = 3,
                             min_similarity: float = 0.1) -> list[HubCenter]:
        """Get cortices most relevant to the query (by centroid similarity).

        This replaces O(n_cortices) scanning in CortexRouter.route() with
        a pre-computed index lookup.
        """
        if not self._hub_centers or not query_embedding:
            return list(self._hub_centers.values())[:top_k]

        scored: list[tuple[HubCenter, float]] = []
        for center in self._hub_centers.values():
            if center.entry_embedding:
                sim = _cosine_sim(query_embedding, center.entry_embedding)
                if sim >= min_similarity:
                    scored.append((center, sim))

        scored.sort(key=lambda x: -x[1])
        for center, _ in scored:
            center.touch()
        return [c for c, _ in scored[:top_k]]

    # ── Common node cache ───────────────────────────────

    def cache_node(self, anchor):
        """Add or refresh a node in the common-node cache.

        If the cache is full, evicts the least-recently-used node.
        """
        # Move to end (most recently used) if already in cache
        if anchor.id in self._common_nodes:
            self._common_nodes.move_to_end(anchor.id)
            self._common_nodes[anchor.id] = anchor
            return

        # Evict LRU if at capacity
        while len(self._common_nodes) >= self.max_common_nodes:
            self._common_nodes.popitem(last=False)

        self._common_nodes[anchor.id] = anchor

    def query_common_nodes(self, query_embedding: list[float] | None = None,
                           query_text: str = "",
                           top_k: int = 5,
                           min_similarity: float = 0.5) -> list:
        """Search the common-node cache for direct hits.

        Fast path — if the query matches a cached hot memory, return it
        immediately without any cortex search.
        """
        self.total_queries += 1

        if not self._common_nodes:
            self.cache_misses += 1
            return []

        candidates: list[tuple[object, float]] = []
        for anchor in reversed(self._common_nodes.values()):
            if not anchor.is_retrievable:
                continue

            score = 0.0
            if query_embedding and anchor.embedding:
                score = _cosine_sim(query_embedding, anchor.embedding)
            elif query_text and anchor.text:
                q_words = set(query_text.lower().split())
                a_words = set(anchor.text.lower().split())
                overlap = len(q_words & a_words)
                if overlap > 0:
                    score = overlap / max(len(q_words), 1) * 0.5
                else:
                    score = 0.1

            if score >= min_similarity:
                candidates.append((anchor, score))

        candidates.sort(key=lambda x: -x[1])
        results = [a for a, _ in candidates[:top_k]]

        if results:
            self.cache_hits += 1
            # Touch accessed nodes (move to MRU end)
            for anchor in results:
                if anchor.id in self._common_nodes:
                    self._common_nodes.move_to_end(anchor.id)
        else:
            self.cache_misses += 1

        return results

    def evict_node(self, anchor_id: str):
        """Remove a specific node from the cache."""
        self._common_nodes.pop(anchor_id, None)

    def refresh_cache(self, all_cortices: list):
        """Rebuild the common-node cache from all cortices' HOT memories.

        Called during sleep consolidation Phase 8 (index rebuild).
        Scans all cortices, selects HOT nodes with highest activation_potential,
        and repopulates the cache.
        """
        self._common_nodes.clear()

        candidates: list[tuple[object, float]] = []
        for cortex in all_cortices:
            for anchor in cortex.graph.anchors.values():
                if anchor.thermal_state.value == "hot" and anchor.is_retrievable:
                    candidates.append((anchor, anchor.activation_potential))

        # Keep top N by activation potential
        candidates.sort(key=lambda x: -x[1])
        for anchor, _ in candidates[:self.max_common_nodes]:
            self._common_nodes[anchor.id] = anchor

    # ── Health ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        hit_rate = self.cache_hits / max(1, self.total_queries)
        return {
            "cached_nodes": len(self._common_nodes),
            "max_nodes": self.max_common_nodes,
            "hub_centers": len(self._hub_centers),
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 3),
            "cortices": list(self._hub_centers.keys()),
        }

    @property
    def is_full(self) -> bool:
        return len(self._common_nodes) >= self.max_common_nodes


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
