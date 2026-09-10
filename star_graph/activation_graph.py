"""Activation Graph — memory retrieval through historical activation spreading.

Core paradigm:
  Before (broken): query → LLM computes brightness → retrieve
  After (correct): query → embedding → seed nodes → activation spread → retrieve

Brightness is NOT computed by LLM per-query. It emerges from:
  - Historical access frequency (access_count)
  - Recency (last_accessed)  
  - Relationship strength (edge weight × co_activation_count)
  - Natural decay (decay_rate over time)

This produces stable, cognitively-plausible retrieval paths that strengthen
with use and decay with disuse — just like human memory.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from star_graph.math_utils import cosine_sim as _cosine_sim

# Default decay: ~50% per 30 days for weak edges, ~10% per 30 days for strong
DECAY_RATE_WEAK = 0.023  # ln(2)/30
DECAY_RATE_STRONG = 0.0035  # ln(1.1)/30
ACTIVATION_THRESHOLD = 0.15
MAX_SPREAD_DEPTH = 4
MAX_ACTIVATED_NODES = 50


@dataclass
class ActivatedNode:
    """A node in the graph with its computed activation value."""
    node_id: str
    activation: float
    depth: int
    path: list[str]
    source: str  # "seed", "spread", "reinforced"


def _batch_cosine_sim(query: list[float], embeddings: list[list[float]]) -> np.ndarray | None:
    """Cosine similarity of `query` against every vector in `embeddings`.

    Mirrors math_utils.cosine_sim semantics (0.0 for near-zero denominators).
    Returns None when shapes are ragged so the caller can fall back to the
    per-pair Python loop.
    """
    d = len(query)
    if d == 0 or any(len(e) != d for e in embeddings):
        return None
    mat = np.asarray(embeddings, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    dots = mat @ q
    denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
    sims = np.zeros_like(dots)
    valid = denom >= 1e-12
    sims[valid] = dots[valid] / denom[valid]
    return sims


def compute_edge_strength(edge, now: float | None = None) -> float:
    """Compute the effective strength of an edge considering time decay.

    strength = weight × (1 - decay_rate)^days_since_last_access
    """
    if now is None:
        now = time.time()
    days_since = (now - edge.last_activated_at) / 86400.0
    if days_since <= 0:
        return edge.weight
    decay = math.exp(-edge.decay_rate * days_since)
    return edge.weight * decay


class ActivationGraph:
    """Manages activation-based retrieval from a memory graph.

    Retrieval flow:
      1. Embed query → find seed nodes via cosine similarity
      2. BFS spread from seeds, accumulating activation
      3. Activation = sum(seed_sim × edge_strength × recency_boost)
      4. Return top-activated nodes
    
    When a node is retrieved, its edges are reinforced (strengthen + co_activate).
    """

    def __init__(self, graph, embedder=None):
        self.graph = graph
        self._embedder = embedder
        self._activation_cache: dict[str, float] = {}

    # ── Seed node selection ─────────────────────────────────
    
    def find_seeds(self, query_embedding: list[float],
                   top_k: int = 5,
                   min_similarity: float = 0.3) -> list[tuple[str, float]]:
        """Find seed nodes by embedding similarity.

        Returns list of (node_id, similarity) sorted descending.

        Vectorized: all anchor similarities come from a single
        matrix-vector product instead of a per-anchor Python loop.
        """
        candidates = [
            (aid, anchor.embedding)
            for aid, anchor in self.graph.anchors.items()
            if hasattr(anchor, 'embedding') and anchor.embedding
        ]
        if not candidates:
            return []
        ids = [aid for aid, _ in candidates]
        embs = [emb for _, emb in candidates]
        sims = _batch_cosine_sim(query_embedding, embs)
        if sims is None:  # ragged embedding dims — per-pair fallback
            sims = [_cosine_sim(query_embedding, emb) for emb in embs]
        else:
            sims = sims.tolist()
        scored: list[tuple[float, str]] = [
            (sim, aid) for aid, sim in zip(ids, sims) if sim >= min_similarity
        ]
        scored.sort(key=lambda x: -x[0])
        return [(aid, sim) for sim, aid in scored[:top_k]]

    # ── Activation spread ───────────────────────────────────

    def spread(self, seeds: list[tuple[str, float]], 
               max_depth: int = MAX_SPREAD_DEPTH,
               max_nodes: int = MAX_ACTIVATED_NODES,
               activation_threshold: float = ACTIVATION_THRESHOLD,
               now: float | None = None) -> list[ActivatedNode]:
        """BFS activation spread from seed nodes.
        
        Args:
            seeds: (node_id, seed_similarity) pairs
            max_depth: maximum BFS depth
            max_nodes: cap on returned nodes
            
        Returns:
            Sorted list of ActivatedNode, highest activation first
        """
        if now is None:
            now = time.time()
            
        activated: dict[str, float] = {}  # node_id → cumulative activation
        depths: dict[str, int] = {}       # node_id → BFS depth at first activation
        paths: dict[str, list[str]] = {}  # node_id → activation path from seed
        seen: set[str] = set()
        seed_ids = {nid for nid, _ in seeds}
        queue: deque[tuple[str, float, int, list[str]]] = deque()  # node_id, act, depth, path

        # Initialize queue with seeds at activation = seed_similarity
        for nid, sim in seeds:
            if nid not in seen:
                seen.add(nid)
                activated[nid] = sim
                depths[nid] = 0
                paths[nid] = [nid]
                queue.append((nid, sim, 0, [nid]))

        # BFS spreading — can propagate through dormant memories
        # Dormant memories have low activation but are NOT ignored.
        # They can be "re-lit" by strong activation from neighbors.
        while queue and len(activated) < max_nodes:
            nid, current_act, depth, path = queue.popleft()

            if depth >= max_depth:
                continue

            # Get neighboring nodes via edges
            neighbors = self.graph._adjacency.get(nid, set())
            if not neighbors:
                continue

            for neighbor_id in neighbors:
                if neighbor_id in seen:
                    continue

                # Compute edge strength
                edge_key = self.graph._key(nid, neighbor_id)
                edge = self.graph.edges.get(edge_key)
                if edge is None:
                    continue

                edge_strength = compute_edge_strength(edge, now)
                # Even decayed edges can propagate if the parent activation is strong enough
                if edge_strength < activation_threshold and current_act < 0.5:
                    continue  # only strong activation can cross weak edges

                # Activation = parent_act × edge_strength × recency_boost
                hours_since = (now - edge.last_activated_at) / 3600.0
                recency_boost = math.exp(-hours_since / 720.0)  # 30-day half-life

                child_act = current_act * edge_strength * (0.5 + 0.5 * recency_boost)

                # Allow VERY weak activation (dormant memories can still be found)
                if child_act < 0.001:
                    continue

                seen.add(neighbor_id)
                new_path = path + [neighbor_id]

                if neighbor_id in activated:
                    activated[neighbor_id] = max(activated[neighbor_id], child_act)
                else:
                    activated[neighbor_id] = child_act
                    depths[neighbor_id] = depth + 1
                    paths[neighbor_id] = new_path

                queue.append((neighbor_id, child_act, depth + 1, new_path))

        # Convert to sorted list — carry the real BFS path and depth so
        # downstream consumers (observatory reinforcement, traces) see the
        # actual activation route instead of a stub path of length 1.
        result = [
            ActivatedNode(
                node_id=nid,
                activation=act,
                depth=depths.get(nid, 0),
                path=list(paths.get(nid, [nid])),
                source="seed" if nid in seed_ids else "spread"
            )
            for nid, act in activated.items()
        ]
        result.sort(key=lambda x: -x.activation)
        return result[:max_nodes]

    # ── Edge reinforcement ──────────────────────────────────
    
    def reinforce_path(self, path: list[str], delta: float = 0.05) -> int:
        """Strengthen all edges along a retrieval path.
        
        Returns number of reinforced edges.
        """
        count = 0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            key = self.graph._key(a, b)
            edge = self.graph.edges.get(key)
            if edge:
                edge.strengthen(delta)
                count += 1
            else:
                # Create new edge if it doesn't exist
                self.graph.add_edge(a, b, weight=0.3, edge_type="topical")
                count += 1
        return count

    # ── Decay all edges (reduce weight, NEVER delete) ──────────
    
    def decay_all(self, now: float | None = None) -> int:
        """Apply time decay to all edges.
        
        Edges are NEVER deleted. Their weight is gradually reduced.
        A weight of 0.001 means the connection is virtually dormant
        but can still be traversed by strong activation.
        
        Returns count of edges that fell below 0.01 threshold.
        """
        if now is None:
            now = time.time()
        low_count = 0
        
        for key, edge in self.graph.edges.items():
            days_since = (now - edge.last_activated_at) / 86400.0
            if days_since > 0:
                decay = math.exp(-edge.decay_rate * days_since)
                edge.weight = max(0.001, edge.weight * decay)  # never go to 0
                if edge.weight < 0.01:
                    low_count += 1
        
        return low_count


def get_activation_graph(graph, embedder=None) -> ActivationGraph:
    """Get or create an activation graph for a star graph."""
    if not hasattr(graph, '_activation_graph') or graph._activation_graph is None:
        graph._activation_graph = ActivationGraph(graph, embedder)
    return graph._activation_graph
