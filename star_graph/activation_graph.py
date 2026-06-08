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
from dataclasses import dataclass, field
from typing import Optional

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
        """
        scored: list[tuple[float, str]] = []
        for aid, anchor in self.graph.anchors.items():
            if not hasattr(anchor, 'embedding') or not anchor.embedding:
                continue
            sim = _cosine_sim(query_embedding, anchor.embedding)
            if sim >= min_similarity:
                scored.append((sim, aid))
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
        seen: set[str] = set()
        queue: list[tuple[str, float, int, list[str]]] = []  # node_id, act, depth, path
        
        # Initialize queue with seeds at activation = seed_similarity
        for nid, sim in seeds:
            if nid not in seen:
                seen.add(nid)
                activated[nid] = sim
                queue.append((nid, sim, 0, [nid]))
        
        # BFS spreading — can propagate through dormant memories
        # Dormant memories have low activation but are NOT ignored.
        # They can be "re-lit" by strong activation from neighbors.
        while queue and len(activated) < max_nodes:
            nid, current_act, depth, path = queue.pop(0)
            
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
                
                queue.append((neighbor_id, child_act, depth + 1, new_path))
        
        # Convert to sorted list
        result = [
            ActivatedNode(
                node_id=nid,
                activation=act,
                depth=0 if seeds and nid == seeds[0][0] else 1,
                path=[nid],
                source="seed" if any(nid == s[0] for s in seeds) else "spread"
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
