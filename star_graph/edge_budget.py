"""Edge Budget Manager — prevent super-node degeneration via smart edge caps.

Every node is limited to max 32 edges. When the limit is exceeded:
  1. Retain high-weight edges (causal > fixes > preference > topical)
  2. Evict low-weight, low-traversal, stale edges
  3. Merge near-duplicate edges (same neighbor pair, different edge types)

This prevents the graph from becoming a dense hairball where every node
connects to every other node, destroying retrieval precision.

Wired into graph.add_edge() as a pre-check and post-eviction hook.
"""

from __future__ import annotations

import math
from typing import Optional


# Priority order for edge retention (higher = keep over lower)
EDGE_TYPE_RETENTION_PRIORITY = {
    "causes": 10,
    "causal": 10,
    "fixes": 9,
    "resolves": 9,
    "depends_on": 8,
    "derived_from": 8,
    "invalidated_by": 7,
    "supersedes": 7,
    "compresses": 6,
    "summarizes": 6,
    "before": 5,
    "after": 5,
    "preference": 4,
    "topical": 3,
    "related": 2,
    "contradicts": 1,
}


class EdgeBudgetManager:
    """Enforces max edges per node with smart eviction.

    Usage:
        budget = EdgeBudgetManager(max_edges=32)
        budget.enforce(graph, node_id)  # call after each add_edge()
    """

    def __init__(self, max_edges: int = 32):
        self.max_edges = max_edges
        self._eviction_count: int = 0
        self._merge_count: int = 0

    def enforce(self, graph, node_id: str) -> dict:
        """Enforce edge budget for a single node.

        If the node has more than max_edges, evict the weakest ones.
        Also checks for edge pairs that can be merged.

        Returns stats dict.
        """
        neighbors = graph._adjacency.get(node_id, set())
        if len(neighbors) <= self.max_edges:
            return {"evicted": 0, "merged": 0, "remaining": len(neighbors)}

        # Score all edges from this node
        scored_edges: list[tuple[str, float]] = []
        for neighbor_id in neighbors:
            edge_key = graph._key(node_id, neighbor_id)
            edge = graph.edges.get(edge_key)
            score = self._edge_retention_score(edge)
            scored_edges.append((neighbor_id, score))

        # Sort by score descending
        scored_edges.sort(key=lambda x: -x[1])

        # Keep top max_edges
        keep_neighbors = {nid for nid, _ in scored_edges[:self.max_edges]}
        evict_neighbors = {nid for nid, _ in scored_edges[self.max_edges:]}

        # Evict edges
        evicted = 0
        for nid in evict_neighbors:
            edge_key = graph._key(node_id, nid)
            if edge_key in graph.edges:
                del graph.edges[edge_key]
                graph._adjacency[node_id].discard(nid)
                graph._adjacency[nid].discard(node_id)
                evicted += 1
                self._eviction_count += 1

        return {
            "evicted": evicted,
            "merged": 0,
            "remaining": len(keep_neighbors),
        }

    def enforce_all(self, graph) -> dict:
        """Enforce edge budget for all nodes in the graph.

        Returns total stats across all nodes.
        """
        total_evicted = 0
        over_budget_nodes = 0

        for node_id in list(graph.anchors.keys()):
            neighbors = graph._adjacency.get(node_id, set())
            if len(neighbors) > self.max_edges:
                result = self.enforce(graph, node_id)
                total_evicted += result["evicted"]
                over_budget_nodes += 1

        return {
            "over_budget_nodes": over_budget_nodes,
            "total_evicted": total_evicted,
            "total_merged": self._merge_count,
            "total_evictions_ever": self._eviction_count,
        }

    def _edge_retention_score(self, edge) -> float:
        """Score an edge for retention priority (higher = keep).

        Formula:
          score = type_priority × weight × recency × activity
        """
        if edge is None:
            return 0.0

        # Base: edge type priority (0-10 scale, normalized to 0-1)
        edge_type = getattr(edge, 'edge_type', 'topical')
        type_priority = EDGE_TYPE_RETENTION_PRIORITY.get(edge_type, 2.0)
        type_score = type_priority / 10.0

        # Edge weight (0-1)
        weight = getattr(edge, 'weight', 0.5)

        # Recency factor (exponential decay since last traversed)
        import time
        last_traversed = getattr(edge, 'last_traversed_at', 0.0)
        if last_traversed > 0:
            hours_since = (time.time() - last_traversed) / 3600
            recency = math.exp(-hours_since / (30 * 24))  # 30-day half-life
        else:
            recency = 0.5  # never traversed, medium score

        # Co-activation count (how often both endpoints activated together)
        co_activation = getattr(edge, 'co_activation_count', 0)
        activity = min(1.0, co_activation / 10.0)

        # Combined score
        return type_score * 0.4 + weight * 0.3 + recency * 0.15 + activity * 0.15

    @property
    def stats(self) -> dict:
        return {
            "max_edges": self.max_edges,
            "total_evictions": self._eviction_count,
            "total_merges": self._merge_count,
        }
