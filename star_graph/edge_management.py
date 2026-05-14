"""Edge Management — budget enforcement + continuous time decay.

Merged from edge_budget.py (#49) and edge_decay.py (#54).

Two concerns, one module:
  1. EdgeBudget — per-node edge cap (default 32) with smart retention scoring
  2. EdgeDecay — continuous time-based weight decay with adaptive rates

Edge budget prevents super-node degeneration. Edge decay ensures stale
connections fade over time. Together they maintain a sparse, high-quality graph.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from .graph import Edge

# ═══════════════════════════════════════════════════════════════════════
# Edge Type Retention Priority (ex-edge_budget)
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
# Edge Decay Rates (ex-edge_decay)
# ═══════════════════════════════════════════════════════════════════════

DECAY_RATE_PER_HOUR = {
    "causes": 0.0003,
    "fixes": 0.0003,
    "resolves": 0.0003,
    "depends_on": 0.0005,
    "derived_from": 0.0005,
    "invalidated_by": 0.0005,
    "supersedes": 0.0005,
    "compresses": 0.001,
    "summarizes": 0.001,
    "before": 0.002,
    "after": 0.002,
    "topical": 0.005,
    "preference": 0.003,
    "related": 0.008,
    "contradicts": 0.004,
}
DEFAULT_DECAY_RATE = 0.005

REINFORCEMENT_EXTEND_HOURS = {
    "causes": 720,
    "fixes": 720,
    "depends_on": 360,
    "topical": 24,
    "related": 12,
}
DEFAULT_REINFORCEMENT_HOURS = 24


# ═══════════════════════════════════════════════════════════════════════
# EdgeBudgetManager
# ═══════════════════════════════════════════════════════════════════════

class EdgeBudgetManager:
    """Enforces max edges per node with smart eviction.

    Usage:
        budget = EdgeBudgetManager(max_edges=32)
        budget.enforce(graph, node_id)
    """

    def __init__(self, max_edges: int = 32):
        self.max_edges = max_edges
        self._eviction_count: int = 0
        self._merge_count: int = 0

    def enforce(self, graph, node_id: str) -> dict:
        neighbors = graph._adjacency.get(node_id, set())
        if len(neighbors) <= self.max_edges:
            return {"evicted": 0, "merged": 0, "remaining": len(neighbors)}

        scored_edges: list[tuple[str, float]] = []
        for neighbor_id in neighbors:
            edge_key = graph._key(node_id, neighbor_id)
            edge = graph.edges.get(edge_key)
            score = self._edge_retention_score(edge)
            scored_edges.append((neighbor_id, score))

        scored_edges.sort(key=lambda x: -x[1])
        keep_neighbors = {nid for nid, _ in scored_edges[:self.max_edges]}
        evict_neighbors = {nid for nid, _ in scored_edges[self.max_edges:]}

        evicted = 0
        for nid in evict_neighbors:
            edge_key = graph._key(node_id, nid)
            if edge_key in graph.edges:
                del graph.edges[edge_key]
                graph._adjacency[node_id].discard(nid)
                graph._adjacency[nid].discard(node_id)
                evicted += 1
                self._eviction_count += 1

        return {"evicted": evicted, "merged": 0, "remaining": len(keep_neighbors)}

    def enforce_all(self, graph) -> dict:
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
        if edge is None:
            return 0.0
        edge_type = getattr(edge, 'edge_type', 'topical')
        type_priority = EDGE_TYPE_RETENTION_PRIORITY.get(edge_type, 2.0)
        type_score = type_priority / 10.0
        weight = getattr(edge, 'weight', 0.5)
        last_traversed = getattr(edge, 'last_traversed_at', 0.0)
        if last_traversed > 0:
            hours_since = (time.time() - last_traversed) / 3600
            recency = math.exp(-hours_since / (30 * 24))
        else:
            recency = 0.5
        co_activation = getattr(edge, 'co_activation_count', 0)
        activity = min(1.0, co_activation / 10.0)
        return type_score * 0.4 + weight * 0.3 + recency * 0.15 + activity * 0.15

    @property
    def stats(self) -> dict:
        return {
            "max_edges": self.max_edges,
            "total_evictions": self._eviction_count,
            "total_merges": self._merge_count,
        }


# ═══════════════════════════════════════════════════════════════════════
# EdgeDecayManager
# ═══════════════════════════════════════════════════════════════════════

class EdgeDecayManager:
    """Continuous edge decay with adaptive rates and TTL extension.

    Usage:
        mgr = EdgeDecayManager()
        mgr.apply_decay(edge)       # call before edge traversal
        mgr.reinforce(edge)         # call after successful traversal
        mgr.decay_all_edges(graph)  # bulk decay during sleep
    """

    def __init__(self, base_decay_multiplier: float = 1.0,
                 min_edge_weight: float = 0.02):
        self.base_decay_multiplier = base_decay_multiplier
        self.min_edge_weight = min_edge_weight
        self._total_decayed = 0
        self._total_evicted = 0
        self._total_reinforced = 0

    def decay_rate_for(self, edge: Edge) -> float:
        edge_type = getattr(edge, 'edge_type', 'topical')
        base_rate = DECAY_RATE_PER_HOUR.get(edge_type, DEFAULT_DECAY_RATE)
        success_rate = getattr(edge, 'success_rate', 0.5)
        if success_rate > 0.7:
            base_rate *= 0.5
        elif success_rate < 0.3:
            base_rate *= 2.0
        if getattr(edge, 'is_stale', False):
            base_rate *= 2.0
        return base_rate * self.base_decay_multiplier

    def apply_decay(self, edge: Edge, now: float | None = None) -> float:
        if now is None:
            now = time.time()
        last_activated = getattr(edge, 'last_activated_at', edge.created_at)
        hours_idle = (now - last_activated) / 3600.0
        if hours_idle <= 0:
            return edge.weight
        rate = self.decay_rate_for(edge)
        new_weight = edge.weight * math.exp(-rate * hours_idle)
        edge.weight = max(self.min_edge_weight, new_weight)
        edge.last_activated_at = now
        self._total_decayed += 1
        return edge.weight

    def reinforce(self, edge: Edge, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        edge.strengthen(0.03)
        edge.last_activated_at = now
        edge_type = getattr(edge, 'edge_type', 'topical')
        extend_hours = REINFORCEMENT_EXTEND_HOURS.get(
            edge_type, DEFAULT_REINFORCEMENT_HOURS)
        new_valid_until = now + extend_hours * 3600
        current_valid = getattr(edge, 'valid_until', 0)
        if new_valid_until > current_valid:
            edge.valid_until = new_valid_until
        self._total_reinforced += 1

    def is_viable(self, edge: Edge, now: float | None = None) -> bool:
        if now is None:
            now = time.time()
        valid_until = getattr(edge, 'valid_until', 0)
        if valid_until > 0 and now > valid_until:
            return False
        if edge.weight < self.min_edge_weight:
            return False
        return True

    def decay_all_edges(self, graph, now: float | None = None) -> dict:
        if now is None:
            now = time.time()
        stats = {"decayed": 0, "evicted": 0, "reinforced": 0}
        to_evict = []
        for key, edge in list(graph.edges.items()):
            new_weight = self.apply_decay(edge, now)
            if new_weight <= self.min_edge_weight:
                to_evict.append(key)
        for key in to_evict:
            a, b = key
            if key in graph.edges:
                del graph.edges[key]
            graph._adjacency[a].discard(b)
            graph._adjacency[b].discard(a)
            stats["evicted"] += 1
            self._total_evicted += 1
        stats["decayed"] = len(graph.edges)
        return stats

    @property
    def stats(self) -> dict:
        return {
            "base_decay_multiplier": self.base_decay_multiplier,
            "min_edge_weight": self.min_edge_weight,
            "total_decayed": self._total_decayed,
            "total_evicted": self._total_evicted,
            "total_reinforced": self._total_reinforced,
        }
