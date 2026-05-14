"""Edge Time Decay — continuous time-based edge weight decay with adaptive rates.

Edges lose weight over time if not reinforced. Strong edges (causal, fixes)
decay slower. Frequently traversed edges extend their lifetime.

Three mechanisms:
  1. Lazy decay on access: when an edge is read, apply accumulated time decay
  2. Reinforcement extends TTL: each traversal extends valid_until
  3. Adaptive decay rate: edges with high success rate get slower decay

Wired into graph traversal and sleep cycles.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from .graph import Edge, EDGE_TRAVERSAL_WEIGHTS

# Base decay rates per hour by edge type category
DECAY_RATE_PER_HOUR = {
    # Strong structural edges — very slow decay (2-3 month half-life)
    "causes": 0.0003,
    "fixes": 0.0003,
    "resolves": 0.0003,
    "depends_on": 0.0005,
    "derived_from": 0.0005,
    "invalidated_by": 0.0005,
    "supersedes": 0.0005,
    # Medium — moderate decay (1-2 month half-life)
    "compresses": 0.001,
    "summarizes": 0.001,
    "before": 0.002,
    "after": 0.002,
    # Weak association edges — fast decay (1-2 week half-life)
    "topical": 0.005,
    "preference": 0.003,
    "related": 0.008,
    "contradicts": 0.004,
}

DEFAULT_DECAY_RATE = 0.005  # per hour for unknown edge types

# --- Reinforcement: traversal extends TTL by this many hours
REINFORCEMENT_EXTEND_HOURS = {
    "causes": 720,     # 30 days
    "fixes": 720,
    "depends_on": 360,  # 15 days
    "topical": 24,      # 1 day
    "related": 12,      # 12 hours
}

DEFAULT_REINFORCEMENT_HOURS = 24


class EdgeDecayManager:
    """Continuous edge decay with adaptive rates and TTL extension.

    Usage:
        mgr = EdgeDecayManager()
        mgr.apply_decay(edge, now)            # call before edge traversal
        mgr.reinforce(edge)                   # call after successful traversal
        mgr.decay_all_edges(graph)            # bulk decay during sleep
    """

    def __init__(self,
                 base_decay_multiplier: float = 1.0,
                 min_edge_weight: float = 0.02):
        self.base_decay_multiplier = base_decay_multiplier
        self.min_edge_weight = min_edge_weight
        self._total_decayed = 0
        self._total_evicted = 0
        self._total_reinforced = 0

    # ── Per-edge operations ──────────────────────────────────

    def decay_rate_for(self, edge: Edge) -> float:
        """Get the decay rate for a specific edge, based on type and history."""
        edge_type = getattr(edge, 'edge_type', 'topical')
        base_rate = DECAY_RATE_PER_HOUR.get(edge_type, DEFAULT_DECAY_RATE)

        # Adaptive: edges with high success rate decay slower
        success_rate = getattr(edge, 'success_rate', 0.5)
        if success_rate > 0.7:
            base_rate *= 0.5  # high-value edges decay half as fast
        elif success_rate < 0.3:
            base_rate *= 2.0  # low-value edges decay twice as fast

        # Stale edges decay faster
        if getattr(edge, 'is_stale', False):
            base_rate *= 2.0

        return base_rate * self.base_decay_multiplier

    def apply_decay(self, edge: Edge, now: float | None = None) -> float:
        """Apply accumulated time decay to an edge. Call before traversal.

        Returns the new weight (0 means edge should be evicted).
        """
        if now is None:
            now = time.time()
        last_activated = getattr(edge, 'last_activated_at', edge.created_at)
        hours_idle = (now - last_activated) / 3600.0

        if hours_idle <= 0:
            return edge.weight

        rate = self.decay_rate_for(edge)
        decay_factor = math.exp(-rate * hours_idle)
        new_weight = edge.weight * decay_factor

        # Update edge in place
        edge.weight = max(self.min_edge_weight, new_weight)
        edge.last_activated_at = now
        self._total_decayed += 1

        return edge.weight

    def reinforce(self, edge: Edge, now: float | None = None) -> None:
        """Reinforce an edge after successful traversal.

        Strengthens weight and extends valid_until.
        """
        if now is None:
            now = time.time()
        edge.strengthen(0.03)
        edge.last_activated_at = now

        # Extend valid_until
        edge_type = getattr(edge, 'edge_type', 'topical')
        extend_hours = REINFORCEMENT_EXTEND_HOURS.get(
            edge_type, DEFAULT_REINFORCEMENT_HOURS)
        new_valid_until = now + extend_hours * 3600
        current_valid = getattr(edge, 'valid_until', 0)
        if new_valid_until > current_valid:
            edge.valid_until = new_valid_until

        self._total_reinforced += 1

    def is_viable(self, edge: Edge, now: float | None = None) -> bool:
        """Check if an edge is still viable for traversal."""
        if now is None:
            now = time.time()
        # Check explicit expiration
        valid_until = getattr(edge, 'valid_until', 0)
        if valid_until > 0 and now > valid_until:
            return False
        # Check weight below minimum
        if edge.weight < self.min_edge_weight:
            return False
        return True

    # ── Bulk operations ──────────────────────────────────────

    def decay_all_edges(self, graph, now: float | None = None) -> dict:
        """Apply time decay to all edges in the graph. Called during sleep.

        Returns stats: {decayed, evicted, reinforced}.
        """
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

    # ── Stats ────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "base_decay_multiplier": self.base_decay_multiplier,
            "min_edge_weight": self.min_edge_weight,
            "total_decayed": self._total_decayed,
            "total_evicted": self._total_evicted,
            "total_reinforced": self._total_reinforced,
        }
