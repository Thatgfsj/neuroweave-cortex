"""Hebbian Edge Learning — co-activation tracking and Hebbian weight updates."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph, Edge


@dataclass
class HebbianConfig:
    """Configuration for Hebbian edge learning."""

    learning_rate: float = 0.01
    decay_rate: float = 0.001
    reinforcement_threshold: int = 3
    max_edge_weight: float = 1.0
    min_edge_weight: float = 0.01
    prune_threshold: float = 0.01
    cooccurrence_window_seconds: float = 300.0
    max_edges_per_node: int = 50


class CoActivationTracker:
    """Tracks which memories are activated together within a sliding window."""

    def __init__(self, window_seconds: float = 300.0) -> None:
        self.window_seconds = window_seconds
        self._sliding_window: deque[tuple[str, float]] = deque()

    def record_activation(self, anchor_id: str, timestamp: float | None = None) -> None:
        """Record that an anchor was activated."""
        ts = timestamp if timestamp is not None else time.time()
        self._sliding_window.append((anchor_id, ts))

    def get_coactivated_pairs(self) -> list[tuple[str, str, float]]:
        """Return pairs of anchors activated within the window, with co-occurrence counts."""
        now = time.time()
        cutoff = now - self.window_seconds
        recent: list[tuple[str, float]] = [
            (aid, ts) for aid, ts in self._sliding_window if ts >= cutoff
        ]
        pair_counts: dict[tuple[str, str], int] = {}
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                aid_a, ts_a = recent[i]
                aid_b, ts_b = recent[j]
                if aid_a == aid_b:
                    continue
                if abs(ts_a - ts_b) > self.window_seconds:
                    continue
                key = (aid_a, aid_b) if aid_a < aid_b else (aid_b, aid_a)
                pair_counts[key] = pair_counts.get(key, 0) + 1
        return [(a, b, float(cnt)) for (a, b), cnt in pair_counts.items()]

    def get_recent_activations(self, n: int = 100) -> list[tuple[str, float]]:
        """Return the most recent activation timestamps."""
        items = list(self._sliding_window)
        return items[-n:][::-1]

    def clear_stale(self, max_age_seconds: float = 3600.0) -> None:
        """Remove activation records older than max_age_seconds."""
        now = time.time()
        cutoff = now - max_age_seconds
        self._sliding_window = deque(
            (aid, ts) for aid, ts in self._sliding_window if ts >= cutoff
        )


@dataclass
class _HebbianStats:
    """Internal accumulator for learner statistics."""

    total_edges: int = 0
    avg_weight: float = 0.0
    reinforced_count: int = 0
    pruned_count: int = 0
    decayed_count: int = 0
    edge_budget_evicted_count: int = 0
    created_count: int = 0

    @property
    def total_weight(self) -> float:
        return self.avg_weight * self.total_edges


class HebbianLearner:
    """Applies Hebbian learning rules to edges in a StarGraph."""

    def __init__(
        self,
        graph: StarGraph,
        config: HebbianConfig | None = None,
        activation_tracker: CoActivationTracker | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or HebbianConfig()
        self.tracker = activation_tracker or CoActivationTracker(
            window_seconds=self.config.cooccurrence_window_seconds
        )
        self._last_decay_time: float = time.time()
        self._pruned_count: int = 0
        self._reinforced_count: int = 0
        self._created_count: int = 0
        self._budget_evicted_count: int = 0

    def _edge_key(self, a: str, b: str) -> tuple[str, str]:
        return self.graph._key(a, b)

    def _find_edge(self, a: str, b: str) -> Edge | None:
        key = self._edge_key(a, b)
        return self.graph.edges.get(key)

    def reinforce(self, anchor_id_a: str, anchor_id_b: str) -> Edge | None:
        """Strengthen the edge between two co-activated anchors. Creates edge if none exists."""
        existing = self._find_edge(anchor_id_a, anchor_id_b)
        if existing is not None:
            old_w = existing.weight
            existing.weight = min(
                self.config.max_edge_weight,
                old_w + self.config.learning_rate,
            )
            existing.co_activation_count += 1
            existing.last_activated_at = time.time()
            if existing.weight > old_w:
                self._reinforced_count += 1
            return existing

        initial_weight = min(
            self.config.max_edge_weight,
            self.config.learning_rate,
        )
        edge = self.graph.add_edge(
            anchor_id_a,
            anchor_id_b,
            weight=initial_weight,
            edge_type="topical",
        )
        if edge is not None:
            self._created_count += 1
        return edge

    def decay(self, anchor_id: str) -> int:
        """Decay all edges connected to this anchor. Returns count of edges decayed."""
        count = 0
        now = time.time()
        neighbors = list(self.graph._adjacency.get(anchor_id, set()))
        for neighbor in neighbors:
            key = self._edge_key(anchor_id, neighbor)
            edge = self.graph.edges.get(key)
            if edge is None:
                continue
            days_since = (now - edge.last_activated_at) / 86400.0
            new_weight = edge.weight * (1.0 - self.config.decay_rate * days_since)
            edge.weight = max(0.0, new_weight)
            count += 1
        return count

    def decay_all(self) -> dict:
        """Decay all edges in graph. Returns {"decayed": N, "pruned": M}."""
        self._last_decay_time = time.time()
        now = self._last_decay_time
        decayed = 0
        pruned = 0
        to_prune: list[tuple[str, str]] = []
        for key, edge in list(self.graph.edges.items()):
            days_since = (now - edge.last_activated_at) / 86400.0
            new_weight = edge.weight * (1.0 - self.config.decay_rate * days_since)
            edge.weight = max(0.0, new_weight)
            decayed += 1
            if edge.weight < self.config.min_edge_weight:
                to_prune.append(key)
        for key in to_prune:
            a, b = key
            self.graph.edges.pop(key, None)
            self.graph._adjacency[a].discard(b)
            self.graph._adjacency[b].discard(a)
            pruned += 1
        self._pruned_count += pruned
        return {"decayed": decayed, "pruned": pruned}

    def hebbian_update(self, anchor_id_a: str, anchor_id_b: str) -> Edge | None:
        """Full Hebbian update: Δw = η × coactivation_count × w_current."""
        existing = self._find_edge(anchor_id_a, anchor_id_b)
        if existing is not None:
            co_count = existing.co_activation_count
            if co_count < self.config.reinforcement_threshold:
                return existing
            delta = self.config.learning_rate * co_count * existing.weight
            existing.weight = min(self.config.max_edge_weight, existing.weight + delta)
            existing.last_activated_at = time.time()
            self._reinforced_count += 1
            return existing

        initial_weight = self.config.learning_rate
        edge = self.graph.add_edge(
            anchor_id_a,
            anchor_id_b,
            weight=min(self.config.max_edge_weight, initial_weight),
            edge_type="topical",
        )
        if edge is not None:
            edge.co_activation_count = 1
            self._created_count += 1
        return edge

    def update_from_tracker(self) -> int:
        """Pull co-activated pairs from tracker and apply Hebbian updates. Returns count of edges updated."""
        pairs = self.tracker.get_coactivated_pairs()
        updated = 0
        for anchor_a, anchor_b, co_count in pairs:
            existing = self._find_edge(anchor_a, anchor_b)
            if existing is not None:
                existing.co_activation_count = max(existing.co_activation_count, int(co_count))
                if co_count >= self.config.reinforcement_threshold:
                    delta = self.config.learning_rate * co_count * existing.weight
                    existing.weight = min(self.config.max_edge_weight, existing.weight + delta)
                    existing.last_activated_at = time.time()
                    self._reinforced_count += 1
                    updated += 1
                continue
            if co_count >= self.config.reinforcement_threshold:
                initial_weight = min(
                    self.config.max_edge_weight,
                    self.config.learning_rate * co_count,
                )
                edge = self.graph.add_edge(
                    anchor_a,
                    anchor_b,
                    weight=initial_weight,
                    edge_type="topical",
                )
                if edge is not None:
                    edge.co_activation_count = int(co_count)
                    self._created_count += 1
                    updated += 1
        return updated

    def prune_weak_edges(self) -> list[str]:
        """Remove edges below prune_threshold. Returns list of removed edge keys."""
        removed: list[str] = []
        to_remove: list[tuple[str, str]] = []
        for key, edge in self.graph.edges.items():
            if edge.weight <= self.config.prune_threshold:
                to_remove.append(key)
        for key in to_remove:
            a, b = key
            self.graph.edges.pop(key, None)
            self.graph._adjacency[a].discard(b)
            self.graph._adjacency[b].discard(a)
            removed.append(f"{a}<->{b}")
        self._pruned_count += len(removed)
        return removed

    def enforce_edge_budget(self, anchor_id: str) -> list[str]:
        """Remove weakest edges if node exceeds max_edges_per_node. Returns removed edge keys."""
        neighbors = self.graph._adjacency.get(anchor_id, set())
        if len(neighbors) <= self.config.max_edges_per_node:
            return []

        scored: list[tuple[str, float]] = []
        for neighbor in neighbors:
            key = self._edge_key(anchor_id, neighbor)
            edge = self.graph.edges.get(key)
            weight = edge.weight if edge else 0.0
            scored.append((neighbor, weight))
        scored.sort(key=lambda x: x[1])

        excess = len(neighbors) - self.config.max_edges_per_node
        removed: list[str] = []
        for neighbor, _ in scored[:excess]:
            key = self._edge_key(anchor_id, neighbor)
            self.graph.edges.pop(key, None)
            self.graph._adjacency[anchor_id].discard(neighbor)
            self.graph._adjacency[neighbor].discard(anchor_id)
            removed.append(f"{anchor_id}<->{neighbor}")
        self._budget_evicted_count += len(removed)
        return removed

    def snapshot(self) -> dict:
        """Return learning statistics."""
        edges = self.graph.edges
        total = len(edges)
        avg_w = sum(e.weight for e in edges.values()) / max(1, total)
        return {
            "total_edges": total,
            "avg_weight": round(avg_w, 4),
            "reinforced_count": self._reinforced_count,
            "pruned_count": self._pruned_count,
            "decayed_count": int(
                (time.time() - self._last_decay_time) / 86400.0 * total
            ),
            "edge_budget_evicted_count": self._budget_evicted_count,
            "created_count": self._created_count,
            "last_decay_time": self._last_decay_time,
            "tracker_window_size": len(self.tracker._sliding_window),
        }
