"""Online micro-consolidation — v0.4 real embeddings.

Lightweight updates between full sleep cycles. Three modes:
- online:  micro-consolidation every N interactions (SWR replay + Hebbian only)
- nightly: full 9-phase sleep at scheduled time
- hybrid:  online for quick updates + nightly for deep consolidation

Now uses real embedding similarity instead of bigram overlap.
"""

from __future__ import annotations

import time
from typing import Optional

from .anchor import Anchor
from .graph import StarGraph, Edge


class OnlineConsolidator:
    """Lightweight consolidator for real-time use. Target: <50ms per cycle."""

    def __init__(self, graph: StarGraph, interval: int = 5,
                 max_anchors_per_cycle: int = 20):
        self.graph = graph
        self.interval = interval
        self.max_anchors = max_anchors_per_cycle
        self.interaction_count = 0
        self.pending_anchors: list[Anchor] = []
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def record_interaction(self, anchor: Anchor | None = None) -> None:
        self.interaction_count += 1
        if anchor:
            self.pending_anchors.append(anchor)
        if self.interaction_count % self.interval == 0:
            self._micro_sleep()

    def _micro_sleep(self) -> dict:
        t0 = time.perf_counter()

        if self.pending_anchors:
            prioritized = sorted(
                self.pending_anchors[-self.max_anchors:],
                key=lambda a: abs(a.vector.emotional_valence) * 0.4
                              + a.vector.surprise * 0.35
                              + a.vector.importance * 0.25,
                reverse=True,
            )

            embedder = self._get_embedder()

            for anchor in prioritized[:self.max_anchors // 2]:
                existing = self.graph.anchors.get(anchor.id)
                if existing:
                    existing.activate()
                    existing.replay_count += 1
                else:
                    if not anchor.embedding:
                        anchor.embedding = embedder.encode(anchor.text)
                    self.graph.add_anchor(anchor)

                anchor_emb = existing.embedding if existing else anchor.embedding
                if not anchor_emb:
                    anchor_emb = embedder.encode(anchor.text)

                # Connect to nearest existing anchors by embedding similarity
                for other_id, other in list(self.graph.anchors.items())[:self.max_anchors]:
                    if other_id == anchor.id or not other.embedding:
                        continue
                    sim = self._cosine_sim(anchor_emb, other.embedding)
                    if sim > 0.6:
                        key = self.graph._key(anchor.id, other_id)
                        if key in self.graph.edges:
                            self.graph.edges[key].strengthen(0.02)
                        else:
                            self.graph.add_edge(anchor.id, other_id, weight=sim,
                                                edge_type="topical")

            self.pending_anchors.clear()

        # Micro Hebbian
        now = time.time()
        updated = 0
        for edge in list(self.graph.edges.values())[:self.max_anchors * 2]:
            hours = (now - edge.last_activated_at) / 3600
            if edge.co_activation_count > 0 and hours < 1:
                edge.strengthen(0.01)
                updated += 1

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "latency_ms": round(latency_ms, 2),
            "edges_updated": updated,
            "anchors_in_graph": len(self.graph.anchors),
        }

    def force_consolidate(self) -> dict:
        self.interaction_count = self.interval
        return self._micro_sleep()

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(min_len))
        na = (sum(x * x for x in a)) ** 0.5
        nb = (sum(x * x for x in b)) ** 0.5
        return dot / (na * nb + 1e-8)
