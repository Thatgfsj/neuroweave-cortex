"""Online micro-consolidation — lightweight updates between full sleep cycles.

In production, you can't wait until 2 AM to consolidate. This module provides
mini sleep cycles that run after every N interactions, taking <50ms.

Three modes:
- online:  micro-consolidation every N interactions (SWR replay + Hebbian only)
- nightly: full 9-phase sleep at scheduled time (existing sleep.py)
- hybrid:  online for quick updates + nightly for deep consolidation
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from .anchor import Anchor
from .graph import StarGraph, Edge


class OnlineConsolidator:
    """Lightweight consolidator for real-time use.

    Runs a stripped-down sleep cycle (SWR replay + Hebbian update)
    after every `interval` interactions. Target latency: <50ms.
    """

    def __init__(self, graph: StarGraph, interval: int = 5,
                 max_anchors_per_cycle: int = 20):
        self.graph = graph
        self.interval = interval
        self.max_anchors = max_anchors_per_cycle
        self.interaction_count = 0
        self.pending_anchors: list[Anchor] = []

    def record_interaction(self, anchor: Anchor | None = None) -> None:
        """Called after each user interaction."""
        self.interaction_count += 1
        if anchor:
            self.pending_anchors.append(anchor)

        if self.interaction_count % self.interval == 0:
            self._micro_sleep()

    def _micro_sleep(self) -> dict:
        """Run a micro sleep cycle — SWR replay + Hebbian only."""
        t0 = time.perf_counter()

        # Phase 1: Mini SWR replay (only top-N most salient pending anchors)
        if self.pending_anchors:
            prioritized = sorted(
                self.pending_anchors[-self.max_anchors:],
                key=lambda a: abs(a.vector.emotional_valence) * 0.4
                              + a.vector.surprise * 0.35
                              + a.vector.importance * 0.25,
                reverse=True,
            )

            for anchor in prioritized[:self.max_anchors // 2]:
                existing = self.graph.anchors.get(anchor.id)
                if existing:
                    existing.activate()
                    existing.replay_count += 1
                else:
                    self.graph.add_anchor(anchor)

                # Quick edge update: connect to nearest existing anchors
                for other_id, other in list(self.graph.anchors.items())[:self.max_anchors]:
                    if other_id == anchor.id:
                        continue
                    overlap = _text_overlap(anchor.text, other.text)
                    if overlap > 0.6:
                        key = self.graph._key(anchor.id, other_id)
                        if key in self.graph.edges:
                            self.graph.edges[key].strengthen(0.02)
                        else:
                            self.graph.add_edge(
                                anchor.id, other_id, weight=overlap,
                                edge_type="topical",
                            )

            # Clear pending (they're now in the graph)
            self.pending_anchors.clear()

        # Phase 2: Micro Hebbian update (only recently active edges)
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
        """Force a consolidation cycle regardless of interval."""
        self.interaction_count = self.interval  # trigger on next check
        return self._micro_sleep()


def _text_overlap(a: str, b: str) -> float:
    def bigrams(s):
        return {s[i:i+2] for i in range(len(s)-1)}
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)
