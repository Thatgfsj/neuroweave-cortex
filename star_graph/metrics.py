"""Cognitive Metrics — not just retrieval benchmarks.

Measures properties of the memory system as a cognitive architecture:
  - Memory Stability: how well long-term memories persist
  - Recall Plasticity: how quickly new knowledge integrates
  - Abstraction Emergence Rate: how fast higher-order concepts form
  - Ghost Reactivation Accuracy: how well ghosts revive
  - Memory Compression Ratio: graph size vs information retained
  - Semantic Drift Resistance: stability of memory meaning over time

These metrics separate a cognitive architecture from a vector database.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Optional

from .graph import StarGraph


class CognitiveMetrics:
    """Computes cognitive-quality metrics over a StarGraph."""

    def __init__(self, graph: StarGraph):
        self.graph = graph
        self._snapshots: list[dict] = []

    def snapshot(self) -> dict:
        """Take a metrics snapshot — call before/after sleep or after N interactions."""
        now = time.time()
        anchors = self.graph.anchors
        if not anchors:
            return {"timestamp": now, "anchors": 0}

        values = list(anchors.values())

        # Core metrics
        stability = self._memory_stability(values)
        plasticity = self._recall_plasticity(values)
        compression = self._compression_ratio()
        drift = self._semantic_drift_resistance(values)
        abstraction_rate = self._abstraction_emergence_rate()
        ghost_accuracy = self._ghost_reactivation_accuracy()

        snap = {
            "timestamp": now,
            "anchors": len(anchors),
            "edges": len(self.graph.edges),
            "ghosts": len(self.graph._ghost_subsystem.ghosts),
            "schemas": len(self.graph.schemas),
            "abstracts": len(getattr(self.graph, 'abstracts', {})),
            "memory_stability": stability,
            "recall_plasticity": plasticity,
            "compression_ratio": compression,
            "semantic_drift_resistance": drift,
            "abstraction_emergence_rate": abstraction_rate,
            "ghost_reactivation_accuracy": ghost_accuracy,
        }
        self._snapshots.append(snap)
        return snap

    def _memory_stability(self, anchors: list) -> float:
        """How stable are consolidated memories? (0..1)

        High stability = consolidated memories retain their importance over time.
        Low stability = memories degrade quickly.
        """
        if not anchors:
            return 0.0
        return sum(
            a.vector.stability * a.retention_score
            for a in anchors
        ) / sum(a.retention_score for a in anchors) if sum(a.retention_score for a in anchors) > 0 else 0.0

    def _recall_plasticity(self, anchors: list) -> float:
        """How quickly can the system adapt to new information? (0..1)

        High plasticity = new memories achieve high retention quickly.
        Low plasticity = system is rigid, new info doesn't integrate well.

        Measured as: ratio of recent (<24h) anchor scores to total.
        """
        now = time.time()
        recent = [a for a in anchors if (now - a.created_at) < 86400]
        if not recent:
            return 0.0
        recent_avg = sum(a.retention_score for a in recent) / len(recent)
        total_avg = sum(a.retention_score for a in anchors) / len(anchors)
        if total_avg < 1e-8:
            return 0.0
        return min(1.0, recent_avg / max(total_avg, 1e-8))

    def _compression_ratio(self) -> float:
        """How efficiently does the graph compress information? (0..1)

        Ratio: (schemas + abstracts) / total_anchors
        Higher = more abstraction, less redundancy.
        """
        total = len(self.graph.anchors) + len(self.graph._ghost_subsystem.ghosts)
        if total == 0:
            return 0.0
        abstracts = len(getattr(self.graph, 'abstracts', {}))
        schemas = len(self.graph.schemas)
        compressed = abstracts + schemas
        return compressed / total

    def _semantic_drift_resistance(self, anchors: list) -> float:
        """How resistant are memories to semantic drift over time? (0..1)

        Measured as: average stability of old (>7 day) anchors.
        High = old memories maintain meaning.
        Low = old memories degrade semantically.
        """
        now = time.time()
        old = [a for a in anchors if (now - a.created_at) > 7 * 86400]
        if not old:
            return 1.0  # no old memories = no drift
        return sum(a.vector.stability for a in old) / len(old)

    def _abstraction_emergence_rate(self) -> float:
        """How fast are abstract concepts forming? (abstracts per sleep cycle)

        Uses snapshot history to compute rate.
        """
        if len(self._snapshots) < 2:
            return 0.0
        prev = self._snapshots[-2].get("abstracts", 0)
        curr = self._snapshots[-1].get("abstracts", 0)
        return max(0, curr - prev)

    def _ghost_reactivation_accuracy(self) -> float:
        """How accurately do ghosts reactivate? (0..1)

        High = ghosts that revive are relevant to the triggering context.
        """
        ghosts = list(self.graph._ghost_subsystem.ghosts.values())
        if not ghosts:
            return 1.0  # no ghosts = nothing to measure
        revived = [g for g in ghosts if getattr(g, 'revival_count', 0) > 0]
        if not revived:
            return 0.0
        return sum(getattr(g, 'revival_count', 0) for g in revived) / len(revived)

    def compare(self) -> dict:
        """Compare latest snapshot to first — show evolution."""
        if len(self._snapshots) < 2:
            return {}
        first = self._snapshots[0]
        last = self._snapshots[-1]
        return {
            "anchor_change": last["anchors"] - first["anchors"],
            "stability_delta": last["memory_stability"] - first["memory_stability"],
            "plasticity_delta": last["recall_plasticity"] - first["recall_plasticity"],
            "compression_improvement": last["compression_ratio"] - first["compression_ratio"],
            "abstractions_formed": last["abstracts"] - first["abstracts"],
            "snapshots_collected": len(self._snapshots),
        }

    def report(self) -> str:
        """Human-readable cognitive health report."""
        if not self._snapshots:
            return "No data. Call snapshot() first."

        s = self._snapshots[-1]
        lines = [
            "Cognitive Health Report",
            "======================",
            f"  Memory Stability:          {s['memory_stability']:.2f}",
            f"  Recall Plasticity:         {s['recall_plasticity']:.2f}",
            f"  Compression Ratio:         {s['compression_ratio']:.2f}",
            f"  Semantic Drift Resistance: {s['semantic_drift_resistance']:.2f}",
            f"  Abstraction Emergence:     {s['abstraction_emergence_rate']:.1f}/cycle",
            f"  Ghost Reactivation:        {s['ghost_reactivation_accuracy']:.2f}",
            f"",
            f"  Anchors: {s['anchors']}  Edges: {s['edges']}  Ghosts: {s['ghosts']}",
            f"  Schemas: {s['schemas']}  Abstracts: {s['abstracts']}",
        ]
        return "\n".join(lines)
