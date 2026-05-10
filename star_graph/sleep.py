"""Sleep-based memory consolidation.

Inspired by hippocampal replay during sleep:
1. Replay recent anchors against the existing graph
2. Merge highly similar anchors into core+variant structures
3. Prune weak anchors and dormant edges
4. Bridge surprising connections between distant constellations
5. Update all weights (Hebbian strengthening, decay of unused)
"""

from __future__ import annotations

import time
import math
from collections import defaultdict
from typing import Optional

from .anchor import Anchor
from .graph import StarGraph, Edge, Constellation


class SleepCycle:
    """One complete sleep/consolidation cycle."""

    def __init__(self, graph: StarGraph):
        self.graph = graph
        self.log: list[str] = []

    def run(self, recent_anchors: list[Anchor] | None = None,
            similarity_threshold: float = 0.85,
            retention_threshold: float = 0.15,
            edge_prune_threshold: float = 0.1) -> dict:
        """Execute a full sleep cycle.

        Args:
            recent_anchors: New anchors from today's sessions
            similarity_threshold: Merge anchors with text overlap above this
            retention_threshold: Remove anchors below this retention score
            edge_prune_threshold: Remove edges below this weight
        """
        started = time.time()
        stats_before = self.graph.stats()

        # Phase 1: Replay
        if recent_anchors:
            self._replay(recent_anchors, similarity_threshold)

        # Phase 2: Merge
        merged = self._merge_similar(similarity_threshold)

        # Phase 3: Prune
        pruned_anchors = self._prune_anchors(retention_threshold)
        pruned_edges = self._prune_edges(edge_prune_threshold)

        # Phase 4: Bridge
        bridges = self._bridge_distant()

        # Phase 5: Hebbian update
        self._hebbian_update()

        # Phase 6: Global decay
        self._global_decay()

        stats_after = self.graph.stats()
        elapsed = time.time() - started

        result = {
            "duration_seconds": round(elapsed, 2),
            "merged": merged,
            "pruned_anchors": len(pruned_anchors),
            "pruned_edges": len(pruned_edges),
            "bridges_created": bridges,
            "stats_before": stats_before,
            "stats_after": stats_after,
            "log": self.log,
        }
        return result

    # ── Phase 1: Replay ───────────────────────────────────

    def _replay(self, recent: list[Anchor], threshold: float) -> None:
        """Replay recent anchors: integrate them into the existing graph."""
        for anchor in recent:
            existing = self.graph.anchors.get(anchor.id)
            if existing:
                # Update existing: boost frequency, reset recency
                existing.activate()
                # Merge vector properties gradually
                existing.vector.importance = (
                    0.7 * existing.vector.importance + 0.3 * anchor.vector.importance
                )
                existing.vector.surprise = max(existing.vector.surprise, anchor.vector.surprise)
                existing.tags = list(set(existing.tags + anchor.tags))
            else:
                self.graph.add_anchor(anchor)

            # Connect to similar existing anchors
            for other_id, other in self.graph.anchors.items():
                if other_id == anchor.id:
                    continue
                overlap = self._text_overlap(anchor.text, other.text)
                if overlap > threshold:
                    edge_type = self._infer_edge_type(anchor, other)
                    self.graph.add_edge(anchor.id, other_id, weight=overlap,
                                        edge_type=edge_type)

        self.log.append(f"Replay: processed {len(recent)} anchors")

    # ── Phase 2: Merge ────────────────────────────────────

    def _merge_similar(self, threshold: float) -> int:
        """Merge anchors that are near-duplicates.

        The older anchor becomes the 'core', the newer one becomes a variant.
        We don't delete the variant, but we consolidate its vector into the core
        and redirect its edges.
        """
        merged = 0
        ids = list(self.graph.anchors.keys())
        processed: set[str] = set()

        for i, aid_a in enumerate(ids):
            if aid_a in processed or aid_a not in self.graph.anchors:
                continue
            for aid_b in ids[i + 1:]:
                if aid_b in processed or aid_b not in self.graph.anchors:
                    continue
                a = self.graph.anchors[aid_a]
                b = self.graph.anchors[aid_b]
                overlap = self._text_overlap(a.text, b.text)
                if overlap > threshold:
                    # Merge b into a (keep the older one as core)
                    core, variant = (a, b) if a.created_at < b.created_at else (b, a)
                    core.vector.importance = max(core.vector.importance, variant.vector.importance)
                    core.vector.frequency = (core.vector.frequency + variant.vector.frequency) / 2
                    core.vector.stability = min(1.0, core.vector.stability + 0.15)
                    core.tags = list(set(core.tags + variant.tags))
                    # Redirect edges
                    for neighbor in list(self.graph._adjacency.get(variant.id, set())):
                        key = self.graph._key(variant.id, neighbor)
                        old_edge = self.graph.edges.pop(key, None)
                        if old_edge:
                            new_key = self.graph._key(core.id, neighbor)
                            if new_key not in self.graph.edges:
                                self.graph.edges[new_key] = Edge(
                                    source=new_key[0], target=new_key[1],
                                    weight=old_edge.weight,
                                    edge_type=old_edge.edge_type,
                                )
                                self.graph._adjacency[core.id].add(neighbor)
                                self.graph._adjacency[neighbor].add(core.id)
                    self.graph.remove_anchor(variant.id)
                    processed.add(variant.id)
                    merged += 1
                    self.log.append(f"Merge: '{variant.text[:50]}...' → '{core.text[:50]}...'")

        return merged

    # ── Phase 3: Prune ────────────────────────────────────

    def _prune_anchors(self, threshold: float) -> list[str]:
        """Remove anchors with retention score below threshold."""
        candidates = self.graph.get_prune_candidates(threshold)
        for aid in candidates:
            self.graph.remove_anchor(aid)
        if candidates:
            self.log.append(f"Prune anchors: removed {len(candidates)} ({threshold=})")
        return candidates

    def _prune_edges(self, threshold: float) -> list[tuple[str, str]]:
        """Remove edges that have weakened below threshold."""
        candidates = self.graph.get_dormant_edges(threshold)
        for key in candidates:
            if key in self.graph.edges:
                del self.graph.edges[key]
                a, b = key
                self.graph._adjacency[a].discard(b)
                self.graph._adjacency[b].discard(a)
        if candidates:
            self.log.append(f"Prune edges: removed {len(candidates)} ({threshold=})")
        return candidates

    # ── Phase 4: Bridge ───────────────────────────────────

    def _bridge_distant(self) -> int:
        """Discover surprising connections between distant constellations.

        Uses structural hole theory: if two constellations are semantically
        similar but not connected, add a weak bridge edge.
        """
        from .resonance import Resonator
        resonator = Resonator(self.graph)

        # Find all constellations
        visited: set[str] = set()
        constellations: list[Constellation] = []
        for aid in self.graph.anchors:
            if aid not in visited:
                c = self.graph.find_constellation(aid)
                if c.anchors:
                    constellations.append(c)
                    for a in c.anchors:
                        visited.add(a.id)

        bridges = 0
        for i, c_a in enumerate(constellations):
            for c_b in constellations[i + 1:]:
                score = resonator.bridge_score(c_a, c_b)
                if score > 0.6:
                    # Add a lightweight bridge edge
                    rep_a = c_a.anchors[0]
                    rep_b = c_b.anchors[0]
                    existing = self.graph.edges.get(self.graph._key(rep_a.id, rep_b.id))
                    if not existing or existing.weight < 0.3:
                        self.graph.add_edge(rep_a.id, rep_b.id, weight=0.3,
                                            edge_type="bridge")
                        bridges += 1
                        self.log.append(
                            f"Bridge: '{c_a.anchors[0].text[:40]}...' "
                            f"↔ '{c_b.anchors[0].text[:40]}...' (score={score:.2f})"
                        )

        if bridges:
            self.log.append(f"Bridge: created {bridges} cross-constellation connections")
        return bridges

    # ── Phase 5: Hebbian update ───────────────────────────

    def _hebbian_update(self) -> None:
        """Strengthen edges that were recently co-activated, weaken the rest."""
        now = time.time()
        for edge in self.graph.edges.values():
            hours_since_activation = (now - edge.last_activated_at) / 3600
            if edge.co_activation_count > 0 and hours_since_activation < 24:
                edge.strengthen(0.03)
            else:
                # Decay based on dormancy
                decay = 0.02 * math.log(1 + hours_since_activation / 24)
                edge.weaken(min(0.1, decay))

    # ── Phase 6: Global decay ─────────────────────────────

    def _global_decay(self) -> None:
        """Apply temporal decay to all anchors."""
        now = time.time()
        for anchor in self.graph.anchors.values():
            hours = (now - anchor.last_activated_at) / 3600
            anchor.decay(hours)

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Jaccard similarity on character bigrams."""
        def bigrams(s):
            return {s[i:i + 2] for i in range(len(s) - 1)}
        ba, bb = bigrams(a), bigrams(b)
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)

    @staticmethod
    def _infer_edge_type(a: Anchor, b: Anchor) -> str:
        """Infer relationship type between two anchors."""
        if a.source_session == b.source_session:
            return "temporal"
        if set(a.tags) & set(b.tags):
            return "topical"
        return "topical"
