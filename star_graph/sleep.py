"""Sleep-based memory consolidation — v0.2 deep mechanisms.

A full sleep cycle now implements:
1. SWR Replay — compressed replay of recent anchors
2. Systems Consolidation — hippocampal → cortical transfer
3. Emotional Stripping — decouple emotion from information
4. Schema Extraction — abstract common patterns across episodes
5. Merge Similar — fuse near-duplicate anchors
6. Adaptive Prune — remove weak anchors, leave ghosts for savings
7. Bridge Constellations — discover surprising connections
8. Hebbian Update — strengthen co-activated, weaken dormant
9. Synaptic Homeostasis — global downscaling, keep only strong
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Optional

from .anchor import Anchor, AnchorVector, GhostAnchor, Oscillator
from .graph import StarGraph, Edge, Constellation, Schema


class SleepCycle:
    """One complete sleep/consolidation cycle — deeply biologically inspired."""

    def __init__(self, graph: StarGraph):
        self.graph = graph
        self.log: list[str] = []
        self._cycle_count: int = 0

    def run(self, recent_anchors: list[Anchor] | None = None,
            similarity_threshold: float = 0.85,
            retention_threshold: float = 0.15,
            edge_prune_threshold: float = 0.1) -> dict:
        """Execute a full sleep cycle."""
        started = time.time()
        self._cycle_count += 1
        stats_before = self.graph.stats()

        # Phase 1: SWR Replay
        if recent_anchors:
            self._swr_replay(recent_anchors, similarity_threshold)

        # Phase 2: Systems Consolidation
        self._systems_consolidation()

        # Phase 3: Emotional Stripping
        self._emotional_stripping()

        # Phase 4: Schema Extraction
        schemas_formed = self._schema_extraction()

        # Phase 5: Merge Similar
        merged = self._merge_similar(similarity_threshold)

        # Phase 6: Adaptive Prune
        pruned_anchors = self._prune_anchors(retention_threshold)
        ghosts_created = self._ghost_count
        pruned_edges = self._prune_edges(edge_prune_threshold)

        # Phase 7: Bridge
        bridges = self._bridge_distant()

        # Phase 8: Hebbian Update
        self._hebbian_update()

        # Phase 9: Synaptic Homeostasis
        self._synaptic_homeostasis()

        # Post-cycle: update cortical index
        self._refresh_cortical_index()

        stats_after = self.graph.stats()
        elapsed = time.time() - started

        result = {
            "cycle": self._cycle_count,
            "duration_seconds": round(elapsed, 2),
            "merged": merged,
            "pruned_anchors": len(pruned_anchors),
            "ghosts_created": ghosts_created,
            "pruned_edges": len(pruned_edges),
            "bridges_created": bridges,
            "schemas_formed": schemas_formed,
            "stats_before": stats_before,
            "stats_after": stats_after,
            "log": self.log,
        }
        return result

    _ghost_count = 0

    # ── Phase 1: SWR Replay ─────────────────────────────

    def _swr_replay(self, recent: list[Anchor], threshold: float) -> None:
        """Sharp-wave ripple replay: compressed (~20×) replay of recent anchors.

        During SWRs, the hippocampus replays recent experiences in compressed
        time. Not all memories are replayed equally — those with higher
        emotional valence and surprise are prioritized.
        """
        # Prioritize: high emotion, high surprise, high importance
        prioritized = sorted(recent, key=lambda a: (
            abs(a.vector.emotional_valence) * 0.4
            + a.vector.surprise * 0.35
            + a.vector.importance * 0.25
        ), reverse=True)

        for anchor in prioritized:
            existing = self.graph.anchors.get(anchor.id)
            if existing:
                # Reactivation strengthens existing
                existing.replay_count += 1
                existing.activate()
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
                    # Higher weight for emotionally-charged connections
                    weight = overlap * (1.0 + 0.3 * abs(anchor.vector.emotional_valence))
                    self.graph.add_edge(anchor.id, other_id,
                                        weight=min(1.0, weight),
                                        edge_type=edge_type)

        self.log.append(f"SWR Replay: replayed {len(prioritized)} anchors "
                        f"(compression ~{max(1, len(prioritized)//3)}:1)")

    # ── Phase 2: Systems Consolidation ──────────────────

    def _systems_consolidation(self) -> None:
        """Gradual transfer from hippocampal to cortical storage.

        Each replay during sleep reduces hippocampal dependency.
        Well-consolidated memories can be retrieved directly (cortical lookup)
        without needing graph traversal.
        """
        TAU = 20.0  # consolidation time constant (cycles)

        for anchor in self.graph.anchors.values():
            # Consolidation driven by replay count
            replay_factor = anchor.replay_count / max(1, self._cycle_count)
            anchor.vector.hippocampal_dependency = math.exp(
                -replay_factor * self._cycle_count / TAU
            )
            anchor.vector.hippocampal_dependency = max(0.05,
                anchor.vector.hippocampal_dependency)

            # As consolidation progresses:
            if anchor.is_cortical:
                # Semanticization: gradually simplify text toward gist
                if anchor.vector.stability > 0.8 and len(anchor.text) > 100:
                    # Keep first sentence (usually the core statement)
                    sentences = anchor.text.replace('！', '。').replace('？', '。').split('。')
                    anchor.text = sentences[0].strip()[:200]

                # Weaken episodic edges, strengthen semantic ones
                for neighbor in self.graph._adjacency.get(anchor.id, set()):
                    key = self.graph._key(anchor.id, neighbor)
                    edge = self.graph.edges.get(key)
                    if edge:
                        if edge.edge_type == "temporal":
                            edge.weaken(0.015)
                        elif edge.edge_type in ("topical", "causal"):
                            edge.strengthen(0.01)

            # Stability increases with consolidation
            anchor.vector.stability = min(1.0,
                1.0 - 0.9 * anchor.vector.hippocampal_dependency)

        cortical_count = sum(1 for a in self.graph.anchors.values() if a.is_cortical)
        self.log.append(f"Systems Consolidation: {cortical_count}/{len(self.graph.anchors)} "
                        f"memories cortical")

    # ── Phase 3: Emotional Stripping ────────────────────

    def _emotional_stripping(self) -> None:
        """Decouple emotional charge from informational content.

        Adaptive: you retain that the snake was dangerous without
        re-experiencing the full fear response. Emotional valence decays
        but importance is preserved (the lesson remains).
        """
        DECAY = 0.75  # per sleep cycle

        for anchor in self.graph.anchors.values():
            if anchor.vector.stability > 0.5:
                old_valence = anchor.vector.emotional_valence
                anchor.vector.emotional_valence *= DECAY
                # Preserve importance: "this mattered" persists
                anchor.vector.importance = max(
                    anchor.vector.importance * 0.97,
                    abs(old_valence) * 0.25 + 0.15
                )

        self.log.append("Emotional Stripping: decoupled emotion from consolidated memories")

    # ── Phase 4: Schema Extraction ──────────────────────

    def _schema_extraction(self) -> int:
        """Extract abstract schemas from constellations with enough instances.

        Schemas capture the invariant structure across related episodes.
        They guide encoding of new similar experiences (assimilation).
        """
        MIN_INSTANCES = 3
        MIN_SIMILARITY = 0.7
        formed = 0

        # Group anchors by tag overlap
        tag_groups: dict[str, list[Anchor]] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            for tag in anchor.tags:
                tag_groups[tag].append(anchor)

        for tag, group in tag_groups.items():
            if len(group) < MIN_INSTANCES:
                continue

            # Check if already covered by a schema
            existing_schema = any(
                s for s in self.graph.schemas.values()
                if tag in s.tags
            )
            if existing_schema:
                continue

            # Try to extract a common template
            # For now: take the most stable anchor's text as template
            sorted_group = sorted(group, key=lambda a: -a.vector.stability)
            template_anchor = sorted_group[0]

            # Check similarity within group
            similarities = []
            for i, a in enumerate(sorted_group[:MIN_INSTANCES]):
                for b in sorted_group[i+1:MIN_INSTANCES]:
                    sim = self._text_overlap(a.text, b.text)
                    similarities.append(sim)

            avg_sim = sum(similarities) / max(1, len(similarities))
            if avg_sim < MIN_SIMILARITY:
                continue

            schema_id = f"schema_{tag}_{self._cycle_count}"
            schema = Schema(
                id=schema_id,
                template=template_anchor.text,
                slots={"topic": "specific topic instance", "context": "conversation context"},
                instance_ids=[a.id for a in sorted_group[:MIN_INSTANCES]],
                confidence=avg_sim,
                tags=[tag],
            )
            self.graph.schemas[schema_id] = schema
            formed += 1

            # Tag instances with schema reference
            for a in sorted_group[:MIN_INSTANCES]:
                a.schema_ref = schema_id

        if formed:
            self.log.append(f"Schema Extraction: formed {formed} new schemas")
        return formed

    # ── Phase 5: Merge Similar ──────────────────────────

    def _merge_similar(self, threshold: float) -> int:
        """Merge near-duplicate anchors. Older becomes core, newer enriches it."""
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
                    core, variant = (a, b) if a.created_at < b.created_at else (b, a)
                    core.vector.importance = max(core.vector.importance, variant.vector.importance)
                    core.vector.frequency = (core.vector.frequency + variant.vector.frequency) / 2
                    core.vector.stability = min(1.0, core.vector.stability + 0.15)
                    core.tags = list(set(core.tags + variant.tags))
                    # Redirect edges from variant to core
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

        if merged:
            self.log.append(f"Merge: fused {merged} duplicate anchor pairs")
        return merged

    # ── Phase 6: Adaptive Prune ─────────────────────────

    def _prune_anchors(self, threshold: float) -> list[str]:
        """Remove anchors below retention threshold, but leave ghosts.

        Ghosts enable the savings effect: if similar content reappears,
        relearning is faster than original learning.
        """
        # Check contradictions — weaker anchor in contradiction pairs gets penalty
        contradictions = self.graph.find_contradictions()
        penalties: dict[str, float] = defaultdict(float)
        for aid_a, aid_b, _ in contradictions:
            a = self.graph.anchors.get(aid_a)
            b = self.graph.anchors.get(aid_b)
            if a and b:
                if a.retention_score < b.retention_score:
                    penalties[aid_a] += 0.2
                else:
                    penalties[aid_b] += 0.2

        candidates = []
        for aid in self.graph.get_prune_candidates(threshold):
            score = self.graph.anchors[aid].retention_score - penalties.get(aid, 0.0)
            if score < threshold:
                candidates.append(aid)

        self._ghost_count = 0
        for aid in candidates:
            if aid in self.graph.anchors:
                self.graph.add_ghost(self.graph.anchors[aid])
                self.graph.remove_anchor(aid)
                self._ghost_count += 1

        # Also prune old ghosts (beyond revival)
        now = time.time()
        stale_ghosts = []
        for gid, ghost in self.graph.ghosts.items():
            # Ghosts older than 30 days with no revivals get removed
            if (now - ghost.pruned_at) > 30 * 86400 and ghost.revival_count == 0:
                stale_ghosts.append(gid)
        for gid in stale_ghosts:
            del self.graph.ghosts[gid]

        if candidates:
            self.log.append(f"Adaptive Prune: removed {len(candidates)} anchors "
                            f"({self._ghost_count} ghosts, {len(stale_ghosts)} stale ghosts cleared)")
        return candidates

    def _prune_edges(self, threshold: float) -> list[tuple[str, str]]:
        candidates = self.graph.get_dormant_edges(threshold)
        for key in candidates:
            if key in self.graph.edges:
                del self.graph.edges[key]
                a, b = key
                self.graph._adjacency[a].discard(b)
                self.graph._adjacency[b].discard(a)
        if candidates:
            self.log.append(f"Edge Prune: removed {len(candidates)} dormant edges")
        return candidates

    # ── Phase 7: Bridge Distant ─────────────────────────

    def _bridge_distant(self) -> int:
        """Discover surprising connections between distant constellations."""
        from .resonance import Resonator
        resonator = Resonator(self.graph)

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
                    rep_a = c_a.anchors[0]
                    rep_b = c_b.anchors[0]
                    existing = self.graph.edges.get(self.graph._key(rep_a.id, rep_b.id))
                    if not existing or existing.weight < 0.3:
                        self.graph.add_edge(rep_a.id, rep_b.id, weight=0.3,
                                            edge_type="bridge")
                        bridges += 1

        if bridges:
            self.log.append(f"Bridge: created {bridges} cross-constellation connections")
        return bridges

    # ── Phase 8: Hebbian Update ─────────────────────────

    def _hebbian_update(self) -> None:
        """Strengthen recently co-activated edges, weaken dormant ones."""
        now = time.time()
        for edge in self.graph.edges.values():
            hours = (now - edge.last_activated_at) / 3600
            if edge.co_activation_count > 0 and hours < 24:
                edge.strengthen(0.03)
            else:
                decay = 0.02 * math.log(1 + hours / 24)
                edge.weaken(min(0.1, decay))

    # ── Phase 9: Synaptic Homeostasis ───────────────────

    def _synaptic_homeostasis(self) -> None:
        """Global downscaling: keep only the strongest connections.

        During sleep, overall synaptic weights are downscaled to maintain
        energy efficiency. Only the strongest weights survive. This prevents
        saturation and maintains the dynamic range for new learning.
        """
        if not self.graph.edges:
            return

        # Compute mean weight
        weights = [e.weight for e in self.graph.edges.values()]
        mean_w = sum(weights) / len(weights)

        # Downscale proportional to mean
        # Target: keep mean around 0.3
        target_mean = 0.3
        if mean_w > target_mean:
            scale = target_mean / mean_w
            for edge in self.graph.edges.values():
                edge.weight *= (0.5 + 0.5 * scale)  # partial scaling

        # Global decay on all anchors
        for anchor in self.graph.anchors.values():
            hours = (time.time() - anchor.last_activated_at) / 3600
            anchor.decay(hours)

    # ── Helpers ─────────────────────────────────────────

    def _refresh_cortical_index(self) -> None:
        """Update the cortical index after consolidation."""
        self.graph.cortical_index = [
            (a.embedding, a.id)
            for a in self.graph.anchors.values()
            if a.embedding and a.is_cortical
        ]

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        def bigrams(s):
            return {s[i:i + 2] for i in range(len(s) - 1)}
        ba, bb = bigrams(a), bigrams(b)
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)

    @staticmethod
    def _infer_edge_type(a: Anchor, b: Anchor) -> str:
        if a.source_session == b.source_session:
            return "temporal"
        if set(a.tags) & set(b.tags):
            return "topical"
        return "topical"
