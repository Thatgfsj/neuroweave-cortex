"""Sleep-based memory consolidation — v0.4 real mechanisms.

A full sleep cycle implements:
1. Prioritized SWR Replay — priority = f(emotion, novelty, freq, centrality, unresolved)
2. Systems Consolidation — hippocampal → cortical transfer
3. Emotional Stripping — decouple emotion from information
4. Schema Extraction — abstract common patterns via embedding similarity
5. Merge Similar — fuse near-duplicate anchors by semantic similarity
6. Adaptive Prune — remove weak anchors, leave ghosts (interference-aware)
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
    """One complete sleep/consolidation cycle."""

    def __init__(self, graph: StarGraph):
        self.graph = graph
        self.log: list[str] = []
        self._cycle_count: int = 0
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def run(self, recent_anchors: list[Anchor] | None = None,
            similarity_threshold: float = 0.85,
            retention_threshold: float = 0.15,
            edge_prune_threshold: float = 0.1) -> dict:
        started = time.time()
        self._cycle_count += 1
        stats_before = self.graph.stats()

        if recent_anchors:
            self._swr_replay(recent_anchors, similarity_threshold)

        self._systems_consolidation()
        self._emotional_stripping()
        schemas_formed = self._schema_extraction()
        merged = self._merge_similar(similarity_threshold)
        pruned_anchors = self._prune_anchors(retention_threshold)
        ghosts_created = self._ghost_count
        pruned_edges = self._prune_edges(edge_prune_threshold)
        bridges = self._bridge_distant()
        self._hebbian_update()
        self._synaptic_homeostasis()
        self._refresh_cortical_index()

        stats_after = self.graph.stats()
        elapsed = time.time() - started

        return {
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

    _ghost_count = 0

    # ── Phase 1: Prioritized SWR Replay ──────────────────

    def _swr_replay(self, recent: list[Anchor], threshold: float) -> None:
        """Prioritized experience replay — like RL's PER but biologically motivated.

        Priority = weighted combination of:
          |emotional_valence| × 0.25  — emotional salience
          surprise × 0.25              — novelty (unexpected = needs consolidation)
          retrieval frequency × 0.20   — how often accessed
          graph centrality × 0.15      — hub position in the graph
          |1 - stability| × 0.15       — unresolved / still-labile memories
        """
        for anchor in recent:
            centrality = len(self.graph._adjacency.get(anchor.id, set()))
            centrality_norm = min(1.0, centrality / max(1, len(self.graph.anchors)))

            anchor._replay_priority = (
                abs(anchor.vector.emotional_valence) * 0.25
                + anchor.vector.surprise * 0.25
                + anchor.vector.frequency * 0.20
                + centrality_norm * 0.15
                + abs(1.0 - anchor.vector.stability) * 0.15
            )

        prioritized = sorted(recent, key=lambda a: a._replay_priority, reverse=True)

        # Stochastic sampling: top 50% always replayed, rest sampled by priority
        top_half = max(1, len(prioritized) // 2)
        guaranteed = prioritized[:top_half]
        remaining = prioritized[top_half:]
        if remaining:
            import random
            weights = [a._replay_priority for a in remaining]
            total_w = sum(weights)
            if total_w > 0:
                probs = [w / total_w for w in weights]
                sample_count = max(1, len(remaining) // 3)
                sampled = random.choices(remaining, weights=probs, k=sample_count)
                prioritized = guaranteed + sampled
            else:
                prioritized = guaranteed

        embedder = self._get_embedder()

        for anchor in prioritized:
            existing = self.graph.anchors.get(anchor.id)
            if existing:
                existing.transition('replay')  # ACTIVE/DORMANT → REHEARSING
                existing.activate()
                existing.vector.importance = (
                    0.7 * existing.vector.importance + 0.3 * anchor.vector.importance
                )
                existing.vector.surprise = max(existing.vector.surprise, anchor.vector.surprise)
                existing.tags = list(set(existing.tags + anchor.tags))
            else:
                if not anchor.embedding:
                    anchor.embedding = embedder.encode(anchor.text)
                self.graph.add_anchor(anchor)

            # Connect using real embedding similarity (not bigrams)
            anchor_emb = existing.embedding if existing else anchor.embedding
            if not anchor_emb:
                anchor_emb = embedder.encode(anchor.text)

            for other_id, other in self.graph.anchors.items():
                if other_id == anchor.id:
                    continue
                if not other.embedding:
                    continue
                sim = self._embedding_similarity(anchor_emb, other.embedding)
                if sim > threshold:
                    edge_type = self._infer_edge_type(anchor, other)
                    weight = sim * (1.0 + 0.3 * abs(anchor.vector.emotional_valence))
                    self.graph.add_edge(anchor.id, other_id,
                                        weight=min(1.0, weight),
                                        edge_type=edge_type)

        self.log.append(f"SWR Replay: replayed {len(prioritized)} anchors "
                        f"(priority-based sampling, compression ~{max(1, len(recent)//3)}:1)")

    # ── Phase 2: Systems Consolidation ──────────────────

    def _systems_consolidation(self) -> None:
        TAU = 20.0

        for anchor in self.graph.anchors.values():
            replay_factor = anchor.replay_count / max(1, self._cycle_count)
            anchor.vector.hippocampal_dependency = math.exp(
                -replay_factor * self._cycle_count / TAU
            )
            anchor.vector.hippocampal_dependency = max(0.05,
                anchor.vector.hippocampal_dependency)

            if anchor.is_cortical:
                if anchor.vector.stability > 0.8 and len(anchor.text) > 100:
                    sentences = anchor.text.replace('！', '。').replace('？', '。').split('。')
                    anchor.text = sentences[0].strip()[:200]

                for neighbor in self.graph._adjacency.get(anchor.id, set()):
                    key = self.graph._key(anchor.id, neighbor)
                    edge = self.graph.edges.get(key)
                    if edge:
                        if edge.edge_type == "temporal":
                            edge.weaken(0.015)
                        elif edge.edge_type in ("topical", "causal"):
                            edge.strengthen(0.01)

            anchor.vector.stability = min(1.0,
                1.0 - 0.9 * anchor.vector.hippocampal_dependency)

        cortical_count = sum(1 for a in self.graph.anchors.values() if a.is_cortical)
        self.log.append(f"Systems Consolidation: {cortical_count}/{len(self.graph.anchors)} "
                        f"memories cortical")

    # ── Phase 3: Emotional Stripping ────────────────────

    def _emotional_stripping(self) -> None:
        DECAY = 0.75

        for anchor in self.graph.anchors.values():
            if anchor.vector.stability > 0.5:
                old_valence = anchor.vector.emotional_valence
                anchor.vector.emotional_valence *= DECAY
                anchor.vector.importance = max(
                    anchor.vector.importance * 0.97,
                    abs(old_valence) * 0.25 + 0.15
                )

        self.log.append("Emotional Stripping: decoupled emotion from consolidated memories")

    # ── Phase 4: Schema Extraction + Abstraction Emergence ─

    def _schema_extraction(self) -> int:
        """Extract schemas AND discover emergent abstract categories.

        Two-stage process:
        1. Tag-based schema extraction (legacy, for tagged anchors)
        2. Embedding-cluster-based abstraction (new — emergent categories)
        """
        MIN_INSTANCES = 3
        MIN_SIMILARITY = 0.6
        formed = 0

        tag_groups: dict[str, list[Anchor]] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            for tag in anchor.tags:
                tag_groups[tag].append(anchor)

        for tag, group in tag_groups.items():
            if len(group) < MIN_INSTANCES:
                continue

            existing_schema = any(
                s for s in self.graph.schemas.values()
                if tag in s.tags
            )
            if existing_schema:
                continue

            sorted_group = sorted(group, key=lambda a: -a.vector.stability)

            # Use embedding similarity for schema validation
            similarities = []
            for i in range(min(MIN_INSTANCES, len(sorted_group))):
                for j in range(i + 1, min(MIN_INSTANCES, len(sorted_group))):
                    if sorted_group[i].embedding and sorted_group[j].embedding:
                        sim = self._embedding_similarity(
                            sorted_group[i].embedding, sorted_group[j].embedding)
                    else:
                        sim = 0.0
                    similarities.append(sim)

            avg_sim = sum(similarities) / max(1, len(similarities))
            if avg_sim < MIN_SIMILARITY:
                continue

            template_anchor = sorted_group[0]
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

            for a in sorted_group[:MIN_INSTANCES]:
                a.schema_ref = schema_id

        if formed:
            self.log.append(f"Schema Extraction: formed {formed} new schemas (embedding-based)")

        # Phase 4b: Abstraction Emergence — discover emergent categories
        abstract_formed = self._abstraction_emergence()

        return formed + abstract_formed

    def _abstraction_emergence(self) -> int:
        """Discover emergent higher-order categories from anchor clusters.

        Uses embedding-cluster-based detection. When multiple anchors share
        a semantic subspace, generates AbstractNode capturing the invariant.

        Example: "likes Python" + "writes Flask" + "deploys FastAPI"
                 -> AbstractNode "Backend Python Developer"
        """
        try:
            from .abstraction import AbstractionEngine
        except ImportError:
            return 0

        if not hasattr(self, '_abstraction_engine'):
            self._abstraction_engine = AbstractionEngine(
                min_cluster_size=3,
                similarity_threshold=0.55,
            )

        # Collect anchor embeddings
        anchors = {}
        embeddings = {}
        for aid, a in self.graph.anchors.items():
            if a.embedding and a.state.name in ('ACTIVE', 'DORMANT', 'CONSOLIDATING'):
                anchors[aid] = a
                embeddings[aid] = a.embedding

        new_abstracts = self._abstraction_engine.discover(anchors, embeddings)

        for abstract in new_abstracts:
            # Store in graph
            self.graph.abstracts[abstract.id] = abstract
            # Tag source anchors with abstraction reference
            for aid in abstract.source_anchor_ids:
                if aid in self.graph.anchors:
                    self.graph.anchors[aid].tags.append(f"abstract:{abstract.label}")

        if new_abstracts:
            self.log.append(
                f"Abstraction Emergence: discovered {len(new_abstracts)} "
                f"new concepts: {[a.label for a in new_abstracts]}"
            )

        return len(new_abstracts)

    # ── Phase 5: Merge Similar ──────────────────────────

    def _merge_similar(self, threshold: float) -> int:
        """Merge near-duplicate anchors using embedding similarity."""
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

                # Prefer embedding similarity, fall back to bigrams
                if a.embedding and b.embedding:
                    overlap = self._embedding_similarity(a.embedding, b.embedding)
                else:
                    overlap = self._text_overlap(a.text, b.text)

                if overlap > threshold:
                    core, variant = (a, b) if a.created_at < b.created_at else (b, a)
                    core.vector.importance = max(core.vector.importance, variant.vector.importance)
                    core.vector.frequency = (core.vector.frequency + variant.vector.frequency) / 2
                    core.vector.stability = min(1.0, core.vector.stability + 0.15)
                    core.tags = list(set(core.tags + variant.tags))
                    for neighbor in list(self.graph._adjacency.get(variant.id, set())):
                        k = self.graph._key(variant.id, neighbor)
                        old = self.graph.edges.pop(k, None)
                        if old:
                            nk = self.graph._key(core.id, neighbor)
                            if nk not in self.graph.edges:
                                self.graph.edges[nk] = Edge(
                                    source=nk[0], target=nk[1],
                                    weight=old.weight, edge_type=old.edge_type,
                                )
                                self.graph._adjacency[core.id].add(neighbor)
                                self.graph._adjacency[neighbor].add(core.id)
                    self.graph.remove_anchor(variant.id)
                    processed.add(variant.id)
                    merged += 1
                    if variant is a:
                        break

        if merged:
            self.log.append(f"Merge: fused {merged} duplicate anchor pairs")
        return merged

    # ── Phase 6: Adaptive Prune ─────────────────────────

    def _prune_anchors(self, threshold: float) -> list[str]:
        """Interference-aware pruning: contradiction penalties + ghosts."""
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
                anchor = self.graph.anchors[aid]
                # Collect residual edges before removal
                residual_edges = {}
                for neighbor in self.graph._adjacency.get(aid, set()):
                    key = self.graph._key(aid, neighbor)
                    edge = self.graph.edges.get(key)
                    if edge:
                        residual_edges[neighbor] = edge.weight * 0.3  # attenuated

                # Create rich ghost via ghost subsystem
                if hasattr(self.graph, '_ghost_subsystem') and self.graph._ghost_subsystem:
                    self.graph._ghost_subsystem.create(anchor, residual_edges)
                else:
                    self.graph.add_ghost(anchor)

                anchor.transition('prune')
                self.graph.remove_anchor(aid)
                self._ghost_count += 1

        # Decay ghosts via subsystem
        if hasattr(self.graph, '_ghost_subsystem') and self.graph._ghost_subsystem:
            stale_count = self.graph._ghost_subsystem.decay_all()
        else:
            now = time.time()
            stale_ghosts = []
            for gid, ghost in self.graph.ghosts.items():
                if isinstance(ghost, GhostAnchor) and (now - ghost.pruned_at) > 30 * 86400 and ghost.revival_count == 0:
                    stale_ghosts.append(gid)
            for gid in stale_ghosts:
                del self.graph.ghosts[gid]
            stale_count = len(stale_ghosts)

        if candidates:
            self.log.append(f"Adaptive Prune: removed {len(candidates)} anchors "
                            f"({self._ghost_count} ghosts, {stale_count} stale ghosts cleared)")
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
        if not self.graph.edges:
            return

        weights = [e.weight for e in self.graph.edges.values()]
        mean_w = sum(weights) / len(weights)

        target_mean = 0.3
        if mean_w > target_mean:
            scale = target_mean / mean_w
            for edge in self.graph.edges.values():
                edge.weight *= (0.5 + 0.5 * scale)

        for anchor in self.graph.anchors.values():
            hours = (time.time() - anchor.last_activated_at) / 3600
            anchor.decay(hours)

    # ── Helpers ─────────────────────────────────────────

    def _refresh_cortical_index(self) -> None:
        self.graph.cortical_index = [
            (a.embedding, a.id)
            for a in self.graph.anchors.values()
            if a.embedding and a.is_cortical
        ]
        # Sync ANN index
        ann = self.graph._get_ann_index() if self.graph._ann_index is not None else None
        if ann is not None and ann.size != len(self.graph.anchors):
            ann.clear()
            for a in self.graph.anchors.values():
                if a.embedding:
                    ann.add(a.id, a.embedding)
            ann.rebuild()

    @staticmethod
    def _embedding_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embeddings."""
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(min_len))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Fallback: character bigram Jaccard (only when embeddings unavailable)."""
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
