"""Sleep-based memory consolidation — v0.4 real mechanisms.

5-Phase Systematized Architecture:
  N1 (Replay Indexing):     SWR replay, priority sampling, centrality analysis
  N2 (Weak Merge):          Merge similar, bridge constellations, edge formation
  N3 (Compression):         Systems consolidation, schema extraction, Hebbian update
  REM (Emotional Decoupling): Emotional stripping, synaptic homeostasis
  Wake-prep (Schema Synthesis): Adaptive prune, edge prune, cortical index refresh

Each phase produces metrics captured in SleepReport for rich, human-readable output.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

from .anchor import Anchor, AnchorVector, GhostAnchor, Oscillator
from .graph import StarGraph, Edge, Constellation, Schema
from .config import Config


# ── Sleep Report ────────────────────────────────────────

@dataclass
class PhaseMetrics:
    """Metrics for one sleep phase."""
    phase: str = ""
    duration_ms: float = 0.0
    items_processed: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class SleepReport:
    """Rich, human-readable report from one full sleep cycle."""
    cycle: int = 0
    total_duration_ms: float = 0.0
    phases: list[PhaseMetrics] = field(default_factory=list)

    # Aggregate counts
    memories_replayed: int = 0
    memories_merged: int = 0
    memories_pruned: int = 0
    ghosts_created: int = 0
    schemas_formed: int = 0
    abstractions_formed: int = 0
    bridges_created: int = 0
    edges_pruned: int = 0
    emotional_decoupled: int = 0
    cortical_transferred: int = 0

    # Before/after
    anchors_before: int = 0
    anchors_after: int = 0
    edges_before: int = 0
    edges_after: int = 0
    avg_retention_before: float = 0.0
    avg_retention_after: float = 0.0
    compression_ratio: float = 1.0

    def summary(self) -> str:
        """One-line summary of the sleep cycle."""
        parts = []
        if self.memories_replayed:
            parts.append(f"Replayed {self.memories_replayed}")
        if self.memories_merged:
            parts.append(f"Merged {self.memories_merged}")
        if self.schemas_formed:
            parts.append(f"Created {self.schemas_formed} schemas")
        if self.memories_pruned:
            parts.append(f"Pruned {self.memories_pruned} ({self.ghosts_created} ghosts)")
        if self.bridges_created:
            parts.append(f"Bridged {self.bridges_created}")
        if not parts:
            return "Sleep cycle complete — no significant changes"
        return " | ".join(parts)

    def detailed(self) -> str:
        """Multi-line detailed report."""
        lines = [
            f"╔══════════════════════════════════════════════════╗",
            f"║  Sleep Cycle #{self.cycle} Report ({self.total_duration_ms:.0f}ms)",
            f"╠══════════════════════════════════════════════════╣",
        ]
        for p in self.phases:
            name = p.phase.ljust(26)
            lines.append(f"║  {name} {p.items_processed:>4} items ({p.duration_ms:>6.0f}ms)")
        lines.append(f"╠══════════════════════════════════════════════════╣")
        lines.append(f"║  Anchors:  {self.anchors_before:>4} → {self.anchors_after:<4} "
                     f"(compression: {self.compression_ratio:.2f}x)")
        lines.append(f"║  Edges:    {self.edges_before:>4} → {self.edges_after:<4}")
        lines.append(f"║  Retention:{self.avg_retention_before:>5.3f} → "
                     f"{self.avg_retention_after:<5.3f}")
        lines.append(f"╠══════════════════════════════════════════════════╣")
        if self.memories_merged:
            lines.append(f"║  Merged:   {self.memories_merged} near-duplicate anchors")
        if self.schemas_formed:
            lines.append(f"║  Schemas:  {self.schemas_formed} new abstractions")
        if self.memories_pruned:
            lines.append(f"║  Pruned:   {self.memories_pruned} low-retention anchors "
                         f"→ {self.ghosts_created} ghosts")
        if self.emotional_decoupled:
            lines.append(f"║  Emotion:  decoupled from {self.emotional_decoupled} memories")
        if self.cortical_transferred:
            lines.append(f"║  Cortical: {self.cortical_transferred} memories transferred")
        if self.bridges_created:
            lines.append(f"║  Bridges:  {self.bridges_created} cross-constellation links")
        lines.append(f"╚══════════════════════════════════════════════════╝")
        return "\n".join(lines)


class SleepCycle:
    """One complete sleep/consolidation cycle."""

    def __init__(self, graph: StarGraph, config: Config | None = None):
        self.graph = graph
        self.cfg = config if config is not None else Config.get()
        self.log: list[str] = []
        self._cycle_count: int = 0
        self._ghost_count: int = 0
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def run(self, recent_anchors: list[Anchor] | None = None,
            similarity_threshold: float | None = None,
            retention_threshold: float | None = None,
            edge_prune_threshold: float | None = None) -> dict:
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self.cfg.sleep.merge.default_threshold
        retention_threshold = retention_threshold if retention_threshold is not None else self.cfg.sleep.prune.default_retention_threshold
        edge_prune_threshold = edge_prune_threshold if edge_prune_threshold is not None else self.cfg.sleep.prune.default_edge_threshold
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

    # ── 5-Phase Systematized Architecture ────────────────

    def run_phased(self, recent_anchors: list[Anchor] | None = None,
                   similarity_threshold: float | None = None,
                   retention_threshold: float | None = None,
                   edge_prune_threshold: float | None = None,
                   brain: object | None = None,
                   hublayer: object | None = None,
                   cortices: list | None = None) -> SleepReport:
        """Run sleep with 8-phase systematized architecture.

        Phase 1: Replay Indexing — SWR replay, priority-weighted sampling
        Phase 2: Conflict Detection — identify contradictions, create edges
        Phase 3: Clustering — group by semantic density threshold
        Phase 4: Summary Generation — compress clusters into abstracts
        Phase 5: Tier Promotion — STM→MTM→LTM based on semantic density
        Phase 6: Hub Connection — cross-cortex pattern detection
        Phase 7: Forgetting/Degradation — thermal lifecycle downgrade
        Phase 8: Index Rebuild — ANN refresh + BrainSphere cache rebuild

        Returns a SleepReport with per-phase metrics and before/after comparison.
        """
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self.cfg.sleep.merge.default_threshold
        retention_threshold = retention_threshold if retention_threshold is not None else self.cfg.sleep.prune.default_retention_threshold
        edge_prune_threshold = edge_prune_threshold if edge_prune_threshold is not None else self.cfg.sleep.prune.default_edge_threshold

        self._cycle_count += 1
        started = time.time()
        stats_before = self.graph.stats()
        report = SleepReport(
            cycle=self._cycle_count,
            anchors_before=stats_before["anchors"],
            edges_before=stats_before["edges"],
            avg_retention_before=stats_before["avg_retention"],
        )

        # ── N1: Replay Indexing ──
        t0 = time.time()
        if recent_anchors:
            self._swr_replay(recent_anchors, similarity_threshold)
        report.phases.append(PhaseMetrics(
            phase="N1 Replay Indexing",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=len(recent_anchors) if recent_anchors else 0,
            details={"method": "priority-weighted sampling"},
        ))
        report.memories_replayed = len(recent_anchors) if recent_anchors else 0

        # ── N2: Weak Merge ──
        t0 = time.time()
        merged = self._merge_similar(similarity_threshold)
        bridges = self._bridge_distant()
        report.phases.append(PhaseMetrics(
            phase="N2 Weak Merge",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=merged + bridges,
            details={"merged": merged, "bridges": bridges},
        ))
        report.memories_merged = merged
        report.bridges_created = bridges

        # ── N3: Compression ──
        t0 = time.time()
        self._systems_consolidation()
        schemas = self._schema_extraction()
        self._hebbian_update()
        # Count abstractions
        abstraction_count = sum(
            1 for a in self.graph.abstracts.values()
            if getattr(a, 'confidence', 0) > self.cfg.abstraction.stable_confidence
        )
        cortical_count = sum(1 for a in self.graph.anchors.values() if a.is_cortical)
        report.phases.append(PhaseMetrics(
            phase="N3 Compression",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=schemas,
            details={"schemas": schemas, "abstractions": abstraction_count,
                     "cortical": cortical_count},
        ))
        report.schemas_formed = schemas
        report.abstractions_formed = abstraction_count
        report.cortical_transferred = cortical_count

        # ── REM: Emotional Decoupling ──
        t0 = time.time()
        self._emotional_stripping()
        self._synaptic_homeostasis()
        emotion_count = sum(
            1 for a in self.graph.anchors.values()
            if abs(a.vector.emotional_valence) < 0.1 and a.vector.stability > 0.5
        )
        report.phases.append(PhaseMetrics(
            phase="REM Emotional Decoupling",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=emotion_count,
        ))
        report.emotional_decoupled = emotion_count

        # ── Wake-prep: Schema Synthesis ──
        t0 = time.time()
        pruned_anchors = self._prune_anchors(retention_threshold)
        ghosts_created = self._ghost_count
        pruned_edges = self._prune_edges(edge_prune_threshold)
        report.phases.append(PhaseMetrics(
            phase="Wake-prep Schema Synthesis",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=len(pruned_anchors) + len(pruned_edges),
            details={"pruned_anchors": len(pruned_anchors),
                     "pruned_edges": len(pruned_edges),
                     "ghosts": ghosts_created},
        ))
        report.memories_pruned = len(pruned_anchors)
        report.ghosts_created = ghosts_created
        report.edges_pruned = len(pruned_edges)

        # ── Phase 6: Hub Connection ──
        t0 = time.time()
        hub_connections = 0
        if hublayer and cortices:
            hub_connections = self._connect_cross_cortex_hubs(hublayer, cortices)
        report.phases.append(PhaseMetrics(
            phase="Phase 6 Hub Connection",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=hub_connections,
            details={"cross_cortex_links": hub_connections},
        ))

        # ── Phase 7: Forgetting/Degradation ──
        t0 = time.time()
        thermal_stats = self._apply_thermal_forgetting()
        report.phases.append(PhaseMetrics(
            phase="Phase 7 Forgetting/Degradation",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=sum(thermal_stats.values()),
            details=thermal_stats,
        ))

        # ── Phase 8: Index Rebuild + BrainSphere refresh ──
        t0 = time.time()
        self._refresh_cortical_index()  # existing ANN index rebuild
        self._rebuild_ann_index()
        if brain and cortices:
            brain.refresh_cache(cortices)
        report.phases.append(PhaseMetrics(
            phase="Phase 8 Index Rebuild",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=0,
            details={"brain_cache_refreshed": brain is not None},
        ))

        # ── Final stats ──
        stats_after = self.graph.stats()
        report.total_duration_ms = (time.time() - started) * 1000
        report.anchors_after = stats_after["anchors"]
        report.edges_after = stats_after["edges"]
        report.avg_retention_after = stats_after["avg_retention"]
        report.compression_ratio = (
            report.anchors_before / max(1, report.anchors_after)
        )

        return report

    # ── Async variants ────────────────────────────────────

    async def run_async(self, recent_anchors: list[Anchor] | None = None,
                        similarity_threshold: float | None = None,
                        retention_threshold: float | None = None,
                        edge_prune_threshold: float | None = None,
                        on_progress: Callable[[str, float], Awaitable[None]] | None = None,
                        ) -> dict:
        """Async version of run() — non-blocking with progress callbacks.

        Args:
            on_progress: async callback(phase_name, progress_0_to_1)
        """
        loop = asyncio.get_running_loop()
        steps = 9
        results = {}

        for i, (name, fn) in enumerate([
            ("SWR Replay", lambda: self._swr_replay(
                recent_anchors or [],
                similarity_threshold or self.cfg.sleep.merge.default_threshold)),
            ("Systems Consolidation", self._systems_consolidation),
            ("Emotional Stripping", self._emotional_stripping),
            ("Schema Extraction", self._schema_extraction),
            ("Merge Similar", lambda: self._merge_similar(
                similarity_threshold or self.cfg.sleep.merge.default_threshold)),
            ("Prune Anchors", lambda: self._prune_anchors(
                retention_threshold or self.cfg.sleep.prune.default_retention_threshold)),
            ("Prune Edges", lambda: self._prune_edges(
                edge_prune_threshold or self.cfg.sleep.prune.default_edge_threshold)),
            ("Bridge Distant", self._bridge_distant),
            ("Hebbian + Homeostasis", self._hebbian_and_homeostasis),
        ]):
            if on_progress:
                await on_progress(name, i / steps)
            await loop.run_in_executor(None, fn)
            if on_progress:
                await on_progress(name, (i + 1) / steps)

        return results

    async def run_phased_async(self,
                               recent_anchors: list[Anchor] | None = None,
                               similarity_threshold: float | None = None,
                               retention_threshold: float | None = None,
                               edge_prune_threshold: float | None = None,
                               on_phase: Callable[[PhaseMetrics], Awaitable[None]] | None = None,
                               ) -> SleepReport:
        """Async 5-phase sleep with per-phase progress callbacks.

        Args:
            on_phase: async callback called after each phase completes
        """
        loop = asyncio.get_running_loop()

        # Run the full phased cycle in executor
        report = await loop.run_in_executor(
            None,
            lambda: self.run_phased(
                recent_anchors, similarity_threshold,
                retention_threshold, edge_prune_threshold,
            )
        )

        if on_phase:
            for phase in report.phases:
                await on_phase(phase)

        return report

    def _hebbian_and_homeostasis(self) -> None:
        self._hebbian_update()
        self._synaptic_homeostasis()
        self._refresh_cortical_index()

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
        c = self.cfg.sleep.swr
        for anchor in recent:
            centrality = len(self.graph._adjacency.get(anchor.id, set()))
            centrality_norm = min(1.0, centrality / max(1, len(self.graph.anchors)))

            anchor._replay_priority = (
                abs(anchor.vector.emotional_valence) * c.valence_weight
                + anchor.vector.surprise * c.surprise_weight
                + anchor.vector.frequency * c.frequency_weight
                + centrality_norm * c.centrality_weight
                + abs(1.0 - anchor.vector.stability) * c.instability_weight
            )

        prioritized = sorted(recent, key=lambda a: a._replay_priority, reverse=True)

        # Stochastic sampling: top fraction always replayed, rest sampled by priority
        top_half = max(1, int(len(prioritized) * c.top_fraction))
        guaranteed = prioritized[:top_half]
        remaining = prioritized[top_half:]
        if remaining:
            import random
            weights = [a._replay_priority for a in remaining]
            total_w = sum(weights)
            if total_w > 0:
                probs = [w / total_w for w in weights]
                sample_count = max(1, int(len(remaining) * c.sample_fraction))
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
                blend = self.cfg.sleep.merge.importance_blend
                existing.vector.importance = (
                    blend * existing.vector.importance
                    + (1 - blend) * anchor.vector.importance
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
                    weight = sim * (1.0 + self.cfg.sleep.edge_formation.emotion_weight_boost * abs(anchor.vector.emotional_valence))
                    self.graph.add_edge(anchor.id, other_id,
                                        weight=min(1.0, weight),
                                        edge_type=edge_type)

        self.log.append(f"SWR Replay: replayed {len(prioritized)} anchors "
                        f"(priority-based sampling, compression ~{max(1, len(recent)//3)}:1)")

    # ── Phase 2: Systems Consolidation ──────────────────

    def _systems_consolidation(self) -> None:
        c = self.cfg.sleep.systems

        for anchor in self.graph.anchors.values():
            replay_factor = anchor.replay_count / max(1, self._cycle_count)
            anchor.vector.hippocampal_dependency = math.exp(
                -replay_factor * self._cycle_count / c.tau
            )
            anchor.vector.hippocampal_dependency = max(c.min_hippocampal_dep,
                anchor.vector.hippocampal_dependency)

            if anchor.is_cortical:
                if anchor.vector.stability > c.schema_stability_threshold and len(anchor.text) > c.schema_text_threshold:
                    sentences = anchor.text.replace('！', '。').replace('？', '。').split('。')
                    anchor.text = sentences[0].strip()[:200]

                for neighbor in self.graph._adjacency.get(anchor.id, set()):
                    key = self.graph._key(anchor.id, neighbor)
                    edge = self.graph.edges.get(key)
                    if edge:
                        if edge.edge_type == "temporal":
                            edge.weaken(c.temporal_edge_weaken)
                        elif edge.edge_type in ("topical", "causal"):
                            edge.strengthen(c.topical_edge_strengthen)

            anchor.vector.stability = min(1.0,
                1.0 - c.cortical_stability_factor * anchor.vector.hippocampal_dependency)

        cortical_count = sum(1 for a in self.graph.anchors.values() if a.is_cortical)
        self.log.append(f"Systems Consolidation: {cortical_count}/{len(self.graph.anchors)} "
                        f"memories cortical")

    # ── Phase 3: Emotional Stripping ────────────────────

    def _emotional_stripping(self) -> None:
        c = self.cfg.sleep.emotional

        for anchor in self.graph.anchors.values():
            if anchor.vector.stability > c.strip_stability_threshold:
                old_valence = anchor.vector.emotional_valence
                anchor.vector.emotional_valence *= c.decay
                anchor.vector.importance = max(
                    anchor.vector.importance * c.importance_min_factor,
                    abs(old_valence) * c.importance_emotional_residual + c.importance_baseline
                )

        self.log.append("Emotional Stripping: decoupled emotion from consolidated memories")

    # ── Phase 4: Schema Extraction + Abstraction Emergence ─

    def _schema_extraction(self) -> int:
        """Extract schemas AND discover emergent abstract categories.

        Two-stage process:
        1. Tag-based schema extraction (legacy, for tagged anchors)
        2. Embedding-cluster-based abstraction (new — emergent categories)
        """
        c = self.cfg.sleep.schema
        formed = 0

        tag_groups: dict[str, list[Anchor]] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            for tag in anchor.tags:
                tag_groups[tag].append(anchor)

        for tag, group in tag_groups.items():
            if len(group) < c.min_instances:
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
            for i in range(min(c.min_instances, len(sorted_group))):
                for j in range(i + 1, min(c.min_instances, len(sorted_group))):
                    if sorted_group[i].embedding and sorted_group[j].embedding:
                        sim = self._embedding_similarity(
                            sorted_group[i].embedding, sorted_group[j].embedding)
                    else:
                        sim = 0.0
                    similarities.append(sim)

            avg_sim = sum(similarities) / max(1, len(similarities))
            if avg_sim < c.min_similarity:
                continue

            template_anchor = sorted_group[0]
            schema_id = f"schema_{tag}_{self._cycle_count}"
            schema = Schema(
                id=schema_id,
                template=template_anchor.text,
                slots={"topic": "specific topic instance", "context": "conversation context"},
                instance_ids=[a.id for a in sorted_group[:c.min_instances]],
                confidence=avg_sim,
                tags=[tag],
            )
            self.graph.schemas[schema_id] = schema
            formed += 1

            for a in sorted_group[:c.min_instances]:
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
                min_cluster_size=self.cfg.abstraction.min_cluster_size,
                similarity_threshold=self.cfg.abstraction.similarity_threshold,
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

                # Gate: require tag overlap to prevent cross-topic cascade merging
                tag_overlap = len(set(a.tags) & set(b.tags))
                min_tag_overlap = getattr(self.cfg.sleep.merge, 'min_tag_overlap', 1)

                if overlap > threshold and tag_overlap >= min_tag_overlap:
                    core, variant = (a, b) if a.created_at < b.created_at else (b, a)
                    core.vector.importance = max(core.vector.importance, variant.vector.importance)
                    core.vector.frequency = (core.vector.frequency + variant.vector.frequency) / 2
                    core.vector.stability = min(1.0, core.vector.stability + self.cfg.sleep.merge.stability_boost)
                    core.tags = list(set(core.tags + variant.tags))
                    for neighbor in list(self.graph._adjacency.get(variant.id, set())):
                        k = self.graph._key(variant.id, neighbor)
                        old = self.graph.edges.pop(k, None)
                        if old:
                            nk = self.graph._key(core.id, neighbor)
                            if nk not in self.graph.edges:
                                new_edge = self._transfer_edge(old, nk)
                                self.graph.edges[nk] = new_edge
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
                    penalties[aid_a] += self.cfg.sleep.prune.contradiction_penalty
                else:
                    penalties[aid_b] += self.cfg.sleep.prune.contradiction_penalty

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
                        residual_edges[neighbor] = edge.weight * self.cfg.sleep.prune.residual_edge_factor

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
                if isinstance(ghost, GhostAnchor) and (now - ghost.pruned_at) > self.cfg.sleep.prune.ghost_stale_days * 86400 and ghost.revival_count == 0:
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
                if score > self.cfg.sleep.bridge.min_score:
                    rep_a = c_a.anchors[0]
                    rep_b = c_b.anchors[0]
                    existing = self.graph.edges.get(self.graph._key(rep_a.id, rep_b.id))
                    if not existing or existing.weight < self.cfg.sleep.bridge.default_weight:
                        self.graph.add_edge(rep_a.id, rep_b.id, weight=self.cfg.sleep.bridge.default_weight,
                                            edge_type="bridge")
                        bridges += 1

        if bridges:
            self.log.append(f"Bridge: created {bridges} cross-constellation connections")
        return bridges

    # ── Phase 8: Hebbian Update ─────────────────────────

    def _hebbian_update(self) -> None:
        c = self.cfg.sleep.hebbian
        now = time.time()
        for edge in self.graph.edges.values():
            hours = (now - edge.last_activated_at) / 3600
            if edge.co_activation_count > 0 and hours < c.active_window_hours:
                edge.strengthen(c.strengthen_delta)
            else:
                decay = c.decay_log_factor * math.log(1 + hours / c.active_window_hours)
                edge.weaken(min(c.max_decay, decay))

    # ── Phase 9: Synaptic Homeostasis ───────────────────

    def _synaptic_homeostasis(self) -> None:
        if not self.graph.edges:
            return

        c = self.cfg.sleep.homeostasis
        weights = [e.weight for e in self.graph.edges.values()]
        mean_w = sum(weights) / len(weights)

        if mean_w > c.target_mean:
            scale = c.target_mean / mean_w
            blend = c.scale_blend
            for edge in self.graph.edges.values():
                edge.weight *= (blend + (1 - blend) * scale)

        for anchor in self.graph.anchors.values():
            hours = (time.time() - anchor.last_activated_at) / 3600
            anchor.decay(hours)

    # ── Phase 6: Hub Connection ────────────────────────────

    def _connect_cross_cortex_hubs(self, hublayer, cortices: list) -> int:
        """Detect cross-cortex co-occurrence patterns and create hub links.

        When the same topic/pattern appears in multiple cortices, create a
        hub-to-hub edge in the HubSphere to enable cross-domain reasoning.
        """
        connections = 0
        # Look for shared tags or similar segments across cortices
        for i, ctx_a in enumerate(cortices):
            seg_a = ctx_a.get_segment_for_hub("compressed")
            if not seg_a or not seg_a.centroid:
                continue
            for ctx_b in cortices[i + 1:]:
                seg_b = ctx_b.get_segment_for_hub("compressed")
                if not seg_b or not seg_b.centroid:
                    continue
                # Check centroid similarity
                sim = _cosine_sim_sleep(seg_a.centroid, seg_b.centroid)
                if sim > 0.6:
                    # Check if there's already a hub for each
                    hub_a_id = seg_a.hub_links[0] if seg_a.hub_links else None
                    hub_b_id = seg_b.hub_links[0] if seg_b.hub_links else None
                    if hub_a_id and hub_b_id:
                        hublayer.add_hub_edge(hub_a_id, hub_b_id, weight=sim, edge_type="cross_domain")
                        connections += 1
        return connections

    # ── Phase 7: Thermal Forgetting ────────────────────────

    def _apply_thermal_forgetting(self) -> dict:
        """Apply thermal lifecycle degradation.

        Scans all anchors and applies:
        - HOT→WARM: retention dropped, reduce priority
        - WARM→COLD: long-unaccessed, offload to index
        - COLD→DEAD: near-zero retention, hash-only
        """
        stats = {"hot": 0, "warm": 0, "cold": 0, "dead": 0,
                 "downgraded": 0, "finalized": 0}
        import time as _time
        now = _time.time()

        from .anchor import MemoryState as MS
        for anchor in self.graph.anchors.values():
            ts = anchor.thermal_state
            stats[ts.value] = stats.get(ts.value, 0) + 1

            if ts.value == "hot":
                # Check if should degrade to WARM
                hours_idle = (now - anchor.last_activated_at) / 3600
                if hours_idle > 72 and anchor.retention_score < 0.4:
                    anchor.vector.stability = max(0.0, anchor.vector.stability - 0.05)
                    stats["downgraded"] += 1

            elif ts.value == "cold":
                # Check COLD→DEAD: retention below 0.03
                if anchor.retention_score < 0.03:
                    anchor.state = MS.GHOST
                    anchor._ghost_reactivation_prob = 0.01
                    stats["finalized"] += 1

        return stats

    # ── Phase 8: Index Rebuild ─────────────────────────────

    def _rebuild_ann_index(self):
        """Rebuild the ANN search index from current anchors."""
        try:
            from .index import ANNIndex
            index = ANNIndex(self.graph)
            index.build()
        except Exception:
            pass  # Index rebuild is best-effort

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
    def _transfer_edge(old: Edge, new_key: tuple[str, str]) -> Edge:
        """Transfer an edge to a new key, preserving RichEdge properties."""
        from .graph import RichEdge
        if isinstance(old, RichEdge):
            return RichEdge(
                source=new_key[0], target=new_key[1],
                weight=old.weight, edge_type=old.edge_type,
                confidence=old.confidence, source_type=old.source_type,
                reinforcement_count=old.reinforcement_count,
                decay_rate=old.decay_rate, is_stale=old.is_stale,
                stale_since=old.stale_since, replaced_by=old.replaced_by,
                version_history=list(old.version_history),
            )
        return Edge(source=new_key[0], target=new_key[1],
                    weight=old.weight, edge_type=old.edge_type)

    @staticmethod
    def _infer_edge_type(a: Anchor, b: Anchor) -> str:
        if a.source_session == b.source_session:
            return "temporal"
        if set(a.tags) & set(b.tags):
            return "topical"
        return "topical"


def _cosine_sim_sleep(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
