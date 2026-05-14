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
import logging
import math
import time
from collections import defaultdict
from typing import Optional, Callable, Awaitable

from .anchor import Anchor, AnchorVector, MemoryState
from .graph import StarGraph, Edge, Constellation, Schema
from .config import Config
from .compression import CompressionLevel
from .atom_facts import FactExtractor, ExtractionResult


# ── Sleep Report — imported from sleep_report.py ──────
from .sleep_report import PhaseMetrics, SleepReport


class SleepCycle:
    """One complete sleep/consolidation cycle."""

    def __init__(self, graph: StarGraph, config: Config | None = None):
        self.graph = graph
        self.cfg = config if config is not None else Config.get()
        self.log: list[str] = []
        self._log = logging.getLogger("star_graph.sleep")
        self._cycle_count: int = 0
        self._ghost_count: int = 0
        self._embedder = None
        self._compressor = None

    def _log_event(self, msg: str) -> None:
        """Log to both the structured report list and the logging system."""
        self._log.info(msg)
        self.log.append(msg)

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _get_compressor(self):
        """Lazy-init the multi-level compressor."""
        if self._compressor is None:
            from .compression import MultiLevelCompressor, SessionCompressor
            self._compressor = MultiLevelCompressor()
        return self._compressor

    def _compress_clusters(self) -> dict:
        """Collect DORMANT/CONSOLIDATING anchors, group by session, and compress.

        This runs the full compression pipeline (RAW → EPISODIC → STRATEGIC → META)
        on anchors that are candidates for consolidation. Only anchors in
        DORMANT or CONSOLIDATING state are eligible — ACTIVE/REHEARSING anchors
        are still being processed and should not be compressed yet.

        Returns compression statistics.
        """
        # Collect eligible anchors: DORMANT or CONSOLIDATING with embeddings
        anchors_by_session: dict[str, list[Anchor]] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            if anchor.state in (MemoryState.DORMANT, MemoryState.CONSOLIDATING):
                if anchor.embedding and anchor.source_session:
                    anchors_by_session[anchor.source_session].append(anchor)

        # Skip if nothing to compress
        total_eligible = sum(len(v) for v in anchors_by_session.values())
        if total_eligible == 0:
            return {"episodic": 0, "strategic": 0, "meta": 0,
                    "eligible_anchors": 0, "compressed_anchors": 0}

        compressor = self._get_compressor()

        # Run full pipeline
        results = compressor.compress_pipeline(anchors_by_session)

        # Insert all summaries into the graph
        total_inserted = 0
        for level, summaries in results.items():
            if summaries:
                inserted = compressor.add_to_graph(self.graph, summaries, edge_type="compresses")
                total_inserted += inserted

        # Count compressed source anchors
        compressed_anchor_ids: set[str] = set()
        for summaries in results.values():
            for s in summaries:
                compressed_anchor_ids.update(s.source_anchor_ids)

        stats = {
            "episodic": len(results.get(CompressionLevel.EPISODIC, [])),
            "strategic": len(results.get(CompressionLevel.STRATEGIC, [])),
            "meta": len(results.get(CompressionLevel.META, [])),
            "eligible_anchors": total_eligible,
            "compressed_anchors": len(compressed_anchor_ids),
            "edges_created": total_inserted,
        }

        if stats["episodic"] > 0:
            self._log_event(
                f"Compression: {stats['episodic']} episodic + {stats['strategic']} strategic "
                f"+ {stats['meta']} meta summaries from {stats['compressed_anchors']} anchors "
                f"({total_inserted} 'compresses' edges)"
            )

        return stats

    def _get_fact_extractor(self) -> FactExtractor:
        """Lazy-init fact extractor. Config-driven with environment fallback."""
        if getattr(self, '_fact_extractor', None) is None:
            import os
            af_cfg = getattr(self.cfg, 'atom_facts', None)
            provider = getattr(af_cfg, 'provider', 'template') if af_cfg else 'template'
            model = getattr(af_cfg, 'model', '') if af_cfg else ''
            min_cluster = getattr(af_cfg, 'min_cluster_size', 3) if af_cfg else 3
            min_conf = getattr(af_cfg, 'min_fact_confidence', 0.4) if af_cfg else 0.4
            max_batch = getattr(af_cfg, 'max_anchors_per_batch', 15) if af_cfg else 15

            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY", "")
                base_url = os.environ.get("OPENAI_BASE_URL", "")
                if not api_key:
                    provider = "template"
                else:
                    self._fact_extractor = FactExtractor(
                        provider="openai", api_key=api_key,
                        base_url=base_url, model=model,
                        min_cluster_size=min_cluster,
                        min_fact_confidence=min_conf,
                        max_anchors_per_batch=max_batch,
                    )
            elif provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                # Also supports Minimax gateway (MINIMAX_API_KEY + MINIMAX_API_BASE_URL)
                if not api_key:
                    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
                    minimax_url = os.environ.get("MINIMAX_API_BASE_URL", "")
                    if minimax_key and minimax_url:
                        self._fact_extractor = FactExtractor(
                            provider="openai", api_key=minimax_key,
                            base_url=minimax_url, model=model or "anthropic/claude-haiku-4-5",
                            min_cluster_size=min_cluster,
                            min_fact_confidence=min_conf,
                            max_anchors_per_batch=max_batch,
                        )
                        return self._fact_extractor
                    provider = "template"
                else:
                    self._fact_extractor = FactExtractor(
                        provider="anthropic", api_key=api_key,
                        model=model or "claude-haiku-4-5",
                        min_cluster_size=min_cluster,
                        min_fact_confidence=min_conf,
                        max_anchors_per_batch=max_batch,
                    )
            if provider == "template":
                self._fact_extractor = FactExtractor(
                    provider="template",
                    min_cluster_size=min_cluster,
                    min_fact_confidence=min_conf,
                    max_anchors_per_batch=max_batch,
                )
        return self._fact_extractor

    def _extract_atom_facts(self) -> dict:
        """Extract atomic entity-centric facts from compressed anchor clusters.

        LLM post-processing during sleep consolidation:
        1. Collect DORMANT/CONSOLIDATING anchors grouped by session
        2. For each session cluster, extract AtomFacts via LLM/Template
        3. Add facts to graph with bidirectional edges to source anchors

        Returns extraction statistics.
        """
        # Collect anchors by session for clustering
        anchors_by_session: dict[str, list[Anchor]] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            if anchor.state in (MemoryState.DORMANT, MemoryState.CONSOLIDATING):
                if anchor.source_session:
                    anchors_by_session[anchor.source_session].append(anchor)

        if not anchors_by_session:
            return {"clusters": 0, "facts_extracted": 0, "provider": "none"}

        extractor = self._get_fact_extractor()

        # Build clusters: (topic, anchors) for each session
        clusters: list[tuple[str, list[Anchor]]] = []
        for session_id, anchors in anchors_by_session.items():
            topic = self._infer_session_topic(anchors)
            clusters.append((topic, anchors))

        # Also include non-session clusters based on tag similarity
        by_tag: dict[str, list[Anchor]] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            if anchor.state in (MemoryState.DORMANT, MemoryState.CONSOLIDATING):
                if anchor.tags:
                    for tag in anchor.tags:
                        if tag not in ('dormant', 'consolidating', 'ghost'):
                            by_tag[tag].append(anchor)
        for tag, anchors in by_tag.items():
            if len(anchors) >= 3:
                clusters.append((f"topic:{tag}", anchors))

        # Extract facts
        all_facts = extractor.extract_from_clusters(clusters)

        # Add to graph
        inserted = extractor.add_facts_to_graph(self.graph, all_facts)

        if inserted > 0:
            self._log_event(
                f"Atom Facts: {inserted} facts extracted from {len(clusters)} clusters "
                f"(provider: {extractor.provider_name})"
            )

        return {
            "clusters": len(clusters),
            "facts_extracted": inserted,
            "provider": extractor.provider_name,
            "extraction_count": extractor._extraction_count,
        }

    def _infer_session_topic(self, anchors: list[Anchor]) -> str:
        """Infer the topic of a session from its anchors' tags and text."""
        tag_counts: dict[str, int] = defaultdict(int)
        for anchor in anchors:
            for tag in anchor.tags:
                if tag not in ('dormant', 'consolidating', 'active', 'ghost'):
                    tag_counts[tag] += 1
        if tag_counts:
            top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:3]
            return " + ".join(top_tags)
        return "general"

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
        rebuild_result = self._sleep_rebuild()
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
            "rebuild": rebuild_result,
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
          Phase 5b: Compression — cluster & compress DORMANT anchors
          Phase 5c: Atom Facts — LLM-assisted entity-centric fact extraction
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
            phase="N1_Replay",
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
            phase="N2_Merge",
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
            phase="N3_Compression",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=schemas,
            details={"schemas": schemas, "abstractions": abstraction_count,
                     "cortical": cortical_count},
        ))
        report.schemas_formed = schemas
        report.abstractions_formed = abstraction_count
        report.cortical_transferred = cortical_count

        # ── N3d: Sleep Rebuild ──
        t0 = time.time()
        rebuild_result = self._sleep_rebuild()
        report.phases.append(PhaseMetrics(
            phase="N3d_SleepRebuild",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=rebuild_result.get("fused_nodes", 0),
            details=rebuild_result,
        ))

        # ── REM: Emotional Decoupling ──
        t0 = time.time()
        self._emotional_stripping()
        self._synaptic_homeostasis()
        emotion_count = sum(
            1 for a in self.graph.anchors.values()
            if abs(a.vector.emotional_valence) < 0.1 and a.vector.stability > 0.5
        )
        report.phases.append(PhaseMetrics(
            phase="REM_Emotion",
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
            phase="N4_Prune",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=len(pruned_anchors) + len(pruned_edges),
            details={"pruned_anchors": len(pruned_anchors),
                     "pruned_edges": len(pruned_edges),
                     "ghosts": ghosts_created},
        ))
        report.memories_pruned = len(pruned_anchors)
        report.ghosts_created = ghosts_created
        report.edges_pruned = len(pruned_edges)

        # ── Phase 5b: Compression ── (after Tier Promotion, before Hub Connection)
        t0 = time.time()
        compression_stats = self._compress_clusters()
        report.phases.append(PhaseMetrics(
            phase="N3b_ClusterCompression",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=compression_stats.get("compressed_anchors", 0),
            details=compression_stats,
        ))

        # ── Phase 5c: Atom Fact Extraction ── (LLM-assisted, after compression)
        t0 = time.time()
        fact_stats = self._extract_atom_facts()
        report.phases.append(PhaseMetrics(
            phase="N3c_AtomFacts",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=fact_stats.get("facts_extracted", 0),
            details=fact_stats,
        ))

        # ── Phase 6: Hub Connection ──
        t0 = time.time()
        hub_connections = 0
        if hublayer and cortices:
            hub_connections = self._connect_cross_cortex_hubs(hublayer, cortices)
        report.phases.append(PhaseMetrics(
            phase="N5_HubConnect",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=hub_connections,
            details={"cross_cortex_links": hub_connections},
        ))

        # ── Phase 7: Forgetting/Degradation ──
        t0 = time.time()
        reinforcement_stats = self._apply_reinforcement_decay()
        thermal_stats = self._apply_thermal_forgetting()
        report.phases.append(PhaseMetrics(
            phase="N6_Forgetting",
            duration_ms=(time.time() - t0) * 1000,
            items_processed=sum(thermal_stats.values()),
            details={**thermal_stats, "reinforcement": reinforcement_stats},
        ))

        # ── Phase 8: Index Rebuild + BrainSphere refresh ──
        t0 = time.time()
        self._refresh_cortical_index()  # existing ANN index rebuild
        self._rebuild_ann_index()
        if brain and cortices:
            brain.refresh_cache(cortices)
        report.phases.append(PhaseMetrics(
            phase="N7_IndexRebuild",
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

    def _constrained_candidates(self, anchor: Anchor, existing: Anchor,
                                 window_hours: float = 168.0) -> dict[str, Anchor]:
        """Build a constrained candidate set for edge creation.

        Instead of full-graph O(n²) similarity search, only consider anchors
        that share: same session, same topic (tag overlap), same time window,
        or same entities. This keeps edge creation O(k) where k << n.
        """
        candidates: dict[str, Anchor] = {}
        now = time.time()
        anchor_time = existing.created_at if existing else now

        for other_id, other in self.graph.anchors.items():
            if other_id == (existing.id if existing else anchor.id):
                continue

            # Constraint 1: same session
            if existing.source_session and other.source_session == existing.source_session:
                candidates[other_id] = other
                continue

            # Constraint 2: same topic (tag overlap)
            if set(existing.tags) & set(other.tags):
                candidates[other_id] = other
                continue

            # Constraint 3: same time window
            time_diff = abs(anchor_time - other.created_at) / 3600
            if time_diff < window_hours:
                candidates[other_id] = other
                continue

        return candidates

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

            # Connect using topology-constrained similarity (NOT full-graph O(n²))
            anchor_emb = existing.embedding if existing else anchor.embedding
            if not anchor_emb:
                anchor_emb = embedder.encode(anchor.text)

            anchor_id = existing.id if existing else anchor.id
            candidates = self._constrained_candidates(anchor, existing or anchor, window_hours=168)
            for other_id, other in candidates.items():
                if other_id == anchor_id:
                    continue
                if not other.embedding:
                    continue
                sim = self._embedding_similarity(anchor_emb, other.embedding)
                if sim > threshold:
                    edge_type = self._infer_edge_type(anchor, other)
                    weight = sim * (1.0 + self.cfg.sleep.edge_formation.emotion_weight_boost * abs(anchor.vector.emotional_valence))
                    self.graph.add_edge(anchor_id, other_id,
                                        weight=min(1.0, weight),
                                        edge_type=edge_type)

        self._log_event(f"SWR Replay: replayed {len(prioritized)} anchors "
                        f"(topology-constrained linking, compression ~{max(1, len(recent)//3)}:1)")

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
        self._log_event(f"Systems Consolidation: {cortical_count}/{len(self.graph.anchors)} "
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

        self._log_event("Emotional Stripping: decoupled emotion from consolidated memories")

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
            self._log_event(f"Schema Extraction: formed {formed} new schemas (embedding-based)")

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
            self._log_event(
                f"Abstraction Emergence: discovered {len(new_abstracts)} "
                f"new concepts: {[a.label for a in new_abstracts]}"
            )

        return len(new_abstracts)

    # ── Phase 4c: Sleep Rebuild ─────────────────────────

    def _sleep_rebuild(self) -> dict:
        """Restructure the entire graph — not just compress, but rebuild.

        Four-step process:
        1. Multi-node fusion: cluster 3+ related anchors → single fused anchor
        2. Graph rewiring: drop dead edges, strengthen success paths, transitive closure
        3. Dynamic rewiring: RL-based edge updates from success/failure history
        4. Abstractive memory: concrete events → pattern memory with faster source decay

        Returns a dict with per-step metrics.
        """
        t0 = time.time()
        fused = self._rebuild_fuse_nodes()
        t1 = time.time()
        rewired = self._rebuild_rewire_graph()
        t2 = time.time()
        dynamic = self._dynamic_rewire()
        t3 = time.time()
        abstracted = self._rebuild_abstractive_memory()
        t4 = time.time()

        result = {
            "fused_nodes": fused,
            "rewired_edges": rewired,
            "dynamic_rewire": dynamic,
            "abstracted_patterns": abstracted,
            "fuse_ms": (t1 - t0) * 1000,
            "rewire_ms": (t2 - t1) * 1000,
            "dynamic_ms": (t3 - t2) * 1000,
            "abstract_ms": (t4 - t3) * 1000,
        }

        parts = []
        if fused:
            parts.append(f"fused {fused} node clusters")
        dropped = rewired.get("dropped", 0)
        strengthened = rewired.get("strengthened", 0)
        transitive = rewired.get("transitive_added", 0)
        if dropped or strengthened or transitive:
            parts.append(f"rewired edges (-{dropped} +{strengthened} +{transitive}t)")
        dyn_boosted = dynamic.get("boosted", 0)
        dyn_weakened = dynamic.get("weakened", 0)
        dyn_formed = dynamic.get("clusters_formed", 0)
        if dyn_boosted or dyn_weakened or dyn_formed:
            parts.append(f"dynamic (+{dyn_boosted} -{dyn_weakened} c{dyn_formed})")
        if abstracted:
            parts.append(f"abstracted {abstracted} patterns")
        if parts:
            self._log_event("Sleep Rebuild: " + ", ".join(parts))

        return result

    def _rebuild_fuse_nodes(self) -> int:
        """Multi-node fusion: cluster 3+ related anchors into a single abstraction.

        Uses community detection to find semantic clusters, then within each
        cluster runs hierarchical merging on embedding similarity. Anchors that
        are near-identical (cosine > 0.85) get fused into the oldest anchor.
        This goes beyond pair-wise merge — it handles chains like:
          "try-except in Python" + "python异常处理" + "错误捕获"
          → fused into "Python Error Handling"

        Returns number of nodes fused (removed from graph).
        """
        fused_count = 0
        threshold = getattr(self.cfg.sleep, 'rebuild_fuse_threshold', 0.85)
        min_cluster = getattr(self.cfg.sleep, 'rebuild_min_cluster', 3)

        # Step 1: detect communities to scope the fusion search
        try:
            from .community import CommunityDetection
            detector = CommunityDetection(min_community_size=min_cluster)
            communities = detector.detect(self.graph)
        except Exception:
            communities = []

        # If community detection fails, fall back to tag-based grouping
        if not communities:
            tag_groups: dict[str, list[str]] = {}
            for aid, a in self.graph.anchors.items():
                for tag in a.tags:
                    if tag not in tag_groups:
                        tag_groups[tag] = []
                    tag_groups[tag].append(aid)
            # Build pseudo-communities from tag groups with 3+ anchors
            communities = []
            for tag, ids in tag_groups.items():
                if len(ids) >= min_cluster:
                    from .community import Community
                    communities.append(Community(
                        id=f"tag_{tag}", anchor_ids=ids,
                        topic_label=tag, size=len(ids),
                    ))

        processed: set[str] = set()

        for community in communities:
            # Get anchors in this community that have embeddings
            community_anchors = {
                aid: self.graph.anchors[aid]
                for aid in community.anchor_ids
                if aid in self.graph.anchors
                and aid not in processed
                and self.graph.anchors[aid].embedding
            }
            if len(community_anchors) < min_cluster:
                continue

            # Step 2: compute pairwise cosine similarity within community
            ids = list(community_anchors.keys())
            n = len(ids)
            # Build adjacency of high-similarity pairs
            pairs: list[tuple[int, int, float]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    a = community_anchors[ids[i]]
                    b = community_anchors[ids[j]]
                    sim = self._embedding_similarity(a.embedding, b.embedding)
                    if sim > threshold:
                        pairs.append((i, j, sim))

            if not pairs:
                continue

            # Step 3: union-find to group transitively similar anchors
            parent = list(range(n))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            for i, j, _ in pairs:
                union(i, j)

            # Step 4: collect groups of 3+ anchors
            groups: dict[int, list[int]] = defaultdict(list)
            for i in range(n):
                groups[find(i)].append(i)

            for root, indices in groups.items():
                if len(indices) < min_cluster:
                    continue

                group_aids = [ids[i] for i in indices]
                group_anchors = [community_anchors[aid] for aid in group_aids]

                # Fuse into the oldest anchor (the "core")
                core = min(group_anchors, key=lambda a: a.created_at)
                others = [a for a in group_anchors if a.id != core.id]

                # Merge properties into core
                for other in others:
                    core.vector.importance = max(core.vector.importance, other.vector.importance)
                    core.vector.frequency = (core.vector.frequency + other.vector.frequency) / 2
                    core.vector.stability = min(1.0, core.vector.stability + 0.05)
                    core.vector.emotional_valence = (
                        core.vector.emotional_valence + other.vector.emotional_valence
                    ) / 2
                    core.tags = list(set(core.tags + other.tags))
                    # Fuse text: append key phrases from other
                    if len(core.text) < 500 and other.text not in core.text:
                        core.text = core.text + "; " + other.text[:200]

                    # Transfer edges from other → core
                    for neighbor in list(self.graph._adjacency.get(other.id, set())):
                        if neighbor == core.id:
                            continue
                        k = self.graph._key(other.id, neighbor)
                        old_edge = self.graph.edges.pop(k, None)
                        if old_edge:
                            nk = self.graph._key(core.id, neighbor)
                            if nk not in self.graph.edges:
                                new_edge = self._transfer_edge(old_edge, nk)
                                self.graph.edges[nk] = new_edge
                                self.graph._adjacency[core.id].add(neighbor)
                                self.graph._adjacency[neighbor].add(core.id)
                            else:
                                # Both core and other connected to same neighbor — reinforce
                                existing = self.graph.edges[nk]
                                existing.strengthen(0.03)

                    # Remove other from adjacency
                    if other.id in self.graph._adjacency:
                        for neighbor in list(self.graph._adjacency[other.id]):
                            self.graph._adjacency[neighbor].discard(other.id)
                        del self.graph._adjacency[other.id]

                    self.graph.remove_anchor(other.id)
                    processed.add(other.id)

                # Mark core as abstractive-fused
                core.tags.append("sleep_rebuilt")
                core.vector.stability = min(1.0, core.vector.stability + 0.1)
                processed.add(core.id)
                fused_count += len(others)

        return fused_count

    def _rebuild_rewire_graph(self) -> dict:
        """Active graph rewiring: drop dead edges, strengthen success paths,
        add transitive closure edges for strong two-hop paths.

        Three operations:
        1. Drop: edges with weight < 0.05 AND co_activation_count == 0
        2. Strengthen: edges with high co-activation (>=3) get a boost
        3. Transitive closure: if A→B and B→C are both strong (weight>0.6),
           create A→C with weight = w_ab * w_bc * 0.7

        Returns dict with counts: {dropped, strengthened, transitive_added}.
        """
        dropped = 0
        strengthened = 0
        transitive_added = 0
        drop_threshold = getattr(self.cfg.sleep, 'rewire_drop_threshold', 0.05)
        strengthen_coactivation = getattr(self.cfg.sleep, 'rewire_strengthen_min_coact', 3)
        transitive_min_weight = getattr(self.cfg.sleep, 'rewire_transitive_min_weight', 0.6)
        transitive_decay = getattr(self.cfg.sleep, 'rewire_transitive_decay', 0.7)

        # ── Step 1: Drop dead edges ──
        dead_keys = []
        for key, edge in self.graph.edges.items():
            if edge.weight < drop_threshold and edge.co_activation_count == 0:
                dead_keys.append(key)

        for key in dead_keys:
            a, b = key
            self.graph._adjacency[a].discard(b)
            self.graph._adjacency[b].discard(a)
            del self.graph.edges[key]
        dropped = len(dead_keys)

        # ── Step 2: Strengthen high-utility edges ──
        for edge in self.graph.edges.values():
            if edge.co_activation_count >= strengthen_coactivation:
                boost = min(0.15, edge.co_activation_count * 0.02)
                edge.strengthen(boost)
                strengthened += 1

        # ── Step 3: Transitive closure for strong two-hop paths ──
        new_edges: list[tuple[str, str, float]] = []
        for a_id in list(self.graph._adjacency.keys()):
            a_neighbors = self.graph._adjacency.get(a_id, set())
            for b_id in a_neighbors:
                ab_key = self.graph._key(a_id, b_id)
                ab_edge = self.graph.edges.get(ab_key)
                if not ab_edge or ab_edge.weight < transitive_min_weight:
                    continue
                b_neighbors = self.graph._adjacency.get(b_id, set())
                for c_id in b_neighbors:
                    if c_id == a_id:
                        continue
                    bc_key = self.graph._key(b_id, c_id)
                    bc_edge = self.graph.edges.get(bc_key)
                    if not bc_edge or bc_edge.weight < transitive_min_weight:
                        continue
                    # A→B strong, B→C strong — check if A→C already exists
                    ac_key = self.graph._key(a_id, c_id)
                    if ac_key in self.graph.edges:
                        continue  # already connected
                    # Don't create self-loops
                    if a_id == c_id:
                        continue
                    transitive_weight = ab_edge.weight * bc_edge.weight * transitive_decay
                    if transitive_weight > 0.15:  # minimum useful weight
                        new_edges.append((a_id, c_id, transitive_weight))

        # Deduplicate and add
        seen = set()
        for a_id, c_id, w in new_edges:
            key = self.graph._key(a_id, c_id)
            if key not in seen and key not in self.graph.edges:
                seen.add(key)
                self.graph.add_edge(a_id, c_id, weight=w, edge_type="topical",
                                    relation="transitive_closure", source_type="inferred")
                transitive_added += 1

        return {
            "dropped": dropped,
            "strengthened": strengthened,
            "transitive_added": transitive_added,
        }

    def _dynamic_rewire(self) -> dict:
        """RL-based dynamic neural rewiring using success/failure history on edges.

        Three operations:
        1. Boost: edges with high success_rate (>=0.7) get strengthened
        2. Weaken: edges with low success_rate (<0.3) with enough trials get weakened
        3. Cluster: high co-activation edges form community clusters

        This is the self-evolving mechanism — the graph learns which connections
        are useful and which are not through reinforcement.

        Returns dict with counts: {boosted, weakened, clusters_formed}.
        """
        boosted = 0
        weakened = 0
        clusters_formed = 0
        min_trials = getattr(self.cfg.sleep, 'rewire_strengthen_min_coact', 3)
        success_boost_threshold = 0.7
        failure_weaken_threshold = 0.3
        cluster_coact_threshold = getattr(self.cfg.sleep, 'dynamic_cluster_coact', 5)

        for key, edge in list(self.graph.edges.items()):
            total_trials = edge.success_count + edge.failure_count

            if total_trials >= min_trials:
                rate = edge.success_rate

                if rate >= success_boost_threshold and edge.success_count >= 2:
                    # Successful reasoning chain — strengthen
                    boost = min(0.1, edge.success_count * 0.015)
                    edge.strengthen(boost)
                    boosted += 1

                elif rate < failure_weaken_threshold and total_trials >= 5:
                    # Failed reasoning chain — significantly weaken
                    penalty = min(0.1, edge.failure_count * 0.01)
                    edge.weaken(penalty)
                    weakened += 1

            # Reset counters after processing (avoid unbounded accumulation)
            edge.success_count = max(0, edge.success_count - 1)  # gradual decay
            edge.failure_count = max(0, edge.failure_count - 1)

            # Form clusters from high co-activation edges
            if edge.co_activation_count >= cluster_coact_threshold:
                # Mark both anchors for community formation
                a, b = key
                if a in self.graph.anchors and b in self.graph.anchors:
                    anchor_a = self.graph.anchors[a]
                    anchor_b = self.graph.anchors[b]
                    # Tag with shared cluster label
                    cluster_tag = f"cluster_{a[:8]}_{b[:8]}"
                    if cluster_tag not in anchor_a.tags:
                        anchor_a.tags.append(cluster_tag)
                    if cluster_tag not in anchor_b.tags:
                        anchor_b.tags.append(cluster_tag)
                    if anchor_a.community_id and anchor_b.community_id:
                        # Both already in communities — bridge them
                        if anchor_a.community_id != anchor_b.community_id:
                            anchor_a.secondary_community_ids.append(anchor_b.community_id)
                            anchor_b.secondary_community_ids.append(anchor_a.community_id)
                    elif anchor_a.community_id:
                        anchor_b.community_id = anchor_a.community_id
                    elif anchor_b.community_id:
                        anchor_a.community_id = anchor_b.community_id
                    clusters_formed += 1

        return {
            "boosted": boosted,
            "weakened": weakened,
            "clusters_formed": clusters_formed,
        }

    def _rebuild_abstractive_memory(self) -> int:
        """Convert groups of concrete events into pattern memory.

        Finds anchors sharing the same schema_ref or tag group, generates a
        higher-level "pattern" anchor, and links concrete anchors to it with
        "instance_of" edges. The concrete anchors get their retention reduced
        (faster decay), while the pattern is stable.

        Example: "chromedriver fix failed on v124" + "chromedriver fix failed on v125"
                 → Pattern: "Browser Driver Version Compatibility Issue"

        Returns number of patterns created.
        """
        abstracted = 0
        min_group_size = getattr(self.cfg.sleep, 'abstractive_min_group', 4)
        decay_factor = getattr(self.cfg.sleep, 'abstractive_decay_factor', 0.6)

        # Step 1: group anchors by tag-based topics
        tag_groups: dict[str, list[str]] = defaultdict(list)
        for aid, a in self.graph.anchors.items():
            if a.state.name in ('DORMANT', 'CONSOLIDATING'):
                for tag in a.tags:
                    if tag not in ('dormant', 'consolidating', 'ghost', 'sleep_rebuilt'):
                        tag_groups[tag].append(aid)

        # Also group by schema_ref
        schema_groups: dict[str, list[str]] = defaultdict(list)
        for aid, a in self.graph.anchors.items():
            if a.schema_ref:
                schema_groups[a.schema_ref].append(aid)

        # Merge tag_groups and schema_groups into unified topic groups
        all_groups: dict[str, set[str]] = {}
        for tag, ids in tag_groups.items():
            if len(ids) >= min_group_size:
                all_groups[f"tag:{tag}"] = set(ids)
        for schema_ref, ids in schema_groups.items():
            if len(ids) >= min_group_size:
                key = f"schema:{schema_ref}"
                if key in all_groups:
                    all_groups[key] |= set(ids)
                else:
                    all_groups[key] = set(ids)

        for group_key, anchor_ids in all_groups.items():
            if len(anchor_ids) < min_group_size:
                continue

            # Get the actual anchors
            group_anchors = [
                self.graph.anchors[aid] for aid in anchor_ids
                if aid in self.graph.anchors and self.graph.anchors[aid].embedding
            ]
            if len(group_anchors) < min_group_size:
                continue

            # Compute centroid embedding
            dim = len(group_anchors[0].embedding)
            centroid = [0.0] * dim
            for a in group_anchors:
                for i in range(dim):
                    centroid[i] += a.embedding[i]
            for i in range(dim):
                centroid[i] /= len(group_anchors)

            # Generate pattern label from tags
            all_tags: list[str] = []
            for a in group_anchors:
                all_tags.extend(a.tags)
            tag_counts = Counter(all_tags)
            top_tags = [t for t, _ in tag_counts.most_common(3)
                       if t not in ('dormant', 'consolidating', 'ghost')]
            pattern_label = " + ".join(top_tags) if top_tags else group_key

            # Generate pattern description from the shortest and longest anchors
            sorted_by_len = sorted(group_anchors, key=lambda a: len(a.text))
            short_desc = sorted_by_len[0].text[:200]
            long_desc = sorted_by_len[-1].text[:300]

            # Create the pattern anchor
            pattern_id = f"pattern_{group_key.replace(':', '_')}_{self._cycle_count}"
            pattern = Anchor.create(
                text=f"[Pattern] {pattern_label}: {short_desc} ... {long_desc}"[:800],
                tags=top_tags + ["pattern", "abstractive_memory"],
                importance=sum(a.vector.importance for a in group_anchors) / len(group_anchors),
                emotional_valence=sum(a.vector.emotional_valence for a in group_anchors) / len(group_anchors),
            )
            pattern.embedding = centroid
            pattern.vector.stability = 0.8
            pattern.vector.recency = max(a.vector.recency for a in group_anchors)
            pattern.id = pattern_id
            pattern.schema_ref = group_key

            self.graph.add_anchor(pattern)

            # Link pattern to concrete anchors with "instance_of" edges
            for a in group_anchors:
                self.graph.add_edge(
                    a.id, pattern_id,
                    weight=0.7, edge_type="topical",
                    relation="instance_of", source_type="explicit",
                    confidence=0.8,
                )
                # Reduce retention on concrete anchors — pattern survives
                a.vector.stability *= decay_factor
                a.vector.importance *= decay_factor

            abstracted += 1

        # Step 2: cross-session pattern detection via AbstractiveMemoryEngine
        try:
            from .abstraction import AbstractiveMemoryEngine
            if not hasattr(self, '_abstractive_engine'):
                self._abstractive_engine = AbstractiveMemoryEngine(
                    min_occurrences=min_group_size,
                    similarity_threshold=0.75,
                )
            engine = self._abstractive_engine

            # Extract new cross-session patterns
            new_patterns = engine.extract_patterns(self.graph)
            if new_patterns:
                # Promote stable patterns to abstract nodes
                promoted = engine.promote_stable_patterns(self.graph)
                # Consolidate: match existing patterns against all anchors
                engine.consolidate_existing_patterns(self.graph)
                abstracted += len(promoted)
                if promoted:
                    self._log_event(
                        f"Abstractive Memory: promoted {len(promoted)} cross-session "
                        f"patterns (engine: {engine.stats['total_patterns']} total, "
                        f"{engine.stats['recurring']} recurring)"
                    )
        except ImportError:
            pass

        return abstracted

    # ── Phase 5: Merge Similar ──────────────────────────

    def _merge_similar(self, threshold: float) -> int:
        """Merge near-duplicate anchors using embedding similarity.

        Uses ANN-index pre-filtering to avoid O(n²) pair enumeration.
        Only checks anchor pairs within cosine-distance reach (top-k per anchor).
        Falls back to O(n²) text-overlap scan for anchors without embeddings.
        """
        merged = 0
        processed: set[str] = set()
        ids = list(self.graph.anchors.keys())
        n = len(ids)
        min_tag_overlap = getattr(self.cfg.sleep.merge, 'min_tag_overlap', 1)

        # Build candidate pairs via ANN pre-filter (O(n log n) instead of O(n²))
        candidate_pairs: set[tuple[str, str]] = set()
        ann = self.graph._get_ann_index()
        ann_k = min(20, max(5, n // 4))

        # Anchors with embeddings: use ANN for candidate discovery
        embed_ids = [aid for aid in ids if aid in self.graph.anchors
                     and self.graph.anchors[aid].embedding]
        no_embed_ids = [aid for aid in ids if aid in self.graph.anchors
                        and not self.graph.anchors[aid].embedding]

        for aid in embed_ids:
            anchor = self.graph.anchors.get(aid)
            if anchor is None or aid in processed:
                continue
            neighbors = ann.query(anchor.embedding, k=ann_k)
            for nid, sim in neighbors:
                if nid != aid and sim > threshold * 0.8:  # pre-filter: near threshold
                    key = (aid, nid) if aid < nid else (nid, aid)
                    candidate_pairs.add(key)

        # Anchors without embeddings: add text-overlap candidates against all others
        for i, aid_a in enumerate(no_embed_ids):
            if aid_a in processed:
                continue
            for aid_b in ids:
                if aid_b in processed or aid_b == aid_a:
                    continue
                # Only include if the other anchor has no embedding either,
                # or skip — embedding anchors are handled above via ANN
                b = self.graph.anchors.get(aid_b)
                if b and b.embedding:
                    continue  # was already covered by ANN scan
                key = (aid_a, aid_b) if aid_a < aid_b else (aid_b, aid_a)
                candidate_pairs.add(key)

        # Process candidate pairs
        for aid_a, aid_b in candidate_pairs:
            if aid_a in processed or aid_b in processed:
                continue
            a = self.graph.anchors.get(aid_a)
            b = self.graph.anchors.get(aid_b)
            if a is None or b is None:
                continue

            # Prefer embedding similarity, fall back to bigrams
            if a.embedding and b.embedding:
                overlap = self._embedding_similarity(a.embedding, b.embedding)
            else:
                overlap = self._text_overlap(a.text, b.text)

            # Gate: require tag overlap to prevent cross-topic cascade merging
            tag_overlap = len(set(a.tags) & set(b.tags))

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

        if merged:
            self._log_event(f"Merge: fused {merged} duplicate anchor pairs")
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
                self.graph._ghost_subsystem.create(anchor, residual_edges)

                anchor.transition('prune')
                self.graph.remove_anchor(aid)
                self._ghost_count += 1

        # Decay ghosts via subsystem
        stale_count, _ = self.graph._ghost_subsystem.decay_all()

        if candidates:
            self._log_event(f"Adaptive Prune: removed {len(candidates)} anchors "
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
            self._log_event(f"Edge Prune: removed {len(candidates)} dormant edges")
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
            self._log_event(f"Bridge: created {bridges} cross-constellation connections")
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
        """Detect cross-cortex patterns and create hub links.

        Compares compressed segments across cortices. When two segments have
        similar centroids (cosine > 0.6), auto-creates leaf hubs from segments
        if none exist, then creates a hub-to-hub edge for cross-domain reasoning.
        """
        connections = 0

        def _ensure_segment_hub(seg, cortex_name: str):
            """Get or create a leaf hub for a segment."""
            if seg.hub_links:
                return seg.hub_links[0]
            if not seg.centroid or not seg.node_ids:
                return None
            topic = seg.summary or f"{cortex_name}_cluster"
            hub = hublayer.create_leaf(
                text=f"[{cortex_name}] {topic}",
                source_anchor_ids=list(seg.node_ids),
                cortex_name=cortex_name,
                importance=seg.importance,
                embedding=seg.centroid,
            )
            seg.link_hub(hub.id)
            return hub.id

        for i, ctx_a in enumerate(cortices):
            seg_a = ctx_a.get_segment_for_hub("compressed")
            if not seg_a or not seg_a.centroid:
                continue
            for ctx_b in cortices[i + 1:]:
                seg_b = ctx_b.get_segment_for_hub("compressed")
                if not seg_b or not seg_b.centroid:
                    continue
                sim = _cosine_sim_sleep(seg_a.centroid, seg_b.centroid)
                if sim > 0.6:
                    hub_a_id = _ensure_segment_hub(seg_a, ctx_a.config.name)
                    hub_b_id = _ensure_segment_hub(seg_b, ctx_b.config.name)
                    if hub_a_id and hub_b_id:
                        hublayer.add_hub_edge(hub_a_id, hub_b_id, weight=sim, edge_type="cross_domain")
                        connections += 1
        return connections

    # ── Phase 7: Thermal Forgetting ────────────────────────

    def _apply_reinforcement_decay(self) -> dict:
        """Adjust anchor decay rates based on success/feedback history.

        Reinforcement-adjusted decay formula:
          decay_rate = base_decay × (1 - success_feedback × 0.5) × (1 - reinforcement × 0.3)

        Anchors with high success_feedback decay slower (better retention).
        Anchors with low success_feedback decay faster (adaptive forgetting).
        Anchors with high confidence resist decay more strongly.

        Returns stats: {adjusted, boosted, penalized}.
        """
        adjusted = 0
        boosted = 0
        penalized = 0

        for anchor in self.graph.anchors.values():
            v = anchor.vector
            # Base decay rate from config or anchor's own rate
            base_rate = getattr(v, 'decay_rate', 0.01)

            # Success feedback: 0..1, higher = slower decay
            success_damping = 1.0 - v.success_feedback * 0.5

            # Confidence: higher confidence = slower decay
            confidence_damping = 1.0 - v.confidence * 0.2

            # Reinforcement from access patterns
            reinforcement = getattr(anchor, 'replay_count', 0) / max(1, self._cycle_count)
            reinforcement_damping = 1.0 - min(0.3, reinforcement * 0.3)

            # Stability slows decay further
            stability_damping = 1.0 - v.stability * 0.4

            # New effective decay rate
            effective_rate = base_rate * success_damping * confidence_damping
            effective_rate *= reinforcement_damping * stability_damping

            # Clamp
            effective_rate = max(0.001, min(0.5, effective_rate))

            if abs(effective_rate - base_rate) > 0.001:
                if effective_rate < base_rate:
                    boosted += 1
                else:
                    penalized += 1
                adjusted += 1

            v.decay_rate = effective_rate

        return {"adjusted": adjusted, "boosted": boosted, "penalized": penalized}

    def _apply_thermal_forgetting(self) -> dict:
        """Apply thermal lifecycle degradation.

        Five-level thermal downgrade:
        - HOT→WARM: retention dropped below 0.4, reduce priority
        - WARM→COLD: long-unaccessed, retention < 0.15, offload to index
        - COLD→FROZEN: retention < 0.06, disk-only archive tier
        - FROZEN→DEAD: retention < 0.01, hash-only audit trail
        """
        stats = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0, "dead": 0,
                 "downgraded": 0, "finalized": 0}
        import time as _time
        now = _time.time()

        from .anchor import ThermalState, MemoryState as MS
        for anchor in self.graph.anchors.values():
            ts = anchor.thermal_state
            stats[ts.value] = stats.get(ts.value, 0) + 1

            if ts == ThermalState.HOT:
                # HOT→WARM: idle > 72h and retention below 0.4
                hours_idle = (now - anchor.last_activated_at) / 3600
                if hours_idle > 72 and anchor.retention_score < 0.4:
                    anchor.vector.stability = max(0.0, anchor.vector.stability - 0.05)
                    stats["downgraded"] += 1

            elif ts == ThermalState.WARM:
                # WARM→COLD: very long idle or low retention
                hours_idle = (now - anchor.last_activated_at) / 3600
                if hours_idle > 720 or anchor.retention_score < 0.15:
                    anchor.vector.stability = max(0.0, anchor.vector.stability - 0.03)
                    anchor.vector.recency *= 0.5
                    stats["downgraded"] += 1

            elif ts == ThermalState.COLD:
                # COLD→FROZEN: retention below 0.06
                if anchor.retention_score < 0.06:
                    anchor.vector.stability = max(0.0, anchor.vector.stability - 0.02)
                    stats["downgraded"] += 1

            elif ts == ThermalState.FROZEN:
                # FROZEN→DEAD: retention below 0.01
                if anchor.retention_score < 0.01:
                    anchor.state = MS.GHOST
                    anchor._ghost_reactivation_prob = 0.005
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
        """Rebuild cortical index, excluding FROZEN and DEAD anchors."""
        from .anchor import ThermalState
        self.graph.cortical_index = [
            (a.embedding, a.id)
            for a in self.graph.anchors.values()
            if a.embedding and a.is_cortical
            and a.thermal_state not in (ThermalState.FROZEN, ThermalState.DEAD)
        ]
        # Sync ANN index — exclude FROZEN/DEAD
        ann = self.graph._get_ann_index() if self.graph._ann_index is not None else None
        if ann is not None:
            ann.clear()
            for a in self.graph.anchors.values():
                if a.embedding and a.thermal_state not in (ThermalState.FROZEN, ThermalState.DEAD):
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
                success_count=old.success_count,
                failure_count=old.failure_count,
                last_success_at=old.last_success_at,
                last_failure_at=old.last_failure_at,
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
