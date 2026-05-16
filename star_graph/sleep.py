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

from .sleep_nrem import SleepNREM
from .sleep_rem import SleepREM
from .sleep_consolidate import SleepConsolidate


class SleepCycle(SleepNREM, SleepREM, SleepConsolidate):
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
