"""RuntimeLifecycle — mixin providing cognitive maintenance, sleep, persistence, and snapshot operations.

Extracted from runtime.py MemoryRuntime to reduce module size.
Inherited by MemoryRuntime alongside RuntimeCore.
"""

from __future__ import annotations

import json
import time

from .anchor import Anchor
from .shard import MemoryShardManager
from .cost_estimator import SleepCostEstimator, CostEstimate
from .snapshot import SnapshotManager, SnapshotMeta


class RuntimeLifecycle:
    """Mixin that provides cognitive maintenance (sleep, evolution) and persistence methods."""

    # ── Cognitive maintenance ─────────────────────────────────

    def _check_auto_sleep(self) -> dict:
        """Called after each remember() to check if auto-sleep thresholds are met.

        Returns a dict with keys 'micro' (MicroSleepResult or None) and 'full' (dict or None).
        Only triggers one sleep type per check (micro takes priority).
        """
        result: dict = {"micro": None, "full": None}
        now = time.time()
        sleep_cfg = getattr(self.cfg, 'sleep', None)

        micro_interval = float(getattr(sleep_cfg, 'auto_micro_interval_hours', 1.0) if sleep_cfg else 1.0) * 3600
        full_interval = float(getattr(sleep_cfg, 'auto_full_interval_hours', 24.0) if sleep_cfg else 24.0) * 3600
        anchor_threshold = int(getattr(sleep_cfg, 'auto_anchor_threshold', 100) if sleep_cfg else 100)

        # Micro-sleep: triggered by anchor count OR time interval
        needs_micro = (
            self._anchors_since_micro >= anchor_threshold
            or (self._last_micro_sleep > 0 and now - self._last_micro_sleep >= micro_interval)
        )
        if needs_micro:
            try:
                result["micro"] = self.micro_consolidate()
                self._auto_micro_count += 1
            except Exception:
                pass
            self._last_micro_sleep = now
            self._anchors_since_micro = 0

        # Full sleep: triggered by time interval since last full sleep
        needs_full = (now - self._last_full_sleep >= full_interval)
        if needs_full and not needs_micro:  # don't do full sleep same cycle as micro
            try:
                result["full"] = self.sleep(current_time=now)
                self._auto_full_count += 1
            except Exception:
                pass
            self._last_full_sleep = now
            self._last_micro_sleep = now
            self._anchors_since_micro = 0

        return result

    def micro_consolidate(self) -> dict:
        """Lightweight online consolidation — call after every few interactions."""
        if self._online_consolidator is None:
            from .online import OnlineConsolidator
            self._online_consolidator = OnlineConsolidator(self.graph)
        self._online_consolidator.record_interaction()

        # Auto-promote well-rehearsed working memory items
        promoted = 0
        for entry in list(self.working_memory._entries):
            if entry.access_count >= 3:
                self.working_memory.promote(entry, self)
                promoted += 1

        result = self._online_consolidator._micro_sleep()
        result["working_memory_promoted"] = promoted
        return result

    def sleep(self, current_time: float | None = None) -> dict:
        """Run a full sleep consolidation cycle.

        Each cortex consolidates independently (own graph + decay params).
        Cross-cortex operations (Hub bridging, global schema extraction)
        run afterwards as a lightweight global pass.
        """
        # 1. Per-cortex independent sleep consolidation
        cortex_reports = {}
        for cortex in self.router.cortices:
            try:
                cortex_reports[cortex.config.name] = cortex.consolidate()
            except Exception as e:
                cortex_reports[cortex.config.name] = {"error": str(e)}

        # 2. Cross-cortex operations (Hub bridging, global schema, edge pruning)
        from .sleep import SleepCycle
        sc = SleepCycle(self.graph)
        global_report = sc.run_phased(
            brain=self.brain,
            hublayer=self.hublayer,
            cortices=self.router.cortices,
        )

        # 3. Offload COLD anchors → disk (tiered storage)
        cold_offloaded = self.offload_cold_anchors()

        # 4. Evolve the graph
        evo = self.evolution.evolve(current_time)
        self.total_evolutions += 1

        # 5. Edge sparsification: evict expired edges
        edges_evicted = self.graph.evict_expired_edges()

        # 6. Hippocampus sleep decisions: promote/discard/keep L2 items
        hc_report = {"promoted": 0, "abstracted": 0, "discarded": 0, "kept": 0}
        if self._hippocampus is not None:
            hc_report = self.hippocampus.sleep_decide(self.graph, self._get_embedder())

        # 6b. Cognitive cache rebuild: topic index + evict expired entries
        cache_rebuild = self.cognitive_cache.rebuild_on_sleep(self.graph)

        # 6c. Cognitive compiler: run worldview emergence pipeline
        compiler_result = {"worldviews": 0, "profile_version": 0}
        try:
            compiler_result_raw = self.compiler.compile(self.graph)
            compiler_result = {
                "worldviews": len(compiler_result_raw.get("worldviews", [])),
                "profile_version": compiler_result_raw.get("stats", {}).get("profile_version", 0),
                "compression_chain": compiler_result_raw.get("stats", {}).get("compression_chain", ""),
            }
        except Exception:
            pass

        # 6d. Self-reflection loop: detect and correct contradictions
        reflection_reports = []
        try:
            if self._reflection_loop is None:
                self._reflection_loop = self.reflection_loop
            new_reports = self._reflection_loop.run(
                self.graph,
                autobiography=self._autobiography,
            )
            reflection_reports = [r.to_dict() for r in new_reports]
        except Exception:
            pass

        # 6e. Edge budget: enforce across all nodes (#49)
        eb_result = {"over_budget_nodes": 0, "total_evicted": 0}
        try:
            eb_result = self.edge_budget.enforce_all(self.graph)
        except Exception:
            pass

        # 6f. Four-layer compression: message→event→semantic→personality (#51)
        fl_result = {"layer0_compressed": 0, "layer1_compressed": 0,
                     "layer2_compressed": 0, "decayed": 0}
        try:
            fl_result["layer0_compressed"] = self.four_layer.compress_layer0()
            fl_result["layer1_compressed"] = self.four_layer.compress_layer1()
            fl_result["layer2_compressed"] = self.four_layer.compress_layer2()
            fl_result["decayed"] = self.four_layer.decay_all()
        except Exception:
            pass

        # 6g. Thermal store demotion scan — hot→cold→archive (#53)
        ts_result = {"hot_to_cold": 0, "cold_to_archive": 0}
        try:
            ts_result = self.thermal_store.demote_scan(self.graph, now=current_time)
            self.thermal_store.flush()
        except Exception:
            pass

        # 6h. Edge time decay — continuous decay across all edges (#54)
        ed_result = {"decayed": 0, "evicted": 0}
        try:
            ed_result = self.edge_decay_mgr.decay_all_edges(self.graph, now=current_time)
        except Exception:
            pass

        # 6i. Self-organization — auto-cluster, merge duplicates, detect topics (#55)
        so_result = {"topics_detected": 0, "merges": 0, "communities_assigned": 0}
        try:
            so_result = self.self_org.organize(self.graph, current_time=current_time)
        except Exception:
            pass

        # 6j. Personality model — full extraction from graph (#56)
        personality_version = 0
        try:
            profile = self.personality.extract_from_graph(self.graph)
            personality_version = profile.version
        except Exception:
            pass

        # 6k. Goal tree — detect new goals, propagate progress, archive stale (#57)
        gt_result = {"new_goals": 0, "propagated": 0, "archived": 0}
        try:
            new_goals = self.goal_tree.detect_from_graph(self.graph)
            gt_result["new_goals"] = len(new_goals)
            gt_result["propagated"] = self.goal_tree.propagate_progress()
            gt_result["archived"] = self.goal_tree.archive_stale(hours=168.0)
        except Exception:
            pass

        # 6l. Cluster memory — rebuild cluster index from communities (A-10)
        cm_result = {"clusters_built": 0}
        try:
            cm_result["clusters_built"] = self.cluster_router.build_index(self.graph)
        except Exception:
            pass

        # 6m. Versioned memory — detect belief evolution chains (A-9)
        vm_result = {"new_beliefs": 0}
        try:
            new_beliefs = self.versioned_memory.detect_from_graph(self.graph)
            vm_result["new_beliefs"] = len(new_beliefs)
        except Exception:
            pass

        # 6n. Episodic memory — summarize completed sessions (B-13)
        em_result = {"sessions_summarized": 0}
        try:
            for session_id in list(self.episodic_memory._session_index.keys()):
                if session_id not in self.episodic_memory._summaries:
                    summary = self.episodic_memory.summarize_session(session_id)
                    if summary:
                        em_result["sessions_summarized"] += 1
        except Exception:
            pass

        # 7. Decay ghosts and clean up cold storage for purged ones
        ghost_purged, purged_ids = self.ghosts.decay_all()
        if purged_ids and self._tiered is not None:
            for gid in purged_ids:
                self._tiered.remove(gid)
            self._tiered.compact()

        # 7. Decay autobiographical narratives
        self_narratives_purged = 0
        if self._autobiography:
            self_narratives_purged = self._autobiography.degrade_all()

        self.sleep_cycles += 1
        return {
            "cortex_reports": cortex_reports,
            "global_report": global_report,
            "cold_offloaded": cold_offloaded,
            "cold_store_size": self.tiered.size,
            "evolution": evo,
            "ghost_stats": self.ghosts.stats,
            "ghosts_purged": ghost_purged,
            "self_narratives_purged": self_narratives_purged,
            "hippocampus": hc_report,
            "cognitive_cache": cache_rebuild,
            "compiler": compiler_result,
            "reflection": {
                "corrections": len(reflection_reports),
                "reports": reflection_reports[:5],
            },
            "reflection_stats": self.reflection_loop.stats,
            "edge_budget": eb_result,
            "four_layer": fl_result,
            "thermal_store": ts_result,
            "edge_decay": ed_result,
            "self_org": so_result,
            "personality_version": personality_version,
            "goal_tree": gt_result,
            "cluster_memory": cm_result,
            "versioned_memory": vm_result,
            "episodic_memory": em_result,
        }

    def micro_sleep(self, steps: int = 2) -> dict:
        """Incremental non-blocking sleep — run 1-2 phases at a time."""
        from .micro_sleep import MicroSleepScheduler

        if self._micro_sleep is None:
            self._micro_sleep = MicroSleepScheduler(
                graph=self.graph, config=self.cfg,
                brain=self.brain, hublayer=self.hublayer,
                cortices=self.router.cortices,
            )

        result = self._micro_sleep.run_next(steps=steps)

        if result.is_complete:
            self._micro_sleep = None
            self.sleep_cycles += 1

        return {
            "phases_run": result.phases_run,
            "phases_processed": result.phases_processed,
            "is_complete": result.is_complete,
            "progress_pct": result.progress.progress_pct if result.progress else 1.0,
            "errors": result.errors,
            "items_processed": result.items_processed,
            "duration_ms": result.duration_ms,
            "summary": self._micro_sleep.get_summary() if self._micro_sleep else "Complete",
        }

    def estimate_sleep_cost(self, dry_run: bool = False) -> CostEstimate:
        estimator = SleepCostEstimator()
        return estimator.estimate(self, dry_run=dry_run)

    def evolve(self, current_time: float | None = None) -> dict:
        result = self.evolution.evolve(current_time)
        self.total_evolutions += 1
        return result

    # ── Persistence ───────────────────────────────────────────

    @property
    def snapshots(self) -> SnapshotManager:
        if self._snapshot_mgr is None:
            self._snapshot_mgr = SnapshotManager()
        return self._snapshot_mgr

    def snapshot(self, description: str = "", force: bool = False) -> SnapshotMeta:
        meta = self.snapshots.snapshot(self.graph, description=description, force=force)
        return meta

    def recover(self) -> tuple:
        graph, log = self.snapshots.recover()
        self.graph = graph
        return graph, log

    def save(self, path: str | None = None) -> str:
        """Persist the entire memory system to disk."""
        from .storage import JSONStorage
        filepath = path or self.storage_path or str(JSONStorage.DEFAULT_PATH)
        storage = JSONStorage(filepath)
        storage.save(self.graph)
        if self._ghosts:
            ghost_path = filepath.replace(".json", "_ghosts.json")
            with open(ghost_path, "w", encoding="utf-8") as f:
                ghost_data = {}
                for gid, g in self._ghosts.ghosts.items():
                    ghost_data[gid] = {
                        "id": g.id,
                        "compressed_embedding": g.compressed_embedding,
                        "residual_edges": g.residual_edges,
                        "emotion_trace": g.emotion_trace,
                        "pruned_at": g.pruned_at,
                        "original_tags": g.original_tags,
                        "original_importance": g.original_importance,
                        "semantic_shadow": g.semantic_shadow,
                        "reactivation_probability": g.reactivation_probability,
                        "revival_count": g.revival_count,
                        "partial_recall_count": g.partial_recall_count,
                    }
                json.dump(ghost_data, f, indent=2, ensure_ascii=False)
        self.storage_path = filepath
        return filepath

    def save_sharded(self, base_dir: str = "memory") -> dict:
        """Save anchors partitioned by domain + time into shard files.

        Returns stats dict with shard_count, total_anchors, total_size_mb.
        """
        sm = MemoryShardManager(base_dir=base_dir, max_file_size_mb=50)
        # Group anchors by domain (inferred from tags or cortex membership)
        batches: dict[str, list[dict]] = {}
        for aid, anchor in self.graph.anchors.items():
            domain = self._infer_domain(anchor)
            file_path = sm.route_anchor(anchor, domain=domain)
            if file_path not in batches:
                batches[file_path] = []
            batches[file_path].append({
                "id": aid,
                "text": anchor.text,
                "embedding": anchor.embedding,
                "tags": anchor.tags,
                "importance": anchor.vector.importance,
                "emotional_valence": anchor.vector.emotional_valence,
                "source_session": anchor.source_session,
                "created_at": anchor.created_at,
                "last_activated_at": anchor.last_activated_at,
                "domain": domain,
            })

        for file_path, data in batches.items():
            sm.save_shard(file_path, data)

        return sm.stats

    def load_sharded(self, base_dir: str = "memory") -> int:
        """Load all anchors from shard files. Returns count loaded."""
        sm = MemoryShardManager(base_dir=base_dir)
        all_data = sm.load_all()
        for data in all_data:
            if data.get("id") and data.get("text"):
                anchor = Anchor.create(
                    text=data["text"],
                    source_session=data.get("source_session", ""),
                    embedding=data.get("embedding"),
                    emotional_valence=data.get("emotional_valence", 0.0),
                    importance=data.get("importance", 0.5),
                    tags=data.get("tags", []),
                )
                anchor.id = data["id"]
                anchor.created_at = data.get("created_at", time.time())
                anchor.last_activated_at = data.get("last_activated_at", time.time())
                self.graph.add_anchor(anchor)
        return len(all_data)
