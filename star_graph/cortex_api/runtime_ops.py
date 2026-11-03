"""RuntimeOps — mixin for persistence, cold tiering, stats, and graph operations.

Extracted verbatim from ``runtime.py`` ``MemoryRuntime`` (issue #9). Bodies are
unchanged; methods still use ``self.graph``/``self.cfg``/``self.storage_path``
and the lazy accessors from :class:`RuntimeSubsystems`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..community import Community
from ..graph import Edge, StarGraph
from ..hub import HubNode
from ..math_utils import cosine_sim as _cosine_sim
from .manager_stats import ManagerStats


class RuntimeOps:
    """Mixin: load, cold-tier offload/thaw, stats/health, graph and hub ops."""

    def load(self, path: str) -> StarGraph:
        """Load the memory system from disk.

        Auto-detects storage backend by file extension (.json → JSON, .db → SQLite).
        """
        from ..storage_backend import StorageBackend
        storage = StorageBackend.detect_and_create(path)
        self.graph = storage.load()
        self.storage_path = path

        # Try to load ghosts
        # NOTE: with_suffix() raises ValueError for non-dot suffixes
        # (e.g. "memory.db" -> "_ghosts.json"), so derive the name from stem.
        _p = Path(path)
        ghost_path = _p.with_name(_p.stem + "_ghosts.json")
        if ghost_path.exists():
            with open(ghost_path, "r", encoding="utf-8") as f:
                ghost_data = json.load(f)
            from ..ghost import GhostNode
            self._ghosts = GhostSubsystem()
            for gid, gd in ghost_data.items():
                ghost = GhostNode(
                    id=gd["id"],
                    compressed_embedding=gd["compressed_embedding"],
                    residual_edges=gd.get("residual_edges", {}),
                    emotion_trace=gd["emotion_trace"],
                    pruned_at=gd["pruned_at"],
                    original_tags=gd.get("original_tags", []),
                    original_importance=gd["original_importance"],
                    semantic_shadow=gd["semantic_shadow"],
                    reactivation_probability=gd["reactivation_probability"],
                    revival_count=gd.get("revival_count", 0),
                    partial_recall_count=gd.get("partial_recall_count", 0),
                )
                self._ghosts.ghosts[gid] = ghost

        # Reset subsystems that cache graph references (or stale derived state)
        self._scheduler = None
        self._evolution = None
        self._metrics = None
        self._community_detection = None
        self._timespine = None
        self._dual_channel = None
        self._cascade = None
        self._spreading = None
        self._bm25 = None
        self._exact_cache = None
        self._hublayer = None
        # _batch_vectorizer holds self.graph; _reflection_loop holds a stale
        # GhostSubsystem when the ghost file above replaced self._ghosts.
        self._batch_vectorizer = None
        self._reflection_loop = None

        # Re-set survival function (class vars don't survive process restart)
        Anchor.set_survival_function(self.survival_function)
        from ..ghost import GhostNode
        GhostNode.set_survival_function(self.survival_function)

        return self.graph

    # ── Tiered storage ─────────────────────────────────────────

    def offload_cold_anchors(self) -> int:
        """Offload COLD thermal-state anchors to disk, freeing RAM.

        Called during sleep N4_Prune. Anchors with thermal_state == COLD are
        serialized to TieredStorage and removed from graph.anchors.
        Returns count of offloaded anchors.
        """
        from ..anchor import ThermalState
        cold_ids = []
        for aid, a in self.graph.anchors.items():
            if getattr(a, '_thermal_state', None) == ThermalState.COLD:
                cold_ids.append(aid)

        if not cold_ids:
            return 0

        from ..tiered import offload_anchor_to_cold
        for aid in cold_ids:
            anchor = self.graph.anchors.get(aid)
            if anchor is None:
                continue
            offload_anchor_to_cold(anchor, self.tiered)
            self.graph.anchors.pop(aid, None)

        self.tiered.flush()
        return len(cold_ids)

    def thaw_anchor(self, anchor_id: str):
        """Reload a COLD anchor from disk into memory. Returns Anchor or None."""
        data = self.tiered.load(anchor_id)
        if data is None:
            return None

        from ..anchor import Anchor
        anchor = Anchor.create(
            text=data.get("text", ""),
            embedding=data.get("embedding"),
            tags=data.get("tags", []),
            source_session=data.get("source_session", ""),
            importance=data.get("importance", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
        )
        anchor.id = anchor_id
        anchor.created_at = data.get("created_at", anchor.created_at)
        anchor.last_activated_at = data.get("last_activated_at", anchor.last_activated_at)
        anchor.community_id = data.get("community_id", "")
        self.graph.anchors[anchor_id] = anchor
        return anchor

    # ── Health & reporting ────────────────────────────────────

    @property
    def stats(self) -> ManagerStats:
        ghost_count = len(self.ghosts.ghosts) if self._ghosts else len(self.graph._ghost_subsystem.ghosts)
        abstract_count = len(self._abstraction.abstracts) if self._abstraction else len(getattr(self.graph, 'abstracts', {}))
        cortex_count = len(self._router.cortices) if self._router else 0
        hub_count = len(self._hublayer.hubs) if self._hublayer else 0
        cluster_count = self._timespine.stats["total_clusters"] if self._timespine else 0
        cold_count = self._tiered.size if self._tiered else 0
        return ManagerStats(
            anchors=len(self.graph.anchors),
            edges=len(self.graph.edges),
            ghosts=ghost_count,
            schemas=len(self.graph.schemas),
            abstracts=abstract_count,
            working_memory=self.working_memory.size,
            cortices=cortex_count,
            hubs=hub_count,
            clusters=cluster_count,
            cold_anchors=cold_count,
            sleep_cycles=self.sleep_cycles,
            total_evolutions=self.total_evolutions,
            auto_micro_sleeps=self._auto_micro_count,
            auto_full_sleeps=self._auto_full_count,
            anchors_since_micro=self._anchors_since_micro,
            uptime_seconds=time.time() - self._started_at,
        )

    def health_report(self) -> str:
        """Full cognitive health report."""
        try:
            cog = self.metrics.snapshot()
        except Exception:
            cog = {}

        s = self.stats
        lines = [
            "=" * 55,
            "  Star Graph Memory — Cognitive Health Report",
            "=" * 55,
            f"  Anchors: {s.anchors}    Edges: {s.edges}    Ghosts: {s.ghosts}",
            f"  Cortices: {s.cortices}    Hubs: {s.hubs}    Clusters: {s.clusters}",
            f"  Schemas: {s.schemas}    Abstracts: {s.abstracts}    Working: {s.working_memory}",
            f"  Self-narratives: {len(self._autobiography._narratives) if self._autobiography else 0}",
            f"  Sleep cycles: {s.sleep_cycles}    Evolutions: {s.total_evolutions}",
            f"  Uptime: {s.uptime_seconds:.0f}s",
            f"",
        ]
        if cog:
            lines += [
                f"  Memory Stability:          {cog.get('memory_stability', 0):.2f}",
                f"  Recall Plasticity:         {cog.get('recall_plasticity', 0):.2f}",
                f"  Compression Ratio:         {cog.get('compression_ratio', 0):.2f}",
                f"  Semantic Drift Resistance: {cog.get('semantic_drift_resistance', 0):.2f}",
                f"  Abstraction Emergence:     {cog.get('abstraction_emergence_rate', 0):.1f}/cycle",
                f"  Ghost Reactivation:        {cog.get('ghost_reactivation_accuracy', 0):.2f}",
            ]
        lines.append("=" * 55)
        return "\n".join(lines)

    def print_health(self) -> None:
        import logging
        logging.getLogger("star_graph.runtime").info("\n" + self.health_report())

    # ── Advanced: graph traversal utilities ───────────────────

    def connect(self, source_id: str, target_id: str,
                weight: float = 0.5, edge_type: str = "topical") -> Edge | None:
        """Explicitly connect two memories. Edge budget enforced post-connection.

        When edge_type is "topical" (default), the causal edge classifier
        attempts to infer a richer causal type from the anchor texts (B-12).
        """
        if source_id not in self.graph.anchors or target_id not in self.graph.anchors:
            return None
        src_emb = self.graph.anchors[source_id].embedding
        tgt_emb = self.graph.anchors[target_id].embedding
        if src_emb and tgt_emb and weight == 0.5:
            weight = _cosine_sim(src_emb, tgt_emb)

        # Infer richer causal edge type from anchor texts
        if edge_type == "topical":
            anchor_a = self.graph.anchors[source_id]
            anchor_b = self.graph.anchors[target_id]
            inferred_type, confidence = self.causal_classifier.infer_from_anchors(anchor_a, anchor_b)
            if confidence > 0.3:
                edge_type = inferred_type

        edge = self.graph.add_edge(source_id, target_id, weight=weight, edge_type=edge_type)
        if edge:
            self.edge_budget.enforce(self.graph, source_id)
            self.edge_budget.enforce(self.graph, target_id)
        return edge

    def reinforce(self, source_id: str, target_id: str) -> bool:
        edge_key = self.graph._key(source_id, target_id)
        edge = self.graph.edges.get(edge_key)
        if edge is None:
            return False
        if hasattr(edge, 'reinforce'):
            edge.reinforce()
        else:
            edge.strengthen(0.05)
        return True

    def discover_abstracts(self) -> list:
        anchors_with_emb = {
            aid: a for aid, a in self.graph.anchors.items()
            if a.embedding
        }
        embeddings = {aid: a.embedding for aid, a in anchors_with_emb.items()}
        return self.abstraction.discover(anchors_with_emb, embeddings)

    def detect_communities(self) -> list[Community]:
        return self.community_detection.detect(self.graph)

    # ── Time spine ────────────────────────────────────────────

    def index_on_timeline(self, anchor_id: str, timestamp: float | None = None,
                          importance: float = 0.5,
                          embedding: list[float] | None = None,
                          topic: str = ""):
        self.timespine.index_anchor(anchor_id, timestamp, importance, embedding, topic)

    def scan_timeline(self, max_days: int = 30,
                      max_clusters: int = 20) -> list:
        clusters = self.timespine.scan_priority(max_days, max_clusters)
        return [
            {"id": c.id, "topic": c.topic, "importance": c.importance,
             "size": c.size, "summary": c.summary,
             "anchor_ids": c.anchor_ids}
            for c in clusters
        ]

    # ── Hub layer ─────────────────────────────────────────────

    def create_topic_hub(self, text: str, source_anchor_ids: list[str],
                         cortex_name: str) -> HubNode:
        return self.hublayer.create_leaf(text, source_anchor_ids, cortex_name)

    def create_domain_hub(self, text: str, child_hub_ids: list[str],
                          cortex_name: str) -> HubNode | None:
        return self.hublayer.create_domain(text, child_hub_ids, cortex_name)

    def bridge_cortices(self, hub_a_id: str, hub_b_id: str) -> bool:
        return self.hublayer.bridge(hub_a_id, hub_b_id)
