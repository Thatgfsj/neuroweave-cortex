"""MemoryRuntime — dependency container and lifecycle manager.

Owns all subsystems (graph, embedder, cortices, ghosts, evolution, etc.)
and provides CRUD, persistence, and cognitive maintenance methods.

MemoryManager delegates to this for all non-retrieval operations.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from .anchor import Anchor, MemoryState
from .graph import StarGraph, Edge
from .config import Config
from .math_utils import cosine_sim as _cosine_sim
from .working_memory import WorkingMemory, WorkingMemoryEntry
from .cortex import MemoryCortex, CortexConfig
from .router import CortexRouter
from .gate import MemoryGate
from .timespine import TimeSpine
from .tiered import TieredStorage
from .hippocampus import HippocampusBuffer
from .shard import MemoryShardManager
from .cascade import CascadeRecall
from .spreading import SpreadingActivation
from .cognitive_cache import CognitiveCacheManager
from .compiler import CognitiveCompiler
from .reflection_loop import SelfReflectionLoop
from .hub import HubLayer, HubNode
from .brain_sphere import BrainSphere, HubCenter
from .symbolic_filter import SymbolicFilter
from .evolution import MemoryEvolutionEngine
from .ghost import GhostSubsystem
from .abstraction import AbstractionEngine
from .community import Community, CommunityDetection
from .metrics import CognitiveMetrics
from .raw_buffer import RawBuffer
from .dual_channel import DualChannelRetriever
from .exact_cache import ExactMatchCache
from .cost_estimator import SleepCostEstimator, CostEstimate
from .snapshot import SnapshotManager, SnapshotMeta
from .tracing import MemoryTracer, get_tracer
from .survival import SurvivalFunction, SurvivalRegistry
from .domain_router import DomainRouter
from .write_gate import MemoryWriteGate, GateDecision
from .edge_budget import EdgeBudgetManager
from .four_layer import FourLayerCompressor
from .thermal_store import ThermalStore
from .edge_decay import EdgeDecayManager
from .multimodal import (
    MultimodalEmbeddingProvider, MultimodalAnchor, CrossModalRetriever, CrossModalResult,
)


@dataclass
class ManagerStats:
    """Snapshot of the memory system state."""
    anchors: int = 0
    edges: int = 0
    ghosts: int = 0
    schemas: int = 0
    abstracts: int = 0
    working_memory: int = 0
    cortices: int = 0
    hubs: int = 0
    clusters: int = 0
    cold_anchors: int = 0
    sleep_cycles: int = 0
    total_evolutions: int = 0
    auto_micro_sleeps: int = 0
    auto_full_sleeps: int = 0
    anchors_since_micro: int = 0
    uptime_seconds: float = 0.0
    cognitive_health: dict | None = None


class MemoryRuntime:
    """Dependency container and lifecycle manager for the cognitive memory system.

    Owns all subsystems, CRUD operations, persistence, and cognitive maintenance.
    Retrieval methods live in RetrievalPipeline, which takes a reference to this runtime.
    """

    def __init__(self, graph: StarGraph | None = None,
                 config: Config | None = None,
                 storage_path: str = ""):
        self.graph = graph or StarGraph()
        self.cfg = config or Config.get()
        self.storage_path = storage_path
        self._started_at = time.time()

        # Edge budget — prevent graph density explosion
        graph_cfg = getattr(self.cfg, 'graph', None)
        edge_budget_enabled = getattr(graph_cfg, 'edge_budget_enabled', True) if graph_cfg else True
        self.graph._max_edges_per_node = getattr(graph_cfg, 'max_edges_per_node', 8) if (graph_cfg and edge_budget_enabled) else 0

        # Working memory — short-term buffer (not persisted)
        wm_cfg = getattr(self.cfg, 'working_memory', None)
        wm_capacity = getattr(wm_cfg, 'max_capacity', 9) if wm_cfg else 9
        wm_ttl = getattr(wm_cfg, 'ttl_seconds', 1800.0) if wm_cfg else 1800.0
        self.working_memory = WorkingMemory(
            max_capacity=wm_capacity,
            ttl_seconds=wm_ttl,
        )

        # Brain sphere — innermost L1 cache (non-lazy, always available)
        self.brain = BrainSphere(max_common_nodes=5000)

        # Symbolic filter — pre-filters by topic/keyword before embedding search
        self.sym_filter = SymbolicFilter()

        # Subsystems — lazily initialized
        self._embedder = None
        self._scheduler: CognitiveMemoryScheduler | None = None
        self._evolution: MemoryEvolutionEngine | None = None
        self._ghosts: GhostSubsystem | None = None
        self._abstraction: AbstractionEngine | None = None
        self._community_detection: CommunityDetection | None = None
        self._metrics: CognitiveMetrics | None = None
        self._online_consolidator = None
        self._router: CortexRouter | None = None
        self._gate: MemoryGate | None = None
        self._timespine: TimeSpine | None = None
        self._cascade: CascadeRecall | None = None
        self._spreading: SpreadingActivation | None = None
        self._cognitive_cache: CognitiveCacheManager | None = None
        self._compiler: CognitiveCompiler | None = None
        self._reflection_loop: SelfReflectionLoop | None = None
        self._hublayer: HubLayer | None = None
        self._bm25 = None  # BM25 keyword index — lazy init
        self._tiered: TieredStorage | None = None
        self._hippocampus = None  # HippocampusBuffer — lazy init
        self._shard_manager = None  # MemoryShardManager — lazy init

        # Raw chunk buffer — uncompressed short-term memory tier (L0)
        self._raw_buffer: RawBuffer | None = None

        # Dual-channel retriever (System-1 + System-2)
        self._dual_channel: DualChannelRetriever | None = None

        # Exact match cache — deterministic O(1) entity-pair lookup bypass
        self._exact_cache: ExactMatchCache | None = None

        # Micro-sleep scheduler — incremental non-blocking consolidation
        self._micro_sleep = None

        # Domain router — hierarchical topic pre-filter (#48)
        self._domain_router: DomainRouter | None = None

        # Write gate — pre-write quality filter (#50)
        self._write_gate: MemoryWriteGate | None = None

        # Edge budget — smart edge cap enforcement (#49)
        self._edge_budget: EdgeBudgetManager | None = None

        # Four-layer compressor — message→event→semantic→personality (#51)
        self._four_layer: FourLayerCompressor | None = None

        # Thermal store — 3-tier hot/cold/archive auto management (#53)
        self._thermal_store: ThermalStore | None = None

        # Edge decay manager — continuous time-based edge decay (#54)
        self._edge_decay_mgr: EdgeDecayManager | None = None

        # Snapshot manager — versioned state snapshots with WAL
        self._snapshot_mgr: SnapshotManager | None = None

        # Tracer — lightweight observability spans
        self.tracer: MemoryTracer = get_tracer(enabled=True)

        # Survival function — configurable memory decay curve
        self._survival_fn: SurvivalFunction | None = None
        Anchor.set_survival_function(self.survival_function)
        from .ghost import GhostNode
        GhostNode.set_survival_function(self.survival_function)

        # Multimodal — optional image/text cross-modal embedding
        self._multimodal_provider: MultimodalEmbeddingProvider | None = None
        self._cross_modal_retriever: CrossModalRetriever | None = None

        # Autobiographical memory — agent's subjective self-model
        self._autobiography = None

        # Streaming — continuous memory ingestion buffer
        self._streaming_buffer = None

        # Stats
        self.sleep_cycles: int = 0
        self.total_evolutions: int = 0
        self._last_micro_sleep: float = time.time()
        self._last_full_sleep: float = time.time()
        self._anchors_since_micro: int = 0
        self._auto_micro_count: int = 0
        self._auto_full_count: int = 0

    # ── Subsystem access (lazy init) ──────────────────────────

    @property
    def scheduler(self):
        if self._scheduler is None:
            from .scheduler import CognitiveMemoryScheduler
            self._scheduler = CognitiveMemoryScheduler(
                self.graph, self.cfg,
                working_memory=self.working_memory,
                sym_filter=self.sym_filter)
        return self._scheduler

    @property
    def evolution(self) -> MemoryEvolutionEngine:
        if self._evolution is None:
            self._evolution = MemoryEvolutionEngine(self.graph, self.cfg)
        return self._evolution

    @property
    def ghosts(self) -> GhostSubsystem:
        if self._ghosts is None:
            self._ghosts = GhostSubsystem()
        return self._ghosts

    @property
    def abstraction(self) -> AbstractionEngine:
        if self._abstraction is None:
            self._abstraction = AbstractionEngine()
        return self._abstraction

    @property
    def community_detection(self) -> CommunityDetection:
        if self._community_detection is None:
            self._community_detection = CommunityDetection()
        return self._community_detection

    @property
    def raw_buffer(self) -> RawBuffer:
        if self._raw_buffer is None:
            rb_cfg = getattr(self.cfg, 'raw_buffer', None)
            max_sessions = getattr(rb_cfg, 'max_sessions', 2) if rb_cfg else 2
            max_chunks = getattr(rb_cfg, 'max_chunks_per_session', 500) if rb_cfg else 500
            self._raw_buffer = RawBuffer(
                max_sessions=max_sessions,
                max_chunks_per_session=max_chunks,
            )
        return self._raw_buffer

    @property
    def dual_channel(self) -> DualChannelRetriever:
        if self._dual_channel is None:
            dc_cfg = getattr(self.cfg, 'dual_channel', None)
            s1_threshold = getattr(dc_cfg, 's1_confidence_threshold', 0.35) if dc_cfg else 0.35
            self._dual_channel = DualChannelRetriever(
                self.graph,
                s1_confidence_threshold=s1_threshold,
                bm25_index=self._bm25,  # shared BM25 index for sparse+dense fusion
            )
        return self._dual_channel

    @property
    def compressor(self):
        """Lazy-init multi-level compression engine."""
        if getattr(self, '_compress_engine', None) is None:
            from .compression import MultiLevelCompressor
            self._compress_engine = MultiLevelCompressor()
        return self._compress_engine

    @property
    def exact_cache(self) -> ExactMatchCache:
        if self._exact_cache is None:
            ec_cfg = getattr(self.cfg, 'exact_cache', None)
            max_per_key = getattr(ec_cfg, 'max_entries_per_key', 5) if ec_cfg else 5
            self._exact_cache = ExactMatchCache(max_entries_per_key=max_per_key)
        return self._exact_cache

    @property
    def survival_function(self) -> SurvivalFunction:
        if self._survival_fn is None:
            self._survival_fn = SurvivalRegistry.from_config(self.cfg)
        return self._survival_fn

    @property
    def autobiography(self):
        """Lazy-init autobiographical memory — agent's subjective self-model."""
        if self._autobiography is None:
            from .autobiography import AutobiographicalMemory
            self._autobiography = AutobiographicalMemory()
        return self._autobiography

    @property
    def streaming_buffer(self):
        """Lazy-init streaming memory buffer for continuous ingestion."""
        if self._streaming_buffer is None:
            from .streaming import StreamingMemoryBuffer
            stream_cfg = getattr(self.cfg, 'streaming', None)
            max_buf = getattr(stream_cfg, 'max_buffer', 500) if stream_cfg else 500
            flush_sec = getattr(stream_cfg, 'flush_interval_s', 30.0) if stream_cfg else 30.0
            batch_sz = getattr(stream_cfg, 'batch_size', 20) if stream_cfg else 20
            dedup_th = getattr(stream_cfg, 'dedup_threshold', 0.85) if stream_cfg else 0.85
            max_ses = getattr(stream_cfg, 'max_sessions', 10) if stream_cfg else 10
            self._streaming_buffer = StreamingMemoryBuffer(
                self,
                max_buffer=max_buf,
                flush_interval_s=flush_sec,
                batch_size=batch_sz,
                dedup_threshold=dedup_th,
                max_sessions=max_ses,
            )
        return self._streaming_buffer

    @property
    def multimodal(self) -> MultimodalEmbeddingProvider:
        if self._multimodal_provider is None:
            self._multimodal_provider = MultimodalEmbeddingProvider()
        return self._multimodal_provider

    @property
    def cross_modal_retriever(self) -> CrossModalRetriever:
        if self._cross_modal_retriever is None:
            self._cross_modal_retriever = CrossModalRetriever(self.multimodal)
        return self._cross_modal_retriever

    @property
    def metrics(self) -> CognitiveMetrics:
        if self._metrics is None:
            self._metrics = CognitiveMetrics(self.graph)
        return self._metrics

    @property
    def router(self) -> CortexRouter:
        if self._router is None:
            self._router = CortexRouter(self.cfg, brain=self.brain)
        return self._router

    @property
    def gate(self) -> MemoryGate:
        if self._gate is None:
            self._gate = MemoryGate(k=self.cfg.gate.k if hasattr(self.cfg, 'gate') else 20, config=self.cfg)
        return self._gate

    @property
    def timespine(self) -> TimeSpine:
        if self._timespine is None:
            self._timespine = TimeSpine(
                max_clusters_per_day=getattr(
                    getattr(self.cfg, 'timespine', None), 'max_clusters_per_day', 10))
        return self._timespine

    @property
    def bm25(self):
        if self._bm25 is None:
            from .bm25 import BM25Index
            self._bm25 = BM25Index()
            # Populate from existing anchors
            for aid, a in self.graph.anchors.items():
                self._bm25.add(aid, a.text)
        return self._bm25

    @property
    def tiered(self) -> TieredStorage:
        if self._tiered is None:
            cold_path = ""
            if self.storage_path:
                cold_path = os.path.join(os.path.dirname(self.storage_path) or ".", "memory_cold.json")
            self._tiered = TieredStorage(path=cold_path)
        return self._tiered

    @property
    def hippocampus(self) -> HippocampusBuffer:
        if self._hippocampus is None:
            hc_cfg = getattr(self.cfg, 'hippocampus', None)
            self._hippocampus = HippocampusBuffer(
                l1_max_items=int(getattr(hc_cfg, 'l1_max_items', 50) if hc_cfg else 50),
                l1_ttl_minutes=float(getattr(hc_cfg, 'l1_ttl_minutes', 30.0) if hc_cfg else 30.0),
                l2_max_items=int(getattr(hc_cfg, 'l2_max_items', 200) if hc_cfg else 200),
                l2_ttl_hours=float(getattr(hc_cfg, 'l2_ttl_hours', 24.0) if hc_cfg else 24.0),
                promote_threshold=int(getattr(hc_cfg, 'promote_threshold', 3) if hc_cfg else 3),
            )
        return self._hippocampus

    @property
    def shard_manager(self) -> MemoryShardManager:
        if self._shard_manager is None:
            base = os.path.dirname(self.storage_path) if self.storage_path else "memory"
            self._shard_manager = MemoryShardManager(base_dir=base, max_file_size_mb=50)
        return self._shard_manager

    @property
    def cascade(self) -> CascadeRecall:
        if self._cascade is None:
            self._cascade = CascadeRecall(self.graph)
        return self._cascade

    @property
    def spreading(self) -> SpreadingActivation:
        if self._spreading is None:
            self._spreading = SpreadingActivation(self.graph, self.cfg)
        return self._spreading

    @property
    def cognitive_cache(self) -> CognitiveCacheManager:
        if self._cognitive_cache is None:
            self._cognitive_cache = CognitiveCacheManager()
        return self._cognitive_cache

    @property
    def compiler(self) -> CognitiveCompiler:
        if self._compiler is None:
            self._compiler = CognitiveCompiler()
        return self._compiler

    @property
    def reflection_loop(self) -> SelfReflectionLoop:
        if self._reflection_loop is None:
            self._reflection_loop = SelfReflectionLoop()
            self._reflection_loop.set_ghost_subsystem(self.ghosts)
        return self._reflection_loop

    @property
    def domain_router(self) -> DomainRouter:
        """Lazy-init hierarchical domain router for retrieval pre-filtering."""
        if self._domain_router is None:
            self._domain_router = DomainRouter()
        return self._domain_router

    @property
    def write_gate(self) -> MemoryWriteGate:
        """Lazy-init pre-write quality filter."""
        if self._write_gate is None:
            self._write_gate = MemoryWriteGate()
        return self._write_gate

    @property
    def edge_budget(self) -> EdgeBudgetManager:
        """Lazy-init edge budget manager for super-node prevention."""
        if self._edge_budget is None:
            eb_cfg = getattr(self.cfg, 'edge_budget', None)
            max_edges = getattr(eb_cfg, 'max_edges', 32) if eb_cfg else 32
            self._edge_budget = EdgeBudgetManager(max_edges=max_edges)
        return self._edge_budget

    @property
    def four_layer(self) -> FourLayerCompressor:
        """Lazy-init four-layer memory compression engine."""
        if self._four_layer is None:
            self._four_layer = FourLayerCompressor()
        return self._four_layer

    @property
    def thermal_store(self) -> ThermalStore:
        """Lazy-init 3-tier hot/cold/archive storage manager."""
        if self._thermal_store is None:
            ts_cfg = getattr(self.cfg, 'thermal_store', None)
            storage_dir = os.path.dirname(self.storage_path) if self.storage_path else "memory"
            self._thermal_store = ThermalStore(storage_dir=storage_dir)
            if ts_cfg:
                self._thermal_store.hot_to_cold_hours = getattr(ts_cfg, 'hot_to_cold_hours', 72.0)
                self._thermal_store.cold_to_archive_hours = getattr(ts_cfg, 'cold_to_archive_hours', 720.0)
        return self._thermal_store

    @property
    def edge_decay_mgr(self) -> EdgeDecayManager:
        """Lazy-init continuous edge time decay manager."""
        if self._edge_decay_mgr is None:
            ed_cfg = getattr(self.cfg, 'edge_decay', None)
            multiplier = getattr(ed_cfg, 'base_multiplier', 1.0) if ed_cfg else 1.0
            min_weight = getattr(ed_cfg, 'min_edge_weight', 0.02) if ed_cfg else 0.02
            self._edge_decay_mgr = EdgeDecayManager(
                base_decay_multiplier=multiplier,
                min_edge_weight=min_weight,
            )
        return self._edge_decay_mgr

    @property
    def hublayer(self) -> HubLayer:
        if self._hublayer is None:
            hub_cfg = getattr(self.cfg, 'hub', None)
            max_deg = getattr(hub_cfg, 'max_degree_per_shard', 50) if hub_cfg else 50
            self._hublayer = HubLayer(max_degree_per_shard=max_deg)
        return self._hublayer

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
            # Create instance-level registry to avoid multi-Manager singleton pollution
            from .anchor import EmbedderRegistry
            self._embedder_registry = EmbedderRegistry(self._embedder)
        return self._embedder

    # ── CRUD: Basic memory operations ─────────────────────────

    def remember(self, text: str, *, source_session: str = "",
                 tags: list[str] | None = None,
                 emotional_valence: float = 0.0,
                 importance: float = 0.5,
                 connect_to: list[str] | None = None,
                 edge_type: str = "topical",
                 anchor_id: str | None = None,
                 skip_gate: bool = False,
                 **vec_kw) -> Anchor | None:
        """Store a new memory. Returns the anchor, or None if rejected by write gate."""
        embedder = self._get_embedder()
        embedding = embedder.encode(text)

        # ── Stage 0: Write Gate pre-filter (#50) ──
        wg_enabled = False
        wg_cfg = getattr(self.cfg, 'write_gate', None)
        if wg_cfg:
            wg_enabled = getattr(wg_cfg, 'enabled', True)
        if wg_enabled and not skip_gate:
            gate_result = self.write_gate.evaluate(
                text=text,
                embedding=embedding,
                tags=tags,
                importance=importance,
                emotional_valence=emotional_valence,
                graph=self.graph,
            )
            if gate_result.decision == GateDecision.REJECT:
                return None
            if gate_result.decision == GateDecision.MERGE and gate_result.merge_target_id:
                self.update(gate_result.merge_target_id, text=text, tags=tags)
                return self.graph.anchors.get(gate_result.merge_target_id)
            if gate_result.decision == GateDecision.DEFER:
                # Only store in hippocampus; skip graph insertion
                self.hippocampus.ingest(
                    text=text, tags=tags or [], importance=importance,
                    emotional_valence=emotional_valence,
                    source_session=source_session, embedding=embedding,
                )
                return None

        if anchor_id:
            anchor = Anchor(
                id=anchor_id, text=text, embedding=embedding,
                source_session=source_session,
                tags=tags or [],
                vector=Anchor.__dataclass_fields__["vector"].default_factory(),
                **vec_kw,
            )
        else:
            anchor = Anchor.create(
                text=text, source_session=source_session,
                embedding=embedding,
                emotional_valence=emotional_valence,
                importance=importance,
                tags=tags,
                **vec_kw,
            )

        # Hard cap: evict before insert if at capacity
        max_total = int(getattr(self.cfg.graph, 'max_total_anchors', 0) if self.cfg else 0)
        if max_total > 0 and len(self.graph.anchors) >= max_total:
            policy = getattr(self.cfg.graph, 'eviction_policy', 'lowest_retention') if self.cfg else 'lowest_retention'
            evicted = self.graph._evict_anchors(count=1, policy=policy)
            if evicted and self._bm25 is not None:
                for eid in evicted:
                    self._bm25.remove(eid)

        # Cortex capacity: auto-consolidate overfull cortices before adding
        for cortex in self.router.cortices:
            if cortex.is_overfull():
                cortex.ensure_capacity()

        # Hippocampus buffer: ingest into L1 (transient cache)
        hc_enabled = True
        hc_cfg = getattr(self.cfg, 'hippocampus', None)
        if hc_cfg:
            hc_enabled = getattr(hc_cfg, 'enabled', True)
        if hc_enabled:
            self.hippocampus.ingest(
                text=text,
                tags=tags or [],
                importance=importance,
                emotional_valence=emotional_valence,
                source_session=source_session,
                embedding=embedding,
            )

        self.graph.add_anchor(anchor)
        self._anchors_since_micro += 1

        # Index on the temporal spine for time-windowed retrieval (Layer 3)
        self.timespine.index_anchor(
            anchor.id,
            timestamp=anchor.created_at,
            importance=importance,
            embedding=embedding,
            topic=(tags[0] if tags else ""),
        )

        # Index in BM25 keyword index for sparse retrieval channel
        if self._bm25 is not None:
            self._bm25.add(anchor.id, text)

        # Index in domain router for hierarchical retrieval pre-filtering (#48)
        dr_enabled = True
        dr_cfg = getattr(self.cfg, 'domain_router', None)
        if dr_cfg:
            dr_enabled = getattr(dr_cfg, 'enabled', True)
        if dr_enabled:
            self.domain_router.index_anchor(anchor.id, text=text, tags=tags)

        # Ingest into four-layer compression pipeline (#51)
        fl_enabled = True
        fl_cfg = getattr(self.cfg, 'four_layer', None)
        if fl_cfg:
            fl_enabled = getattr(fl_cfg, 'enabled', True)
        if fl_enabled:
            self.four_layer.ingest_message(
                text=text,
                embedding=embedding,
                importance=importance,
                tags=tags or [],
            )

        # Auto-sleep check: trigger micro/full consolidation on thresholds
        self._check_auto_sleep()

        # Dual-write: also store raw uncompressed chunk in L0 buffer
        self.raw_buffer.add(
            text=text, session_id=source_session,
            embedding=embedding, tags=tags,
            importance=importance, anchor_id=anchor.id,
        )

        # Harvest exact match keys into deterministic KV cache
        auto_harvest = True
        ec_cfg = getattr(self.cfg, 'exact_cache', None)
        if ec_cfg:
            auto_harvest = getattr(ec_cfg, 'auto_harvest', True)
        if auto_harvest:
            self.exact_cache.harvest_from_anchor(anchor)

        # Auto-connect to specified anchors
        if connect_to:
            for target_id in connect_to:
                if target_id in self.graph.anchors:
                    target_emb = self.graph.anchors[target_id].embedding
                    sim = 0.5
                    if target_emb and embedding:
                        sim = _cosine_sim(embedding, target_emb)
                    edge = self.graph.add_edge(
                        anchor.id, target_id,
                        weight=sim, edge_type=edge_type,
                    )
                    # Enforce edge budget after each edge addition (#49)
                    if edge:
                        self.edge_budget.enforce(self.graph, anchor.id)
                        self.edge_budget.enforce(self.graph, target_id)

        return anchor

    def forget(self, anchor_id: str, create_ghost: bool = True) -> Anchor | None:
        """Remove a memory. If create_ghost=True, preserves a latent trace."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return None

        if create_ghost:
            residual = {}
            for neighbor_id in self.graph._adjacency.get(anchor_id, set()):
                edge_key = self.graph._key(anchor_id, neighbor_id)
                edge = self.graph.edges.get(edge_key)
                if edge:
                    residual[neighbor_id] = edge.weight * 0.3
            self.ghosts.create(anchor, residual)

        self.graph.remove_anchor(anchor_id)
        self.timespine.remove_anchor(anchor_id)
        if self._bm25 is not None:
            self._bm25.remove(anchor_id)
        return anchor

    # ── Working Memory (short-term buffer) ───────────────────

    def remember_working(self, text: str, *,
                         importance: float = 0.5,
                         tags: list[str] | None = None,
                         source_session: str = "",
                         emotional_valence: float = 0.0) -> WorkingMemoryEntry:
        """Add an item to working memory — fast, ephemeral, high-priority."""
        embedder = self._get_embedder()
        embedding = embedder.encode(text)
        return self.working_memory.add(
            text=text,
            embedding=embedding,
            importance=importance,
            tags=tags,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )

    def get_working(self) -> list[WorkingMemoryEntry]:
        return self.working_memory.get_all()

    def promote_working(self, entry: WorkingMemoryEntry) -> str | None:
        if entry not in self.working_memory._entries:
            return None
        return self.working_memory.promote(entry, self)

    def clear_working_memory(self, session_id: str = ""):
        if session_id:
            self.working_memory.clear_session(session_id)
        else:
            self.working_memory.clear()

    def update(self, anchor_id: str, text: str | None = None,
               tags: list[str] | None = None,
               importance: float | None = None) -> Anchor | None:
        """Update an existing memory with new information."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return None

        if text is not None:
            anchor.text = text
            embedder = self._get_embedder()
            anchor.embedding = embedder.encode(text)
        if tags is not None:
            anchor.tags = list(set(anchor.tags + tags))
        if importance is not None:
            anchor.vector.importance = max(0.0, min(1.0, importance))

        anchor.activate()
        return anchor

    # ── Multimodal operations ─────────────────────────────────

    def remember_image(self, image_path: str, caption: str = "",
                      tags: list[str] | None = None,
                      importance: float = 0.5,
                      source_session: str = "",
                      emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Store a memory with an image attachment."""
        anchor = MultimodalAnchor.from_image(
            image_path=image_path,
            provider=self.multimodal,
            caption=caption,
            tags=tags,
            importance=importance,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )
        self.graph.add_anchor(anchor)
        self.raw_buffer.add(
            text=caption or f"[image] {os.path.basename(image_path)}",
            session_id=source_session,
            embedding=anchor.embedding,
            tags=tags,
            importance=importance,
            anchor_id=anchor.id,
        )
        return anchor

    def remember_text_and_image(self, text: str, image_path: str,
                               tags: list[str] | None = None,
                               importance: float = 0.5,
                               source_session: str = "",
                               emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Store a memory with both text content and an associated image."""
        anchor = MultimodalAnchor.from_text_and_image(
            text=text,
            image_path=image_path,
            provider=self.multimodal,
            tags=tags,
            importance=importance,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )
        self.graph.add_anchor(anchor)
        self.raw_buffer.add(
            text=text,
            session_id=source_session,
            embedding=anchor.embedding,
            tags=tags,
            importance=importance,
            anchor_id=anchor.id,
        )
        return anchor

    # ── Cortex management ────────────────────────────────────

    def add_cortex(self, name: str, domain_keywords: list[str],
                   description: str = "", **kwargs) -> MemoryCortex:
        """Create and register a new domain-specific memory cortex."""
        cortex = self.router.find_or_create_cortex(
            name=name,
            domain_keywords=domain_keywords,
            description=description,
            **kwargs,
        )
        self.brain.register_cortex(
            cortex_name=name,
            entry_embedding=cortex.centroid or [],
            summary=description,
            node_count=0,
        )
        return cortex

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

    # ── Ghost operations ──────────────────────────────────────

    def fuzzy_recall(self, embedding: list[float] | None = None,
                     query: str = "", threshold: float | None = None):
        """Fuzzy recall from ghost traces — 'I seem to remember...' """
        if embedding is None:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        return self.ghosts.fuzzy_recall(embedding, threshold)

    def ghost_intensity_recall(self, query: str = "",
                               embedding: list[float] | None = None,
                               top_k: int = 10) -> list[dict]:
        """Rank ghost traces by intensity for retrieval boosting."""
        if embedding is None:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        ranked = self.ghosts.ranked_resonance(embedding, top_k=top_k)
        return [
            {
                "ghost_id": ghost.id,
                "intensity": round(intensity, 4),
                "semantic_shadow": ghost.semantic_shadow,
                "original_tags": ghost.original_tags,
                "original_importance": ghost.original_importance,
                "emotion_trace": ghost.emotion_trace,
                "revival_count": ghost.revival_count,
            }
            for ghost, intensity in ranked
        ]

    def create_negative_ghost(self, original_text: str,
                             contradiction_text: str,
                             target_anchor_id: str = "",
                             original_importance: float = 0.5,
                             contradiction_type: str = "direct") -> str:
        """Create a negative ghost to track a contradiction."""
        embedder = self._get_embedder()
        embedding = embedder.encode(original_text)
        ghost = self.ghosts.create_negative(
            original_text=original_text,
            contradiction_text=contradiction_text,
            target_anchor_id=target_anchor_id,
            original_importance=original_importance,
            contradiction_type=contradiction_type,
            embedding=embedding,
        )
        return ghost.id

    def check_ghost_suppression(self, embedding: list[float] | None = None,
                               query: str = "") -> dict:
        """Check if a query or embedding is suppressed by negative ghosts."""
        if embedding is None and query:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        if embedding is None:
            return {"suppression_factor": 1.0, "active_negatives": []}

        factor = self.ghosts.check_suppression(embedding)
        active = []
        for ghost in self.ghosts.negative_ghosts:
            resonance = ghost.resonance(embedding)
            if resonance > 0.1:
                active.append({
                    "ghost_id": ghost.id,
                    "resonance": round(resonance, 4),
                    "intensity": round(ghost.intensity, 4),
                    "contradiction_type": ghost.contradiction_type,
                    "contradiction_text": ghost.contradiction_text[:120],
                })

        return {
            "suppression_factor": round(factor, 4),
            "active_negatives": active,
        }

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
        filepath = path or self.storage_path or "star_graph_memory.json"
        from .storage import JSONStorage
        storage = JSONStorage(filepath)
        storage.save(self.graph)
        if self._ghosts:
            import json
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

    def _infer_domain(self, anchor) -> str:
        """Infer the memory domain from anchor tags."""
        tags_lower = [t.lower() for t in anchor.tags]
        tag_text = " ".join(tags_lower) + " " + anchor.text.lower()

        procedural_kw = {"how", "fix", "solution", "workflow", "deploy", "install",
                         "config", "setup", "debug", "error", "bug", "patch"}
        semantic_kw = {"preference", "knowledge", "concept", "fact", "opinion",
                       "user", "style", "likes", "dislikes", "background"}
        reflection_kw = {"strategy", "lesson", "pattern", "summary", "insight",
                         "reflection", "meta", "review", "analysis"}

        if any(kw in tag_text for kw in reflection_kw):
            return "reflection"
        if any(kw in tag_text for kw in procedural_kw):
            return "procedural"
        if any(kw in tag_text for kw in semantic_kw):
            return "semantic"
        return "episodic"  # default: conversations and events

    def load(self, path: str) -> StarGraph:
        """Load the memory system from disk."""
        from .storage import JSONStorage
        storage = JSONStorage(path)
        self.graph = storage.load()
        self.storage_path = path

        # Try to load ghosts
        import json
        ghost_path = path.replace(".json", "_ghosts.json")
        if os.path.exists(ghost_path):
            with open(ghost_path, "r", encoding="utf-8") as f:
                ghost_data = json.load(f)
            from .ghost import GhostNode
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

        # Reset subsystems that cache graph references
        self._scheduler = None
        self._evolution = None
        self._metrics = None
        self._community_detection = None

        # Re-set survival function (class vars don't survive process restart)
        Anchor.set_survival_function(self.survival_function)
        from .ghost import GhostNode
        GhostNode.set_survival_function(self.survival_function)

        return self.graph

    # ── Reflection (meta-cognitive insights) ───────────────

    def reflect(self, text: str, source_anchor_ids: list[str],
                reflection_type: str = "lesson_learned",
                confidence: float = 0.6) -> str:
        """Create a meta-cognitive reflection on past memories."""
        from .graph import ReflectionNode
        if reflection_type == "failure_analysis":
            r = ReflectionNode.from_failure(text, source_anchor_ids, confidence)
        elif reflection_type == "success_pattern":
            r = ReflectionNode.from_success(text, source_anchor_ids, confidence)
        elif reflection_type == "root_cause":
            import hashlib
            rid = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
            r = ReflectionNode(id=rid, text=text, reflection_type="root_cause",
                              source_anchor_ids=source_anchor_ids,
                              confidence=confidence, strength=0.5)
        else:
            r = ReflectionNode.from_lesson(text, source_anchor_ids, confidence)
        return self.graph.add_reflection(r)

    def get_reflections(self, anchor_ids: list[str],
                        types: list[str] | None = None) -> list[dict]:
        nodes = self.graph.find_reflections(anchor_ids, types)
        return [
            {"id": r.id, "text": r.text, "type": r.reflection_type,
             "confidence": r.confidence, "strength": r.strength}
            for r in nodes
        ]

    def strengthen_reflection(self, reflection_id: str) -> bool:
        if reflection_id in self.graph.reflections:
            self.graph.reflections[reflection_id].reinforce()
            return True
        return False

    def weaken_reflection(self, reflection_id: str) -> bool:
        if reflection_id in self.graph.reflections:
            self.graph.reflections[reflection_id].weaken()
            return True
        return False

    # ── Autobiographical memory (self-model) ───────────────

    def remember_self(self, episode_summary: str = "",
                      self_belief: str = "",
                      emotional_tone: float = 0.0,
                      source_session: str = "",
                      source_anchor_ids: list[str] | None = None,
                      tags: list[str] | None = None):
        """Record a first-person narrative — the agent's own experience.

        This is NOT a fact about the user/world (use remember() for that).
        This is "I discussed X," "I believe Y about the user," "my tone was Z."
        """
        return self.autobiography.form_from_interaction(
            episode_summary=episode_summary,
            self_belief=self_belief,
            emotional_tone=emotional_tone,
            source_session=source_session,
            source_anchor_ids=source_anchor_ids,
            tags=tags,
        )

    def recall_self(self, query: str = "", *,
                    top_k: int = 10,
                    min_stability: float = 0.05):
        """Recall what the agent knows about itself — 'what do I know about myself?' """
        return self.autobiography.recall_self(
            query=query, top_k=top_k, min_stability=min_stability,
        )

    def get_self_beliefs(self, min_stability: float = 0.2) -> list[dict]:
        """Get the agent's stable self-beliefs — its current self-model."""
        return self.autobiography.get_beliefs(min_stability=min_stability)

    def update_self_belief(self, belief_substring: str, correction: str = ""):
        """Update or correct a self-belief when the agent's understanding changes."""
        return self.autobiography.contradict_belief(belief_substring, correction)

    def self_emotional_profile(self, session_id: str = "") -> dict:
        """Get the agent's emotional profile across interactions."""
        return self.autobiography.get_emotional_profile(session_id)

    # ── Tiered storage ─────────────────────────────────────────

    def offload_cold_anchors(self) -> int:
        """Offload COLD thermal-state anchors to disk, freeing RAM.

        Called during sleep N4_Prune. Anchors with thermal_state == COLD are
        serialized to TieredStorage and removed from graph.anchors.
        Returns count of offloaded anchors.
        """
        from .anchor import ThermalState
        cold_ids = []
        for aid, a in self.graph.anchors.items():
            if getattr(a, '_thermal_state', None) == ThermalState.COLD:
                cold_ids.append(aid)

        if not cold_ids:
            return 0

        from .tiered import offload_anchor_to_cold
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

        from .anchor import Anchor
        anchor = Anchor(
            id=anchor_id,
            text=data.get("text", ""),
            embedding=data.get("embedding"),
            tags=data.get("tags", []),
            source_session=data.get("source_session", ""),
            created_at=data.get("created_at", time.time()),
            last_activated_at=data.get("last_activated_at", time.time()),
            community_id=data.get("community_id", ""),
            importance=data.get("importance", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
        )
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
        """Explicitly connect two memories. Edge budget enforced post-connection."""
        if source_id not in self.graph.anchors or target_id not in self.graph.anchors:
            return None
        src_emb = self.graph.anchors[source_id].embedding
        tgt_emb = self.graph.anchors[target_id].embedding
        if src_emb and tgt_emb and weight == 0.5:
            weight = _cosine_sim(src_emb, tgt_emb)
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

    # ── Compression API ──────────────────────────────────────────

    def compress_session(self, session_id: str) -> list:
        from .compression import SessionCompressor
        anchors_list = list(self.graph.anchors.values())
        compressor = SessionCompressor()
        summaries = compressor.compress(anchors_list, session_id)
        if summaries:
            _mcomp = self.compressor
            _mcomp.add_to_graph(self.graph, summaries, edge_type="compresses")
        return summaries

    def compress_all(self) -> dict:
        from collections import defaultdict
        from .compression import CompressionLevel

        anchors_by_session: dict[str, list] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            if anchor.embedding and anchor.source_session:
                anchors_by_session[anchor.source_session].append(anchor)

        if not anchors_by_session:
            return {
                "levels": {"episodic": 0, "strategic": 0, "meta": 0},
                "total_summaries": 0,
                "total_edges": 0,
            }

        compressor = self.compressor
        results = compressor.compress_pipeline(anchors_by_session)

        total_edges = 0
        level_counts: dict[str, int] = {}
        for level, summaries in results.items():
            level_name = level.name.lower()
            level_counts[level_name] = len(summaries)
            if summaries:
                total_edges += compressor.add_to_graph(self.graph, summaries, edge_type="compresses")

        return {
            "levels": level_counts,
            "total_summaries": sum(level_counts.values()),
            "total_edges": total_edges,
        }

    def get_compressed_view(self,
                            min_level: int = 1,
                            max_summaries: int = 50) -> list[dict]:
        from .compression import CompressionLevel
        level_tags = {
            CompressionLevel.EPISODIC: "level:episodic",
            CompressionLevel.STRATEGIC: "level:strategic",
            CompressionLevel.META: "level:meta",
        }

        summaries: list[dict] = []
        for aid, anchor in self.graph.anchors.items():
            level_tag = None
            level_val = 0
            for lvl, tag in level_tags.items():
                if tag in anchor.tags:
                    level_tag = tag
                    level_val = lvl.value
                    break

            if level_val < min_level:
                continue

            source_count = 0
            for neighbor in self.graph._adjacency.get(aid, set()):
                edge_key = self.graph._key(aid, neighbor)
                edge = self.graph.edges.get(edge_key)
                if edge and edge.edge_type == "compresses":
                    source_count += 1

            summaries.append({
                "id": aid,
                "text": anchor.text,
                "level": level_val,
                "level_name": CompressionLevel(level_val).name.lower() if level_val > 0 else "raw",
                "confidence": anchor.vector.confidence,
                "stability": anchor.vector.stability,
                "source_count": source_count,
                "tags": [t for t in anchor.tags if not t.startswith("level:")],
            })

        summaries.sort(key=lambda x: -x["confidence"])
        return summaries[:max_summaries]

    # ── Cognitive Compiler API (#45) ─────────────────────────

    def compile_worldviews(self) -> dict:
        """Run the full cognitive compilation pipeline and return results."""
        return self.compiler.compile(self.graph)

    def get_worldviews(self, min_stability: float = 0.5) -> list:
        """Get stable worldview beliefs about the user."""
        wvs = self.compiler.get_stable_worldviews(min_stability=min_stability)
        return [
            {
                "id": wv.id,
                "label": wv.label,
                "description": wv.description,
                "type": wv.worldview_type,
                "confidence": round(wv.confidence, 3),
                "stability": round(wv.stability, 3),
                "evidence_count": wv.evidence_count,
                "domain": wv.domain,
                "tags": wv.tags,
            }
            for wv in wvs
        ]

    def get_user_profile(self) -> dict | None:
        """Get the synthesized user profile from worldview consensus."""
        profile = self.compiler.profile
        if profile is None:
            return None
        return {
            "summary": profile.summary,
            "preferences": profile.preferences,
            "expertise_areas": profile.expertise_areas,
            "working_style": profile.working_style,
            "values": profile.values,
            "habits": profile.habits,
            "confidence": round(profile.confidence, 3),
            "version": profile.version,
        }

    # ── Self-Reflection API (#46) ────────────────────────────

    def get_corrections(self, topic: str = "") -> list[dict]:
        """Get correction reports. 'what did I get wrong about X?'"""
        if topic:
            return self.reflection_loop.get_corrections_for_topic(topic)
        return self.reflection_loop.get_recent_corrections()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for dedup comparison."""
        import re
        return re.sub(r'\s+', ' ', text.lower().strip())[:100]


