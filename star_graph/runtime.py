"""MemoryRuntime — dependency container and lifecycle manager.

Owns all subsystems (graph, embedder, cortices, ghosts, evolution, etc.)
and provides CRUD, persistence, and cognitive maintenance methods.

MemoryManager delegates to this for all non-retrieval operations.
"""

from __future__ import annotations

import math
import os
import time
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
from .edge_management import EdgeBudgetManager, EdgeDecayManager
from .four_layer import FourLayerCompressor
from .thermal_store import ThermalStore
from .self_org import SelfOrganization
from .personality import PersonalityModel
from .goal_tree import GoalTree
from .retrieval_budget import RetrievalBudget
from .versioned_memory import CognitiveTrajectory
from .cluster_memory import ClusterRouter
from .causal_edges import CausalEdgeClassifier
from .episodic_memory import EpisodicMemory
from .multimodal import (
    MultimodalEmbeddingProvider, MultimodalAnchor, CrossModalRetriever, CrossModalResult,
)


from .manager_stats import ManagerStats
from .runtime_core import RuntimeCore
from .runtime_lifecycle import RuntimeLifecycle


class MemoryRuntime(RuntimeCore, RuntimeLifecycle):
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

        # Self-organization — auto-cluster, merge, topic detection (#55)
        self._self_org: SelfOrganization | None = None

        # Personality model — deep trait extraction (#56)
        self._personality: PersonalityModel | None = None

        # Goal tree — hierarchical goal decomposition (#57)
        self._goal_tree: GoalTree | None = None

        # Retrieval budget — hop/node/token limits (S-5)
        self._retrieval_budget: RetrievalBudget | None = None

        # Versioned memory — cognitive trajectory tracking (A-9)
        self._versioned_memory: CognitiveTrajectory | None = None

        # Cluster memory — retrieval pre-filtering (A-10)
        self._cluster_router: ClusterRouter | None = None

        # Causal edge classifier — richer edge types (B-12)
        self._causal_classifier: CausalEdgeClassifier | None = None

        # Episodic memory — time + context event streams (B-13)
        self._episodic_memory: EpisodicMemory | None = None

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
    def self_org(self) -> SelfOrganization:
        """Lazy-init self-organization engine for auto-clustering and merging."""
        if self._self_org is None:
            so_cfg = getattr(self.cfg, 'self_org', None)
            merge_th = getattr(so_cfg, 'merge_threshold', 0.88) if so_cfg else 0.88
            cluster_sim = getattr(so_cfg, 'cluster_similarity', 0.55) if so_cfg else 0.55
            self._self_org = SelfOrganization(
                merge_threshold=merge_th,
                cluster_similarity=cluster_sim,
            )
        return self._self_org

    @property
    def personality(self) -> PersonalityModel:
        """Lazy-init personality model for deep trait extraction."""
        if self._personality is None:
            self._personality = PersonalityModel()
        return self._personality

    @property
    def goal_tree(self) -> GoalTree:
        """Lazy-init goal tree for hierarchical goal tracking."""
        if self._goal_tree is None:
            self._goal_tree = GoalTree()
        return self._goal_tree

    @property
    def retrieval_budget(self) -> RetrievalBudget:
        """Lazy-init retrieval budget for hop/node/token limits."""
        if self._retrieval_budget is None:
            rb_cfg = getattr(self.cfg, 'retrieval_budget', None)
            max_hops = getattr(rb_cfg, 'max_hops', 3) if rb_cfg else 3
            max_nodes = getattr(rb_cfg, 'max_nodes', 24) if rb_cfg else 24
            max_tokens = getattr(rb_cfg, 'max_tokens', 6000) if rb_cfg else 6000
            self._retrieval_budget = RetrievalBudget(
                max_hops=max_hops, max_nodes=max_nodes, max_tokens=max_tokens)
        return self._retrieval_budget

    @property
    def versioned_memory(self) -> CognitiveTrajectory:
        """Lazy-init cognitive trajectory for belief evolution tracking."""
        if self._versioned_memory is None:
            self._versioned_memory = CognitiveTrajectory()
        return self._versioned_memory

    @property
    def cluster_router(self) -> ClusterRouter:
        """Lazy-init cluster router for retrieval pre-filtering."""
        if self._cluster_router is None:
            cr_cfg = getattr(self.cfg, 'cluster_memory', None)
            min_size = getattr(cr_cfg, 'min_cluster_size', 5) if cr_cfg else 5
            self._cluster_router = ClusterRouter(min_cluster_size=min_size)
        return self._cluster_router

    @property
    def causal_classifier(self) -> CausalEdgeClassifier:
        """Lazy-init causal edge classifier for richer edge types."""
        if self._causal_classifier is None:
            self._causal_classifier = CausalEdgeClassifier()
        return self._causal_classifier

    @property
    def episodic_memory(self) -> EpisodicMemory:
        """Lazy-init episodic memory for event stream recording."""
        if self._episodic_memory is None:
            self._episodic_memory = EpisodicMemory()
        return self._episodic_memory

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

    # (CRUD / working memory / multimodal / cortex / ghost / lifecycle / persistence methods
    #  moved to RuntimeCore and RuntimeLifecycle mixins)

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
