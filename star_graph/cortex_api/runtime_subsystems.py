"""RuntimeSubsystems — mixin with lazily initialized subsystem accessors.

Extracted verbatim from ``runtime.py`` ``MemoryRuntime`` (issue #9) so every
module stays within the 1000-line budget. Property bodies are unchanged: each
subsystem is still constructed on first access and cached on ``self``.
"""

from __future__ import annotations

import os

from ..router import CortexRouter
from ..timespine import TimeSpine
from ..tiered import TieredStorage
from ..hippocampus import HippocampusBuffer
from ..shard import MemoryShardManager
from ..cascade import CascadeRecall
from ..spreading import SpreadingActivation
from ..cognitive_cache import CognitiveCacheManager
from ..compiler import CognitiveCompiler
from ..reflection_loop import SelfReflectionLoop
from ..hub import HubLayer
from ..evolution import MemoryEvolutionEngine
from ..ghost import GhostSubsystem
from ..abstraction import AbstractionEngine
from ..community import CommunityDetection
from ..metrics import CognitiveMetrics
from ..raw_buffer import RawBuffer
from ..dual_channel import DualChannelRetriever
from ..exact_cache import ExactMatchCache
from ..survival import SurvivalFunction, SurvivalRegistry
from ..domain_router import DomainRouter
from .gate import MemoryGate
from .write_gate import MemoryWriteGate
from ..edge_management import EdgeBudgetManager, EdgeDecayManager
from ..four_layer import FourLayerCompressor
from ..thermal_store import ThermalStore
from ..self_org import SelfOrganization
from ..personality import PersonalityModel
from ..goal_tree import GoalTree
from ..retrieval_budget import RetrievalBudget
from ..versioned_memory import CognitiveTrajectory
from ..cluster_memory import ClusterRouter
from ..causal_edges import CausalEdgeClassifier
from ..episodic_memory import EpisodicMemory
from ..multimodal import MultimodalEmbeddingProvider, CrossModalRetriever


class RuntimeSubsystems:
    """Mixin: lazy subsystem accessors (construct on first access, then cached)."""

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
            from ..compression import MultiLevelCompressor
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
            from ..autobiography import AutobiographicalMemory
            self._autobiography = AutobiographicalMemory()
        return self._autobiography

    @property
    def streaming_buffer(self):
        """Lazy-init streaming memory buffer for continuous ingestion."""
        if self._streaming_buffer is None:
            from ..streaming import StreamingMemoryBuffer
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
            from ..bm25 import BM25Index
            self._bm25 = BM25Index()
            # Populate from existing anchors
            for aid, a in self.graph.anchors.items():
                self._bm25.add(aid, a.text)
        return self._bm25

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            from ..cross_encoder import CrossEncoderReranker
            cfg = self.cfg
            rerank_cfg = cfg.get_path('retrieval.rerank', {}) if hasattr(cfg, 'get_path') else {}
            self._cross_encoder = CrossEncoderReranker(
                model_name=rerank_cfg.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                top_k=rerank_cfg.get('top_k', 10),
                threshold=rerank_cfg.get('threshold', 0.0),
                enabled=rerank_cfg.get('enabled', False),
            )
        return self._cross_encoder

    @property
    def batch_vectorizer(self):
        if self._batch_vectorizer is None:
            from ..batch_vectorizer import BatchVectorizer
            bv_cfg = getattr(self.cfg, 'batch_vectorization', None) or {}
            journal_path = getattr(bv_cfg, 'journal_path', "") if hasattr(bv_cfg, 'get') else bv_cfg.get('journal_path', "")
            if journal_path and self.storage_path:
                import os
                journal_path = os.path.join(os.path.dirname(self.storage_path) or ".", "pending_embeddings.db")
            self._batch_vectorizer = BatchVectorizer(
                self.graph,
                self._get_embedder(),
                batch_size=getattr(bv_cfg, 'batch_size', 32) if hasattr(bv_cfg, 'get') else bv_cfg.get('batch_size', 32),
                flush_interval=getattr(bv_cfg, 'flush_interval_seconds', 30.0) if hasattr(bv_cfg, 'get') else bv_cfg.get('flush_interval_seconds', 30.0),
                db_path=journal_path,
            )
        return self._batch_vectorizer

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
        """Lazy-init retrieval budget for hop/node/token limits.

        Delegates to RetrievalBudget.from_config — the single budget
        construction entry point (values from defaults.yaml
        `retrieval_budget:` section, constants as fallback).
        """
        if self._retrieval_budget is None:
            self._retrieval_budget = RetrievalBudget.from_config(self.cfg)
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
            from ..embedding import get_embedder
            self._embedder = get_embedder()
            # Create instance-level registry to avoid multi-Manager singleton pollution
            from ..anchor import EmbedderRegistry
            self._embedder_registry = EmbedderRegistry(self._embedder)
        return self._embedder
