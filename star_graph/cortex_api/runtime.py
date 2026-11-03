"""MemoryRuntime — dependency container and lifecycle manager.

Owns all subsystems (graph, embedder, cortices, ghosts, evolution, etc.)
and provides CRUD, persistence, and cognitive maintenance methods.

MemoryManager delegates to this for all non-retrieval operations.

Method bodies live in mixin modules (``runtime_core``, ``runtime_lifecycle``,
``runtime_subsystems``, ``runtime_ops``, ``runtime_cognitive``); this module
keeps the ``MemoryRuntime`` composition class and its historical import
surface (issue #9).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

from ..anchor import Anchor, MemoryState
from ..graph import StarGraph, Edge
from ..config import Config
from ..math_utils import cosine_sim as _cosine_sim
from .working_memory import WorkingMemory, WorkingMemoryEntry
from ..cortex import MemoryCortex, CortexConfig
from ..router import CortexRouter
from .gate import MemoryGate
from ..timespine import TimeSpine
from ..tiered import TieredStorage
from ..hippocampus import HippocampusBuffer
from ..shard import MemoryShardManager
from ..cascade import CascadeRecall
from ..spreading import SpreadingActivation
from ..cognitive_cache import CognitiveCacheManager
from ..compiler import CognitiveCompiler
from ..reflection_loop import SelfReflectionLoop
from ..hub import HubLayer, HubNode
from ..brain_sphere import BrainSphere, HubCenter
from ..symbolic_filter import SymbolicFilter
from ..evolution import MemoryEvolutionEngine
from ..ghost import GhostSubsystem
from ..abstraction import AbstractionEngine
from ..community import Community, CommunityDetection
from ..metrics import CognitiveMetrics
from ..raw_buffer import RawBuffer
from ..dual_channel import DualChannelRetriever
from ..exact_cache import ExactMatchCache
from ..cost_estimator import SleepCostEstimator, CostEstimate
from ..snapshot import SnapshotManager, SnapshotMeta
from ..tracing import MemoryTracer, get_tracer
from ..survival import SurvivalFunction, SurvivalRegistry
from ..domain_router import DomainRouter
from .write_gate import MemoryWriteGate, GateDecision
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
from ..multimodal import (
    MultimodalEmbeddingProvider, MultimodalAnchor, CrossModalRetriever, CrossModalResult,
)


from .manager_stats import ManagerStats
from .runtime_core import RuntimeCore
from .runtime_lifecycle import RuntimeLifecycle


from .manager_stats import ManagerStats
from .runtime_core import RuntimeCore
from .runtime_lifecycle import RuntimeLifecycle
from .runtime_subsystems import RuntimeSubsystems
from .runtime_ops import RuntimeOps
from .runtime_cognitive import RuntimeCognitive


class MemoryRuntime(RuntimeSubsystems, RuntimeOps, RuntimeCognitive,
                    RuntimeCore, RuntimeLifecycle):
    """Dependency container and lifecycle manager for the cognitive memory system.

    Owns all subsystems, CRUD operations, persistence, and cognitive maintenance.
    Retrieval methods live in RetrievalPipeline, which takes a reference to this runtime.
    """

    def __init__(self, graph: StarGraph | None = None,
                 config: Config | None = None,
                 storage_path: str = ""):
        self._lock = threading.RLock()
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
        self._cross_encoder = None  # CrossEncoder reranker — lazy init
        self._batch_vectorizer = None  # BatchVectorizer — lazy init
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
        from ..ghost import GhostNode
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
