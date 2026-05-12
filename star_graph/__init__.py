"""Star Graph Memory System — v0.6 cognitive architecture.

A sparse-activated, cortex-partitioned cognitive memory runtime for AI agents.

Architecture:
  Layer 3 (Behavior):  Cortex routing, memory gating, working memory,
                        dimensional reduction retrieval, adaptive replay
  Layer 2 (Cognitive): Hub abstraction, cascade recall, time spine,
                        sleep consolidation, evolution, ghost revival
  Layer 1 (Storage):   CRUD, persistence, indexing, ANN lookup
"""

__version__ = "1.0.6"

from .anchor import Anchor, AnchorVector, GhostAnchor, Oscillator, MemoryState, ThermalState
from .graph import StarGraph, Edge, RichEdge, Constellation, Schema, ReflectionNode
from .sleep import SleepCycle, SleepReport, PhaseMetrics
from .online import OnlineConsolidator
from .retriever import (
    RetrievalResult,
    RetrievalTrace,
    RetrievalTraceEntry,
    Retriever,
    OscillationResonanceRetriever,
    VectorSimilarityRetriever,
    HybridFusionRetriever,
    ExplainableScore,
    personalized_pagerank,
    compare_retrievers,
)
from .embedding import EmbeddingProvider, get_embedder
from .index import ANNIndex
from .seed import seed_everything, is_deterministic
from .ghost import GhostNode, NegativeGhost, GhostSubsystem
from .abstraction import AbstractNode, AbstractionEngine
from .community import Community, CommunityHealth, CommunityDetection
from .raw_buffer import RawBuffer, RawChunk
from .dual_channel import DualChannelRetriever, DualChannelOutput, ChannelResult
from .atom_facts import FactExtractor, AtomFact, ExtractionResult, check_llm_availability
from .metrics import CognitiveMetrics
from .competition import MemoryCompetition
from .config import Config, config, override, reload_defaults, load_config
from .layers import enforce_layer_boundaries, layer_summary, get_layer, check_import
from .anchor import EmbedderRegistry
from .storage import Storage, JSONStorage
from .storage_backend import StorageBackend
from .sqlite_storage import SQLiteStorage
from .evolution import MemoryEvolutionEngine, EvolutionEvent, BeliefTransition
from .scheduler import CognitiveMemoryScheduler, AgentContext, MemoryType, MemoryItem, MemoryContext
from .working_memory import WorkingMemory, WorkingMemoryEntry
from .cortex import MemoryCortex, CortexConfig
from .router import CortexRouter, RouteResult
from .gate import MemoryGate, GateScore
from .timespine import TimeSpine, TimeBucket, MemoryCluster
from .cascade import CascadeRecall, CausalChain
from .hub import HubLayer, HubNode, HubEdge, HubShard
from .brain_sphere import BrainSphere, HubCenter
from .manager import MemoryManager, ManagerStats
from .compression import (
    CompressionLevel,
    SummaryAnchor,
    SessionCompressor,
    MultiLevelCompressor,
)
from .exact_cache import ExactMatchCache, ExactMatchEntry, extract_entity_keys
from .micro_sleep import MicroSleepScheduler, MicroSleepProgress, MicroSleepResult
from .cost_estimator import SleepCostEstimator, CostEstimate
from .snapshot import SnapshotManager, SnapshotMeta
from .async_manager import AsyncMemoryManager, AsyncManagerStats
from .tracing import MemoryTracer, TraceSpan, Trace, get_tracer, trace_recall
from .survival import (
    SurvivalFunction,
    EbbinghausSurvival,
    PowerLawSurvival,
    ExponentialSurvival,
    CustomSurvival,
    SurvivalRegistry,
    SurvivalState,
    derive_strength,
)
from .multimodal import (
    MultimodalEmbeddingProvider,
    MultimodalAnchor,
    CrossModalRetriever,
    CrossModalResult,
)
from .resonance import Resonator
from .symbolic_filter import SymbolicFilter, FilterResult
from .streaming import (
    StreamItem,
    StreamStats,
    StreamingMemoryBuffer,
)
from .benchmark import (
    BenchmarkSuite,
    BenchmarkScenario,
    BenchmarkResult,
    ScenarioResult,
    Category,
    run_benchmark,
    compare_systems,
)

# MCP server is optional — requires `pip install mcp`
try:
    from .mcp_server import server as mcp_server
except ImportError:
    mcp_server = None
