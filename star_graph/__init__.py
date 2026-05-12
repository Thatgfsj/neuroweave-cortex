"""Star Graph Memory System — v0.6 cognitive architecture.

A sparse-activated, cortex-partitioned cognitive memory runtime for AI agents.

Architecture:
  Layer 3 (Behavior):  Cortex routing, memory gating, working memory,
                        dimensional reduction retrieval, adaptive replay
  Layer 2 (Cognitive): Hub abstraction, cascade recall, time spine,
                        sleep consolidation, evolution, ghost revival
  Layer 1 (Storage):   CRUD, persistence, indexing, ANN lookup
"""

__version__ = "0.6.0-dev"

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
from .ghost import GhostNode, GhostSubsystem
from .abstraction import AbstractNode, AbstractionEngine
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
from .hub import HubLayer, HubNode, HubEdge
from .brain_sphere import BrainSphere, HubCenter
from .manager import MemoryManager, ManagerStats

# MCP server is optional — requires `pip install mcp`
try:
    from .mcp_server import server as mcp_server
except ImportError:
    mcp_server = None
