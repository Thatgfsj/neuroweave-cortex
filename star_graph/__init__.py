"""Star Graph Memory System — v0.4 real mechanisms.

A hippocampal-inspired cognitive memory architecture for AI agents.

Three-layer architecture:
  Layer 1 (Storage): CRUD, persistence, indexing (graph.py, index.py, storage.py)
  Layer 2 (Cognitive): resonance, abstraction, replay, consolidation
                       (retriever.py, sleep.py, resonance.py, abstraction.py)
  Layer 3 (Behavior): retrieval policy, forgetting policy, adaptive replay
                       (seed.py, embedding.py)

Key v0.4 additions:
  - MemoryState machine: ACTIVE→REHEARSING→CONSOLIDATING→DORMANT→GHOST→REACTIVATED
  - Ghost Subsystem: latent memory traces with fuzzy recall
  - Abstraction Engine: emergent higher-order concepts from anchor clusters
  - Real embeddings: sentence-transformers with meaningful phase derivation
  - ANN-indexed sub-linear retrieval
  - Prioritized Experience Replay
  - Deterministic mode for reproducible benchmarks
"""

__version__ = "0.4.0-dev"

from .anchor import Anchor, AnchorVector, GhostAnchor, Oscillator, MemoryState
from .graph import StarGraph, Edge, Constellation, Schema
from .sleep import SleepCycle
from .online import OnlineConsolidator
from .retriever import (
    RetrievalResult,
    RetrievalTrace,
    RetrievalTraceEntry,
    Retriever,
    OscillationResonanceRetriever,
    VectorSimilarityRetriever,
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
