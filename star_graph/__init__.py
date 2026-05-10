"""Star Graph Memory System — v0.4 real mechanisms.

A hippocampal-inspired cognitive memory architecture for AI agents.

Two retrieval pathways:
  Hippocampal: oscillatory phase-locking + spreading activation
  Cortical:    ANN-indexed embedding lookup (consolidated memories)

Sleep cycles implement 9 phases:
  Prioritized SWR Replay → Systems Consolidation → Emotional Stripping →
  Schema Extraction → Merge → Adaptive Prune → Bridge →
  Hebbian Update → Synaptic Homeostasis

v0.4 — mechanism, not metaphor:
  - Real embeddings (sentence-transformers / sklearn TF-IDF)
  - Meaningful phase derivation: f(timestamp, importance, emotion)
  - Prioritized Experience Replay (like RL PER, biologically motivated)
  - ANN-indexed sub-linear retrieval (sklearn NearestNeighbors)
  - Embedding similarity replaces character bigrams throughout
  - Deterministic mode for reproducible benchmarks (seed.py)
"""

__version__ = "0.4.0-dev"

from .anchor import Anchor, AnchorVector, GhostAnchor, Oscillator
from .graph import StarGraph, Edge, Constellation, Schema
from .sleep import SleepCycle
from .online import OnlineConsolidator
from .retriever import (
    Retriever,
    OscillationResonanceRetriever,
    VectorSimilarityRetriever,
    compare_retrievers,
)
from .embedding import EmbeddingProvider, get_embedder
from .index import ANNIndex
from .seed import seed_everything, is_deterministic
