"""Embedding Provider — multi-source embedding abstraction.

Pluggable embedding backends with unified async API:
  LocalProvider   — sentence-transformers (offline, free)
  OpenAIProvider  — text-embedding-3-small/large (API)
  ZhipuProvider   — embedding-2 (API, semaphore rate-limited)
  MixedProvider   — primary/fallback auto-failover

Usage:
    from star_graph.embedding_provider import LocalProvider, OpenAIProvider

    local = LocalProvider("all-MiniLM-L6-v2")
    vec = await local.embed_single("hello world")

    openai = OpenAIProvider(dimensions=256)
    vecs = await openai.embed(["text1", "text2"])
"""

from .providers import (
    EmbeddingProvider,
    LocalProvider,
    OpenAIProvider,
    ZhipuProvider,
    MixedProvider,
    ProviderMetrics,
    create_provider,
)

# Also re-export from legacy embedding.py for backward compat
from star_graph.embedding import EmbeddingProvider as LegacyEmbeddingProvider  # noqa: F401
from star_graph.embedding import get_embedder, reset_embedder

__all__ = [
    "EmbeddingProvider",
    "LocalProvider",
    "OpenAIProvider",
    "ZhipuProvider",
    "MixedProvider",
    "ProviderMetrics",
    "create_provider",
    "get_embedder",
    "reset_embedder",
]
