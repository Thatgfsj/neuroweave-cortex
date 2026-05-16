"""Embedding Provider — multi-source embedding abstraction with unified async API.

Four providers:
  LocalProvider   — sentence-transformers / ONNX (offline-capable)
  OpenAIProvider  — text-embedding-3-small/large (API, dimensions truncation)
  ZhipuProvider   — embedding-2 (API, semaphore rate limiting)
  MixedProvider   — primary/fallback auto-failover with dimension validation
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("star_graph.embedding")


# ── Abstract base ──────────────────────────────────────────────────────


class EmbeddingProvider(ABC):
    """Unified embedding interface.

    All implementations are async. Sync wrappers use asyncio.to_thread
    internally for local models.
    """

    dimension: int
    max_batch_size: int = 32

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns vectors with shape [len(texts), dimension]."""

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        results = await self.embed([text])
        return results[0]


# ── LocalProvider ───────────────────────────────────────────────────────


class LocalProvider(EmbeddingProvider):
    """Local embedding via sentence-transformers with optional ONNX acceleration.

    Falls back through: sentence-transformers → TF-IDF → hash (SHA256).
    Models are downloaded on first use, not at install time.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 dim: int = 384,
                 device: str = "cpu",
                 onnx: bool = False):
        self._model_name = model_name
        self._dim = dim
        self._device = device
        self._onnx = onnx
        self._model = None
        self._tfidf = None
        self._tfidf_texts: list[str] = []
        self._backend: str = "none"

    @property
    def dimension(self) -> int:
        return self._dim

    max_batch_size: int = 64

    def _ensure_model(self) -> None:
        if self._backend != "none":
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            try:
                self._dim = self._model.get_embedding_dimension()
            except AttributeError:
                self._dim = self._model.get_sentence_embedding_dimension()
            self._backend = "sentence-transformers"
            logger.info("LocalProvider: loaded %s (dim=%d)", self._model_name, self._dim)
        except Exception as e:
            logger.warning("LocalProvider: model load failed (%s), falling back to TF-IDF", e)
            self._backend = "tfidf"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        loop = asyncio.get_running_loop()
        if self._backend == "sentence-transformers":
            return await loop.run_in_executor(None, self._encode_st, texts)
        elif self._backend == "tfidf":
            return await loop.run_in_executor(None, self._encode_tfidf, texts)
        else:
            return await loop.run_in_executor(None, self._encode_hash_batch, texts)

    def _encode_st(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False)
        return vecs.tolist()

    def _encode_tfidf(self, texts: list[str]) -> list[list[float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=self._dim)
            self._dim = min(self._dim, 384)
        self._tfidf_texts.extend(texts)
        self._tfidf.fit(self._tfidf_texts)
        results = []
        for t in texts:
            mat = self._tfidf.transform([t])
            dense = mat.toarray()[0]
            norm = math.sqrt(sum(x * x for x in dense))
            if norm > 1e-8:
                dense = dense / norm
            vec = dense.tolist()
            if len(vec) < self._dim:
                vec += [0.0] * (self._dim - len(vec))
            results.append(vec[:self._dim])
        return results

    @staticmethod
    def _hash_embed(text: str, dim: int) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode())
        digest = int(h.hexdigest(), 16)
        rng = np.random.RandomState(digest % (2**31))
        vec = rng.randn(dim).tolist()
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def _encode_hash_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t, self._dim) for t in texts]


# ── OpenAIProvider ─────────────────────────────────────────────────────


class OpenAIProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small/large with proxy support and dimensions truncation.

    Uses the OpenAI REST API. Dimensions parameter only valid for text-embedding-3-* models.
    """

    def __init__(self, model: str = "text-embedding-3-small",
                 dimensions: int = 512,
                 api_key: str | None = None,
                 base_url: str | None = None):
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self._dim = dimensions

    @property
    def dimension(self) -> int:
        return self._dim

    max_batch_size: int = 2048

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import aiohttp
        url = f"{self._base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
        }
        if "text-embedding-3" in self._model:
            payload["dimensions"] = self._dimensions

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenAI embedding API error {resp.status}: {text[:500]}")
                data = await resp.json()
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [e["embedding"] for e in embeddings]


# ── ZhipuProvider ──────────────────────────────────────────────────────


class ZhipuProvider(EmbeddingProvider):
    """Zhipu (智谱) embedding-2 with semaphore-based concurrency control.

    Zhipu API has strict QPS limits. The semaphore enforces max_concurrent
    requests, and embed() splits large batches automatically.
    """

    def __init__(self, model: str = "embedding-2",
                 api_key: str | None = None,
                 max_concurrent: int = 3):
        self._model = model
        self._api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # embedding-2 outputs 1024 or 2048 dims
        self._dim: int = 1024

    @property
    def dimension(self) -> int:
        return self._dim

    max_batch_size: int = 16  # Zhipu batch limit

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import aiohttp
        url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        all_embeddings: list[list[float]] = []

        async def _do_batch(batch: list[str], start_idx: int):
            async with self._semaphore:
                payload = {"model": self._model, "input": batch}
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"Zhipu embedding API error {resp.status}: {text[:500]}")
                        data = await resp.json()
                        items = sorted(data["data"], key=lambda x: x["index"])
                        return [(start_idx + i, items[i]["embedding"]) for i in range(len(items))]

        # Split into batches respecting max_batch_size
        tasks = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            tasks.append(_do_batch(batch, i))

        results = await asyncio.gather(*tasks)
        all_items = []
        for r in results:
            all_items.extend(r)
        all_items.sort(key=lambda x: x[0])
        return [item[1] for item in all_items]


# ── MixedProvider ──────────────────────────────────────────────────────


@dataclass
class ProviderMetrics:
    """Per-provider runtime metrics for monitoring."""
    primary_errors: int = 0
    fallback_count: int = 0
    total_latency_ms: float = 0.0
    call_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0


class MixedProvider(EmbeddingProvider):
    """Primary/fallback embedding provider with auto-failover.

    Validates dimension alignment between primary and fallback on init.
    Logs full exception details on failover for debugging.
    Tracks metrics for monitoring (embedding_fallback_count, etc.).
    """

    def __init__(self, primary: EmbeddingProvider,
                 fallback: EmbeddingProvider,
                 timeout: float = 8.0):
        if primary.dimension != fallback.dimension:
            raise ValueError(
                f"Dimension mismatch: primary={primary.dimension}, "
                f"fallback={fallback.dimension}. Index corruption would result."
            )
        self._primary = primary
        self._fallback = fallback
        self._timeout = timeout
        self._dim = primary.dimension
        self.metrics = ProviderMetrics()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def max_batch_size(self) -> int:
        return min(self._primary.max_batch_size, self._fallback.max_batch_size)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            t0 = time.perf_counter()
            result = await asyncio.wait_for(
                self._primary.embed(texts), timeout=self._timeout
            )
            self.metrics.call_count += 1
            self.metrics.total_latency_ms += (time.perf_counter() - t0) * 1000
            return result
        except Exception as e:
            self.metrics.primary_errors += 1
            self.metrics.fallback_count += 1
            self.metrics.last_error = f"{type(e).__name__}: {e}"
            self.metrics.last_error_time = time.time()
            logger.warning(
                "MixedProvider: primary failed (%s), falling back to %s",
                self.metrics.last_error, type(self._fallback).__name__
            )
            try:
                t0 = time.perf_counter()
                result = await self._fallback.embed(texts)
                self.metrics.call_count += 1
                self.metrics.total_latency_ms += (time.perf_counter() - t0) * 1000
                return result
            except Exception as e2:
                self.metrics.last_error = f"{type(e2).__name__}: {e2}"
                self.metrics.last_error_time = time.time()
                logger.error("MixedProvider: fallback also failed: %s", e2)
                raise


# ── Factory ─────────────────────────────────────────────────────────────


def create_provider(config: dict) -> EmbeddingProvider:
    """Create an EmbeddingProvider from a configuration dict.

    Config structure (matches YAML section):
      embedding:
        provider: mixed|openai|local|zhipu
        mixed:
          primary: openai
          fallback: local
          timeout: 8
        openai:
          model: text-embedding-3-small
          api_key: ${OPENAI_API_KEY}
          base_url: null
          dimensions: 512
        local:
          model: all-MiniLM-L6-v2
          device: cpu
          onnx: false
        zhipu:
          model: embedding-2
          api_key: ${ZHIPU_API_KEY}
          max_concurrent: 3
    """
    provider_type = config.get("provider", "local")

    if provider_type == "openai":
        oai = config.get("openai", {})
        return OpenAIProvider(
            model=oai.get("model", "text-embedding-3-small"),
            dimensions=oai.get("dimensions", 512),
            api_key=_resolve_env(oai.get("api_key", "")),
            base_url=oai.get("base_url"),
        )

    elif provider_type == "zhipu":
        zhi = config.get("zhipu", {})
        return ZhipuProvider(
            model=zhi.get("model", "embedding-2"),
            api_key=_resolve_env(zhi.get("api_key", "")),
            max_concurrent=zhi.get("max_concurrent", 3),
        )

    elif provider_type == "local":
        loc = config.get("local", {})
        return LocalProvider(
            model_name=loc.get("model", "all-MiniLM-L6-v2"),
            dim=loc.get("dim", 384),
            device=loc.get("device", "cpu"),
            onnx=loc.get("onnx", False),
        )

    elif provider_type == "mixed":
        mixed = config.get("mixed", {})
        primary_type = mixed.get("primary", "openai")
        fallback_type = mixed.get("fallback", "local")
        timeout = mixed.get("timeout", 8)

        primary = create_provider({"provider": primary_type, **config})
        fallback = create_provider({"provider": fallback_type, **config})
        return MixedProvider(primary, fallback, timeout=timeout)

    else:
        raise ValueError(f"Unknown embedding provider type: {provider_type}")


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} patterns in a config value."""
    if value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, "")
    return value
