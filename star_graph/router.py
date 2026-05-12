"""Cortex Router — routes queries to the right brain regions.

Phase 1 of sparse activation: given a query, activate only 1-3 relevant
cortices out of potentially dozens. This is the first and most critical
gate — it prevents O(n²) global search by constraining retrieval to
domain-specific subgraphs.

Routing combines:
- Semantic similarity (query embedding vs cortex centroid)
- Keyword matching (query terms vs domain keywords)
- Recency (recently active cortices get priority)
- Default fallback when no cortex matches well
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .cortex import MemoryCortex, CortexConfig
from .config import Config


@dataclass
class RouteResult:
    """Result of routing a query to cortices."""
    cortex: MemoryCortex
    score: float
    reasoning: str = ""


class CortexRouter:
    """Routes queries to the most relevant cortices.

    Only 1-3 cortices are activated per query — this is the fundamental
    mechanism that keeps retrieval cost bounded regardless of total memory count.

    Usage:
        router = CortexRouter()
        router.add_cortex(dev_cortex)
        router.add_cortex(personal_cortex)

        # Route a query
        active = router.route("Redis connection pool config")
        # Returns top-2 cortices: [dev_cortex (0.82), infra_cortex (0.45)]

        # Recall from activated cortices only
        results = router.recall("Redis pool", context=ctx)
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config.get()
        self.cortices: list[MemoryCortex] = []
        self._default_cortex: MemoryCortex | None = None
        self._embedder = None

        # Routing stats
        self.total_routes: int = 0
        self.route_history: list[tuple[str, list[str], float]] = []  # (query, cortices, latency)

    # ── Cortex management ────────────────────────────────

    def add_cortex(self, cortex: MemoryCortex):
        """Register a new cortex."""
        if cortex.config.name in {c.config.name for c in self.cortices}:
            raise ValueError(f"Cortex '{cortex.config.name}' already exists")
        self.cortices.append(cortex)

    def remove_cortex(self, name: str):
        """Remove a cortex by name."""
        self.cortices = [c for c in self.cortices if c.config.name != name]

    def get_cortex(self, name: str) -> MemoryCortex | None:
        for c in self.cortices:
            if c.config.name == name:
                return c
        return None

    @property
    def default_cortex(self) -> MemoryCortex:
        """Lazily create a default fallback cortex."""
        if self._default_cortex is None:
            self._default_cortex = MemoryCortex(CortexConfig(
                name="general",
                description="Default fallback cortex for uncategorized memories",
                domain_keywords=[],
            ))
        return self._default_cortex

    # ── Routing ──────────────────────────────────────────

    def route(self, query: str = "",
              query_embedding: list[float] | None = None,
              max_cortices: int = 3,
              min_score: float = 0.1) -> list[RouteResult]:
        """Route a query to the most relevant cortices.

        Only 1-3 cortices are activated. If no cortex scores above min_score,
        the default cortex is used as fallback.

        Returns list of (cortex, score), sorted by score descending.
        """
        t0 = time.perf_counter()

        if not self.cortices:
            # No cortices registered — use default
            result = RouteResult(
                cortex=self.default_cortex,
                score=1.0,
                reasoning="No cortices registered, using default",
            )
            return [result]

        # Score each cortex
        embedder = self._get_embedder()
        if query_embedding is None and query:
            query_embedding = embedder.encode(query)

        scored: list[RouteResult] = []
        for cortex in self.cortices:
            score = cortex.route(
                query_embedding=query_embedding,
                query_text=query,
            )
            if score >= min_score:
                scored.append(RouteResult(
                    cortex=cortex,
                    score=score,
                    reasoning=f"route_score={score:.3f}",
                ))

        scored.sort(key=lambda r: -r.score)
        scored = scored[:max_cortices]

        # Fallback to default cortex if nothing matched
        if not scored:
            scored = [RouteResult(
                cortex=self.default_cortex,
                score=0.5,
                reasoning="Fallback — no cortex matched query",
            )]

        latency = (time.perf_counter() - t0) * 1000
        self.total_routes += 1
        self.route_history.append((
            query[:80],
            [r.cortex.config.name for r in scored],
            latency,
        ))

        return scored

    def recall(self, query: str = "",
               context=None,
               max_items: int = 10,
               query_embedding: list[float] | None = None) -> list[tuple[MemoryCortex, 'MemoryContext']]:
        """Route and recall from activated cortices.

        Only the activated cortices participate in retrieval — everything
        else is completely dormant.

        Returns list of (cortex, MemoryContext) from each activated cortex.
        """
        from .scheduler import AgentContext
        if context is None:
            context = AgentContext(task_type="conversation")

        routes = self.route(query, query_embedding=query_embedding)

        results: list[tuple[MemoryCortex, 'MemoryContext']] = []
        for route_result in routes:
            ctx = route_result.cortex.recall(
                query=query,
                context=context,
                max_items=max_items,
            )
            results.append((route_result.cortex, ctx))

        return results

    # ── Cortex creation ──────────────────────────────────

    def find_or_create_cortex(self, name: str,
                              domain_keywords: list[str] | None = None,
                              **kwargs) -> MemoryCortex:
        """Find an existing cortex or create a new one.

        If the cortex already exists, returns it. Otherwise creates a new
        cortex with the given domain keywords and config overrides.
        """
        existing = self.get_cortex(name)
        if existing:
            return existing

        cortex_config = CortexConfig(
            name=name,
            domain_keywords=domain_keywords or [],
            **kwargs,
        )
        cortex = MemoryCortex(cortex_config, self.cfg)
        self.add_cortex(cortex)
        return cortex

    # ── Health ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total_anchors = sum(len(c.graph.anchors) for c in self.cortices)
        return {
            "cortices": len(self.cortices),
            "total_anchors": total_anchors,
            "total_routes": self.total_routes,
            "recent_routes": self.route_history[-5:] if self.route_history else [],
        }

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder
