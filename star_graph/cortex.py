"""Memory Cortex — independent domain-specific memory brain region.

Each Cortex is a self-contained memory system:
- Own StarGraph + ANN index
- Own decay curve and retention thresholds
- Own retrieval policy and token budget
- Independent sleep/consolidation cycle

Cortices do NOT directly connect to each other. Cross-cortex links
go through HubNodes in the HubLayer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, MemoryState
from .graph import StarGraph
from .config import Config
from .index import ANNIndex
from .scheduler import CognitiveMemoryScheduler, AgentContext, MemoryContext


@dataclass
class CortexConfig:
    """Per-cortex hyperparameters — each cortex tunes its own memory dynamics."""
    name: str = "default"
    description: str = ""
    # Domain keywords for routing
    domain_keywords: list[str] = field(default_factory=list)
    domain_embedding: list[float] | None = None  # centroid of this domain
    # Decay parameters (can differ from global)
    decay_half_life_days: float = 30.0
    retention_threshold: float = 0.15
    # Token budget for this cortex's retrieval output
    token_budget: int = 2000
    # Retrieval policy
    retrieval_top_k: int = 20
    phase_weight: float = 0.3
    # Consolidation frequency (how often to run local sleep)
    consolidate_interval_hours: float = 24.0
    # Max anchors before triggering auto-consolidation
    max_anchors_before_consolidate: int = 1000
    # Route matching
    route_semantic_weight: float = 0.6
    route_keyword_weight: float = 0.3
    route_recency_weight: float = 0.1


class MemoryCortex:
    """Independent memory brain region with its own graph and retrieval.

    Each cortex models a cognitive domain: "dev", "finance", "personal", etc.
    The cortex has its own embedding space, so "Python" in DevCortex and
    "Python" in FinanceCortex are semantically distinct.

    Usage:
        dev_cortex = MemoryCortex(CortexConfig(
            name="dev",
            domain_keywords=["code", "python", "debug", "api", "docker"],
        ))
        dev_cortex.remember("Debugged Redis timeout — pool size was 10")
        result = dev_cortex.recall("Redis connection issues")
    """

    def __init__(self, config: CortexConfig,
                 global_cfg: Config | None = None):
        self.config = config
        self.global_cfg = global_cfg or Config.get()
        self.graph = StarGraph()
        self._index: ANNIndex | None = None
        self._scheduler: CognitiveMemoryScheduler | None = None
        self._embedder = None
        self._centroid: list[float] | None = None
        self._centroid_stale = True

        # Stats
        self.created_at: float = time.time()
        self.last_accessed_at: float = time.time()
        self.total_anchors_added: int = 0
        self.total_recalls: int = 0
        self.total_sleep_cycles: int = 0

    # ── Routing ──────────────────────────────────────────

    def route(self, query_embedding: list[float] | None = None,
              query_text: str = "") -> float:
        """Score how well a query matches this cortex (0..1).

        Combines:
        - Semantic similarity to cortex centroid
        - Keyword overlap with domain keywords
        - Recency bonus (recently active cortices get boost)
        """
        scores: list[float] = []
        weights = self.config.route_semantic_weight, \
                  self.config.route_keyword_weight, \
                  self.config.route_recency_weight
        w_sem, w_kw, w_rec = weights

        # 1. Semantic similarity to cortex centroid
        if query_embedding and self.centroid:
            sem_score = _cosine_sim(query_embedding, self.centroid)
        elif query_embedding:
            sem_score = 0.3  # no centroid yet, neutral
        else:
            sem_score = 0.3
        scores.append((sem_score, w_sem))

        # 2. Keyword match with domain keywords
        if self.config.domain_keywords and query_text:
            q_words = set(query_text.lower().split())
            kw_matches = sum(1 for kw in self.config.domain_keywords
                            if kw.lower() in q_words
                            or any(w in q_words for w in kw.lower().split()))
            kw_score = min(1.0, kw_matches / max(1, len(self.config.domain_keywords)) * 2)
        else:
            kw_score = 0.0
        scores.append((kw_score, w_kw))

        # 3. Recency bonus
        hours_idle = (time.time() - self.last_accessed_at) / 3600
        recency = math.exp(-hours_idle / 24)  # decay over 24 hours
        scores.append((recency, w_rec))

        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in scores) / total_weight

    @property
    def centroid(self) -> list[float] | None:
        """Lazily compute the embedding centroid of all anchors in this cortex.

        The centroid represents the "center of mass" of this domain's knowledge.
        """
        if self._centroid_stale or self._centroid is None:
            self._recompute_centroid()
        return self._centroid

    def _recompute_centroid(self):
        """Recompute centroid from all anchor embeddings."""
        embeddings = [a.embedding for a in self.graph.anchors.values()
                     if a.embedding and a.is_retrievable]
        if not embeddings:
            self._centroid = None
        else:
            dim = len(embeddings[0])
            self._centroid = [0.0] * dim
            for emb in embeddings:
                for i, v in enumerate(emb):
                    self._centroid[i] += v
            for i in range(dim):
                self._centroid[i] /= len(embeddings)
        self._centroid_stale = False

    # ── Core operations ──────────────────────────────────

    def remember(self, text: str, *,
                 source_session: str = "",
                 tags: list[str] | None = None,
                 emotional_valence: float = 0.0,
                 importance: float = 0.5,
                 connect_to: list[str] | None = None,
                 **kwargs) -> Anchor:
        """Store a memory in this cortex."""
        embedder = self._get_embedder()
        embedding = embedder.encode(text)

        anchor = Anchor.create(
            text=text,
            source_session=source_session,
            embedding=embedding,
            emotional_valence=emotional_valence,
            importance=importance,
            tags=tags,
            **kwargs,
        )
        self.graph.add_anchor(anchor)
        self._centroid_stale = True
        self.total_anchors_added += 1

        if connect_to:
            for target_id in connect_to:
                if target_id in self.graph.anchors:
                    tgt_emb = self.graph.anchors[target_id].embedding
                    sim = _cosine_sim(embedding, tgt_emb) if embedding and tgt_emb else 0.5
                    self.graph.add_edge(anchor.id, target_id, weight=sim)

        return anchor

    def recall(self, query: str = "",
               context: AgentContext | None = None,
               max_items: int = 10) -> MemoryContext:
        """Retrieve memories from this cortex."""
        if context is None:
            context = AgentContext(task_type="conversation")
        self.last_accessed_at = time.time()
        self.total_recalls += 1
        return self.scheduler.retrieve(context, query, max_items)

    def forget(self, anchor_id: str) -> Anchor | None:
        """Remove a memory from this cortex."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor:
            self.graph.remove_anchor(anchor_id)
            self._centroid_stale = True
        return anchor

    def consolidate(self) -> dict:
        """Run a local sleep cycle on this cortex."""
        from .sleep import SleepCycle
        sc = SleepCycle(self.graph)
        report = sc.run_phased()
        self.total_sleep_cycles += 1
        self._centroid_stale = True
        self._index = None  # invalidate index after graph changes
        return report

    # ── Lazy-loaded subsystems ───────────────────────────

    @property
    def scheduler(self) -> CognitiveMemoryScheduler:
        if self._scheduler is None:
            self._scheduler = CognitiveMemoryScheduler(
                self.graph, self.global_cfg,
                working_memory=None,
            )
        return self._scheduler

    @property
    def index(self) -> ANNIndex:
        if self._index is None:
            self._index = ANNIndex(self.graph)
        return self._index

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    # ── Health ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "name": self.config.name,
            "anchors": len(self.graph.anchors),
            "edges": len(self.graph.edges),
            "schemas": len(self.graph.schemas),
            "sleep_cycles": self.total_sleep_cycles,
            "total_added": self.total_anchors_added,
            "total_recalls": self.total_recalls,
            "centroid_dim": len(self._centroid) if self._centroid else 0,
            "hours_idle": (time.time() - self.last_accessed_at) / 3600,
        }

    @property
    def is_overdue_for_consolidation(self) -> bool:
        hours_since = (time.time() - self.created_at) / 3600
        return hours_since >= self.config.consolidate_interval_hours

    @property
    def is_near_capacity(self) -> bool:
        return len(self.graph.anchors) >= self.config.max_anchors_before_consolidate


# ── Helper ───────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
