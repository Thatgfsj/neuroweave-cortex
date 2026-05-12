"""Memory Cortex — independent domain-specific memory brain region.

Each Cortex is a self-contained memory system:
- Own StarGraph + ANN index
- Own decay curve and retention thresholds
- Own retrieval policy and token budget
- Independent sleep/consolidation cycle
- Internal Segments (西瓜块) — clusters by semantic density, bridge to hub nodes

Cortices do NOT directly connect to each other. Cross-cortex links
go through HubNodes in the HubLayer, connected via Segments.
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


@dataclass
class Segment:
    """A cluster of memories within a cortex — the "西瓜块" (watermelon slice).

    Segments group memories by semantic density range within a cortex.
    Each segment has a centroid embedding and can link to HubNodes in the
    HubSphere. Segments are the bridge between cortex-local memory and
    cross-domain abstraction.

    A cortex typically has 3-10 segments, each representing a density band:
    - Low density (0.0-0.3): raw episodic events
    - Medium density (0.3-0.7): compressed summaries
    - High density (0.7-1.0): abstract rules/patterns
    """
    id: str
    cortex_name: str
    centroid: list[float] | None = None       # mean embedding of nodes in this segment
    density_band: tuple[float, float] = (0.0, 1.0)  # (min_density, max_density)
    node_ids: list[str] = field(default_factory=list)
    hub_links: list[str] = field(default_factory=list)  # linked HubNode IDs
    summary: str = ""
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.node_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.node_ids) == 0

    def add_node(self, anchor_id: str, embedding: list[float] | None = None):
        """Add a node to this segment and update centroid."""
        if anchor_id not in self.node_ids:
            self.node_ids.append(anchor_id)
        self.last_updated = time.time()

    def remove_node(self, anchor_id: str):
        if anchor_id in self.node_ids:
            self.node_ids.remove(anchor_id)
        self.last_updated = time.time()

    def link_hub(self, hub_id: str):
        """Link a HubNode to this segment."""
        if hub_id not in self.hub_links:
            self.hub_links.append(hub_id)


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

        # Segments — clusters by semantic density
        self._segments: dict[str, Segment] = {}
        self._init_default_segments()

        # Stats
        self.created_at: float = time.time()
        self.last_accessed_at: float = time.time()
        self.total_anchors_added: int = 0
        self.total_recalls: int = 0
        self.total_sleep_cycles: int = 0

    def _init_default_segments(self):
        """Create default density-band segments."""
        bands = [
            ("seg_raw", (0.0, 0.3), "Raw episodic events"),
            ("seg_compressed", (0.3, 0.7), "Compressed summaries"),
            ("seg_abstract", (0.7, 1.0), "Abstract rules and patterns"),
        ]
        for seg_id, band, desc in bands:
            self._segments[seg_id] = Segment(
                id=f"{self.config.name}_{seg_id}",
                cortex_name=self.config.name,
                density_band=band,
                summary=desc,
            )

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
                 cortex_path: str = "",
                 **kwargs) -> Anchor:
        """Store a memory in this cortex. Auto-assigns to a density segment."""
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
        anchor.cortex_path = cortex_path or self.config.name
        self.graph.add_anchor(anchor)
        self._assign_to_segment(anchor)
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

    # ── Segment management ──────────────────────────────

    def _assign_to_segment(self, anchor: Anchor) -> str:
        """Assign an anchor to the appropriate density-band segment.

        Returns the segment ID the anchor was assigned to.
        """
        density = anchor.semantic_density
        for seg in self._segments.values():
            lo, hi = seg.density_band
            if lo <= density <= hi:
                seg.add_node(anchor.id, anchor.embedding)
                anchor.segment_id = seg.id
                return seg.id

        # Fallback to raw segment
        raw_seg = self._segments.get(f"{self.config.name}_seg_raw")
        if raw_seg:
            raw_seg.add_node(anchor.id, anchor.embedding)
            anchor.segment_id = raw_seg.id
            return raw_seg.id
        return ""

    def get_segments(self) -> list[Segment]:
        """Get all segments in this cortex."""
        return list(self._segments.values())

    def get_segment(self, segment_id: str) -> Segment | None:
        return self._segments.get(segment_id)

    def get_segment_for_hub(self, density_band: str = "compressed") -> Segment | None:
        """Get the segment that should connect to hub nodes.

        Typically the 'compressed' segment (medium density), as it represents
        summarized knowledge most useful for cross-domain abstraction.
        """
        band_map = {"raw": "seg_raw", "compressed": "seg_compressed", "abstract": "seg_abstract"}
        seg_key = band_map.get(density_band, "seg_compressed")
        return self._segments.get(f"{self.config.name}_{seg_key}")

    def rebuild_segments(self):
        """Rebuild all segments from current anchors. Called after sleep."""
        # Clear all segment node lists
        for seg in self._segments.values():
            seg.node_ids.clear()

        # Re-assign all anchors
        for anchor in self.graph.anchors.values():
            if anchor.is_retrievable:
                self._assign_to_segment(anchor)

        # Update centroids
        for seg in self._segments.values():
            if seg.node_ids:
                embeddings = []
                for nid in seg.node_ids:
                    anchor = self.graph.anchors.get(nid)
                    if anchor and anchor.embedding:
                        embeddings.append(anchor.embedding)
                if embeddings:
                    dim = len(embeddings[0])
                    seg.centroid = [0.0] * dim
                    for emb in embeddings:
                        for i, v in enumerate(emb):
                            seg.centroid[i] += v
                    for i in range(dim):
                        seg.centroid[i] /= len(embeddings)

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
