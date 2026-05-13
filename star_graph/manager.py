"""MemoryManager — high-level facade for the cognitive memory runtime.

Single entry point for AI agents. Wraps graph CRUD, retrieval, sleep, evolution,
online consolidation, ghost revival, abstraction, and metrics.

Usage:
    manager = MemoryManager()
    manager.remember("User prefers concise answers", tags=["preference", "style"])
    manager.remember("Debugged Redis timeout — pool size was 10, fixed to 20",
                     tags=["debug", "redis"])

    # Context-aware recall
    ctx = AgentContext(task_type="debugging", active_goals=["fix Redis"])
    memories = manager.recall("Redis connection issues", context=ctx)

    # Cognitive maintenance
    manager.micro_consolidate()   # light online update
    report = manager.sleep()      # full 5-phase consolidation

    # Persistence
    manager.save("agent_memory.db")
    manager.load("agent_memory.db")

    # Health
    manager.print_health()
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, MemoryState
from .graph import StarGraph, Edge
from .config import Config
from .scheduler import CognitiveMemoryScheduler, AgentContext, MemoryContext, MemoryItem, MemoryType
from .working_memory import WorkingMemory, WorkingMemoryEntry
from .cortex import MemoryCortex, CortexConfig
from .router import CortexRouter, RouteResult
from .gate import MemoryGate
from .timespine import TimeSpine
from .cascade import CascadeRecall
from .hub import HubLayer, HubNode
from .brain_sphere import BrainSphere, HubCenter
from .symbolic_filter import SymbolicFilter
from .evolution import MemoryEvolutionEngine
from .ghost import GhostSubsystem
from .abstraction import AbstractionEngine
from .community import Community, CommunityHealth, CommunityDetection
from .metrics import CognitiveMetrics
from .raw_buffer import RawBuffer, RawChunk
from .dual_channel import DualChannelRetriever, DualChannelOutput
from .exact_cache import ExactMatchCache
from .cost_estimator import SleepCostEstimator, CostEstimate
from .snapshot import SnapshotManager, SnapshotMeta
from .tracing import MemoryTracer, get_tracer, trace_recall
from .survival import SurvivalFunction, SurvivalRegistry
from .multimodal import (
    MultimodalEmbeddingProvider, MultimodalAnchor, CrossModalRetriever, CrossModalResult,
)


@dataclass
class ManagerStats:
    """Snapshot of the memory system state."""
    anchors: int = 0
    edges: int = 0
    ghosts: int = 0
    schemas: int = 0
    abstracts: int = 0
    working_memory: int = 0
    cortices: int = 0
    hubs: int = 0
    clusters: int = 0
    sleep_cycles: int = 0
    total_evolutions: int = 0
    uptime_seconds: float = 0.0
    cognitive_health: dict | None = None


class MemoryManager:
    """High-level facade over the entire cognitive memory system.

    This is the API that AI agents use — everything else is implementation detail.
    """

    def __init__(self, graph: StarGraph | None = None,
                 config: Config | None = None,
                 storage_path: str = ""):
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
        self._hublayer: HubLayer | None = None

        # Raw chunk buffer — uncompressed short-term memory tier (L0)
        self._raw_buffer: RawBuffer | None = None

        # Dual-channel retriever (System-1 + System-2)
        self._dual_channel: DualChannelRetriever | None = None

        # Exact match cache — deterministic O(1) entity-pair lookup bypass
        self._exact_cache: ExactMatchCache | None = None

        # Micro-sleep scheduler — incremental non-blocking consolidation
        self._micro_sleep = None

        # Snapshot manager — versioned state snapshots with WAL
        self._snapshot_mgr: SnapshotManager | None = None

        # Tracer — lightweight observability spans
        self.tracer: MemoryTracer = get_tracer(enabled=True)

        # Survival function — configurable memory decay curve
        self._survival_fn: SurvivalFunction | None = None
        # Set as class-level default for all Anchor and GhostNode instances
        Anchor.set_survival_function(self.survival_function)
        from .ghost import GhostNode
        GhostNode.set_survival_function(self.survival_function)

        # Multimodal — optional image/text cross-modal embedding
        self._multimodal_provider: MultimodalEmbeddingProvider | None = None
        self._cross_modal_retriever: CrossModalRetriever | None = None

        # Streaming — continuous memory ingestion buffer
        self._streaming_buffer = None

        # Stats
        self.sleep_cycles: int = 0
        self.total_evolutions: int = 0

    # ── Subsystem access (lazy init) ──────────────────────────

    @property
    def scheduler(self) -> CognitiveMemoryScheduler:
        if self._scheduler is None:
            self._scheduler = CognitiveMemoryScheduler(
                self.graph, self.cfg,
                working_memory=self.working_memory,
                sym_filter=self.sym_filter)
        return self._scheduler

    @property
    def evolution(self) -> MemoryEvolutionEngine:
        if self._evolution is None:
            self._evolution = MemoryEvolutionEngine(self.graph, self.cfg)
        return self._evolution

    @property
    def ghosts(self) -> GhostSubsystem:
        if self._ghosts is None:
            self._ghosts = GhostSubsystem()
        return self._ghosts

    @property
    def abstraction(self) -> AbstractionEngine:
        if self._abstraction is None:
            self._abstraction = AbstractionEngine()
        return self._abstraction

    @property
    def community_detection(self) -> CommunityDetection:
        if self._community_detection is None:
            self._community_detection = CommunityDetection()
        return self._community_detection

    @property
    def raw_buffer(self) -> RawBuffer:
        if self._raw_buffer is None:
            rb_cfg = getattr(self.cfg, 'raw_buffer', None)
            max_sessions = getattr(rb_cfg, 'max_sessions', 2) if rb_cfg else 2
            max_chunks = getattr(rb_cfg, 'max_chunks_per_session', 500) if rb_cfg else 500
            self._raw_buffer = RawBuffer(
                max_sessions=max_sessions,
                max_chunks_per_session=max_chunks,
            )
        return self._raw_buffer

    @property
    def dual_channel(self) -> DualChannelRetriever:
        if self._dual_channel is None:
            dc_cfg = getattr(self.cfg, 'dual_channel', None)
            s1_threshold = getattr(dc_cfg, 's1_confidence_threshold', 0.35) if dc_cfg else 0.35
            self._dual_channel = DualChannelRetriever(
                self.graph,
                s1_confidence_threshold=s1_threshold,
            )
        return self._dual_channel

    @property
    def compressor(self):
        """Lazy-init multi-level compression engine."""
        if getattr(self, '_compress_engine', None) is None:
            from .compression import MultiLevelCompressor
            self._compress_engine = MultiLevelCompressor()
        return self._compress_engine

    @property
    def exact_cache(self) -> ExactMatchCache:
        if self._exact_cache is None:
            ec_cfg = getattr(self.cfg, 'exact_cache', None)
            max_per_key = getattr(ec_cfg, 'max_entries_per_key', 5) if ec_cfg else 5
            self._exact_cache = ExactMatchCache(max_entries_per_key=max_per_key)
        return self._exact_cache

    @property
    def survival_function(self) -> SurvivalFunction:
        """Lazy-init survival function from config."""
        if self._survival_fn is None:
            self._survival_fn = SurvivalRegistry.from_config(self.cfg)
        return self._survival_fn

    @property
    def streaming_buffer(self):
        """Lazy-init streaming memory buffer for continuous ingestion."""
        if self._streaming_buffer is None:
            from .streaming import StreamingMemoryBuffer
            stream_cfg = getattr(self.cfg, 'streaming', None)
            max_buf = getattr(stream_cfg, 'max_buffer', 500) if stream_cfg else 500
            flush_sec = getattr(stream_cfg, 'flush_interval_s', 30.0) if stream_cfg else 30.0
            batch_sz = getattr(stream_cfg, 'batch_size', 20) if stream_cfg else 20
            dedup_th = getattr(stream_cfg, 'dedup_threshold', 0.85) if stream_cfg else 0.85
            max_ses = getattr(stream_cfg, 'max_sessions', 10) if stream_cfg else 10
            self._streaming_buffer = StreamingMemoryBuffer(
                self,
                max_buffer=max_buf,
                flush_interval_s=flush_sec,
                batch_size=batch_sz,
                dedup_threshold=dedup_th,
                max_sessions=max_ses,
            )
        return self._streaming_buffer

    @property
    def multimodal(self) -> MultimodalEmbeddingProvider:
        """Lazy-init multimodal embedding provider."""
        if self._multimodal_provider is None:
            self._multimodal_provider = MultimodalEmbeddingProvider()
        return self._multimodal_provider

    @property
    def cross_modal_retriever(self) -> CrossModalRetriever:
        """Lazy-init cross-modal retriever."""
        if self._cross_modal_retriever is None:
            self._cross_modal_retriever = CrossModalRetriever(self.multimodal)
        return self._cross_modal_retriever

    @property
    def metrics(self) -> CognitiveMetrics:
        if self._metrics is None:
            self._metrics = CognitiveMetrics(self.graph)
        return self._metrics

    @property
    def router(self) -> CortexRouter:
        if self._router is None:
            self._router = CortexRouter(self.cfg, brain=self.brain)
        return self._router

    @property
    def gate(self) -> MemoryGate:
        if self._gate is None:
            self._gate = MemoryGate(k=self.cfg.gate.k if hasattr(self.cfg, 'gate') else 20, config=self.cfg)
        return self._gate

    @property
    def timespine(self) -> TimeSpine:
        if self._timespine is None:
            self._timespine = TimeSpine(
                max_clusters_per_day=getattr(
                    getattr(self.cfg, 'timespine', None), 'max_clusters_per_day', 10))
        return self._timespine

    @property
    def cascade(self) -> CascadeRecall:
        if self._cascade is None:
            self._cascade = CascadeRecall(self.graph)
        return self._cascade

    @property
    def hublayer(self) -> HubLayer:
        if self._hublayer is None:
            hub_cfg = getattr(self.cfg, 'hub', None)
            max_deg = getattr(hub_cfg, 'max_degree_per_shard', 50) if hub_cfg else 50
            self._hublayer = HubLayer(max_degree_per_shard=max_deg)
        return self._hublayer

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    # ── CRUD: Basic memory operations ─────────────────────────

    def remember(self, text: str, *, source_session: str = "",
                 tags: list[str] | None = None,
                 emotional_valence: float = 0.0,
                 importance: float = 0.5,
                 connect_to: list[str] | None = None,
                 edge_type: str = "topical",
                 anchor_id: str | None = None,
                 **vec_kw) -> Anchor:
        """Store a new memory. Returns the anchor.

        Args:
            text: Memory content
            source_session: Which session/conversation this came from
            tags: Semantic tags for classification and retrieval
            emotional_valence: -1..+1 emotional charge
            importance: 0..1 initial importance
            connect_to: List of existing anchor IDs to connect this to
            edge_type: Type of edge to create ('topical', 'causal', 'temporal')
            anchor_id: Optional explicit ID (auto-generated if None)
        """
        embedder = self._get_embedder()
        embedding = embedder.encode(text)

        if anchor_id:
            anchor = Anchor(
                id=anchor_id, text=text, embedding=embedding,
                source_session=source_session,
                tags=tags or [],
                vector=Anchor.__dataclass_fields__["vector"].default_factory(),
                **vec_kw,
            )
        else:
            anchor = Anchor.create(
                text=text, source_session=source_session,
                embedding=embedding,
                emotional_valence=emotional_valence,
                importance=importance,
                tags=tags,
                **vec_kw,
            )

        self.graph.add_anchor(anchor)

        # Dual-write: also store raw uncompressed chunk in L0 buffer
        self.raw_buffer.add(
            text=text, session_id=source_session,
            embedding=embedding, tags=tags,
            importance=importance, anchor_id=anchor.id,
        )

        # Harvest exact match keys into deterministic KV cache
        auto_harvest = True
        ec_cfg = getattr(self.cfg, 'exact_cache', None)
        if ec_cfg:
            auto_harvest = getattr(ec_cfg, 'auto_harvest', True)
        if auto_harvest:
            self.exact_cache.harvest_from_anchor(anchor)

        # Auto-connect to specified anchors
        if connect_to:
            for target_id in connect_to:
                if target_id in self.graph.anchors:
                    target_emb = self.graph.anchors[target_id].embedding
                    sim = 0.5
                    if target_emb and embedding:
                        sim = _cosine_sim(embedding, target_emb)
                    self.graph.add_edge(
                        anchor.id, target_id,
                        weight=sim, edge_type=edge_type,
                    )

        return anchor

    # ── Structural intent keywords for System-2 auto-trigger ──
    _SYSTEM2_KEYWORDS = {
        'all', 'every', 'list', 'enumerate', 'each',
        'which', 'select', 'what kind', 'what type',
        'before', 'after', 'last', 'first', 'previous', 'next',
        'earlier', 'later', 'since', 'until',
        'how many', 'what caused', 'why did', 'steps', 'sequence',
        'summarize', 'summary', 'overview', 'history', 'timeline', 'pattern',
        'across sessions', 'previously', 'past conversations',
    }

    def recall(self, query: str = "",
               context: AgentContext | None = None,
               max_items: int = 10) -> MemoryContext:
        """Multi-path retrieval with tracing instrumentation and auto System-2 trigger.

        Path 0 (Exact Cache): Deterministic O(1) entity-pair lookup.
        Path A (L0 Raw Buffer): BM25 + vector search on uncompressed chunks.
        Path B (L1-L5 Graph): Dimensional descent through structured anchors.

        Auto-triggers System-2 (goal-directed traversal) when:
        - Query contains structural intent keywords (all/which/before/last/...)
        - System-1 confidence drops below 0.35
        """
        if context is None:
            context = AgentContext(task_type="conversation")

        # Detect if System-2 is warranted before running System-1
        # Check multi-word phrases first, then single-word with word boundaries
        query_lower = query.lower()
        s2_keyword = False
        for kw in self._SYSTEM2_KEYWORDS:
            if ' ' in kw:
                if kw in query_lower:
                    s2_keyword = True
                    break
            else:
                import re
                if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                    s2_keyword = True
                    break

        if s2_keyword:
            return self._system2_recall(query, context, max_items,
                                        trigger_reason="structural_keyword")

        embedder = self._get_embedder()
        query_emb = embedder.encode(query) if query else None
        t_start = time.time()

        # Path 0: Exact cache lookup (deterministic O(1) bypass)
        t0 = time.time()
        exact_results: list[MemoryItem] = []
        if query:
            query_keys = self.exact_cache.query_keys(query)
            for key in query_keys:
                entries = self.exact_cache.get(key)
                for entry in entries:
                    anchor = self.graph.anchors.get(entry.anchor_id)
                    if anchor and anchor.is_retrievable:
                        exact_results.append(MemoryItem(
                            anchor=anchor,
                            relevance_score=0.95 + entry.confidence * 0.05,
                            memory_type=MemoryType.SEMANTIC,
                            compression_level=0,
                            compressed_text=entry.text,
                        ))
            for key in query_keys:
                wm_entries = self.working_memory.get_exact(key)
                for wme in wm_entries:
                    exact_results.append(MemoryItem(
                        anchor=None,
                        relevance_score=0.92,
                        memory_type=MemoryType.WORKING,
                        compression_level=0,
                        compressed_text=wme.text[:200],
                    ))
        exact_ms = (time.time() - t0) * 1000

        # Path A: Raw buffer search (L0 — highest priority after exact cache)
        t0 = time.time()
        raw_results = self.raw_buffer.search(
            query=query, query_embedding=query_emb,
            top_k=max_items,
            session_id=context.session_id if context else "",
        )
        raw_ms = (time.time() - t0) * 1000

        # Path B: Graph dimensional descent (L1-L5)
        t0 = time.time()
        graph_result = self.retrieve_with_descent(
            query=query, context=context, max_items=max_items)
        graph_ms = (time.time() - t0) * 1000

        # Merge: exact cache → raw buffer (recent, uncompressed) → graph (compressed)
        merged_items = list(exact_results)
        seen_texts = {self._normalize_text(item.compressed_text) for item in merged_items
                      if hasattr(item, 'compressed_text') and item.compressed_text}

        for chunk, score in raw_results:
            norm_text = self._normalize_text(chunk.text[:120])
            if norm_text and norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            merged_items.append(MemoryItem(
                anchor=None,
                relevance_score=score,
                memory_type=MemoryType.WORKING,
                compression_level=0,
                compressed_text=chunk.text[:200],
            ))

        for item in graph_result.get("items", []):
            text = getattr(item, 'compressed_text', '') or ''
            norm_text = self._normalize_text(text)
            if norm_text and norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            merged_items.append(item)

        merged_items.sort(key=lambda i: i.relevance_score, reverse=True)
        merged_items = merged_items[:max_items]

        # Auto-trigger System-2 if System-1 confidence is low
        s1_confidence = graph_result.get("confidence", 1.0)
        if s1_confidence < 0.35 and len(merged_items) < max_items:
            return self._system2_recall(query, context, max_items,
                                        trigger_reason="low_confidence")

        layers_visited = (["exact_cache"] if exact_results else []) + ["raw_buffer"] + graph_result.get("layers_visited", [])
        total_ms = (time.time() - t_start) * 1000

        # Trace: record full recall as a completed span
        self.tracer.record("recall", attributes={
            "query": query[:200],
            "max_items": max_items,
            "exact_hits": len(exact_results),
            "raw_chunks": len(raw_results),
            "graph_items": len(graph_result.get("items", [])),
            "final_count": len(merged_items),
            "layers_visited": str(layers_visited)[:300],
            "channel": "system1",
            "exact_ms": round(exact_ms, 3),
            "raw_ms": round(raw_ms, 3),
            "graph_ms": round(graph_ms, 3),
            "total_ms": round(total_ms, 3),
        }, duration_ms=total_ms)

        return MemoryContext(
            items=merged_items,
            memory_summary=f"System-1 | Layer: {graph_result.get('layer_used', 'graph')}, visited: {layers_visited}",
            active_patterns=[],
            relevant_facts=[],
            reasoning_traces=[
                f"exact_hits={len(exact_results)}",
                f"raw_chunks={len(raw_results)}",
                f"graph_items={len(graph_result.get('items', []))}",
                f"merged={len(merged_items)}",
            ],
        )

    def _system2_recall(self, query: str, context: AgentContext,
                        max_items: int, trigger_reason: str) -> MemoryContext:
        """Run System-2 (goal-directed) recall via DualChannelRetriever.

        Triggered automatically when structural keywords are detected
        or System-1 confidence is below threshold.
        """
        t_start = time.time()
        dc_output = self.dual_channel.retrieve(
            query=query, context=context,
            query_embedding=None, max_items=max_items,
        )
        total_ms = (time.time() - t_start) * 1000

        self.tracer.record("recall", attributes={
            "query": query[:200],
            "max_items": max_items,
            "final_count": len(dc_output.items),
            "channel": dc_output.active_channel,
            "s2_triggered": dc_output.s2_triggered,
            "trigger_reason": trigger_reason,
            "s1_confidence": round(dc_output.s1_confidence, 3),
            "s2_confidence": round(dc_output.s2_confidence, 3),
            "total_ms": round(total_ms, 3),
        }, duration_ms=total_ms)

        channel_label = dc_output.active_channel
        if dc_output.s2_triggered:
            channel_label += f" (S2: {trigger_reason})"

        return MemoryContext(
            items=dc_output.items[:max_items],
            memory_summary=f"Dual-Channel [{channel_label}]",
            active_patterns=[],
            relevant_facts=[],
            reasoning_traces=[
                f"channel={dc_output.active_channel}",
                f"s2_triggered={dc_output.s2_triggered}",
                f"reason={trigger_reason}",
                f"s1_confidence={dc_output.s1_confidence:.3f}",
                f"s2_confidence={dc_output.s2_confidence:.3f}",
            ],
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for dedup comparison."""
        import re
        return re.sub(r'\s+', ' ', text.lower().strip())[:100]

    def dual_recall(self, query: str = "",
                    context: AgentContext | None = None,
                    max_items: int = 10) -> DualChannelOutput:
        """Dual-channel recall: System-1 (fast association) + System-2 (goal-directed).

        System-2 auto-triggers when:
        - Query contains structural intent words (all, which, before, list, ...)
        - System-1 confidence drops below threshold (< 0.35)

        Returns DualChannelOutput with items, channel info, and trigger metadata.
        """
        if context is None:
            context = AgentContext(task_type="conversation")
        embedder = self._get_embedder()
        query_emb = embedder.encode(query) if query else None

        return self.dual_channel.retrieve(
            query=query, context=context,
            query_embedding=query_emb, max_items=max_items,
        )

    def forget(self, anchor_id: str, create_ghost: bool = True) -> Anchor | None:
        """Remove a memory. If create_ghost=True, preserves a latent trace."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return None

        if create_ghost:
            residual = {}
            for neighbor_id in self.graph._adjacency.get(anchor_id, set()):
                edge_key = self.graph._key(anchor_id, neighbor_id)
                edge = self.graph.edges.get(edge_key)
                if edge:
                    residual[neighbor_id] = edge.weight * 0.3
            self.ghosts.create(anchor, residual)

        self.graph.remove_anchor(anchor_id)
        return anchor

    # ── Working Memory (short-term buffer) ───────────────────

    def remember_working(self, text: str, *,
                         importance: float = 0.5,
                         tags: list[str] | None = None,
                         source_session: str = "",
                         emotional_valence: float = 0.0) -> WorkingMemoryEntry:
        """Add an item to working memory — fast, ephemeral, high-priority.

        Working memory is checked BEFORE long-term memory during recall.
        Items auto-expire after the TTL (default 30 min).
        """
        embedder = self._get_embedder()
        embedding = embedder.encode(text)
        return self.working_memory.add(
            text=text,
            embedding=embedding,
            importance=importance,
            tags=tags,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )

    def get_working(self) -> list[WorkingMemoryEntry]:
        """Get all active working memory items."""
        return self.working_memory.get_all()

    def promote_working(self, entry: WorkingMemoryEntry) -> str | None:
        """Promote a working memory item to long-term storage.

        Returns the new anchor ID, or None if the entry is no longer valid.
        """
        if entry not in self.working_memory._entries:
            return None
        return self.working_memory.promote(entry, self)

    def clear_working_memory(self, session_id: str = ""):
        """Clear working memory. If session_id given, only clears that session."""
        if session_id:
            self.working_memory.clear_session(session_id)
        else:
            self.working_memory.clear()

    def update(self, anchor_id: str, text: str | None = None,
               tags: list[str] | None = None,
               importance: float | None = None) -> Anchor | None:
        """Update an existing memory with new information."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return None

        if text is not None:
            anchor.text = text
            embedder = self._get_embedder()
            anchor.embedding = embedder.encode(text)
        if tags is not None:
            anchor.tags = list(set(anchor.tags + tags))
        if importance is not None:
            anchor.vector.importance = max(0.0, min(1.0, importance))

        anchor.activate()
        return anchor

    # ── Multimodal operations ─────────────────────────────────

    def remember_image(self, image_path: str, caption: str = "",
                      tags: list[str] | None = None,
                      importance: float = 0.5,
                      source_session: str = "",
                      emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Store a memory with an image attachment.

        Creates a MultimodalAnchor with both text (caption) and image embeddings.
        With CLIP, enables cross-modal retrieval (text→image, image→text).
        Without CLIP, falls back to caption/tag-based search.
        """
        anchor = MultimodalAnchor.from_image(
            image_path=image_path,
            provider=self.multimodal,
            caption=caption,
            tags=tags,
            importance=importance,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )
        self.graph.add_anchor(anchor)

        # Also index raw buffer for L0 recall
        self.raw_buffer.add(
            text=caption or f"[image] {os.path.basename(image_path)}",
            session_id=source_session,
            embedding=anchor.embedding,
            tags=tags,
            importance=importance,
            anchor_id=anchor.id,
        )
        return anchor

    def remember_text_and_image(self, text: str, image_path: str,
                               tags: list[str] | None = None,
                               importance: float = 0.5,
                               source_session: str = "",
                               emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Store a memory with both text content and an associated image."""
        anchor = MultimodalAnchor.from_text_and_image(
            text=text,
            image_path=image_path,
            provider=self.multimodal,
            tags=tags,
            importance=importance,
            source_session=source_session,
            emotional_valence=emotional_valence,
        )
        self.graph.add_anchor(anchor)
        self.raw_buffer.add(
            text=text,
            session_id=source_session,
            embedding=anchor.embedding,
            tags=tags,
            importance=importance,
            anchor_id=anchor.id,
        )
        return anchor

    def cross_modal_recall(self, query: str = "",
                          query_image_path: str = "",
                          max_items: int = 10,
                          text_weight: float = 0.6,
                          image_weight: float = 0.4) -> list[CrossModalResult]:
        """Retrieve memories across text and image modalities.

        Combines anchors from main graph and all cortices.
        Text→text, text→image, image→text, image→image all supported.
        True cross-modal alignment requires CLIP.

        Args:
            query: Text query (optional if query_image_path given)
            query_image_path: Image query (optional if query_text given)
            max_items: Max results to return
            text_weight: Weight for text similarity in combined score
            image_weight: Weight for image similarity in combined score
        """
        # Collect all anchors from graph + cortices
        all_anchors = dict(self.graph.anchors)
        for cortex in self.router.cortices:
            all_anchors.update(cortex.graph.anchors)

        return self.cross_modal_retriever.retrieve(
            anchors=all_anchors,
            query_text=query,
            query_image_path=query_image_path,
            top_k=max_items,
            text_weight=text_weight,
            image_weight=image_weight,
        )

    def image_search(self, image_path: str,
                    max_items: int = 10) -> list[CrossModalResult]:
        """Find images visually similar to a query image.

        Uses perceptual hash matching (always available) or CLIP (if installed).
        """
        all_anchors = dict(self.graph.anchors)
        for cortex in self.router.cortices:
            all_anchors.update(cortex.graph.anchors)
        return self.cross_modal_retriever.image_search(
            anchors=all_anchors,
            query_image_path=image_path,
            top_k=max_items,
        )

    def text_to_image(self, query: str,
                     max_items: int = 10) -> list[CrossModalResult]:
        """Find images matching a text description (CLIP required for best results)."""
        all_anchors = dict(self.graph.anchors)
        for cortex in self.router.cortices:
            all_anchors.update(cortex.graph.anchors)
        return self.cross_modal_retriever.text_to_image(
            anchors=all_anchors,
            query_text=query,
            top_k=max_items,
        )

    # ── Cortex management ────────────────────────────────────

    def add_cortex(self, name: str, domain_keywords: list[str],
                   description: str = "", **kwargs) -> MemoryCortex:
        """Create and register a new domain-specific memory cortex.

        Also registers the cortex entry point in BrainSphere for O(1) routing.
        """
        cortex = self.router.find_or_create_cortex(
            name=name,
            domain_keywords=domain_keywords,
            description=description,
            **kwargs,
        )
        # Register in BrainSphere for O(1) routing (centroid may be None initially)
        self.brain.register_cortex(
            cortex_name=name,
            entry_embedding=cortex.centroid or [],
            summary=description,
            node_count=0,
        )
        return cortex

    def route_to_cortex(self, query: str = "",
                        query_embedding: list[float] | None = None
                        ) -> list[RouteResult]:
        """Route a query to the most relevant cortices (1-3)."""
        return self.router.route(query, query_embedding=query_embedding)

    def sparse_recall(self, query: str = "",
                      context: AgentContext | None = None,
                      max_items: int = 20) -> dict:
        """Full sparse-activation recall pipeline.

        1. Cortex routing — activate only 1-3 cortices
        2. Local recall within each activated cortex
        3. Memory gating — winner-take-all selection
        """
        if context is None:
            context = AgentContext(task_type="conversation")

        embedder = self._get_embedder()
        query_emb = embedder.encode(query) if query else None

        # Phase 1: Cortex routing
        routes = self.router.route(query, query_embedding=query_emb)
        active_cortices = [r.cortex for r in routes]

        # Phase 2: Local recall from activated cortices only
        all_items: list = []
        for cortex in active_cortices:
            ctx = cortex.recall(query=query, context=context, max_items=max_items)
            all_items.extend(ctx.items)

        # Phase 3: Memory gating
        gated = self.gate.gate(all_items, context, query_emb, query)

        return {
            "items": gated,
            "active_cortices": [c.config.name for c in active_cortices],
            "route_scores": {r.cortex.config.name: r.score for r in routes},
            "total_candidates": len(all_items),
            "gated_count": len(gated),
        }

    def retrieve_with_descent(self, query: str = "",
                               context: AgentContext | None = None,
                               max_items: int = 15) -> dict:
        """5-layer dimensional descent retrieval pipeline.

        Layer 0: BrainSphere cache (fast path — common nodes)
            ↓ miss or insufficient results
        Layer 1: CortexSphere 3D search (semantic + gating within target cortex)
            ↓ need cross-domain or insufficient
        Layer 2: HubSphere traversal (cross-cortex hub navigation, max 2 hops)
            ↓ still insufficient
        Layer 3: 2D planar (time × importance projection, all cortices)
            ↓ still insufficient
        Layer 4: Pseudo-2D timeline scan (TimeSpine, guaranteed results)

        Returns dict with:
        - items: final selected MemoryItems
        - layer_used: which layer produced the final results
        - layers_visited: all layers that were queried
        - brain_hit: whether BrainSphere had a direct hit
        """
        if context is None:
            context = AgentContext(task_type="conversation")
        embedder = self._get_embedder()
        query_emb = embedder.encode(query) if query else None

        result = {
            "items": [],
            "layer_used": "",
            "layers_visited": [],
            "brain_hit": False,
            "total_candidates": 0,
            "active_cortices": [],
        }

        # ── Layer 0: BrainSphere (fast cache) ────────────
        result["layers_visited"].append("brain")
        brain_hits = self.brain.query_common_nodes(query_emb, query, top_k=5)
        if brain_hits:
            result["brain_hit"] = True
            items = []
            for anchor in brain_hits:
                from .scheduler import MemoryItem
                items.append(MemoryItem(
                    anchor=anchor,
                    relevance_score=0.9,
                    memory_type=MemoryType.WORKING,
                    compression_level=0,
                    compressed_text=anchor.text,
                ))
            if len(items) >= max_items:
                result["items"] = items[:max_items]
                result["layer_used"] = "brain"
                return result
            result["items"].extend(items)

        # ── Layer 1: CortexSphere 3D search ──────────────
        result["layers_visited"].append("cortex")
        routes = self.router.route(query, query_embedding=query_emb)
        active_cortices = [r.cortex for r in routes if r.score > 0.1]
        result["active_cortices"] = [c.config.name for c in active_cortices]

        all_candidates: list = list(result["items"])
        for cortex in active_cortices[:3]:
            ctx = cortex.recall(query=query, context=context, max_items=max_items)
            all_candidates.extend(ctx.items)

        result["total_candidates"] = len(all_candidates)

        if all_candidates:
            gated = self.gate.gate(all_candidates, context, query_emb, query)
            if len(gated) >= max_items:
                result["items"] = gated[:max_items]
                result["layer_used"] = "cortex"
                return result
            all_candidates = gated

        # ── Layer 1.5: Community-aware retrieval ──────────
        if self._community_detection and self._community_detection.communities:
            result["layers_visited"].append("community")
            # Match query to best community by centroid similarity
            best_community = None
            best_csim = -1.0
            route_weight = getattr(
                getattr(self.cfg, 'community', None),
                'route_centroid_weight', 0.7)
            for c in self._community_detection.communities:
                if c.centroid_embedding and query_emb:
                    sim = _cosine_sim(query_emb, c.centroid_embedding)
                    if sim * route_weight > best_csim:
                        best_csim = sim * route_weight
                        best_community = c.id

            if best_community and best_csim > 0.2:
                community_filter = {best_community}
                comm_items = self.scheduler._community_aware_retrieve(
                    context, query, query_emb,
                    self.scheduler._select_memory_types(context),
                    max_items,
                    community_filter=community_filter,
                    community_detection=self._community_detection,
                )
                existing_ids = {item.anchor.id for item in all_candidates}
                for item in comm_items:
                    if item.anchor.id not in existing_ids:
                        all_candidates.append(item)
                        existing_ids.add(item.anchor.id)

                if len(all_candidates) >= max_items:
                    from .scheduler import MemoryItem
                    gated = self.gate.gate(all_candidates, context, query_emb, query)
                    result["items"] = gated[:max_items]
                    result["layer_used"] = "community"
                    return result

        # ── Layer 2: HubSphere traversal ─────────────────
        result["layers_visited"].append("hub")
        existing_ids = {item.anchor.id for item in all_candidates}
        for cortex in active_cortices[:2]:
            seg = cortex.get_segment_for_hub("compressed")
            if seg and seg.hub_links:
                for hub_id in seg.hub_links[:2]:
                    related_hubs = self.hublayer.traverse_hubs(hub_id, max_hops=2, max_results=3)
                    for hub in related_hubs:
                        if hub.id not in existing_ids:
                            # Create a synthetic anchor from hub summary
                            hub_anchor = Anchor.create(
                                text=hub.text,
                                embedding=hub.embedding,
                                importance=hub.importance,
                            )
                            from .scheduler import MemoryItem
                            all_candidates.append(MemoryItem(
                                anchor=hub_anchor,
                                relevance_score=0.6,
                                memory_type=MemoryType.SEMANTIC,
                                compression_level=1,
                                compressed_text=hub.text[:120],
                            ))
                            existing_ids.add(hub.id)

        if len(all_candidates) >= max_items:
            gated = self.gate.gate(all_candidates, context, query_emb, query)
            result["items"] = gated[:max_items]
            result["layer_used"] = "hub"
            return result

        # ── Layer 3: 2D planar (time × importance) ───────
        result["layers_visited"].append("2d_plane")
        plane_items = []
        import time as _time
        now = _time.time()
        for cortex in active_cortices:
            for anchor in cortex.graph.anchors.values():
                if not anchor.is_retrievable or anchor.id in existing_ids:
                    continue
                hours = (now - anchor.last_activated_at) / 3600
                recency = math.exp(-hours / 168)
                score = 0.6 * recency + 0.4 * anchor.retention_score
                plane_items.append((anchor, score))

        plane_items.sort(key=lambda x: -x[1])
        from .scheduler import MemoryItem
        for anchor, score in plane_items[:max_items]:
            if anchor.id not in existing_ids:
                all_candidates.append(MemoryItem(
                    anchor=anchor, relevance_score=score,
                    memory_type=MemoryType.EPISODIC, compression_level=2,
                    compressed_text=anchor.text[:80],
                ))
                existing_ids.add(anchor.id)

        if len(all_candidates) >= max_items:
            result["items"] = [item for item in all_candidates if item.anchor is not None][:max_items]
            result["layer_used"] = "2d_plane"
            return result

        # ── Layer 4: Timeline scan (guaranteed results) ───
        result["layers_visited"].append("timeline")
        timeline_results = self.scan_timeline(max_days=90, max_clusters=max_items)
        from .scheduler import MemoryItem
        for tc in timeline_results:
            for anchor_id in tc.get("anchor_ids", []):
                if anchor_id in existing_ids:
                    continue
                # Look up anchor in main graph or any cortex
                anchor = self.graph.anchors.get(anchor_id)
                if anchor is None:
                    for cortex in active_cortices:
                        anchor = cortex.graph.anchors.get(anchor_id)
                        if anchor:
                            break
                if anchor and anchor.is_retrievable:
                    all_candidates.append(MemoryItem(
                        anchor=anchor,
                        relevance_score=0.3 + 0.2 * tc.get("importance", 0.5),
                        memory_type=MemoryType.EPISODIC,
                        compression_level=3,
                        compressed_text=anchor.text[:60],
                    ))
                    existing_ids.add(anchor_id)

        # Final fallback: return whatever we have
        if all_candidates:
            result["items"] = [item for item in all_candidates[:max_items] if item.anchor is not None]
        result["layer_used"] = "timeline"
        return result

    def cascade_recall(self, query: str = "",
                       max_chains: int = 5,
                       max_depth: int = 5) -> list:
        """Causal chain recall — trace event chains from query-relevant seeds."""
        embedder = self._get_embedder()
        query_emb = embedder.encode(query)

        # Find seed anchors
        seeds = self.scheduler._discover_seeds(
            AgentContext(task_type="reflection"), query, query_emb,
            self.scheduler._select_memory_types(AgentContext()))

        seed_ids = [a.id for a in seeds[:3]]
        chains = self.cascade.cascade_from_seeds(seed_ids, max_depth, max_chains)
        return [
            {"narrative": c.narrative, "confidence": c.total_confidence,
             "depth": c.depth, "type": c.chain_type}
            for c in chains if c.is_valid
        ]

    # ── Time spine ────────────────────────────────────────────

    def index_on_timeline(self, anchor_id: str, timestamp: float | None = None,
                          importance: float = 0.5,
                          embedding: list[float] | None = None,
                          topic: str = ""):
        """Index an anchor into the TimeSpine."""
        self.timespine.index_anchor(anchor_id, timestamp, importance, embedding, topic)

    def scan_timeline(self, max_days: int = 30,
                      max_clusters: int = 20) -> list:
        """'Upper-right to lower-left' priority scan of the TimeSpine."""
        clusters = self.timespine.scan_priority(max_days, max_clusters)
        return [
            {"id": c.id, "topic": c.topic, "importance": c.importance,
             "size": c.size, "summary": c.summary,
             "anchor_ids": c.anchor_ids}
            for c in clusters
        ]

    # ── Hub layer ─────────────────────────────────────────────

    def create_topic_hub(self, text: str, source_anchor_ids: list[str],
                         cortex_name: str) -> HubNode:
        """Create a leaf hub from a topic cluster within a cortex."""
        return self.hublayer.create_leaf(text, source_anchor_ids, cortex_name)

    def create_domain_hub(self, text: str, child_hub_ids: list[str],
                          cortex_name: str) -> HubNode | None:
        """Create a domain hub aggregating leaf hubs in the same cortex."""
        return self.hublayer.create_domain(text, child_hub_ids, cortex_name)

    def bridge_cortices(self, hub_a_id: str, hub_b_id: str) -> bool:
        """Create a cross-cortex bridge between two hubs.

        This is the ONLY mechanism for cross-cortex associations.
        """
        return self.hublayer.bridge(hub_a_id, hub_b_id)

    # ── Cognitive maintenance ─────────────────────────────────

    def micro_consolidate(self) -> dict:
        """Lightweight online consolidation — call after every few interactions.

        Includes working memory maintenance:
        - Items accessed 3+ times get auto-promoted to long-term memory
        - Stale items are cleared
        """
        if self._online_consolidator is None:
            from .online import OnlineConsolidator
            self._online_consolidator = OnlineConsolidator(self.graph)
        self._online_consolidator.record_interaction()

        # Auto-promote well-rehearsed working memory items
        promoted = 0
        for entry in list(self.working_memory._entries):
            if entry.access_count >= 3:
                self.working_memory.promote(entry, self)
                promoted += 1

        result = self._online_consolidator._micro_sleep()
        result["working_memory_promoted"] = promoted
        return result

    def sleep(self, current_time: float | None = None) -> dict:
        """Run a full sleep consolidation cycle.

        Each cortex consolidates independently (own graph + decay params).
        Cross-cortex operations (Hub bridging, global schema extraction)
        run afterwards as a lightweight global pass.
        """
        # 1. Per-cortex independent sleep consolidation
        cortex_reports = {}
        for cortex in self.router.cortices:
            try:
                cortex_reports[cortex.config.name] = cortex.consolidate()
            except Exception as e:
                cortex_reports[cortex.config.name] = {"error": str(e)}

        # 2. Cross-cortex operations (Hub bridging, global schema, edge pruning)
        from .sleep import SleepCycle
        sc = SleepCycle(self.graph)
        global_report = sc.run_phased(
            brain=self.brain,
            hublayer=self.hublayer,
            cortices=self.router.cortices,
        )

        # 3. Evolve the graph
        evo = self.evolution.evolve(current_time)
        self.total_evolutions += 1

        # 4. Decay ghosts
        ghost_purged = self.ghosts.decay_all()

        self.sleep_cycles += 1
        return {
            "cortex_reports": cortex_reports,
            "global_report": global_report,
            "evolution": evo,
            "ghost_stats": self.ghosts.stats,
            "ghosts_purged": ghost_purged,
        }

    def micro_sleep(self, steps: int = 2) -> dict:
        """Incremental non-blocking sleep — run 1-2 phases at a time.

        Call this during agent idle time. Each call runs 'steps' phases
        and returns immediately. The next call resumes from the checkpoint.

        Returns dict with phases_run, is_complete, progress_pct, errors.
        """
        from .micro_sleep import MicroSleepScheduler

        if self._micro_sleep is None:
            self._micro_sleep = MicroSleepScheduler(
                graph=self.graph, config=self.cfg,
                brain=self.brain, hublayer=self.hublayer,
                cortices=self.router.cortices,
            )

        result = self._micro_sleep.run_next(steps=steps)

        if result.is_complete:
            self._micro_sleep = None  # reset for next cycle
            self.sleep_cycles += 1

        return {
            "phases_run": result.phases_run,
            "phases_processed": result.phases_processed,
            "is_complete": result.is_complete,
            "progress_pct": result.progress.progress_pct if result.progress else 1.0,
            "errors": result.errors,
            "items_processed": result.items_processed,
            "duration_ms": result.duration_ms,
            "summary": self._micro_sleep.get_summary() if self._micro_sleep else "Complete",
        }

    def estimate_sleep_cost(self, dry_run: bool = False) -> CostEstimate:
        """Estimate resource consumption before running sleep.

        Returns a CostEstimate with predicted LLM calls, token counts,
        dollar cost, and wall-clock time. Use dry_run=True to mark
        the estimate as non-execution.
        """
        estimator = SleepCostEstimator()
        return estimator.estimate(self, dry_run=dry_run)

    def evolve(self, current_time: float | None = None) -> dict:
        """Run an evolution cycle without sleep (decay, boost, conflict, interference)."""
        result = self.evolution.evolve(current_time)
        self.total_evolutions += 1
        return result

    # ── Ghost operations ──────────────────────────────────────

    def fuzzy_recall(self, embedding: list[float] | None = None,
                     query: str = "", threshold: float | None = None):
        """Fuzzy recall from ghost traces — 'I seem to remember...' """
        if embedding is None:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        return self.ghosts.fuzzy_recall(embedding, threshold)

    def ghost_intensity_recall(self, query: str = "",
                               embedding: list[float] | None = None,
                               top_k: int = 10) -> list[dict]:
        """Rank ghost traces by intensity for retrieval boosting.

        Returns top ghosts sorted by intensity (descending), each with
        resonance score and semantic shadow for partial recall.
        """
        if embedding is None:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        ranked = self.ghosts.ranked_resonance(embedding, top_k=top_k)
        return [
            {
                "ghost_id": ghost.id,
                "intensity": round(intensity, 4),
                "semantic_shadow": ghost.semantic_shadow,
                "original_tags": ghost.original_tags,
                "original_importance": ghost.original_importance,
                "emotion_trace": ghost.emotion_trace,
                "revival_count": ghost.revival_count,
            }
            for ghost, intensity in ranked
        ]

    def create_negative_ghost(self, original_text: str,
                             contradiction_text: str,
                             target_anchor_id: str = "",
                             original_importance: float = 0.5,
                             contradiction_type: str = "direct") -> str:
        """Create a negative ghost to track a contradiction.

        During future retrievals, this ghost will suppress memories similar
        to the contradicted one. Returns the ghost ID.
        """
        embedder = self._get_embedder()
        embedding = embedder.encode(original_text)
        ghost = self.ghosts.create_negative(
            original_text=original_text,
            contradiction_text=contradiction_text,
            target_anchor_id=target_anchor_id,
            original_importance=original_importance,
            contradiction_type=contradiction_type,
            embedding=embedding,
        )
        return ghost.id

    def check_ghost_suppression(self, embedding: list[float] | None = None,
                               query: str = "") -> dict:
        """Check if a query or embedding is suppressed by negative ghosts.

        Returns a dict with the overall suppression factor and details
        about which negative ghosts are contributing.
        """
        if embedding is None and query:
            embedder = self._get_embedder()
            embedding = embedder.encode(query)
        if embedding is None:
            return {"suppression_factor": 1.0, "active_negatives": []}

        factor = self.ghosts.check_suppression(embedding)
        active = []
        for ghost in self.ghosts.negative_ghosts:
            resonance = ghost.resonance(embedding)
            if resonance > 0.1:
                active.append({
                    "ghost_id": ghost.id,
                    "resonance": round(resonance, 4),
                    "intensity": round(ghost.intensity, 4),
                    "contradiction_type": ghost.contradiction_type,
                    "contradiction_text": ghost.contradiction_text[:120],
                })

        return {
            "suppression_factor": round(factor, 4),
            "active_negatives": active,
        }

    # ── Persistence ───────────────────────────────────────────

    @property
    def snapshots(self) -> SnapshotManager:
        """Lazy-init snapshot manager for versioned state persistence."""
        if self._snapshot_mgr is None:
            self._snapshot_mgr = SnapshotManager()
        return self._snapshot_mgr

    def snapshot(self, description: str = "", force: bool = False) -> SnapshotMeta:
        """Create a versioned state snapshot (crash-safe checkpoint).

        Keeps last N snapshots by default, auto-cleans old versions.
        Use recover() to restore from latest snapshot after a crash.
        """
        meta = self.snapshots.snapshot(self.graph, description=description, force=force)
        return meta

    def recover(self) -> tuple:
        """Crash recovery: load latest snapshot + replay WAL.

        Returns (graph, recovery_log). Use this after unexpected shutdown.
        """
        graph, log = self.snapshots.recover()
        # Replace in-memory graph with recovered state
        self.graph = graph
        return graph, log

    def save(self, path: str | None = None) -> str:
        """Persist the entire memory system to disk."""
        filepath = path or self.storage_path or "star_graph_memory.json"
        from .storage import JSONStorage
        storage = JSONStorage(filepath)
        storage.save(self.graph)
        if self._ghosts:
            # Save ghosts alongside the main graph in metadata
            import json
            ghost_path = filepath.replace(".json", "_ghosts.json")
            with open(ghost_path, "w", encoding="utf-8") as f:
                ghost_data = {}
                for gid, g in self._ghosts.ghosts.items():
                    ghost_data[gid] = {
                        "id": g.id,
                        "compressed_embedding": g.compressed_embedding,
                        "residual_edges": g.residual_edges,
                        "emotion_trace": g.emotion_trace,
                        "pruned_at": g.pruned_at,
                        "original_tags": g.original_tags,
                        "original_importance": g.original_importance,
                        "semantic_shadow": g.semantic_shadow,
                        "reactivation_probability": g.reactivation_probability,
                        "revival_count": g.revival_count,
                        "partial_recall_count": g.partial_recall_count,
                    }
                json.dump(ghost_data, f, indent=2, ensure_ascii=False)
        self.storage_path = filepath
        return filepath

    def load(self, path: str) -> StarGraph:
        """Load the memory system from disk."""
        from .storage import JSONStorage
        storage = JSONStorage(path)
        self.graph = storage.load()
        self.storage_path = path

        # Try to load ghosts
        import json
        import os
        ghost_path = path.replace(".json", "_ghosts.json")
        if os.path.exists(ghost_path):
            with open(ghost_path, "r", encoding="utf-8") as f:
                ghost_data = json.load(f)
            from .ghost import GhostNode
            self._ghosts = GhostSubsystem()
            for gid, gd in ghost_data.items():
                ghost = GhostNode(
                    id=gd["id"],
                    compressed_embedding=gd["compressed_embedding"],
                    residual_edges=gd.get("residual_edges", {}),
                    emotion_trace=gd["emotion_trace"],
                    pruned_at=gd["pruned_at"],
                    original_tags=gd.get("original_tags", []),
                    original_importance=gd["original_importance"],
                    semantic_shadow=gd["semantic_shadow"],
                    reactivation_probability=gd["reactivation_probability"],
                    revival_count=gd.get("revival_count", 0),
                    partial_recall_count=gd.get("partial_recall_count", 0),
                )
                self._ghosts.ghosts[gid] = ghost

        # Reset subsystems that cache graph references
        self._scheduler = None
        self._evolution = None
        self._metrics = None
        self._community_detection = None

        # Re-set survival function (class vars don't survive process restart)
        Anchor.set_survival_function(self.survival_function)
        from .ghost import GhostNode
        GhostNode.set_survival_function(self.survival_function)

        return self.graph

    # ── Reflection (meta-cognitive insights) ───────────────

    def reflect(self, text: str, source_anchor_ids: list[str],
                reflection_type: str = "lesson_learned",
                confidence: float = 0.6) -> str:
        """Create a meta-cognitive reflection on past memories.

        Args:
            text: The insight or lesson text.
            source_anchor_ids: Which anchor IDs this insight is based on.
            reflection_type: One of "failure_analysis", "success_pattern",
                            "root_cause", "lesson_learned".
            confidence: How confident we are in this insight (0..1).
        Returns:
            The reflection node ID.
        """
        from .graph import ReflectionNode
        if reflection_type == "failure_analysis":
            r = ReflectionNode.from_failure(text, source_anchor_ids, confidence)
        elif reflection_type == "success_pattern":
            r = ReflectionNode.from_success(text, source_anchor_ids, confidence)
        elif reflection_type == "root_cause":
            import hashlib
            rid = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
            r = ReflectionNode(id=rid, text=text, reflection_type="root_cause",
                              source_anchor_ids=source_anchor_ids,
                              confidence=confidence, strength=0.5)
        else:
            r = ReflectionNode.from_lesson(text, source_anchor_ids, confidence)
        return self.graph.add_reflection(r)

    def get_reflections(self, anchor_ids: list[str],
                        types: list[str] | None = None) -> list[dict]:
        """Get reflection nodes connected to given anchors."""
        nodes = self.graph.find_reflections(anchor_ids, types)
        return [
            {"id": r.id, "text": r.text, "type": r.reflection_type,
             "confidence": r.confidence, "strength": r.strength}
            for r in nodes
        ]

    def strengthen_reflection(self, reflection_id: str) -> bool:
        """Reinforce a reflection when confirmed by new evidence."""
        if reflection_id in self.graph.reflections:
            self.graph.reflections[reflection_id].reinforce()
            return True
        return False

    def weaken_reflection(self, reflection_id: str) -> bool:
        """Weaken a reflection when contradicted."""
        if reflection_id in self.graph.reflections:
            self.graph.reflections[reflection_id].weaken()
            return True
        return False

    # ── Health & reporting ────────────────────────────────────

    @property
    def stats(self) -> ManagerStats:
        ghost_count = len(self.ghosts.ghosts) if self._ghosts else len(self.graph._ghost_subsystem.ghosts)
        abstract_count = len(self._abstraction.abstracts) if self._abstraction else len(getattr(self.graph, 'abstracts', {}))
        cortex_count = len(self._router.cortices) if self._router else 0
        hub_count = len(self._hublayer.hubs) if self._hublayer else 0
        cluster_count = self._timespine.stats["total_clusters"] if self._timespine else 0
        return ManagerStats(
            anchors=len(self.graph.anchors),
            edges=len(self.graph.edges),
            ghosts=ghost_count,
            schemas=len(self.graph.schemas),
            abstracts=abstract_count,
            working_memory=self.working_memory.size,
            cortices=cortex_count,
            hubs=hub_count,
            clusters=cluster_count,
            sleep_cycles=self.sleep_cycles,
            total_evolutions=self.total_evolutions,
            uptime_seconds=time.time() - self._started_at,
        )

    def health_report(self) -> str:
        """Full cognitive health report."""
        try:
            cog = self.metrics.snapshot()
        except Exception:
            cog = {}

        s = self.stats
        lines = [
            "=" * 55,
            "  Star Graph Memory — Cognitive Health Report",
            "=" * 55,
            f"  Anchors: {s.anchors}    Edges: {s.edges}    Ghosts: {s.ghosts}",
            f"  Cortices: {s.cortices}    Hubs: {s.hubs}    Clusters: {s.clusters}",
            f"  Schemas: {s.schemas}    Abstracts: {s.abstracts}    Working: {s.working_memory}",
            f"  Sleep cycles: {s.sleep_cycles}    Evolutions: {s.total_evolutions}",
            f"  Uptime: {s.uptime_seconds:.0f}s",
            f"",
        ]
        if cog:
            lines += [
                f"  Memory Stability:          {cog.get('memory_stability', 0):.2f}",
                f"  Recall Plasticity:         {cog.get('recall_plasticity', 0):.2f}",
                f"  Compression Ratio:         {cog.get('compression_ratio', 0):.2f}",
                f"  Semantic Drift Resistance: {cog.get('semantic_drift_resistance', 0):.2f}",
                f"  Abstraction Emergence:     {cog.get('abstraction_emergence_rate', 0):.1f}/cycle",
                f"  Ghost Reactivation:        {cog.get('ghost_reactivation_accuracy', 0):.2f}",
            ]
        lines.append("=" * 55)
        return "\n".join(lines)

    def print_health(self) -> None:
        print(self.health_report())

    # ── Advanced: graph traversal utilities ───────────────────

    def connect(self, source_id: str, target_id: str,
                weight: float = 0.5, edge_type: str = "topical") -> Edge | None:
        """Explicitly connect two memories."""
        if source_id not in self.graph.anchors or target_id not in self.graph.anchors:
            return None
        src_emb = self.graph.anchors[source_id].embedding
        tgt_emb = self.graph.anchors[target_id].embedding
        if src_emb and tgt_emb and weight == 0.5:
            weight = _cosine_sim(src_emb, tgt_emb)
        return self.graph.add_edge(source_id, target_id, weight=weight, edge_type=edge_type)

    def reinforce(self, source_id: str, target_id: str) -> bool:
        """Reinforce an edge — called when the connection is confirmed."""
        edge_key = self.graph._key(source_id, target_id)
        edge = self.graph.edges.get(edge_key)
        if edge is None:
            return False
        if hasattr(edge, 'reinforce'):
            edge.reinforce()
        else:
            edge.strengthen(0.05)
        return True

    def discover_abstracts(self) -> list:
        """Run abstraction discovery on current anchors."""
        anchors_with_emb = {
            aid: a for aid, a in self.graph.anchors.items()
            if a.embedding
        }
        embeddings = {aid: a.embedding for aid, a in anchors_with_emb.items()}
        return self.abstraction.discover(anchors_with_emb, embeddings)

    def detect_communities(self) -> list[Community]:
        """Detect communities in the main graph via label propagation.

        Updates anchor community fields in-place and returns the list of
        detected communities. Call this after significant graph growth
        to enable community-aware retrieval.
        """
        return self.community_detection.detect(self.graph)

    # ── Compression API ──────────────────────────────────────────

    def compress_session(self, session_id: str) -> list:
        """Run Level 0→1 compression on a single session's anchors.

        Groups anchors by embedding similarity, generates EPISODIC summaries,
        inserts proxy anchors into the graph, and down-weights source anchors.

        Args:
            session_id: Session to compress

        Returns:
            List of SummaryAnchor objects created
        """
        from .compression import SessionCompressor
        anchors_list = list(self.graph.anchors.values())
        compressor = SessionCompressor()
        summaries = compressor.compress(anchors_list, session_id)

        if summaries:
            # Insert summaries into graph
            _mcomp = self.compressor
            _mcomp.add_to_graph(self.graph, summaries, edge_type="compresses")

        return summaries

    def compress_all(self) -> dict:
        """Run the full multi-level compression pipeline on all sessions.

        Groups all anchors by session, then runs EPISODIC → STRATEGIC → META
        compression, inserting all resulting summary anchors into the graph.

        Returns:
            Dict with counts per compression level and total stats
        """
        from collections import defaultdict
        from .compression import CompressionLevel

        # Group anchors by session
        anchors_by_session: dict[str, list] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            if anchor.embedding and anchor.source_session:
                anchors_by_session[anchor.source_session].append(anchor)

        if not anchors_by_session:
            return {
                "levels": {"episodic": 0, "strategic": 0, "meta": 0},
                "total_summaries": 0,
                "total_edges": 0,
            }

        compressor = self.compressor
        results = compressor.compress_pipeline(anchors_by_session)

        # Insert all summaries into graph
        total_edges = 0
        level_counts: dict[str, int] = {}
        for level, summaries in results.items():
            level_name = level.name.lower()
            level_counts[level_name] = len(summaries)
            if summaries:
                total_edges += compressor.add_to_graph(self.graph, summaries, edge_type="compresses")

        return {
            "levels": level_counts,
            "total_summaries": sum(level_counts.values()),
            "total_edges": total_edges,
        }

    def get_compressed_view(self,
                            min_level: int = 1,
                            max_summaries: int = 50) -> list[dict]:
        """Get a view of compressed memories at or above a given compression level.

        Scans the graph for summary anchors (identified by their tags containing
        "level:episodic", "level:strategic", or "level:meta") and returns them
        as dict records sorted by confidence descending.

        Args:
            min_level: Minimum compression level (1=episodic, 2=strategic, 3=meta)
            max_summaries: Maximum number of summaries to return

        Returns:
            List of dicts with id, text, level, confidence, source_count, tags
        """
        from .compression import CompressionLevel
        level_tags = {
            CompressionLevel.EPISODIC: "level:episodic",
            CompressionLevel.STRATEGIC: "level:strategic",
            CompressionLevel.META: "level:meta",
        }

        summaries: list[dict] = []
        for aid, anchor in self.graph.anchors.items():
            # Identify summary anchors by their "level:*" tags
            level_tag = None
            level_val = 0
            for lvl, tag in level_tags.items():
                if tag in anchor.tags:
                    level_tag = tag
                    level_val = lvl.value
                    break

            if level_val < min_level:
                continue

            # Count source anchors by inspecting "compresses" edges
            source_count = 0
            for neighbor in self.graph._adjacency.get(aid, set()):
                edge_key = self.graph._key(aid, neighbor)
                edge = self.graph.edges.get(edge_key)
                if edge and edge.edge_type == "compresses":
                    source_count += 1

            summaries.append({
                "id": aid,
                "text": anchor.text,
                "level": level_val,
                "level_name": CompressionLevel(level_val).name.lower() if level_val > 0 else "raw",
                "confidence": anchor.vector.confidence,
                "stability": anchor.vector.stability,
                "source_count": source_count,
                "tags": [t for t in anchor.tags if not t.startswith("level:")],
            })

        # Sort by confidence descending
        summaries.sort(key=lambda x: -x["confidence"])
        return summaries[:max_summaries]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
