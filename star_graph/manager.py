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

import time
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, MemoryState
from .graph import StarGraph, Edge
from .config import Config
from .scheduler import CognitiveMemoryScheduler, AgentContext, MemoryContext
from .working_memory import WorkingMemory, WorkingMemoryEntry
from .cortex import MemoryCortex, CortexConfig
from .router import CortexRouter, RouteResult
from .gate import MemoryGate
from .timespine import TimeSpine
from .cascade import CascadeRecall
from .hub import HubLayer, HubNode
from .evolution import MemoryEvolutionEngine
from .ghost import GhostSubsystem
from .abstraction import AbstractionEngine
from .metrics import CognitiveMetrics


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

        # Working memory — short-term buffer (not persisted)
        wm_cfg = getattr(self.cfg, 'working_memory', None)
        wm_capacity = getattr(wm_cfg, 'max_capacity', 9) if wm_cfg else 9
        wm_ttl = getattr(wm_cfg, 'ttl_seconds', 1800.0) if wm_cfg else 1800.0
        self.working_memory = WorkingMemory(
            max_capacity=wm_capacity,
            ttl_seconds=wm_ttl,
        )

        # Subsystems — lazily initialized
        self._embedder = None
        self._scheduler: CognitiveMemoryScheduler | None = None
        self._evolution: MemoryEvolutionEngine | None = None
        self._ghosts: GhostSubsystem | None = None
        self._abstraction: AbstractionEngine | None = None
        self._metrics: CognitiveMetrics | None = None
        self._online_consolidator = None
        self._router: CortexRouter | None = None
        self._gate: MemoryGate | None = None
        self._timespine: TimeSpine | None = None
        self._cascade: CascadeRecall | None = None
        self._hublayer: HubLayer | None = None

        # Stats
        self.sleep_cycles: int = 0
        self.total_evolutions: int = 0

    # ── Subsystem access (lazy init) ──────────────────────────

    @property
    def scheduler(self) -> CognitiveMemoryScheduler:
        if self._scheduler is None:
            self._scheduler = CognitiveMemoryScheduler(
                self.graph, self.cfg, working_memory=self.working_memory)
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
    def metrics(self) -> CognitiveMetrics:
        if self._metrics is None:
            self._metrics = CognitiveMetrics(self.graph)
        return self._metrics

    @property
    def router(self) -> CortexRouter:
        if self._router is None:
            self._router = CortexRouter(self.cfg)
        return self._router

    @property
    def gate(self) -> MemoryGate:
        if self._gate is None:
            self._gate = MemoryGate(k=self.cfg.gate.k if hasattr(self.cfg, 'gate') else 20)
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
            self._hublayer = HubLayer()
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

    def recall(self, query: str = "",
               context: AgentContext | None = None,
               max_items: int = 10) -> MemoryContext:
        """Retrieve memories relevant to the query and agent context.

        Args:
            query: What the agent is looking for
            context: Agent's current task, goals, emotional state
            max_items: Maximum memories to return
        """
        if context is None:
            context = AgentContext(task_type="conversation")
        return self.scheduler.retrieve(context, query, max_items)

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

    # ── Cortex management ────────────────────────────────────

    def add_cortex(self, name: str, domain_keywords: list[str],
                   description: str = "", **kwargs) -> MemoryCortex:
        """Create and register a new domain-specific memory cortex."""
        cortex = self.router.find_or_create_cortex(
            name=name,
            domain_keywords=domain_keywords,
            description=description,
            **kwargs,
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
             "size": c.size, "summary": c.summary}
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
        """Run a full 5-phase sleep cycle.

        Combines: SWR replay, Hebbian plasticity, schema extraction,
        ghost management, and memory evolution.

        Returns a dict with keys: sleep_report, evolution_summary, ghost_stats.
        """
        from .sleep import SleepCycle

        # 1. Full sleep consolidation
        sc = SleepCycle(self.graph)
        sleep_report = sc.run_phased()

        # 2. Evolve the graph
        evo = self.evolution.evolve(current_time)
        self.total_evolutions += 1

        # 3. Decay ghosts
        ghost_purged = self.ghosts.decay_all()

        self.sleep_cycles += 1
        return {
            "sleep_report": sleep_report,
            "evolution": evo,
            "ghost_stats": self.ghosts.stats,
            "ghosts_purged": ghost_purged,
        }

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

    # ── Persistence ───────────────────────────────────────────

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
        ghost_count = len(self.ghosts.ghosts) if self._ghosts else len(self.graph.ghosts)
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


def _cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
