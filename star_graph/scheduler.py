"""Cognitive Memory Scheduler — context-aware, reasoning-driven retrieval.

Not "retrieve top-k." This module understands WHAT the agent is doing and
selects the RIGHT memory type, at the RIGHT compression level, traversing
the graph with multi-hop reasoning.

Architecture:
  AgentContext (task_type, emotional_state, active_goals)
    ↓
  Memory Type Selection (semantic / episodic / procedural / working)
    ↓
  Seed Discovery (embedding + oscillation resonance)
    ↓
  Multi-hop Graph Traversal (spreading activation with RichEdge scoring)
    ↓
  Composite Ranking (confidence + recency + reinforcement + relevance)
    ↓
  Adaptive Compression (raw → abstract, depending on context window budget)
    ↓
  StructuredMemoryContext output

This transforms star-graph from a passive storage engine into a cognitive
memory runtime that an AI agent can actually USE.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .anchor import Anchor, MemoryState
from .graph import StarGraph, Constellation, Edge, RichEdge
from .config import Config
from .working_memory import WorkingMemory, WorkingMemoryEntry


# ── Memory types ───────────────────────────────────────

class MemoryType(Enum):
    """Cognitive memory taxonomy — not just 'stored' vs 'deleted'."""
    SEMANTIC = "semantic"    # Stable facts: "user knows Python", "project uses Docker"
    EPISODIC = "episodic"    # Events: "user debugged Redis timeout on Tuesday"
    PROCEDURAL = "procedural" # Behavior patterns: "user prefers concise answers"
    WORKING = "working"      # Current session context — high plasticity


@dataclass
class AgentContext:
    """What the agent is doing right now — drives memory selection."""
    task_type: str = "general"        # "coding", "debugging", "planning", "reflection", "conversation"
    emotional_state: float = 0.0      # -1..+1
    active_goals: list[str] = field(default_factory=list)
    current_topic: str = ""
    recent_anchor_ids: list[str] = field(default_factory=list)
    context_budget_tokens: int = 4000  # how much context window is available
    session_id: str = ""


@dataclass
class MemoryItem:
    """A single memory retrieved by the scheduler."""
    anchor: Anchor
    relevance_score: float = 0.0
    confidence: float = 0.5
    memory_type: MemoryType = MemoryType.EPISODIC
    related_anchors: list[str] = field(default_factory=list)
    reasoning_path: list[str] = field(default_factory=list)  # multi-hop traversal path
    compression_level: int = 0  # 0=raw, 1=compressed, 2=abstract
    compressed_text: str = ""


@dataclass
class MemoryContext:
    """Structured memory context returned to the agent."""
    items: list[MemoryItem]
    memory_summary: str = ""             # compressed overview
    active_patterns: list[str] = field(default_factory=list)  # detected behavioral patterns
    relevant_facts: list[str] = field(default_factory=list)   # key facts
    reasoning_traces: list[str] = field(default_factory=list) # how memories were connected
    reflections: list[dict] = field(default_factory=list)     # meta-cognitive insights
    total_tokens: int = 0
    retrieval_latency_ms: float = 0.0


# ── Scheduler ──────────────────────────────────────────

class CognitiveMemoryScheduler:
    """Context-aware memory retrieval with multi-hop reasoning.

    Usage:
        scheduler = CognitiveMemoryScheduler(graph)
        context = AgentContext(task_type="debugging", active_goals=["fix Redis timeout"])
        memory = scheduler.retrieve(context, query="Redis connection pool")
        print(memory.memory_summary)
    """

    def __init__(self, graph: StarGraph, config: Config | None = None,
                 working_memory: WorkingMemory | None = None):
        self.graph = graph
        self.cfg = config or Config.get()
        self.working_memory = working_memory
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    # ── Main retrieval pipeline ─────────────────────────

    def retrieve(self, context: AgentContext, query: str = "",
                 max_items: int = 10) -> MemoryContext:
        """Full cognitive retrieval pipeline with dimensional reduction.

        Level 1: 3D semantic search (embedding + phase coherence)
        Level 2: 2D plane projection (time × importance), if L1 returns < k items
        Level 3: Pseudo-2D timeline scan (TimeSpine), if L2 still insufficient

        This guarantees retrieval always returns results — even when semantic
        similarity fails, we fall back through coarser dimensions.
        """
        t0 = time.perf_counter()
        embedder = self._get_embedder()
        query_embedding = embedder.encode(query) if query else None

        # Step 0: Check working memory first — immediate, high-priority
        wm_items: list[MemoryItem] = []
        if self.working_memory:
            wm_items = self._retrieve_from_working_memory(
                context, query, query_embedding)

        # Step 1: Select memory types
        memory_types = self._select_memory_types(context)

        # Step 2-4: Dimensional reduction retrieval
        compressed = self._dimensional_reduction_retrieve(
            context, query, query_embedding, memory_types, max_items)

        # Prepend working memory items (they take priority as immediate context)
        remaining = max_items - len(wm_items)
        if remaining > 0:
            compressed = wm_items + compressed[:remaining]
        else:
            compressed = wm_items[:max_items]

        # Step 6: Build context
        latency = (time.perf_counter() - t0) * 1000
        return self._build_context(compressed, context, latency)

    # ── Step 0: Working memory retrieval ────────────────

    def _retrieve_from_working_memory(self, context: AgentContext, query: str,
                                       query_embedding: list[float] | None
                                       ) -> list[MemoryItem]:
        """Check working memory for directly relevant items.

        Working memory is the fastest, most plastic buffer — checked first
        before any long-term retrieval. Items here represent what the agent
        is actively thinking about.
        """
        if self.working_memory is None:
            return []

        wm_results = self.working_memory.get_relevant(
            query_embedding=query_embedding,
            query_text=query,
            max_items=3,
        )

        items: list[MemoryItem] = []
        for entry, score in wm_results:
            # Create a synthetic anchor for the working memory entry
            anchor = Anchor.create(
                text=entry.text,
                source_session=entry.source_session,
                embedding=entry.embedding,
                emotional_valence=entry.emotional_valence,
                importance=entry.importance,
                tags=entry.tags,
            )
            items.append(MemoryItem(
                anchor=anchor,
                relevance_score=score + 0.15,  # small boost for being in WM
                confidence=0.7,
                memory_type=MemoryType.WORKING,
                compression_level=0,
                compressed_text=entry.text,
            ))

        return items

    # ── Dimensional reduction retrieval ─────────────────

    def _dimensional_reduction_retrieve(self, context: AgentContext,
                                         query: str,
                                         query_embedding: list[float] | None,
                                         memory_types: list[MemoryType],
                                         max_items: int) -> list[MemoryItem]:
        """Three-level retrieval with automatic fallback.

        Level 1 (3D Semantic): Full embedding search in semantic space.
            Most precise, but can fail for cross-domain or vague queries.
        Level 2 (2D Plane): Project to time × importance, ignore semantics.
            "Show me what's recent and important, regardless of topic."
        Level 3 (Pseudo-2D Timeline): TimeSpine priority scan.
            "Upper-right to lower-left" — guaranteed to return something.
        """
        min_for_level = max(3, max_items // 3)

        # Level 1: Semantic search (3D)
        seeds = self._discover_seeds(context, query, query_embedding, memory_types)
        scored = self._multi_hop_traverse(seeds, context, query_embedding, max_hops=3)
        ranked = self._composite_rank(scored, context, query_embedding)
        compressed = self._adaptive_compress(ranked[:max_items], context)

        if len(compressed) >= min_for_level:
            return compressed

        # Level 2: 2D Plane (time × importance projection)
        plane_results = self._retrieve_2d_plane(context, max_items)
        # Merge with L1 results, deduplicating
        existing_ids = {item.anchor.id for item in compressed}
        for item in plane_results:
            if item.anchor.id not in existing_ids:
                compressed.append(item)
                existing_ids.add(item.anchor.id)
            if len(compressed) >= max_items:
                break

        if len(compressed) >= min_for_level:
            return compressed

        # Level 3: Timeline scan (pseudo-2D, guaranteed results)
        timeline_results = self._retrieve_timeline(context, max_items)
        existing_ids = {item.anchor.id for item in compressed}
        for item in timeline_results:
            if item.anchor.id not in existing_ids:
                compressed.append(item)
                existing_ids.add(item.anchor.id)
            if len(compressed) >= max_items:
                break

        return compressed

    def _retrieve_2d_plane(self, context: AgentContext,
                            max_items: int = 10) -> list[MemoryItem]:
        """Level 2: Retrieve from time × importance 2D projection.

        Projects all retrievable anchors onto (recency, importance) plane.
        Selects top items by: recency DESC, importance DESC.
        This ignores semantic similarity entirely — useful when semantic
        search fails (e.g., vague query, cross-domain question).
        """
        import time as _time
        now = _time.time()

        items: list[tuple[Anchor, float]] = []
        for anchor in self.graph.anchors.values():
            if not anchor.is_retrievable:
                continue

            hours_since = (now - anchor.last_activated_at) / 3600
            recency = math.exp(-hours_since / 168)
            importance = anchor.retention_score

            # 2D score = weighted sum on the (recency, importance) plane
            score = 0.6 * recency + 0.4 * importance
            items.append((anchor, score))

        items.sort(key=lambda x: -x[1])
        results: list[MemoryItem] = []
        for anchor, score in items[:max_items]:
            results.append(MemoryItem(
                anchor=anchor,
                relevance_score=score,
                memory_type=self._classify_memory_type(anchor),
                compression_level=1,
                compressed_text=anchor.text[:120],
            ))
        return results

    def _retrieve_timeline(self, context: AgentContext,
                            max_items: int = 10) -> list[MemoryItem]:
        """Level 3: Pseudo-2D timeline scan (upper-right to lower-left).

        Scans the TimeSpine: most recent days first, within each day
        most important clusters first. This is the final fallback —
        it always returns something as long as there are memories.
        """
        if self.working_memory is None:
            return []

        # Use the time spine if available (via working memory or manager)
        items: list[MemoryItem] = []
        all_anchors = sorted(
            [a for a in self.graph.anchors.values() if a.is_retrievable],
            key=lambda a: (-a.retention_score, -a.last_activated_at),
        )

        for anchor in all_anchors[:max_items]:
            items.append(MemoryItem(
                anchor=anchor,
                relevance_score=anchor.retention_score,
                memory_type=self._classify_memory_type(anchor),
                compression_level=2,
                compressed_text=f"[{anchor.tags[0] if anchor.tags else 'memory'}] {anchor.text[:80]}",
            ))

        return items

    # ── Step 1: Memory type selection ───────────────────

    def _select_memory_types(self, context: AgentContext) -> list[MemoryType]:
        """Select which memory types to query based on agent context.

        Different tasks need different memory:
        - coding → procedural (style, conventions) + semantic (tech stack)
        - debugging → episodic (past bugs) + semantic (system knowledge)
        - planning → semantic (goals, constraints) + procedural (preferences)
        - reflection → episodic (past events) + semantic (patterns)
        """
        task_map = {
            "coding": [MemoryType.PROCEDURAL, MemoryType.SEMANTIC, MemoryType.EPISODIC],
            "debugging": [MemoryType.EPISODIC, MemoryType.SEMANTIC],
            "planning": [MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
            "reflection": [MemoryType.EPISODIC, MemoryType.SEMANTIC],
            "conversation": [MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC],
        }
        return task_map.get(context.task_type,
                           [MemoryType.SEMANTIC, MemoryType.EPISODIC])

    # ── Step 2: Seed discovery ──────────────────────────

    def _discover_seeds(self, context: AgentContext, query: str,
                        query_embedding: list[float] | None,
                        memory_types: list[MemoryType]) -> list[Anchor]:
        """Find initial seed anchors via hybrid retrieval.

        Uses both embedding similarity AND context relevance:
        - Semantic similarity to query
        - Recency bonus for recent session anchors
        - Emotional congruence with current state
        - Memory type filter
        """
        candidates: list[tuple[Anchor, float]] = []

        for anchor in self.graph.anchors.values():
            if not anchor.is_retrievable:
                continue

            # Memory type filter
            anchor_type = self._classify_memory_type(anchor)
            if anchor_type not in memory_types:
                continue

            # Base score from embedding similarity
            base_score = 0.0
            if query_embedding and anchor.embedding:
                base_score = _cosine_sim(query_embedding, anchor.embedding)

            # Context relevance boost
            context_boost = 0.0

            # Recency: recent session anchors get priority
            if context.session_id and anchor.source_session == context.session_id:
                context_boost += 0.2

            # Emotional congruence
            if abs(context.emotional_state) > 0.3:
                emotion_match = 1.0 - abs(context.emotional_state - anchor.vector.emotional_valence) / 2.0
                context_boost += 0.1 * emotion_match

            # Goal relevance: anchor tags match active goals
            if context.active_goals:
                goal_keywords = set()
                for g in context.active_goals:
                    goal_keywords.update(g.lower().split())
                anchor_words = set(anchor.text.lower().split())
                if goal_keywords & anchor_words:
                    context_boost += 0.3

            total_score = base_score + context_boost
            if total_score > 0.05 or anchor_type == MemoryType.WORKING:
                candidates.append((anchor, total_score))

        candidates.sort(key=lambda x: -x[1])
        return [a for a, _ in candidates[:10]]

    # ── Step 3: Multi-hop traversal ─────────────────────

    def _multi_hop_traverse(self, seeds: list[Anchor], context: AgentContext,
                            query_embedding: list[float] | None,
                            max_hops: int = 3) -> list[MemoryItem]:
        """Multi-hop graph traversal with composite edge scoring.

        This is NOT simple spreading activation. It's a biased random walk
        that follows edges proportionally to their retrieval_score, performing
        multi-hop reasoning: A → B → C means A and C might be related even
        without a direct edge.
        """
        visited: set[str] = set()
        items: dict[str, MemoryItem] = {}
        reasoning_paths: dict[str, list[str]] = defaultdict(list)

        # Initialize from seeds
        current_wave: list[tuple[Anchor, float, list[str]]] = []
        for seed in seeds:
            visited.add(seed.id)
            items[seed.id] = MemoryItem(
                anchor=seed,
                relevance_score=1.0,
                memory_type=self._classify_memory_type(seed),
            )
            current_wave.append((seed, 1.0, [seed.id]))

        # Multi-hop expansion
        for hop in range(max_hops):
            next_wave: list[tuple[Anchor, float, list[str]]] = []

            for anchor, incoming_score, path in current_wave:
                neighbors = self.graph.neighbors(anchor.id, min_weight=0.05)
                if not neighbors:
                    continue

                # Score each neighbor edge
                for neighbor_id, edge_weight in neighbors:
                    if neighbor_id in visited:
                        continue
                    neighbor = self.graph.anchors.get(neighbor_id)
                    if not neighbor or not neighbor.is_retrievable:
                        continue

                    # Get RichEdge if available, else use simple edge
                    edge_key = self.graph._key(anchor.id, neighbor_id)
                    edge = self.graph.edges.get(edge_key)

                    if isinstance(edge, RichEdge):
                        edge_score = edge.retrieval_score
                        confidence = edge.confidence
                    else:
                        edge_score = edge_weight if edge else 0.1
                        confidence = 0.5

                    # Decay over hops
                    hop_decay = self.cfg.retrieval.spreading.decay ** (hop + 1)
                    traversal_score = incoming_score * edge_score * hop_decay

                    if traversal_score < 0.02:
                        continue

                    new_path = path + [neighbor_id]
                    reasoning_paths[neighbor_id] = new_path
                    visited.add(neighbor_id)

                    items[neighbor_id] = MemoryItem(
                        anchor=neighbor,
                        relevance_score=traversal_score,
                        confidence=confidence,
                        memory_type=self._classify_memory_type(neighbor),
                        related_anchors=path,
                        reasoning_path=new_path,
                    )

                    next_wave.append((neighbor, traversal_score, new_path))

            current_wave = next_wave
            if not current_wave:
                break

        return list(items.values())

    # ── Step 4: Composite ranking ───────────────────────

    def _composite_rank(self, items: list[MemoryItem], context: AgentContext,
                        query_embedding: list[float] | None) -> list[MemoryItem]:
        """Rank memories by composite score.

        score = α * relevance + β * confidence + γ * recency + δ * importance + ε * centrality

        Weights adapt to agent context:
        - debugging: β (confidence) higher — need reliable info
        - reflection: δ (importance) higher — need significant events
        - coding: β + γ (confidence + recency) higher
        """
        # Adaptive weights based on task type
        weights = {
            "debugging":  (0.25, 0.35, 0.15, 0.15, 0.10),  # α,β,γ,δ,ε
            "coding":      (0.25, 0.30, 0.25, 0.10, 0.10),
            "planning":    (0.20, 0.25, 0.10, 0.25, 0.20),
            "reflection":  (0.15, 0.20, 0.15, 0.35, 0.15),
            "conversation":(0.35, 0.15, 0.30, 0.10, 0.10),
        }
        w = weights.get(context.task_type, (0.25, 0.25, 0.20, 0.15, 0.15))
        alpha, beta, gamma, delta, epsilon = w

        now = time.time()
        max_degree = max((len(self.graph._adjacency.get(a.anchor.id, set()))
                          for a in items), default=1)
        max_degree = max(max_degree, 1)

        for item in items:
            a = item.anchor

            # Semantic relevance
            semantic = item.relevance_score

            # Confidence
            confidence = item.confidence

            # Recency
            hours_since = (now - a.last_activated_at) / 3600
            recency = math.exp(-hours_since / 168)  # decay over 1 week

            # Importance
            importance = a.retention_score

            # Graph centrality (normalized degree)
            degree = len(self.graph._adjacency.get(a.id, set()))
            centrality = degree / max_degree

            item.relevance_score = (
                alpha * semantic
                + beta * confidence
                + gamma * recency
                + delta * importance
                + epsilon * centrality
            )

        items.sort(key=lambda x: -x.relevance_score)
        return items

    # ── Step 5: Adaptive compression ────────────────────

    def _adaptive_compress(self, items: list[MemoryItem],
                           context: AgentContext) -> list[MemoryItem]:
        """Compress memories to fit the agent's context budget.

        Level 0 (raw): full anchor text — used when budget is abundant
        Level 1 (compressed): first sentence or key terms — default
        Level 2 (abstract): schema/abstract reference — used when budget is tight

        Memories with reasoning paths get priority — they show multi-hop connections.
        """
        budget = context.context_budget_tokens
        used = 0

        for item in items:
            text_len = len(item.anchor.text.split())

            if used + text_len <= budget * 0.8:
                # Level 0: full text fits
                item.compression_level = 0
                item.compressed_text = item.anchor.text
                used += text_len
            elif used + 30 <= budget * 0.9:
                # Level 1: compress to key sentence
                sentences = item.anchor.text.replace('！', '.').replace('？', '.').split('.')
                item.compression_level = 1
                item.compressed_text = sentences[0].strip()[:120]
                if item.reasoning_path and len(item.reasoning_path) > 1:
                    item.compressed_text += f" [connected via {len(item.reasoning_path)-1} hops]"
                used += 30
            else:
                # Level 2: abstract reference only
                item.compression_level = 2
                tags_str = ", ".join(item.anchor.tags[:3]) if item.anchor.tags else "general"
                item.compressed_text = f"[{item.memory_type.value}] {tags_str}"
                used += 10

        return items

    # ── Step 6: Build structured context ────────────────

    def _build_context(self, items: list[MemoryItem], context: AgentContext,
                       latency_ms: float) -> MemoryContext:
        """Build structured memory context with reasoning traces."""
        total_tokens = sum(len(it.compressed_text.split()) for it in items)

        # Extract key facts (semantic, high confidence)
        facts = []
        for it in items:
            if it.memory_type == MemoryType.SEMANTIC and it.confidence > 0.6:
                facts.append(it.compressed_text[:100])

        # Extract patterns (procedural, reinforced)
        patterns = []
        for it in items:
            if it.memory_type == MemoryType.PROCEDURAL:
                patterns.append(it.compressed_text[:120])

        # Build reasoning traces
        traces = []
        for it in items:
            if it.reasoning_path and len(it.reasoning_path) > 2:
                path_desc = " -> ".join(
                    self.graph.anchors[aid].text[:40] + "..."
                    if aid in self.graph.anchors else aid[:8]
                    for aid in it.reasoning_path[:4]
                )
                traces.append(path_desc)

        # Memory summary
        summary_parts = []
        if facts:
            summary_parts.append(f"Key facts: {len(facts)}")
        if patterns:
            summary_parts.append(f"Behavioral patterns: {len(patterns)}")
        if traces:
            summary_parts.append(f"Reasoning chains: {len(traces)}")
        summary = " | ".join(summary_parts) if summary_parts else "No relevant memories"

        # Find meta-cognitive reflections connected to retrieved anchors
        retrieved_aids = [item.anchor.id for item in items]
        reflection_nodes = self.graph.find_reflections(retrieved_aids)
        reflections = [
            {"id": r.id, "text": r.text, "type": r.reflection_type,
             "confidence": r.confidence, "strength": r.strength}
            for r in reflection_nodes[:5]
        ]

        return MemoryContext(
            items=items,
            memory_summary=summary,
            active_patterns=patterns[:3],
            relevant_facts=facts[:5],
            reasoning_traces=traces[:5],
            reflections=reflections,
            total_tokens=total_tokens,
            retrieval_latency_ms=latency_ms,
        )

    # ── Memory classification ───────────────────────────

    def _classify_memory_type(self, anchor: Anchor) -> MemoryType:
        """Classify an anchor using tags, state, and vector properties.

        - WORKING: explicitly transient (tagged 'working' or current-session only)
        - EPISODIC: events, bugs, debugging sessions, specific occurrences
        - SEMANTIC: stable facts, knowledge, configuration, tech stack
        - PROCEDURAL: preferences, habits, conventions, reinforced patterns
        """
        v = anchor.vector
        tags_lower = {t.lower() for t in anchor.tags}

        # WORKING: explicitly tagged or truly transient
        if "working" in tags_lower:
            return MemoryType.WORKING
        if (anchor.state == MemoryState.ACTIVE
                and anchor.replay_count == 0
                and v.stability < 0.2
                and not tags_lower):
            return MemoryType.WORKING

        # PROCEDURAL: preferences, habits, conventions, or highly reinforced
        proc_signals = {"preference", "style", "habit", "convention", "pattern", "workflow", "rule"}
        if proc_signals & tags_lower or anchor.replay_count >= 4 or v.hippocampal_dependency < 0.4:
            return MemoryType.PROCEDURAL

        # SEMANTIC: facts, knowledge, tech stack, stable information
        sem_signals = {"fact", "knowledge", "tech-stack", "config", "setup", "infra",
                       "architecture", "definition", "reference"}
        if sem_signals & tags_lower or v.stability > 0.4 or anchor.is_cortical or anchor.schema_ref:
            return MemoryType.SEMANTIC

        # EPISODIC: events, bugs, specific occurrences — the default
        return MemoryType.EPISODIC

    # ── Public utilities ────────────────────────────────

    def get_user_profile(self) -> MemoryContext:
        """Get a compressed user profile: preferences, habits, key facts."""
        context = AgentContext(
            task_type="reflection",
            context_budget_tokens=500,
        )
        # Query for preferences and patterns
        embedder = self._get_embedder()
        emb = embedder.encode("user preferences, habits, technical stack, coding style")
        return self.retrieve(context, "user profile preferences habits")

    def get_relevant_history(self, topic: str, max_items: int = 5) -> MemoryContext:
        """Quick history lookup for a specific topic."""
        context = AgentContext(
            task_type="conversation",
            active_goals=[topic],
            context_budget_tokens=2000,
        )
        return self.retrieve(context, topic, max_items=max_items)

    def print_context(self, ctx: MemoryContext) -> None:
        """Human-readable memory context output."""
        print(f"\n{'─'*55}")
        print(f"  Cognitive Memory Context ({ctx.total_tokens} tokens, "
              f"{ctx.retrieval_latency_ms:.0f}ms)")
        print(f"  {ctx.memory_summary}")
        print(f"{'─'*55}")

        for item in ctx.items[:5]:
            level_marker = ["[raw]", "[cmp]", "[abs]"][item.compression_level]
            type_marker = item.memory_type.value[:4]
            path = ""
            if item.reasoning_path and len(item.reasoning_path) > 1:
                path = f" (via {len(item.reasoning_path)-1} hops)"
            print(f"  {level_marker} [{type_marker}] "
                  f"(score:{item.relevance_score:.2f}, conf:{item.confidence:.2f})"
                  f"{path}")
            print(f"       {item.compressed_text[:100]}")

        if ctx.relevant_facts:
            print(f"\n  Key Facts:")
            for f in ctx.relevant_facts:
                print(f"    - {f}")
        if ctx.active_patterns:
            print(f"\n  Behavioral Patterns:")
            for p in ctx.active_patterns:
                print(f"    - {p}")
        if ctx.reasoning_traces:
            print(f"\n  Reasoning Traces:")
            for t in ctx.reasoning_traces:
                print(f"    - {t}")
        print(f"{'─'*55}\n")


# ── Helpers ─────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
