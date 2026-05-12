"""Dual-Channel Retrieval — System-1 (fast association) + System-2 (goal-directed).

Inspired by Mnemis (93.9% on LoCoMo): two complementary retrieval channels that
switch based on query characteristics.

System-1 (Fast Association): Existing graph similarity + resonance search.
  Answers "what's similar to X?" — fast, embedding-based, associative.

System-2 (Goal-Directed): Hierarchical top-down traversal through summary→detail.
  Answers "all X", "which Y", "what happened before Z", "list everything about W"
  — structured, exhaustive, edge-following.

Triggers for System-2:
  - Structural intent words: all, every, list, which, before, after, last, first, ...
  - System-1 confidence below threshold (< 0.35)
  - Multi-hop/chained reasoning needed
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor
from .graph import StarGraph
from .scheduler import AgentContext, MemoryItem, MemoryType


# ── Structural intent keywords ─────────────────────────────

_STRUCTURAL_INTENT = {
    # Exhaustive enumeration
    'all', 'every', 'list', 'enumerate', 'each', 'any of',
    # Discriminative selection
    'which', 'which one', 'what kind', 'what type', 'select',
    # Temporal comparison
    'before', 'after', 'last', 'first', 'previous', 'next',
    'earlier', 'later', 'since', 'until', 'recently',
    # Causal / sequence
    'how many', 'what caused', 'what led to', 'why did',
    'steps', 'sequence', 'chain', 'flow', 'process',
    # Summarization / aggregation
    'summarize', 'summary', 'overview', 'what happened',
    'history', 'timeline', 'pattern', 'trend',
    # Cross-session
    'across sessions', 'past conversations', 'over time',
    'previously', 'earlier discussion',
}


# ── Data structures ────────────────────────────────────────

@dataclass
class ChannelResult:
    """Result from a single retrieval channel."""
    items: list[MemoryItem]
    channel: str  # "system1" or "system2"
    confidence: float  # 0..1
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class DualChannelOutput:
    """Merged output from dual-channel retrieval."""
    items: list[MemoryItem]
    active_channel: str  # "system1", "system2", or "merged"
    s1_confidence: float
    s2_confidence: float
    s2_triggered: bool
    trigger_reason: str
    total_latency_ms: float


class DualChannelRetriever:
    """System-1 + System-2 dual-channel cognitive retrieval.

    System-1: fast embedding-based associative search (the existing mechanism).
    System-2: goal-directed hierarchical traversal through summary→detail graph.

    Usage:
        retriever = DualChannelRetriever(graph, hub_layer)
        result = retriever.retrieve(query, context, embedding)

        if result.s2_triggered:
            print(f"System-2 engaged: {result.trigger_reason}")
    """

    def __init__(self, graph: StarGraph,
                 s1_confidence_threshold: float = 0.35,
                 s2_max_depth: int = 4,
                 s2_max_items: int = 20,
                 merge_weight_s1: float = 0.4,
                 merge_weight_s2: float = 0.6):
        self.graph = graph
        self.s1_threshold = s1_confidence_threshold
        self.s2_max_depth = s2_max_depth
        self.s2_max_items = s2_max_items
        self.merge_w1 = merge_weight_s1
        self.merge_w2 = merge_weight_s2

        # Cached structural analysis of the graph
        self._abstract_nodes: dict[str, Anchor] = {}  # summary/abstract anchors
        self._hierarchy: dict[str, list[str]] = {}  # parent_id -> [child_ids]
        self._reverse_hierarchy: dict[str, str] = {}  # child_id -> parent_id
        self._last_indexed: float = 0.0

    # ── Intent Detection ────────────────────────────────────

    def _detect_structural_intent(self, query: str) -> tuple[bool, str, list[str]]:
        """Check if query requires goal-directed (System-2) search.

        Returns (needs_s2, reason, matched_keywords).
        """
        query_lower = query.lower()
        matched = []

        # Check multi-word phrases first (longer match = stronger signal)
        phrases = sorted(
            [p for p in _STRUCTURAL_INTENT if ' ' in p],
            key=len, reverse=True,
        )
        for phrase in phrases:
            if phrase in query_lower:
                matched.append(phrase)

        # Check single words
        for word in _STRUCTURAL_INTENT:
            if ' ' not in word and word not in matched:
                # Word boundary check
                import re
                if re.search(r'\b' + re.escape(word) + r'\b', query_lower):
                    matched.append(word)

        if matched:
            # Classify intent type
            if any(m in {'all', 'every', 'list', 'enumerate', 'each'} for m in matched):
                reason = "exhaustive_enumeration"
            elif any(m in {'which', 'select', 'what kind', 'what type'} for m in matched):
                reason = "discriminative_selection"
            elif any(m in {'before', 'after', 'last', 'first', 'previous', 'next',
                          'earlier', 'later', 'since', 'until'} for m in matched):
                reason = "temporal_comparison"
            elif any(m in {'how many', 'what caused', 'why did', 'steps', 'sequence',
                          'chain', 'process'} for m in matched):
                reason = "causal_chain"
            elif any(m in {'summarize', 'summary', 'overview', 'history', 'timeline',
                          'pattern', 'trend'} for m in matched):
                reason = "aggregation"
            else:
                reason = "structural_intent"
            return True, reason, matched

        return False, "", []

    def _needs_structural_search(self, query: str) -> tuple[bool, str]:
        """Determine if System-2 is warranted based on query characteristics.

        Also checks for question-words that imply multi-hop reasoning.
        """
        has_intent, reason, keywords = self._detect_structural_intent(query)
        if has_intent:
            return True, reason

        # Check for temporal query patterns (dates, durations, sequences)
        import re
        temporal_patterns = [
            r'\b\d{4}[-/]\d{1,2}\b',           # dates like 2024-01
            r'\b\d+\s*(?:days?|weeks?|months?|years?)\s*(?:ago|before|earlier)\b',
            r'\b(?:today|yesterday|tomorrow|last week|next month)\b',
        ]
        for pat in temporal_patterns:
            if re.search(pat, query, re.IGNORECASE):
                return True, "temporal_pattern"

        return False, ""

    # ── Hierarchy Indexing ──────────────────────────────────

    def _index_hierarchy(self, force: bool = False) -> None:
        """Build the summary→detail hierarchy from graph edges.

        Summary anchors (compression summaries, hub-linked nodes) are parents.
        Their source anchors (linked via 'compresses' edges) are children.

        This is lazy-indexed: only rebuilds if graph changed since last index.
        """
        now = time.time()
        if not force and self._last_indexed > 0 and \
           (now - self._last_indexed) < 60:
            return

        self._abstract_nodes.clear()
        self._hierarchy.clear()
        self._reverse_hierarchy.clear()

        # Find abstract/summary anchors (tagged with compression level or schema_ref)
        for aid, anchor in self.graph.anchors.items():
            is_abstract = False
            # Check for compression level tags
            for tag in anchor.tags:
                if tag.startswith('level:') or tag in ('summary', 'abstract', 'pattern'):
                    is_abstract = True
                    break
            # Check schema_ref — indicates abstraction
            if anchor.schema_ref:
                is_abstract = True
            # High semantic density indicates abstraction
            if anchor.semantic_density > 0.6:
                is_abstract = True

            if is_abstract:
                self._abstract_nodes[aid] = anchor

        # Find parent-child relationships via edges
        for edge_key, edge in self.graph.edges.items():
            src, tgt = edge_key
            edge_type = getattr(edge, 'edge_type', '')

            # 'compresses' edges: parent (summary) -> child (source detail)
            if edge_type == 'compresses':
                if src in self._abstract_nodes:
                    parent, child = src, tgt
                elif tgt in self._abstract_nodes:
                    parent, child = tgt, src
                else:
                    continue
                self._hierarchy.setdefault(parent, []).append(child)
                self._reverse_hierarchy[child] = parent

            # 'causal' / 'temporal' edges: sequential relationships
            elif edge_type in ('causal', 'temporal', 'sequence', 'state_transition'):
                self._hierarchy.setdefault(src, []).append(tgt)

        self._last_indexed = now

    # ── System-2: Goal-Directed Traversal ───────────────────

    def _system2_search(self, query: str, query_embedding: list[float] | None,
                        context: AgentContext, max_items: int,
                        intent_reason: str) -> ChannelResult:
        """Goal-directed hierarchical traversal.

        Strategy depends on intent type:
        - exhaustive_enumeration → collect all children of matching abstract nodes
        - temporal_comparison → follow temporal edges in sequence
        - causal_chain → follow causal edges forward/backward
        - aggregation → traverse up to find abstract summaries
        - discriminative → narrow down by filtering children
        """
        t0 = time.perf_counter()
        self._index_hierarchy()

        results: list[MemoryItem] = []
        seen_ids: set[str] = set()

        if not query_embedding:
            results_latency = (time.perf_counter() - t0) * 1000
            return ChannelResult(items=[], channel="system2",
                                confidence=0.0, latency_ms=results_latency)

        # Step 1: Find best-matching abstract/parent nodes
        scored_parents = self._score_abstract_nodes(query_embedding, top_k=5)

        if intent_reason in ('exhaustive_enumeration', 'aggregation'):
            # Collect ALL children of matching abstract nodes (exhaustive)
            for parent_id, parent_score in scored_parents:
                children = self._get_all_descendants(parent_id, max_depth=self.s2_max_depth)
                for child_id in children:
                    if child_id in seen_ids or child_id not in self.graph.anchors:
                        continue
                    seen_ids.add(child_id)
                    child = self.graph.anchors[child_id]
                    if child.is_retrievable:
                        results.append(MemoryItem(
                            anchor=child,
                            relevance_score=parent_score * 0.9,
                            memory_type=MemoryType.EPISODIC,
                            compression_level=0,
                            compressed_text=child.text[:120],
                        ))

        elif intent_reason == 'temporal_comparison':
            # Follow temporal edges to build event sequence
            for parent_id, _ in scored_parents:
                timeline = self._follow_temporal_chain(
                    parent_id, max_steps=self.s2_max_depth, max_items=max_items)
                for anchor, step_score in timeline:
                    if anchor.id in seen_ids:
                        continue
                    seen_ids.add(anchor.id)
                    results.append(MemoryItem(
                        anchor=anchor,
                        relevance_score=step_score,
                        memory_type=MemoryType.EPISODIC,
                        compression_level=0,
                        compressed_text=anchor.text[:120],
                    ))

        elif intent_reason == 'causal_chain':
            # Bidirectional causal traversal
            for parent_id, _ in scored_parents:
                causal_chain = self._follow_causal_chain(
                    parent_id, max_steps=self.s2_max_depth)
                for anchor, step_score in causal_chain:
                    if anchor.id in seen_ids:
                        continue
                    seen_ids.add(anchor.id)
                    results.append(MemoryItem(
                        anchor=anchor,
                        relevance_score=step_score,
                        memory_type=MemoryType.EPISODIC,
                        compression_level=0,
                        compressed_text=anchor.text[:120],
                    ))

        else:  # discriminative_selection or general
            # Narrow-down: score individual children of matching parents
            for parent_id, parent_score in scored_parents:
                children = self._get_direct_children(parent_id)
                for child_id in children:
                    if child_id in seen_ids or child_id not in self.graph.anchors:
                        continue
                    child = self.graph.anchors[child_id]
                    if not child.is_retrievable:
                        continue
                    # Score child by semantic match to query
                    child_sim = 0.0
                    if child.embedding and query_embedding:
                        dot = sum(a * b for a, b in zip(query_embedding, child.embedding))
                        na = math.sqrt(sum(x**2 for x in query_embedding))
                        nb = math.sqrt(sum(x**2 for x in child.embedding))
                        child_sim = max(0.0, dot / (na * nb + 1e-8))
                    combined = 0.3 * parent_score + 0.7 * child_sim
                    if combined > 0.15:
                        seen_ids.add(child_id)
                        results.append(MemoryItem(
                            anchor=child,
                            relevance_score=combined,
                            memory_type=MemoryType.EPISODIC,
                            compression_level=0,
                            compressed_text=child.text[:120],
                        ))

        # Sort and truncate
        results.sort(key=lambda i: i.relevance_score, reverse=True)
        results = results[:max_items]

        latency = (time.perf_counter() - t0) * 1000
        confidence = min(0.9, len(results) / max(1, max_items)) if results else 0.1

        return ChannelResult(
            items=results, channel="system2",
            confidence=confidence, latency_ms=latency,
            metadata={
                "intent": intent_reason,
                "parents_scored": len(scored_parents),
                "children_collected": len(seen_ids),
            },
        )

    # ── Hierarchy Navigation Helpers ────────────────────────

    def _score_abstract_nodes(self, query_embedding: list[float],
                              top_k: int = 5) -> list[tuple[str, float]]:
        """Score abstract nodes by embedding similarity to query."""
        scored = []
        for aid, anchor in self._abstract_nodes.items():
            if not anchor.embedding:
                continue
            dot = sum(a * b for a, b in zip(query_embedding, anchor.embedding))
            na = math.sqrt(sum(x**2 for x in query_embedding))
            nb = math.sqrt(sum(x**2 for x in anchor.embedding))
            sim = max(0.0, dot / (na * nb + 1e-8))
            if sim > 0.2:
                scored.append((aid, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _get_direct_children(self, parent_id: str) -> list[str]:
        """Get immediate children of a parent node."""
        return list(self._hierarchy.get(parent_id, []))

    def _get_all_descendants(self, parent_id: str,
                             max_depth: int = 4) -> list[str]:
        """BFS to collect all descendants up to max_depth."""
        visited = set()
        queue = [parent_id]
        for _ in range(max_depth):
            if not queue:
                break
            next_queue = []
            for node_id in queue:
                for child_id in self._hierarchy.get(node_id, []):
                    if child_id not in visited:
                        visited.add(child_id)
                        next_queue.append(child_id)
            queue = next_queue
        return list(visited)

    def _follow_temporal_chain(self, start_id: str, max_steps: int = 4,
                                max_items: int = 20) -> list[tuple[Anchor, float]]:
        """Follow temporal/sequential edges to build an event timeline."""
        results: list[tuple[Anchor, float]] = []
        visited: set[str] = set()
        queue = [(start_id, 0)]
        score_decay = 1.0

        while queue and len(results) < max_items:
            node_id, depth = queue.pop(0)
            if depth >= max_steps or node_id in visited:
                continue
            visited.add(node_id)

            if node_id in self.graph.anchors:
                anchor = self.graph.anchors[node_id]
                if anchor.is_retrievable:
                    results.append((anchor, score_decay))
                    score_decay *= 0.85

            # Follow temporal edges
            for child_id in self._hierarchy.get(node_id, []):
                if child_id not in visited and self.graph.anchors.get(child_id):
                    queue.append((child_id, depth + 1))

        return results

    def _follow_causal_chain(self, start_id: str, max_steps: int = 4
                              ) -> list[tuple[Anchor, float]]:
        """Follow causal edges bidirectionally (causes → effects)."""
        results: list[tuple[Anchor, float]] = []
        visited: set[str] = set()

        # Forward chain: start_id → effects
        queue = [(start_id, 0)]
        score = 1.0
        while queue:
            node_id, depth = queue.pop(0)
            if depth >= max_steps or node_id in visited:
                continue
            visited.add(node_id)
            if node_id in self.graph.anchors:
                anchor = self.graph.anchors[node_id]
                if anchor.is_retrievable:
                    results.append((anchor, score))
                    score *= 0.8
            for child_id in self._hierarchy.get(node_id, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        # Backward chain: causes → start_id
        current = start_id
        rev_score = 0.7
        for _ in range(max_steps):
            parent = self._reverse_hierarchy.get(current)
            if not parent or parent in visited:
                break
            visited.add(parent)
            if parent in self.graph.anchors:
                anchor = self.graph.anchors[parent]
                if anchor.is_retrievable:
                    results.append((anchor, rev_score))
                    rev_score *= 0.8
            current = parent

        return results

    # ── System-1: Fast Association (delegates to existing) ──

    def _system1_search(self, query: str, query_embedding: list[float] | None,
                        context: AgentContext, max_items: int) -> ChannelResult:
        """System-1 fast associative search using existing graph mechanisms.

        This is the default channel — embedding similarity + resonance.
        """
        t0 = time.perf_counter()
        results: list[MemoryItem] = []

        if not query_embedding:
            latency = (time.perf_counter() - t0) * 1000
            return ChannelResult(items=[], channel="system1",
                                confidence=0.0, latency_ms=latency)

        # Score all anchors by embedding similarity
        scored = []
        for anchor in self.graph.anchors.values():
            if not anchor.is_retrievable or not anchor.embedding:
                continue
            dot = sum(a * b for a, b in zip(query_embedding, anchor.embedding))
            na = math.sqrt(sum(x**2 for x in query_embedding))
            nb = math.sqrt(sum(x**2 for x in anchor.embedding))
            sim = max(0.0, dot / (na * nb + 1e-8))

            # Apply retention weighting
            score = sim * anchor.retention_score
            if score > 0.05:
                scored.append((anchor, score))

        scored.sort(key=lambda x: -x[1])

        for anchor, score in scored[:max_items]:
            results.append(MemoryItem(
                anchor=anchor,
                relevance_score=score,
                memory_type=MemoryType.SEMANTIC,
                compression_level=0,
                compressed_text=anchor.text[:120],
            ))

        latency = (time.perf_counter() - t0) * 1000

        # Compute confidence: average top-3 semantic similarity
        top_sims = [s for _, s in scored[:3]]
        avg_sim = sum(top_sims) / max(1, len(top_sims))
        confidence = min(1.0, avg_sim * 1.5)  # scale up so 0.5 sim → 0.75 confidence

        return ChannelResult(
            items=results, channel="system1",
            confidence=confidence, latency_ms=latency,
            metadata={"top_sim": top_sims[0] if top_sims else 0.0},
        )

    # ── Main Entry Point ────────────────────────────────────

    def retrieve(self, query: str,
                 context: AgentContext | None = None,
                 query_embedding: list[float] | None = None,
                 max_items: int = 10) -> DualChannelOutput:
        """Dual-channel retrieval with automatic System-2 triggering.

        Flow:
        1. Run System-1 (always, fast)
        2. Check if System-2 should be triggered
        3. If triggered, run System-2 and merge results
        4. Return merged + metadata
        """
        if context is None:
            context = AgentContext(task_type="conversation")

        t_total = time.perf_counter()

        # System-1: always runs (fast associative search)
        s1_result = self._system1_search(query, query_embedding, context, max_items)

        # Check if System-2 is needed
        need_s2, s2_reason = self._needs_structural_search(query)
        low_confidence = s1_result.confidence < self.s1_threshold

        if not need_s2 and not low_confidence:
            # System-1 only
            total_ms = (time.perf_counter() - t_total) * 1000
            return DualChannelOutput(
                items=s1_result.items, active_channel="system1",
                s1_confidence=s1_result.confidence, s2_confidence=0.0,
                s2_triggered=False, trigger_reason="",
                total_latency_ms=total_ms,
            )

        # System-2 triggered
        if need_s2:
            trigger_reason = f"structural_intent:{s2_reason}"
        else:
            trigger_reason = f"low_confidence:{s1_result.confidence:.2f}"

        s2_result = self._system2_search(
            query, query_embedding, context, max_items, s2_reason)

        # Merge: S1 + S2 results, deduplicated
        merged = self._merge_channels(s1_result, s2_result, max_items)

        total_ms = (time.perf_counter() - t_total) * 1000
        return DualChannelOutput(
            items=merged, active_channel="merged",
            s1_confidence=s1_result.confidence,
            s2_confidence=s2_result.confidence,
            s2_triggered=True, trigger_reason=trigger_reason,
            total_latency_ms=total_ms,
        )

    def _merge_channels(self, s1: ChannelResult, s2: ChannelResult,
                        max_items: int) -> list[MemoryItem]:
        """Merge and re-rank results from both channels.

        System-2 results get higher weight when it was triggered for
        structural/exhaustive queries. System-1 results get boost for
        pure similarity queries.

        Deduplication: if an anchor appears in both channels, take the
        higher-scored version and boost its score.
        """
        merged: dict[str, MemoryItem] = {}

        # Add S2 first (higher priority when triggered)
        for item in s2.items:
            if item.anchor:
                merged[item.anchor.id] = item
            else:
                # Items without anchors use text as key
                merged[item.compressed_text[:80]] = item

        # Add S1, boosting if already in S2
        for item in s1.items:
            key = item.anchor.id if item.anchor else item.compressed_text[:80]
            if key in merged:
                # Boost: item found by both channels
                existing = merged[key]
                existing.relevance_score = max(
                    existing.relevance_score,
                    item.relevance_score,
                ) * 1.15  # 15% boost for cross-channel confirmation
                merged[key] = existing
            else:
                merged[key] = item

        # Sort by relevance, return top max_items
        sorted_items = sorted(merged.values(),
                             key=lambda i: i.relevance_score, reverse=True)
        return sorted_items[:max_items]

    # ── Utilities ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "abstract_nodes_indexed": len(self._abstract_nodes),
            "hierarchy_edges": sum(len(v) for v in self._hierarchy.values()),
            "max_hierarchy_depth": self._max_depth() if self._hierarchy else 0,
            "last_indexed_ago_sec": time.time() - self._last_indexed,
        }

    def _max_depth(self) -> int:
        """Compute maximum depth of the hierarchy tree."""
        depths: dict[str, int] = {}

        def dfs(node_id: str) -> int:
            if node_id in depths:
                return depths[node_id]
            max_child_depth = 0
            for child_id in self._hierarchy.get(node_id, []):
                max_child_depth = max(max_child_depth, dfs(child_id) + 1)
            depths[node_id] = max_child_depth
            return max_child_depth

        for parent in self._hierarchy:
            if parent not in self._reverse_hierarchy:  # root nodes
                dfs(parent)
        return max(depths.values()) if depths else 0
