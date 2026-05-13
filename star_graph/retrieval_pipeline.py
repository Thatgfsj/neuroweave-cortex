"""RetrievalPipeline — all retrieval orchestration for the cognitive memory system.

Handles multi-path recall: exact cache, raw buffer, graph dimensional descent,
dual-channel (System-1 + System-2), cross-modal, and cascade recall.

Takes a MemoryRuntime reference for subsystem access.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from .scheduler import AgentContext, MemoryContext, MemoryItem, MemoryType
from .anchor import Anchor
from .router import RouteResult
from .dual_channel import DualChannelOutput
from .multimodal import CrossModalResult


class RetrievalPipeline:
    """All retrieval orchestration for a MemoryRuntime.

    Instantiated by MemoryManager with a reference to the runtime.
    """

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

    def __init__(self, runtime):
        self._rt = runtime  # MemoryRuntime instance

    # ── Primary recall entry point ────────────────────────────

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

        embedder = self._rt._get_embedder()
        query_emb = embedder.encode(query) if query else None
        t_start = time.time()

        # Path 0: Exact cache lookup (deterministic O(1) bypass)
        t0 = time.time()
        exact_results: list[MemoryItem] = []
        if query:
            query_keys = self._rt.exact_cache.query_keys(query)
            for key in query_keys:
                entries = self._rt.exact_cache.get(key)
                for entry in entries:
                    anchor = self._rt.graph.anchors.get(entry.anchor_id)
                    if anchor and anchor.is_retrievable:
                        exact_results.append(MemoryItem(
                            anchor=anchor,
                            relevance_score=0.95 + entry.confidence * 0.05,
                            memory_type=MemoryType.SEMANTIC,
                            compression_level=0,
                            compressed_text=entry.text,
                        ))
            for key in query_keys:
                wm_entries = self._rt.working_memory.get_exact(key)
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
        raw_results = self._rt.raw_buffer.search(
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
        seen_texts = {self._rt._normalize_text(item.compressed_text) for item in merged_items
                      if hasattr(item, 'compressed_text') and item.compressed_text}

        for chunk, score in raw_results:
            norm_text = self._rt._normalize_text(chunk.text[:120])
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
            norm_text = self._rt._normalize_text(text)
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

        self._rt.tracer.record("recall", attributes={
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
        """Run System-2 (goal-directed) recall via DualChannelRetriever."""
        t_start = time.time()
        dc_output = self._rt.dual_channel.retrieve(
            query=query, context=context,
            query_embedding=None, max_items=max_items,
        )
        total_ms = (time.time() - t_start) * 1000

        self._rt.tracer.record("recall", attributes={
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

    def dual_recall(self, query: str = "",
                    context: AgentContext | None = None,
                    max_items: int = 10) -> DualChannelOutput:
        """Dual-channel recall: System-1 (fast association) + System-2 (goal-directed).

        System-2 auto-triggers when:
        - Query contains structural intent words (all, which, before, list, ...)
        - System-1 confidence drops below threshold (< 0.35)
        """
        if context is None:
            context = AgentContext(task_type="conversation")
        embedder = self._rt._get_embedder()
        query_emb = embedder.encode(query) if query else None

        return self._rt.dual_channel.retrieve(
            query=query, context=context,
            query_embedding=query_emb, max_items=max_items,
        )

    # ── Sparse recall ─────────────────────────────────────────

    def route_to_cortex(self, query: str = "",
                        query_embedding: list[float] | None = None
                        ) -> list[RouteResult]:
        """Route a query to the most relevant cortices (1-3)."""
        return self._rt.router.route(query, query_embedding=query_embedding)

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

        embedder = self._rt._get_embedder()
        query_emb = embedder.encode(query) if query else None

        # Phase 1: Cortex routing
        routes = self._rt.router.route(query, query_embedding=query_emb)
        active_cortices = [r.cortex for r in routes]

        # Phase 2: Local recall from activated cortices only
        all_items: list = []
        for cortex in active_cortices:
            ctx = cortex.recall(query=query, context=context, max_items=max_items)
            all_items.extend(ctx.items)

        # Phase 3: Memory gating
        gated = self._rt.gate.gate(all_items, context, query_emb, query)

        return {
            "items": gated,
            "active_cortices": [c.config.name for c in active_cortices],
            "route_scores": {r.cortex.config.name: r.score for r in routes},
            "total_candidates": len(all_items),
            "gated_count": len(gated),
        }

    # ── 5-Layer dimensional descent ───────────────────────────

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
        """
        if context is None:
            context = AgentContext(task_type="conversation")
        embedder = self._rt._get_embedder()
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
        brain_hits = self._rt.brain.query_common_nodes(query_emb, query, top_k=5)
        if brain_hits:
            result["brain_hit"] = True
            items = []
            for anchor in brain_hits:
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
        routes = self._rt.router.route(query, query_embedding=query_emb)
        active_cortices = [r.cortex for r in routes if r.score > 0.1]
        result["active_cortices"] = [c.config.name for c in active_cortices]

        all_candidates: list = list(result["items"])
        for cortex in active_cortices[:3]:
            ctx = cortex.recall(query=query, context=context, max_items=max_items)
            all_candidates.extend(ctx.items)

        result["total_candidates"] = len(all_candidates)

        if all_candidates:
            gated = self._rt.gate.gate(all_candidates, context, query_emb, query)
            if len(gated) >= max_items:
                result["items"] = gated[:max_items]
                result["layer_used"] = "cortex"
                return result
            all_candidates = gated

        # ── Layer 1.5: Community-aware retrieval ──────────
        if self._rt._community_detection and self._rt._community_detection.communities:
            result["layers_visited"].append("community")
            best_community = None
            best_csim = -1.0
            route_weight = getattr(
                getattr(self._rt.cfg, 'community', None),
                'route_centroid_weight', 0.7)
            for c in self._rt._community_detection.communities:
                if c.centroid_embedding and query_emb:
                    sim = _cosine_sim(query_emb, c.centroid_embedding)
                    if sim * route_weight > best_csim:
                        best_csim = sim * route_weight
                        best_community = c.id

            if best_community and best_csim > 0.2:
                community_filter = {best_community}
                comm_items = self._rt.scheduler._community_aware_retrieve(
                    context, query, query_emb,
                    self._rt.scheduler._select_memory_types(context),
                    max_items,
                    community_filter=community_filter,
                    community_detection=self._rt._community_detection,
                )
                existing_ids = {item.anchor.id for item in all_candidates}
                for item in comm_items:
                    if item.anchor.id not in existing_ids:
                        all_candidates.append(item)
                        existing_ids.add(item.anchor.id)

                if len(all_candidates) >= max_items:
                    gated = self._rt.gate.gate(all_candidates, context, query_emb, query)
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
                    related_hubs = self._rt.hublayer.traverse_hubs(hub_id, max_hops=2, max_results=3)
                    for hub in related_hubs:
                        if hub.id not in existing_ids:
                            hub_anchor = Anchor.create(
                                text=hub.text,
                                embedding=hub.embedding,
                                importance=hub.importance,
                            )
                            all_candidates.append(MemoryItem(
                                anchor=hub_anchor,
                                relevance_score=0.6,
                                memory_type=MemoryType.SEMANTIC,
                                compression_level=1,
                                compressed_text=hub.text[:120],
                            ))
                            existing_ids.add(hub.id)

        if len(all_candidates) >= max_items:
            gated = self._rt.gate.gate(all_candidates, context, query_emb, query)
            result["items"] = gated[:max_items]
            result["layer_used"] = "hub"
            return result

        # ── Layer 3: 2D planar (time × importance) ───────
        result["layers_visited"].append("2d_plane")
        plane_items = []
        now = time.time()
        for cortex in active_cortices:
            for anchor in cortex.graph.anchors.values():
                if not anchor.is_retrievable or anchor.id in existing_ids:
                    continue
                hours = (now - anchor.last_activated_at) / 3600
                recency = math.exp(-hours / 168)
                score = 0.6 * recency + 0.4 * anchor.retention_score
                plane_items.append((anchor, score))

        plane_items.sort(key=lambda x: -x[1])
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
        timeline_results = self._rt.scan_timeline(max_days=90, max_clusters=max_items)
        for tc in timeline_results:
            for anchor_id in tc.get("anchor_ids", []):
                if anchor_id in existing_ids:
                    continue
                anchor = self._rt.graph.anchors.get(anchor_id)
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

        if all_candidates:
            result["items"] = [item for item in all_candidates[:max_items] if item.anchor is not None]
        result["layer_used"] = "timeline"
        return result

    # ── Cascade recall ────────────────────────────────────────

    def cascade_recall(self, query: str = "",
                       max_chains: int = 5,
                       max_depth: int = 5) -> list:
        """Causal chain recall — trace event chains from query-relevant seeds."""
        embedder = self._rt._get_embedder()
        query_emb = embedder.encode(query)

        seeds = self._rt.scheduler._discover_seeds(
            AgentContext(task_type="reflection"), query, query_emb,
            self._rt.scheduler._select_memory_types(AgentContext()))

        seed_ids = [a.id for a in seeds[:3]]
        chains = self._rt.cascade.cascade_from_seeds(seed_ids, max_depth, max_chains)
        return [
            {"narrative": c.narrative, "confidence": c.total_confidence,
             "depth": c.depth, "type": c.chain_type}
            for c in chains if c.is_valid
        ]

    # ── Cross-modal retrieval ─────────────────────────────────

    def cross_modal_recall(self, query: str = "",
                          query_image_path: str = "",
                          max_items: int = 10,
                          text_weight: float = 0.6,
                          image_weight: float = 0.4) -> list[CrossModalResult]:
        """Retrieve memories across text and image modalities."""
        all_anchors = dict(self._rt.graph.anchors)
        for cortex in self._rt.router.cortices:
            all_anchors.update(cortex.graph.anchors)

        return self._rt.cross_modal_retriever.retrieve(
            anchors=all_anchors,
            query_text=query,
            query_image_path=query_image_path,
            top_k=max_items,
            text_weight=text_weight,
            image_weight=image_weight,
        )

    def image_search(self, image_path: str,
                    max_items: int = 10) -> list[CrossModalResult]:
        """Find images visually similar to a query image."""
        all_anchors = dict(self._rt.graph.anchors)
        for cortex in self._rt.router.cortices:
            all_anchors.update(cortex.graph.anchors)
        return self._rt.cross_modal_retriever.image_search(
            anchors=all_anchors,
            query_image_path=image_path,
            top_k=max_items,
        )

    def text_to_image(self, query: str,
                     max_items: int = 10) -> list[CrossModalResult]:
        """Find images matching a text description (CLIP required for best results)."""
        all_anchors = dict(self._rt.graph.anchors)
        for cortex in self._rt.router.cortices:
            all_anchors.update(cortex.graph.anchors)
        return self._rt.cross_modal_retriever.text_to_image(
            anchors=all_anchors,
            query_text=query,
            top_k=max_items,
        )


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
