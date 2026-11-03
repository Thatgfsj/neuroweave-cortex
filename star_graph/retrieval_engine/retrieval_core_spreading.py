"""RetrievalCore spreading mixin — spreading activation and descent paths.

Extracted verbatim from the original retrieval_core.py (issue #7) — pure move,
no behaviour change. Contains:
  - _spreading_recall (Path C spreading activation graph walk)
  - recall_with_spreading (spreading-first retrieval)
  - dual_recall (dual-channel System-1 + System-2)
  - route_to_cortex (cortex routing)
  - sparse_recall (sparse-activation pipeline)
  - retrieve_with_descent (5-layer dimensional descent)
  - cascade_recall (causal chain recall)

Expects ``self._rt`` (a MemoryRuntime reference) to be available, set by the
composing class.
"""

from __future__ import annotations

import math
import time

from ..scheduler import AgentContext, MemoryContext, MemoryItem, MemoryType
from ..anchor import Anchor
from ..router import RouteResult
from .dual_channel import DualChannelOutput
from ..math_utils import cosine_sim as _cosine_sim


class SpreadingMixin:
    """Mixin with spreading, sparse, descent and cascade retrieval paths.

    Expects self._rt to be a MemoryRuntime instance (set by the inheriting class).
    """

    def _spreading_recall(self, query_emb: list[float] | None = None,
                          query: str = "", max_items: int = 10) -> list[MemoryItem]:
        """Path C: Spreading activation graph walk (#52).

        Finds seeds via ANN, then BFS-walks the graph with edge-type-weighted
        traversal. Returns MemoryItem list for merging into the main recall path.
        """
        if query_emb is None and query:
            embedder = self._rt._get_embedder()
            query_emb = embedder.encode(query)
        if query_emb is None:
            return []

        try:
            activated = self._rt.spreading.activate(
                query_embedding=query_emb, top_k=max_items)
        except Exception:
            return []

        items: list[MemoryItem] = []
        for node in activated:
            anchor = node.anchor
            if anchor is None or not anchor.is_retrievable:
                continue
            # Scale activation (0..~1.0 range) to relevance score (0..1.0)
            score = min(1.0, node.accumulated_activation)
            items.append(MemoryItem(
                anchor=anchor,
                relevance_score=score,
                memory_type=MemoryType.SEMANTIC,
                compression_level=node.activation_depth,
                compressed_text=anchor.text[:200],
            ))
        return items

    def recall_with_spreading(self, query: str = "",
                              context: AgentContext | None = None,
                              max_items: int = 10,
                              spreading_weight: float = 0.6) -> MemoryContext:
        """Spreading-first recall: graph walk as primary signal, embedding as refinement.

        Uses the SpreadingActivation BFS walk as the primary retrieval mechanism,
        with embedding similarity as a secondary re-ranking factor. This favours
        structurally-connected memories over pure semantic similarity.
        """
        if context is None:
            context = AgentContext(task_type="conversation")
        t_start = time.time()
        embedder = self._rt._get_embedder()
        query_emb = embedder.encode(query) if query else None

        # Run spreading activation as primary path
        activated = self._rt.spreading.activate(
            query_embedding=query_emb, top_k=max_items * 2)

        # Convert to MemoryItem with combined topology+embedding score
        items: list[MemoryItem] = []
        for node in activated:
            anchor = node.anchor
            if anchor is None or not anchor.is_retrievable:
                continue
            topo_score = min(1.0, node.accumulated_activation)
            emb_score = 0.0
            if query_emb and anchor.embedding:
                emb_score = _cosine_sim(query_emb, anchor.embedding)
            combined = spreading_weight * topo_score + (1 - spreading_weight) * emb_score
            items.append(MemoryItem(
                anchor=anchor,
                relevance_score=combined,
                memory_type=MemoryType.SEMANTIC,
                compression_level=node.activation_depth,
                compressed_text=anchor.text[:200],
            ))

        items.sort(key=lambda i: -i.relevance_score)
        items = items[:max_items]

        total_ms = (time.time() - t_start) * 1000
        self._rt.tracer.record("recall", attributes={
            "query": query[:200],
            "max_items": max_items,
            "final_count": len(items),
            "channel": "spreading_first",
            "total_ms": round(total_ms, 3),
        }, duration_ms=total_ms)

        return MemoryContext(
            items=items,
            memory_summary=f"Spreading-First | activated {len(activated)} nodes, top {len(items)}",
            active_patterns=[],
            relevant_facts=[],
            reasoning_traces=[
                f"spreading_first",
                f"weight={spreading_weight}",
                f"activated={len(activated)}",
                f"returned={len(items)}",
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
        Layer 3: 2D planar (time x importance projection, all cortices)
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

        # ── Layer 3: 2D planar (time x importance) ───────
        # Use TimeSpine-indexed scan for O(days*buckets) instead of O(all_anchors).
        # Falls back to time-limited cortex scan when TimeSpine is sparse/empty.
        result["layers_visited"].append("2d_plane")
        plane_items = []
        now = time.time()

        max_days = self._rt.cfg.get_path('timespine.max_days_scan', 30) if hasattr(self._rt.cfg, 'get_path') else 30
        spine_clusters = self._rt.scan_timeline(max_days=max_days, max_clusters=max_items * 2)
        spine_ids: set[str] = set()
        if spine_clusters:
            for sc in spine_clusters:
                for aid in sc.get("anchor_ids", []):
                    spine_ids.add(aid)

        candidates_from_spine = 0
        if spine_ids:
            # Look up spine-indexed anchors from global graph + all cortices
            for aid in spine_ids:
                if aid in existing_ids:
                    continue
                anchor = self._rt.graph.anchors.get(aid)
                if anchor is None:
                    for cortex in self._rt.router.cortices:
                        anchor = cortex.graph.anchors.get(aid)
                        if anchor:
                            break
                if anchor and anchor.is_retrievable:
                    hours = (now - anchor.last_activated_at) / 3600
                    recency = math.exp(-hours / 168)
                    score = 0.6 * recency + 0.4 * anchor.retention_score
                    plane_items.append((anchor, score))
                    candidates_from_spine += 1

        # Fallback: if TimeSpine is sparse, scan active cortices limited to recent window
        if candidates_from_spine < max_items:
            cutoff = now - max_days * 86400
            for cortex in active_cortices:
                for anchor in cortex.graph.anchors.values():
                    if not anchor.is_retrievable or anchor.id in existing_ids:
                        continue
                    if anchor.id in spine_ids:
                        continue  # already scored above
                    if anchor.last_activated_at < cutoff:
                        continue  # outside time window
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
