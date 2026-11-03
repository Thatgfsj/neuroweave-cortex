"""RetrievalCore recall mixin — primary multi-path recall.

Extracted verbatim from the original retrieval_core.py (issue #7) — pure move,
no behaviour change. Contains:
  - recall (primary RRF entry point with System-2 auto-trigger)
  - _system2_recall (goal-directed System-2 fallback)
  - temporal-query detection helpers used by recall()

Expects ``self._rt`` (a MemoryRuntime reference) to be available, set by the
composing class.
"""

from __future__ import annotations

import re
import time

from ..scheduler import AgentContext, MemoryContext, MemoryItem, MemoryType
from ..anchor import Anchor
from .retriever import RetrievalTrace, RetrievalTraceEntry


# ── Temporal query detection ──────────────────────────────
# Identifies queries containing time/date references for
# TimeSpine-based pre-filtering. Used by recall() to boost
# Category 1 (temporal QA) recall on LoCoMo.

_TEMPORAL_PATTERNS = re.compile(
    r'\b('
    r'yesterday|today|tomorrow|'
    r'last\s+\w+|next\s+\w+|'
    r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*|'
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|'
    r'\d{4}[/-]\d{1,2}[/-]\d{1,2}|'
    r'\d{1,2}:\d{2}\s*(am|pm)|'
    r'ago|earlier|later|before|after|'
    r'morning|afternoon|evening|night|'
    r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
    r'this\s+(week|month|year)|'
    r'previous\s+(week|month|year|session)|'
    r'\d+\s+(days?|weeks?|months?|years?|hours?|minutes?)\s+(ago|earlier|before|later)'
    r')\b',
    re.IGNORECASE,
)


def _detect_temporal_query(query: str) -> tuple[bool, list[str]]:
    """Detect whether query contains temporal/date references.

    Returns:
        (is_temporal: bool, matched_keywords: list[str])
    """
    if not query:
        return False, []
    matches = _TEMPORAL_PATTERNS.findall(query)
    return len(matches) > 0, list(set(matches))


class RecallMixin:
    """Mixin with the primary recall entry points.

    Expects self._rt to be a MemoryRuntime instance (set by the inheriting class).
    """

    # ── Primary recall entry point ────────────────────────────

    def recall(self, query: str = "",
               context: AgentContext | None = None,
               max_items: int = 10,
               goals: list[str] | None = None) -> MemoryContext:
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

        # Path 0.5: Cognitive cache lookup (query/session/topic/activation caches)
        t0 = time.time()
        cache_hit_anchors: list[tuple[Anchor, float, str]] = []
        if query:
            cache_result = self._rt.cognitive_cache.lookup(
                query=query,
                tags=list(context.task_tags) if context and getattr(context, 'task_tags', None) else None,
                seed_ids=[item.anchor.id for item in exact_results if item.anchor],
            )
            seen_ids: set[str] = set()
            for source, anchor_ids in cache_result.items():
                for aid in anchor_ids:
                    if aid in seen_ids:
                        continue
                    anchor = self._rt.graph.anchors.get(aid)
                    if anchor and anchor.is_retrievable:
                        seen_ids.add(aid)
                        score = 0.88 if source == "query_cache" else 0.82
                        cache_hit_anchors.append((anchor, score, source))
        cache_ms = (time.time() - t0) * 1000

        # Path A: Raw buffer search (L0 — highest priority after exact cache)
        t0 = time.time()
        raw_results = self._rt.raw_buffer.search(
            query=query, query_embedding=query_emb,
            top_k=max_items,
            session_id=context.session_id if context else "",
        )
        raw_ms = (time.time() - t0) * 1000

        # Path A2: BM25 anchor search — keyword-first, always reliable
        t0 = time.time()
        bm25_items = self._bm25_anchor_search(
            query=query, query_emb=query_emb,
            max_items=max_items, context=context)
        bm25_ms = (time.time() - t0) * 1000

        # Path B: Graph dimensional descent (L1-L5)
        t0 = time.time()
        graph_result = self.retrieve_with_descent(
            query=query, context=context, max_items=max_items)
        graph_ms = (time.time() - t0) * 1000

        # Path C: Spreading activation graph walk (#52)
        t0 = time.time()
        spreading_items = self._spreading_recall(
            query_emb=query_emb, query=query, max_items=max_items)
        spreading_ms = (time.time() - t0) * 1000

        # Path C2: Activation-graph walk (v1.3) — uses historical edge activation
        # instead of LLM-determined relevance. Brightness = f(access, recency, edge_strength).
        t0 = time.time()
        activation_items: list[MemoryItem] = []
        try:
            from star_graph.activation_graph import get_activation_graph
            ag = get_activation_graph(self._rt.graph)
            emb = query_emb or (self._rt._get_embedder().encode(query) if query else None)
            if emb is not None:
                act_results = ag.find_seeds(emb, top_k=max_items, min_similarity=0.15)
                if act_results:
                    # Spread caps come from retrieval_budget config (single
                    # source); nodes stay ≤ the diffusion ceiling (50) and
                    # default behaviour is unchanged (depth 2, nodes 20).
                    rb_cfg = getattr(self._rt.cfg, "retrieval_budget", None)
                    _spread_depth = getattr(rb_cfg, "spread_max_depth", 2) if rb_cfg else 2
                    _spread_nodes = min(
                        getattr(rb_cfg, "spread_max_nodes", 50) if rb_cfg else 50,
                        max_items * 2)
                    act_spread = ag.spread(act_results, max_depth=_spread_depth, max_nodes=_spread_nodes)
                    seen_aids = set()
                    for node in act_spread:
                        if node.node_id in seen_aids:
                            continue
                        anchor = self._rt.graph.anchors.get(node.node_id)
                        if anchor and anchor.is_retrievable:
                            seen_aids.add(node.node_id)
                            activation_items.append(MemoryItem(
                                anchor=anchor,
                                relevance_score=node.activation,
                                memory_type=MemoryType.SEMANTIC,
                                compression_level=1,
                                compressed_text=anchor.text[:200],
                            ))
        except Exception:
            pass
        activation_ms = (time.time() - t0) * 1000

        # ── RRF Fusion: multi-path reciprocal rank fusion ──
        # Collect anchored, independently-ranked lists from each retrieval
        # path. Reciprocal Rank Fusion rewards items that appear highly in
        # multiple paths, naturally boosting consensus results and
        # suppressing single-path noise.
        from .bm25 import weighted_reciprocal_rank_fusion

        ranked_lists: list[list[tuple[str, float]]] = []
        path_weights: list[float] = []
        # id → (anchor, text, mem_type, comp_level, pathway_label)
        id_meta: dict[str, tuple] = {}

        # Path 0: Exact cache — deterministic, highest weight
        exact_anchored = [(item.anchor.id, item.relevance_score)
                          for item in exact_results if item.anchor]
        if exact_anchored:
            ranked_lists.append(exact_anchored)
            path_weights.append(1.5)
            for item in exact_results:
                if item.anchor:
                    id_meta[item.anchor.id] = (
                        item.anchor, item.compressed_text or item.anchor.text[:200],
                        item.memory_type, item.compression_level, "exact_cache")

        # Path A2: BM25 keyword — reliable for entity/term matching
        if bm25_items:
            ranked_lists.append([(item.anchor.id, item.relevance_score)
                                 for item in bm25_items if item.anchor])
            path_weights.append(1.5)  # increased: BM25 is strongest signal for exact token matching
            for item in bm25_items:
                if item.anchor and item.anchor.id not in id_meta:
                    id_meta[item.anchor.id] = (
                        item.anchor, item.compressed_text or item.anchor.text[:200],
                        item.memory_type, item.compression_level, "bm25_keyword")

        # Path 0.5: Cognitive cache — session/query/topic cache hits
        if cache_hit_anchors:
            ranked_lists.append([(anchor.id, score)
                                 for anchor, score, _ in cache_hit_anchors])
            path_weights.append(1.1)
            for anchor, score, source in cache_hit_anchors:
                if anchor.id not in id_meta:
                    id_meta[anchor.id] = (
                        anchor, anchor.text[:200], MemoryType.SEMANTIC, 1,
                        f"cognitive_{source}")

        # Path B: Graph dimensional descent — L1-L5 structural retrieval
        graph_items = graph_result.get("items", [])
        if graph_items:
            ranked_lists.append([(item.anchor.id, item.relevance_score)
                                 for item in graph_items if item.anchor])
            path_weights.append(0.9)
            for item in graph_items:
                if item.anchor and item.anchor.id not in id_meta:
                    id_meta[item.anchor.id] = (
                        item.anchor,
                        getattr(item, 'compressed_text', '') or item.anchor.text[:200],
                        item.memory_type, item.compression_level, "graph_descent")

        # Path C: Spreading activation — BFS graph walk with edge-type weighting
        if spreading_items:
            ranked_lists.append([(item.anchor.id, item.relevance_score)
                                 for item in spreading_items if item.anchor])
            path_weights.append(0.6)  # reduced: spreading adds breadth but low precision for exact matching
            for item in spreading_items:
                if item.anchor and item.anchor.id not in id_meta:
                    id_meta[item.anchor.id] = (
                        item.anchor,
                        getattr(item, 'compressed_text', '') or item.anchor.text[:200],
                        item.memory_type, item.compression_level, "spreading_activation")

        # Path C2: Activation-graph walk (v1.3) — historical edge activation
        if activation_items:
            ranked_lists.append([(item.anchor.id, item.relevance_score)
                                 for item in activation_items if item.anchor])
            path_weights.append(0.7)  # activation history, moderate weight
            for item in activation_items:
                if item.anchor and item.anchor.id not in id_meta:
                    id_meta[item.anchor.id] = (
                        item.anchor,
                        getattr(item, 'compressed_text', '') or item.anchor.text[:200],
                        item.memory_type, item.compression_level, "activation_graph")


        # ── Full semantic scan (auxiliary path) ──
        # Scores ALL anchors by embedding cosine to catch items missed by
        # BM25 pre-filtering. Low weight: provides breadth, not precision.
        # Vectorized: one matrix-vector product instead of a per-anchor
        # Python cosine loop.
        t_sem = time.time()
        sem_ids: list[tuple[str, float]] = []
        if query_emb is not None and len(self._rt.graph.anchors) > 0:
            try:
                sem_ids = self._semantic_scan(query_emb, max_items * 10)
            except Exception:
                sem_ids = []
        if sem_ids:
            ranked_lists.append(sem_ids)
            path_weights.append(1.0)  # semantic breadth, lower than BM25/exact to avoid diluting keyword precision
            for aid, sim in sem_ids:
                if aid not in id_meta:
                    anchor = self._rt.graph.anchors.get(aid)
                    if anchor:
                        id_meta[aid] = (anchor, anchor.text[:200], MemoryType.SEMANTIC, 1, "semantic_scan")
        semantic_ms = (time.time() - t_sem) * 1000


        # Execute weighted RRF fusion
        fused = weighted_reciprocal_rank_fusion(ranked_lists, path_weights)

        # Normalize RRF scores to [0, 1] range
        theoretical_max = sum(w / 61.0 for w in path_weights) if path_weights else 1.0

        # Build MemoryItems from fused ranking
        merged_items: list[MemoryItem] = []
        seen_ids: set[str] = set()

        for anchor_id, rrf_score in fused:
            if anchor_id in seen_ids:
                continue
            seen_ids.add(anchor_id)
            meta = id_meta.get(anchor_id)
            if meta is None:
                anchor = self._rt.graph.anchors.get(anchor_id)
                if anchor is None or not anchor.is_retrievable:
                    continue
                text = anchor.text[:200]
                mem_type = MemoryType.SEMANTIC
                comp_level = 1
            else:
                anchor, text, mem_type, comp_level, _ = meta

            norm_score = min(1.0, rrf_score / theoretical_max) if theoretical_max > 0 else 0.0
            merged_items.append(MemoryItem(
                anchor=anchor,
                relevance_score=round(norm_score, 4),
                memory_type=mem_type,
                compression_level=comp_level,
                compressed_text=text or (anchor.text[:200] if anchor else ""),
            ))

        # Append unanchored items (raw buffer chunks + WM exact hits —
        # these have no stable graph ID so can't participate in RRF)
        seen_texts = {self._rt._normalize_text(item.compressed_text)
                      for item in merged_items
                      if hasattr(item, 'compressed_text') and item.compressed_text}
        # Raw buffer chunks keep their natural score — previously penalized
        # by 0.7 multiplier, but these are the most recent un-compressed
        # memories and often the best match for short-term factual queries.
        # They compete on merit alongside anchored items.
        for chunk, score in raw_results:
            norm_text = self._rt._normalize_text(chunk.text[:120])
            if norm_text and norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            merged_items.append(MemoryItem(
                anchor=None, relevance_score=score,
                memory_type=MemoryType.WORKING, compression_level=0,
                compressed_text=chunk.text[:200],
            ))
        for item in exact_results:
            if not item.anchor:
                norm_text = self._rt._normalize_text(item.compressed_text or "")
                if norm_text not in seen_texts:
                    seen_texts.add(norm_text)
                    merged_items.append(item)

        # ── Domain-based re-ranking (#48) ──
        dr_enabled = True
        dr_cfg = getattr(self._rt.cfg, 'domain_router', None)
        if dr_cfg:
            dr_enabled = getattr(dr_cfg, 'enabled', True)
        if dr_enabled and query:
            try:
                domain_ids, _ = self._rt.domain_router.get_candidate_scope(query)
                if domain_ids:
                    for item in merged_items:
                        if item.anchor and item.anchor.id in domain_ids:
                            item.relevance_score = min(1.0, item.relevance_score + 0.08)
            except Exception:
                pass

        # ── Recency boost — reward recently created/accessed memories ──
        # Directly addresses LoCoMo Category 2 (Short Memory) and 3 (Long Memory)
        # where pure embedding fails on older but contextually relevant anchors.
        now = time.time()
        for item in merged_items:
            if item.anchor:
                age_hours = (now - item.anchor.last_activated_at) / 3600.0
                if age_hours < 24:          # last 24 hours → strong boost
                    item.relevance_score = min(1.0, item.relevance_score + 0.10)
                elif age_hours < 168:       # last 7 days → moderate boost
                    item.relevance_score = min(1.0, item.relevance_score + 0.05)

        # ── Temporal query boosting ──
        # For queries with temporal keywords (yesterday, last week, date patterns),
        # scan TimeSpine for matching time windows and boost matching items.
        # Directly addresses LoCoMo Category 1 (Temporal QA)
        is_temporal, _ = _detect_temporal_query(query)
        if is_temporal and query:
            try:
                temporal_aids = self._rt.timespine.query_window_anchor_ids(
                    start_time=0, end_time=now, top_k=50)
                temporal_set = set(temporal_aids)
                for item in merged_items:
                    if item.anchor and item.anchor.id in temporal_set:
                        item.relevance_score = min(1.0, item.relevance_score + 0.15)
            except Exception:
                pass

        # Sort by fused relevance score (RRF already weights paths, no
        # need for manual BM25-first bias)
        merged_items.sort(key=lambda i: -i.relevance_score)

        # Enforce retrieval budget — hard limits on node count + token budget (S-5)
        budget_state = self._rt.retrieval_budget.begin()
        merged_items = self._rt.retrieval_budget.enforce_nodes(merged_items, budget_state)
        merged_items = self._rt.retrieval_budget.enforce_tokens(merged_items, budget_state)
        merged_items = merged_items[:max_items]

        # ── Cross-Encoder rerank (#62) ──
        ce = self._rt.cross_encoder
        if ce.enabled and query and len(merged_items) > 1:
            t_ce = time.time()
            # Prepare candidates: (anchor_id, text) for anchored items
            ce_candidates: list[tuple[str, str]] = []
            for item in merged_items:
                text = (item.compressed_text or
                        (item.anchor.text[:200] if item.anchor else ""))
                if text.strip():
                    key = item.anchor.id if item.anchor else text[:40]
                    ce_candidates.append((key, text))
            if len(ce_candidates) > 1:
                try:
                    ce_scored = ce.rerank(query, ce_candidates)
                    ce_scores = dict(ce_scored)
                    for item in merged_items:
                        key = item.anchor.id if item.anchor else (
                            (item.compressed_text or "")[:40])
                        if key in ce_scores:
                            # Blend: 60% cross-encoder, 40% original RRF
                            ce_score = ce_scores[key]
                            item.relevance_score = round(
                                0.6 * ce_score + 0.4 * item.relevance_score, 4)
                    merged_items.sort(key=lambda i: -i.relevance_score)
                    ce_ms = (time.time() - t_ce) * 1000
                except Exception:
                    ce_ms = 0
            else:
                ce_ms = 0
        else:
            ce_ms = 0

        # ── Goal-Aware re-ranking (Phase 3) ───────────────────
        # If goals are provided, boost items semantically aligned with goals.
        if goals and merged_items:
            goal_text = " ".join(goals)
            goal_lower = goal_text.lower()
            goal_words = set(goal_lower.split())
            for item in merged_items:
                item_text = (item.compressed_text or
                            (item.anchor.text[:200] if item.anchor else "")).lower()
                item_words = set(item_text.split())
                overlap = len(item_words & goal_words)
                if overlap > 0:
                    goal_boost = min(0.15, overlap * 0.03)
                    item.relevance_score = min(1.0, item.relevance_score + goal_boost)
            merged_items.sort(key=lambda i: -i.relevance_score)

        # Auto-trigger System-2 if System-1 confidence is low
        s1_confidence = graph_result.get("confidence", 1.0)
        if s1_confidence < 0.35 and len(merged_items) < max_items:
            return self._system2_recall(query, context, max_items,
                                        trigger_reason="low_confidence")

        # Record results in cognitive caches
        result_ids = [item.anchor.id for item in merged_items if item.anchor]
        result_scores = [item.relevance_score for item in merged_items]
        if query and result_ids:
            self._rt.cognitive_cache.record_query(query, result_ids, result_scores)
        for aid in result_ids:
            self._rt.cognitive_cache.record_access(aid)

        layers_visited = (["exact_cache"] if exact_results else [])
        if cache_hit_anchors:
            layers_visited.append("cognitive_cache")
        layers_visited.append("raw_buffer")
        layers_visited.extend(graph_result.get("layers_visited", []))
        total_ms = (time.time() - t_start) * 1000

        # Build retrieval trace for explainability — track which pathways
        # contributed to each result and at what rank
        pathway_labels = ["exact_cache", "bm25_keyword", "cognitive_cache",
                          "graph_descent", "spreading_activation"]
        pathway_ranked = [exact_anchored,
                          [(item.anchor.id, item.relevance_score) for item in bm25_items if item.anchor],
                          [(anchor.id, score) for anchor, score, _ in cache_hit_anchors],
                          [(item.anchor.id, item.relevance_score) for item in graph_items if item.anchor],
                          [(item.anchor.id, item.relevance_score) for item in spreading_items if item.anchor]]
        # Build id → {pathway: rank} lookup
        id_pathways: dict[str, dict[str, int]] = {}
        for label, ranked in zip(pathway_labels, pathway_ranked):
            if not ranked:
                continue
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                if doc_id not in id_pathways:
                    id_pathways[doc_id] = {}
                id_pathways[doc_id][label] = rank

        rrf_trace_entries: list[RetrievalTraceEntry] = []
        for item in merged_items[:max_items]:
            anchor_id = item.anchor.id if item.anchor else None
            pathways = id_pathways.get(anchor_id, {}) if anchor_id else {}
            pathway_str = "+".join(sorted(pathways.keys())) if pathways else "unanchored"
            reason_parts = []
            if "exact_cache" in pathways:
                reason_parts.append(f"exact_hit(#{pathways['exact_cache']})")
            if "bm25_keyword" in pathways:
                reason_parts.append(f"bm25(#{pathways['bm25_keyword']})")
            if "cognitive_cache" in pathways:
                reason_parts.append(f"cache(#{pathways['cognitive_cache']})")
            if "graph_descent" in pathways:
                reason_parts.append(f"graph(#{pathways['graph_descent']})")
            if "spreading_activation" in pathways:
                reason_parts.append(f"spread(#{pathways['spreading_activation']})")
            if not reason_parts:
                reason_parts.append("unanchored")

            rrf_trace_entries.append(RetrievalTraceEntry(
                memory_id=anchor_id or "unanchored",
                score=item.relevance_score,
                reason=", ".join(reason_parts),
                pathway=pathway_str,
                components={p: round(1.0 / max(r, 1), 3) for p, r in pathways.items()},
            ))

        self._rt.tracer.record("recall", attributes={
            "query": query[:200],
            "max_items": max_items,
            "exact_hits": len(exact_results),
            "cache_hits": len(cache_hit_anchors),
            "raw_chunks": len(raw_results),
            "bm25_items": len(bm25_items),
            "graph_items": len(graph_result.get("items", [])),
            "spreading_items": len(spreading_items),
            "final_count": len(merged_items),
            "is_temporal_query": is_temporal if 'is_temporal' in dir() else False,
            "layers_visited": str(layers_visited)[:300],
            "channel": "system1_rrf",
            "rrf_paths_fused": len(ranked_lists),
            "rrf_path_weights": str([round(w, 1) for w in path_weights])[:80],
            "exact_ms": round(exact_ms, 3),
            "cache_ms": round(cache_ms, 3),
            "raw_ms": round(raw_ms, 3),
            "bm25_ms": round(bm25_ms, 3),
            "graph_ms": round(graph_ms, 3),
            "spreading_ms": round(spreading_ms, 3),
            "semantic_ms": round(semantic_ms, 3),
            "ce_ms": round(ce_ms, 3),
            "budget_nodes": budget_state.nodes_activated,
            "budget_tokens": budget_state.tokens_used,
            "budget_truncated": budget_state.truncated,
            "total_ms": round(total_ms, 3),
        }, duration_ms=total_ms)

        return MemoryContext(
            items=merged_items,
            memory_summary=f"System-1 RRF | {len(ranked_lists)} paths fused, visited: {layers_visited}",
            active_patterns=[],
            relevant_facts=[],
            reasoning_traces=[
                f"rrf_paths={len(ranked_lists)}",
                f"exact_hits={len(exact_results)}",
                f"cache_hits={len(cache_hit_anchors)}",
                f"bm25_items={len(bm25_items)}",
                f"graph_items={len(graph_result.get('items', []))}",
                f"spreading_items={len(spreading_items)}",
                f"rrf_fused={len(fused)}",
                f"merged={len(merged_items)}",
            ],
            retrieval_trace=RetrievalTrace(
                query=query[:200],
                method="weighted_rrf",
                retrieved_memories=rrf_trace_entries,
            ).to_dict(),
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
