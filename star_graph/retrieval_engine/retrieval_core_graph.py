"""RetrievalCore graph mixin — topology-first and cross-modal retrieval.

Extracted verbatim from the original retrieval_core.py (issue #7) — pure move,
no behaviour change. Contains:
  - graph_first_recall (seed by embedding, then walk the graph)
  - topology_rank (graph centrality / edge-type ranking)
  - cross_modal_recall (text + image retrieval)
  - image_search (image-to-image search)
  - text_to_image (text-to-image search)

Expects ``self._rt`` (a MemoryRuntime reference) to be available, set by the
composing class.
"""

from __future__ import annotations

import time

from ..scheduler import AgentContext, MemoryContext, MemoryItem, MemoryType
from ..multimodal import CrossModalResult
from ..topology import graph_first_recall, topology_rank


class GraphMixin:
    """Mixin with graph-first, topology and cross-modal retrieval methods.

    Expects self._rt to be a MemoryRuntime instance (set by the inheriting class).
    """

    # ── Graph-first retrieval ──────────────────────────────────

    def graph_first_recall(self, query: str = "",
                           max_items: int = 10,
                           max_depth: int = 2) -> MemoryContext:
        """Graph-first retrieval: seed by embedding, then walk the graph.

        Unlike the standard recall() which ranks primarily by embedding similarity,
        this path uses graph topology as the primary signal:
        1. Find seed nodes via cheap ANN embedding search
        2. BFS walk from seeds following typed edges (causal > preference > topical)
        3. Rank all visited nodes by combined topology + embedding score

        The graph_first_weight / embedding_weight balance is configurable.
        """
        t_start = time.time()
        embedder = self._rt._get_embedder()
        query_emb = embedder.encode(query) if query else None

        if query_emb is None:
            return MemoryContext(
                items=[],
                memory_summary="Graph-First | No embedding available",
            )

        # Get graph-first weight config
        gf_cfg = getattr(self._rt.cfg, 'graph_first', None)
        graph_weight = getattr(gf_cfg, 'graph_weight', 0.6) if gf_cfg else 0.6
        embedding_weight = getattr(gf_cfg, 'embedding_weight', 0.4) if gf_cfg else 0.4
        max_depth_cfg = getattr(gf_cfg, 'max_depth', 2) if gf_cfg else max_depth

        # Run graph-first recall
        results = graph_first_recall(
            self._rt.graph,
            query_embedding=query_emb,
            top_k=max_items * 2,  # oversample for gate
            max_depth=max_depth_cfg,
            graph_weight=graph_weight,
            embedding_weight=embedding_weight,
        )

        if not results:
            return MemoryContext(
                items=[],
                memory_summary="Graph-First | No results",
            )

        # Build MemoryItems from results
        items = []
        for aid, score, depth in results:
            anchor = self._rt.graph.anchors.get(aid)
            if anchor is None or not anchor.is_retrievable:
                continue
            items.append(MemoryItem(
                anchor=anchor,
                relevance_score=score,
                memory_type=MemoryType.SEMANTIC,
                compression_level=depth,
                compressed_text=anchor.text[:200],
            ))

        # Gate the results
        if len(items) > max_items:
            items = self._rt.gate.gate(
                items,
                AgentContext(task_type="conversation"),
                query_emb,
                query,
            )[:max_items]

        total_ms = (time.time() - t_start) * 1000

        self._rt.tracer.record("graph_first_recall", attributes={
            "query": query[:200],
            "max_items": max_items,
            "max_depth": max_depth_cfg,
            "graph_weight": graph_weight,
            "embedding_weight": embedding_weight,
            "final_count": len(items),
            "total_ms": round(total_ms, 3),
        }, duration_ms=total_ms)

        return MemoryContext(
            items=items,
            memory_summary=f"Graph-First | {len(items)} items, depth<={max_depth_cfg}, "
                          f"g={graph_weight:.1f}/e={embedding_weight:.1f}",
            active_patterns=[],
            relevant_facts=[],
            reasoning_traces=[
                f"graph_first=True",
                f"depth={max_depth_cfg}",
                f"graph_weight={graph_weight}",
                f"results={len(results)}->gated->{len(items)}",
            ],
        )

    def topology_rank(self, candidate_ids: list[str] | None = None,
                      top_k: int = 20,
                      graph_weight: float = 0.6) -> list[tuple[str, float]]:
        """Rank anchors by graph topology score (centrality + edge type richness).

        Returns list of (anchor_id, topology_score) sorted descending.
        """
        return topology_rank(
            self._rt.graph,
            candidate_ids=candidate_ids,
            top_k=top_k,
            graph_weight=graph_weight,
        )

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
