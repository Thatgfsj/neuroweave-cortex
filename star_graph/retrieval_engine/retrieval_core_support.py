"""RetrievalCore support mixin — scoring, compression and explainability.

Extracted verbatim from the original retrieval_core.py (issue #7) — pure move,
no behaviour change. Contains:
  - _semantic_scan (vectorized full-graph cosine scan)
  - compress_context (semantic dedup + relevance filtering)
  - explain (human-readable retrieval explanations)
  - _bm25_anchor_search (BM25 keyword path with composite scoring)

Expects ``self._rt`` (a MemoryRuntime reference) to be available, set by the
composing class.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..scheduler import AgentContext, MemoryItem, MemoryType
from ..math_utils import cosine_sim as _cosine_sim


class SupportMixin:
    """Mixin with scoring, compression and explainability helpers.

    Expects self._rt to be a MemoryRuntime instance (set by the inheriting class).
    """

    # ── Vectorized full-graph semantic scan ───────────────────

    def _semantic_scan(self, query_emb: list[float],
                       limit: int) -> list[tuple[str, float]]:
        """Score ALL retrievable anchors by embedding cosine similarity.

        Single [n, d] · [d] matmul replaces the per-anchor Python loop.
        Mirrors math_utils.cosine_sim semantics (0.0 for near-zero norms).
        Returns up to `limit` (anchor_id, similarity) pairs sorted descending;
        ties keep anchor insertion order (stable), matching the previous
        sort-based implementation.
        """
        if limit <= 0 or not query_emb:
            return []
        ids: list[str] = []
        embs: list[list[float]] = []
        for anchor in self._rt.graph.anchors.values():
            if not anchor.is_retrievable or not anchor.embedding:
                continue
            ids.append(anchor.id)
            embs.append(anchor.embedding)
        if not ids:
            return []

        d = len(query_emb)
        if d > 0 and all(len(e) == d for e in embs):
            mat = np.asarray(embs, dtype=np.float64)
            q = np.asarray(query_emb, dtype=np.float64)
            dots = mat @ q
            denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
            sims = np.zeros(dots.shape[0], dtype=np.float64)
            valid = denom >= 1e-12
            sims[valid] = dots[valid] / denom[valid]
            if limit < sims.shape[0]:
                top = np.argpartition(-sims, limit - 1)[:limit]
            else:
                top = np.arange(sims.shape[0])
            top = top[np.argsort(-sims[top], kind="stable")]
            return [(ids[int(i)], float(sims[i])) for i in top]

        # Ragged embedding dims — fall back to the per-pair computation.
        scored = [(aid, _cosine_sim(query_emb, emb))
                  for aid, emb in zip(ids, embs)]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ── Phase 3: Context Compression ──────────────────────────
    # Compresses a large result set (100+) into a compact useful set (~10)
    # via semantic dedup + relevance filtering.

    def compress_context(self, merged_items: list,
                         max_items: int = 10) -> list:
        """Compress a list of MemoryItems into a compact, diverse set.

        Strategy:
          1. Filter low-relevance items (score < 0.15)
          2. Deduplicate near-duplicate text (cosine-sim via word overlap)
          3. Keep top-k by relevance score
        """
        # Step 1: Filter low relevance
        items = [i for i in merged_items if i.relevance_score >= 0.15]
        if not items:
            return merged_items[:max_items]

        # Step 2: Deduplicate by text overlap
        kept: list = []
        seen_normalized: set = set()
        for item in items:
            text = (item.compressed_text or
                    (item.anchor.text[:200] if item.anchor else "")).strip().lower()
            if not text:
                kept.append(item)
                continue
            # Normalize: keep only alphanumeric tokens
            import re
            tokens = set(re.findall(r'\w+', text))
            # Check overlap with existing items
            is_dup = False
            for existing in kept:
                ext_text = (existing.compressed_text or
                           (existing.anchor.text[:200] if existing.anchor else "")).strip().lower()
                ext_tokens = set(re.findall(r'\w+', ext_text))
                if not tokens or not ext_tokens:
                    continue
                jaccard = len(tokens & ext_tokens) / max(1, len(tokens | ext_tokens))
                if jaccard > 0.85:  # near-duplicate
                    is_dup = True
                    break
            if not is_dup:
                kept.append(item)

        # Step 3: Take top-k by score
        kept.sort(key=lambda i: -i.relevance_score)
        return kept[:max_items]

    # ── Explainability API (#62) ──────────────────────────────────

    def explain(self, query: str = "",
                context: AgentContext | None = None,
                max_items: int = 10,
                memory_id: str | None = None) -> dict:
        """Return a human-readable explanation of WHY each memory was retrieved.

        Args:
            query: The recall query to explain.
            context: Optional agent context.
            max_items: Max results to retrieve and explain.
            memory_id: Optional specific memory ID to explain in detail.

        Returns:
            Dict with 'query', 'method', and 'retrieved_memories' list,
            each entry containing memory_id, score, reason, pathway, and
            per-channel component scores.
        """
        result = self.recall(query=query, context=context, max_items=max_items)
        trace = result.retrieval_trace or {}

        if memory_id:
            # Filter to just the requested memory
            memories = trace.get("retrieved_memories", [])
            filtered = [m for m in memories if m.get("memory_id") == memory_id]
            return {
                "query": query[:200],
                "method": "weighted_rrf_explain",
                "target_memory": memory_id,
                "retrieved_memories": filtered,
                "total_memories": len(memories),
            }

        return trace

    # ── BM25 anchor search ───────────────────────────────────

    def _bm25_anchor_search(self, query: str, query_emb: list[float] | None,
                             max_items: int, context: AgentContext | None) -> list[MemoryItem]:
        """BM25 keyword search over graph anchors with composite scoring.

        Uses the shared BM25Index on the runtime to find keyword-matching
        anchors, then computes a composite relevance score from:
          - BM25 term match (weight: 0.50)
          - Tag overlap with query terms (weight: 0.20)
          - Semantic cosine similarity (weight: 0.15)
          - Recency (weight: 0.10)
          - Importance (weight: 0.05)

        When the embedder uses hash fallback (no real semantic model),
        the semantic weight is redistributed to BM25.
        """
        if not query or not query.strip():
            return []

        bm25 = getattr(self._rt, 'bm25', None)
        if bm25 is None or bm25.size == 0:
            return []

        # Fetch 2x candidates for reranking headroom
        # Search BM25 with generous pool — more candidates = better RRF fusion
        search_k = max(max_items * 15, 200)  # at least 200 candidates for better recall
        hits = bm25.search(query, top_k=search_k)
        if not hits:
            return []

        # Detect hash-embedding fallback (no real semantic signal)
        embedder = self._rt._get_embedder()
        is_hash = getattr(embedder, '_backend', '') == 'hash'

        # Adjust weights: BM25 gets semantic weight when embeddings are degraded
        if is_hash:
            w_bm25, w_tag, w_sem, w_recency, w_importance = 0.80, 0.12, 0.00, 0.05, 0.03
        else:
            w_bm25, w_tag, w_sem, w_recency, w_importance = 0.55, 0.18, 0.15, 0.08, 0.04

        # Normalize BM25 scores to [0, 1] via exponential saturation
        max_bm25 = max(s for _, s in hits) if hits else 1.0

        # Tokenize query once for tag matching
        import re
        _tokenize = getattr(bm25, '_tokenize', None)
        if _tokenize is None:
            from .bm25 import _tokenize
        query_tokens = set(_tokenize(query))

        now = time.time()
        items: list[MemoryItem] = []
        for anchor_id, bm25_score in hits:
            anchor = self._rt.graph.anchors.get(anchor_id)
            if anchor is None or not anchor.is_retrievable:
                continue

            # 1. BM25 normalized score — calibrated so a good keyword
            #    match (BM25 ≥ 4) reaches ≥ 0.90, competing with the
            #    brain-hit baseline of 0.9 used by graph descent path.
            bm25_norm = 1.0 - math.exp(-bm25_score / 1.5)

            # 2. Tag overlap score
            tag_score = 0.0
            if anchor.tags and query_tokens:
                tag_tokens = set()
                for t in anchor.tags:
                    tag_tokens.update(_tokenize(t))
                overlap = query_tokens & tag_tokens
                if overlap:
                    tag_score = min(1.0, len(overlap) / max(len(query_tokens), 1))

            # 3. Semantic similarity
            sem_score = 0.0
            if not is_hash and query_emb and anchor.embedding:
                sem_score = max(0.0, _cosine_sim(query_emb, anchor.embedding))

            # 4. Recency (exponential decay, half-life ~7 days)
            age_seconds = now - getattr(anchor, 'created_at', now)
            half_life = 7 * 24 * 3600  # 7 days
            recency = math.exp(-math.log(2) * age_seconds / half_life) if age_seconds > 0 else 1.0

            # 5. Importance (from AnchorVector)
            importance = anchor.vector.importance if anchor.vector else 0.5

            # Composite score — scale to [0.82, 1.0] range so BM25
            # items outrank graph-descent brain hits (fixed 0.9).
            raw = (w_bm25 * bm25_norm +
                   w_tag * tag_score +
                   w_sem * sem_score +
                   w_recency * recency +
                   w_importance * importance)
            composite = raw  # raw is already calibrated to compete with graph descent

            items.append(MemoryItem(
                anchor=anchor,
                relevance_score=round(composite, 4),
                memory_type=MemoryType.SEMANTIC,
                compression_level=0,
                compressed_text=anchor.text[:200],
            ))

        items.sort(key=lambda i: -i.relevance_score)
        return_items = items[:max_items * 5]  # generous pool for RRF fusion
        return return_items
