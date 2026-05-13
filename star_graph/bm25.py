"""BM25 keyword index — incremental, sparse retrieval for multi-channel fusion.

Provides IDF-weighted term scoring that complements embedding-based semantic search.
Supports incremental add/remove so the index stays synchronized with the graph.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict


def _tokenize(text: str) -> list[str]:
    """Extract alphanumeric tokens (2+ chars) from lowercase text."""
    return re.findall(r'[a-zA-Z0-9_]\w{1,}', text.lower())


class BM25Index:
    """Incremental BM25 index for text documents keyed by string ID.

    Usage:
        idx = BM25Index()
        idx.add("a1", "Redis connection timeout is 30 seconds")
        idx.add("a2", "MySQL query timeout is 60 seconds")
        results = idx.search("redis timeout", top_k=5)
        # → [("a1", 2.35), ("a2", 1.12)]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: dict[str, str] = {}       # id → text
        self._tokens: dict[str, list[str]] = {}  # id → token list
        self._df: dict[str, int] = {}         # document frequency per term
        self._avg_dl: float = 0.0

    @property
    def size(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str) -> None:
        """Index or re-index a document."""
        if not text:
            return
        # Remove old tokens first if re-indexing
        if doc_id in self._docs:
            self.remove(doc_id)
        tokens = _tokenize(text)
        self._docs[doc_id] = text
        self._tokens[doc_id] = tokens
        for t in set(tokens):
            self._df[t] = self._df.get(t, 0) + 1
        self._update_avg_dl()

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self._docs:
            return
        tokens = self._tokens.pop(doc_id, [])
        self._docs.pop(doc_id, None)
        for t in set(tokens):
            if t in self._df:
                self._df[t] -= 1
                if self._df[t] <= 0:
                    del self._df[t]
        self._update_avg_dl()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return (doc_id, BM25_score) ranked by relevance."""
        q_tokens = _tokenize(query)
        N = len(self._docs)
        if N == 0 or not q_tokens:
            return []

        doc_lens = [len(t) for t in self._tokens.values()]
        avg_dl = self._avg_dl

        scores: list[tuple[str, float]] = []
        for doc_id, tokens in self._tokens.items():
            score = 0.0
            for qt in q_tokens:
                if qt not in self._df:
                    continue
                tf = tokens.count(qt)
                if tf == 0:
                    continue
                df = self._df[qt]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * len(tokens) / max(1.0, avg_dl))
                score += idf * numerator / max(0.01, denominator)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _update_avg_dl(self) -> None:
        N = len(self._tokens)
        self._avg_dl = sum(len(t) for t in self._tokens.values()) / max(1, N)

    def clear(self) -> None:
        self._docs.clear()
        self._tokens.clear()
        self._df.clear()
        self._avg_dl = 0.0


def reciprocal_rank_fusion(result_lists: list[list[tuple[str, float]]],
                           k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF_score(d) = Σ 1/(k + rank_i(d))  across all lists i that contain d.

    Args:
        result_lists: List of [(id, score), ...] ranked lists.
        k: Damping constant (default 60, per TREC recommendation).

    Returns:
        Fused [(id, rrf_score), ...] sorted descending.
    """
    rrf: dict[str, float] = defaultdict(float)
    for ranked_list in result_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            rrf[doc_id] += 1.0 / (k + rank)
    fused = sorted(rrf.items(), key=lambda x: -x[1])
    return fused
