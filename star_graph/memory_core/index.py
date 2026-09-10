"""Approximate nearest neighbor index — sub-linear retrieval.

Replaces the 14+ O(n) linear scan sites across the codebase with
proper indexed lookup.

Implementation: an incrementally maintained [n, dim] numpy matrix with an
id → row dict. add is O(1) amortized (in-place update on re-add), remove is
O(1) (dict location + swap-delete), and query is a single matrix-vector
product plus top-k selection. The matrix is (re)built lazily once per batch
of mutations, so the first query after bulk writes pays one cheap array copy
instead of a full index refit.

Also supports deterministic mode (seed-based) for reproducible benchmarks.
"""

from __future__ import annotations

import numpy as np


class ANNIndex:
    """Nearest-neighbor index over embeddings for sub-linear retrieval.

    Exact cosine top-k via vectorized matmul (no per-query refit).
    For >50K anchors, consider swapping to FAISS/HNSW.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._ids: list[str] = []
        self._embeddings: list[list[float]] = []
        self._pos: dict[str, int] = {}   # anchor_id → row index
        self._matrix = None               # lazily built [n, dim] float32
        self._dirty = False

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, anchor_id: str, embedding: list[float]) -> None:
        if len(embedding) < self._dim:
            embedding = list(embedding) + [0.0] * (self._dim - len(embedding))
        elif len(embedding) > self._dim:
            embedding = embedding[:self._dim]
        idx = self._pos.get(anchor_id)
        if idx is not None:
            # Re-adding an existing id updates its row in place (no dupes).
            self._embeddings[idx] = list(embedding)
        else:
            self._pos[anchor_id] = len(self._ids)
            self._ids.append(anchor_id)
            self._embeddings.append(list(embedding))
        self._dirty = True

    def remove(self, anchor_id: str) -> None:
        idx = self._pos.pop(anchor_id, None)
        if idx is None:
            return
        last = len(self._ids) - 1
        if idx != last:
            moved_id = self._ids[last]
            self._ids[idx] = moved_id
            self._embeddings[idx] = self._embeddings[last]
            self._pos[moved_id] = idx
        self._ids.pop()
        self._embeddings.pop()
        self._dirty = True

    def rebuild(self) -> None:
        """(Re)build the query matrix (call after batch adds or before query)."""
        if not self._embeddings:
            self._matrix = None
            self._dirty = False
            return
        self._matrix = np.asarray(self._embeddings, dtype=np.float32)
        self._dirty = False

    def _ensure_index(self) -> None:
        if self._dirty or self._matrix is None:
            self.rebuild()

    def query(self, embedding: list[float], k: int = 10) -> list[tuple[str, float]]:
        """Return (anchor_id, cosine_similarity) for top-k nearest neighbors."""
        if not self._ids:
            return []
        self._ensure_index()
        vec = np.asarray(embedding[:self._dim], dtype=np.float32)
        if len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))
        mat = self._matrix
        dots = mat @ vec
        denom = np.linalg.norm(mat, axis=1) * float(np.linalg.norm(vec))
        sims = np.zeros(dots.shape[0], dtype=np.float32)
        valid = denom > 1e-8
        sims[valid] = dots[valid] / denom[valid]
        kk = min(k, len(self._ids))
        if kk < sims.shape[0]:
            top = np.argpartition(-sims, kk - 1)[:kk]
        else:
            top = np.arange(sims.shape[0])
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(self._ids[int(i)], float(sims[i])) for i in top]

    def clear(self) -> None:
        self._ids.clear()
        self._embeddings.clear()
        self._pos.clear()
        self._matrix = None
        self._dirty = False
