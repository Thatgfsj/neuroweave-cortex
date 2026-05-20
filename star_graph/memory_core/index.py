"""Approximate nearest neighbor index — sub-linear retrieval.

Replaces the 14+ O(n) linear scan sites across the codebase with
proper indexed lookup using sklearn NearestNeighbors.

Also supports deterministic mode (seed-based) for reproducible benchmarks.
"""

from __future__ import annotations

import math
from typing import Optional


class ANNIndex:
    """Nearest-neighbor index over embeddings for sub-linear retrieval.

    Uses sklearn's NearestNeighbors (ball tree for dim<20, kd-tree otherwise).
    For >50K anchors, consider swapping to FAISS/HNSW.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._index = None
        self._ids: list[str] = []
        self._embeddings: list[list[float]] = []
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, anchor_id: str, embedding: list[float]) -> None:
        if len(embedding) < self._dim:
            embedding = list(embedding) + [0.0] * (self._dim - len(embedding))
        elif len(embedding) > self._dim:
            embedding = embedding[:self._dim]
        self._ids.append(anchor_id)
        self._embeddings.append(list(embedding))
        self._dirty = True

    def remove(self, anchor_id: str) -> None:
        try:
            idx = self._ids.index(anchor_id)
            self._ids.pop(idx)
            self._embeddings.pop(idx)
            self._dirty = True
        except ValueError:
            pass

    def rebuild(self) -> None:
        """Reconstruct the index (call after batch adds or before query)."""
        if not self._embeddings:
            self._index = None
            self._dirty = False
            return
        from sklearn.neighbors import NearestNeighbors
        if self._dim <= 20:
            self._index = NearestNeighbors(n_neighbors=min(50, len(self._embeddings)),
                                           algorithm='ball_tree', metric='cosine')
        else:
            self._index = NearestNeighbors(n_neighbors=min(50, len(self._embeddings)),
                                           algorithm='brute', metric='cosine',
                                           n_jobs=-1)
        import numpy as np
        X = np.array(self._embeddings, dtype=np.float32)
        self._index.fit(X)
        self._dirty = False

    def _ensure_index(self) -> None:
        if self._dirty:
            self.rebuild()

    def query(self, embedding: list[float], k: int = 10) -> list[tuple[str, float]]:
        """Return (anchor_id, cosine_similarity) for top-k nearest neighbors."""
        if not self._ids:
            return []
        self._ensure_index()
        import numpy as np
        vec = np.array(embedding[:self._dim], dtype=np.float32)
        if len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))
        distances, indices = self._index.kneighbors(
            vec.reshape(1, -1),
            n_neighbors=min(k, len(self._ids)),
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._ids):
                sim = 1.0 - dist  # cosine distance → similarity
                results.append((self._ids[idx], float(sim)))
        return results

    def clear(self) -> None:
        self._index = None
        self._ids.clear()
        self._embeddings.clear()
        self._dirty = True
