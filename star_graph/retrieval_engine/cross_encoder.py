"""Cross-Encoder reranker for post-retrieval precision improvement.

Reranks the top-N candidates from RRF fusion using a lightweight
sentence-transformers CrossEncoder model. The cross-encoder scores
(query, candidate) pairs jointly — more accurate than embedding cosine
but more expensive, so it's only run on the top candidates.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default: ms-marco-MiniLM-L-6-v2 — 384-dim, ~80MB, <1ms per pair on CPU
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Post-retrieval reranker using a sentence-transformers CrossEncoder.

    Usage:
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked = reranker.rerank(
            query="what is redis timeout",
            candidates=[
                ("id1", "Redis connection timeout is 30 seconds"),
                ("id2", "MySQL query timeout is 60 seconds"),
            ],
            top_k=5,
        )
        # → [("id1", 0.92), ("id2", 0.45)]
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                 top_k: int = 10, threshold: float = 0.0,
                 enabled: bool = True):
        self._model_name = model_name
        self._model: Optional[object] = None
        self.top_k = top_k
        self.threshold = threshold
        self.enabled = enabled

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        """Lazy-load the cross-encoder model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder loaded: %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; cross-encoder reranking disabled"
            )
            self.enabled = False
        except Exception:
            logger.exception("Failed to load cross-encoder model")
            self.enabled = False

    def rerank(self, query: str,
               candidates: list[tuple[str, str]],
               top_k: int | None = None,
               threshold: float | None = None
               ) -> list[tuple[str, float]]:
        """Rerank candidates using the cross-encoder.

        Args:
            query: The user's query text.
            candidates: List of (id, text) pairs to score.
            top_k: Max results to return (default: self.top_k).
            threshold: Minimum score to include (default: self.threshold).

        Returns:
            List of (id, cross_encoder_score) sorted descending.
        """
        if not self.enabled or not candidates:
            return [(cid, 0.5) for cid, _ in candidates]

        self._load()
        if not self.enabled or self._model is None:
            return [(cid, 0.5) for cid, _ in candidates]

        if top_k is None:
            top_k = self.top_k
        if threshold is None:
            threshold = self.threshold

        # Build (query, candidate_text) pairs
        pairs = [(query, text) for _, text in candidates]

        try:
            scores: list[float] = self._model.predict(pairs).tolist()
        except Exception:
            logger.exception("CrossEncoder predict failed")
            return [(cid, 0.5) for cid, _ in candidates]

        # Pair scores with IDs
        scored = [(cid, float(s)) for (cid, _), s in zip(candidates, scores)]

        # Filter by threshold, sort, trim to top_k
        if threshold > 0:
            scored = [s for s in scored if s[1] >= threshold]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
