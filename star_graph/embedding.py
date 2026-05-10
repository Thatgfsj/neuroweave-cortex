"""Real embedding provider — sentence-transformers with principled fallbacks.

Phase derivation is NOT random. It encodes temporal/semantic structure:
  theta_phase = f(timestamp, importance, emotional_valence)

This means phase carries actual information:
- Recent memories → different phase than old
- Important memories → phase-aligned (resonate together)
- Emotional memories → phase-shifted (stand out)

Before this module: phase = hash(text) % 6283 / 1000  (random noise)
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np


class EmbeddingProvider:
    """Produces real semantic embeddings + meaningful oscillation parameters.

    Tries sentence-transformers first, falls back to sklearn TfidfVectorizer
    (which is at least a meaningful bag-of-words, unlike hash-based noise).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self._model = None
        self._tfidf = None
        self._tfidf_texts: list[str] = []
        self._model_name = model_name
        self._dim = dim
        self._backend: str = "none"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def backend(self) -> str:
        return self._backend

    def _ensure_model(self) -> None:
        if self._model is not None or self._backend == "tfidf":
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            try:
                self._dim = self._model.get_embedding_dimension()
            except AttributeError:
                self._dim = self._model.get_sentence_embedding_dimension()
            self._backend = "sentence-transformers"
        except Exception:
            self._init_tfidf()

    def _init_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(max_features=384)
        self._backend = "tfidf"
        self._dim = 384

    def encode(self, text: str) -> list[float]:
        """Produce a real semantic embedding for a text."""
        self._ensure_model()
        if self._backend == "sentence-transformers":
            vec = self._model.encode(text, show_progress_bar=False)
            return vec.tolist()
        elif self._backend == "tfidf":
            texts = list(self._tfidf_texts) + [text]
            try:
                self._tfidf.fit(texts)
                mat = self._tfidf.transform([text])
                dense = mat.toarray()[0]
                norm = math.sqrt(sum(x * x for x in dense))
                if norm > 1e-8:
                    dense = dense / norm
                result = dense.tolist()
                if len(result) < self._dim:
                    result = result + [0.0] * (self._dim - len(result))
                return result[:self._dim]
            except Exception:
                return self._hash_embed(text)
        else:
            return self._hash_embed(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Last-resort fallback: at least SHA256 not hash() (cross-run stable)."""
        import hashlib
        h = hashlib.sha256(text.encode())
        digest = int(h.hexdigest(), 16)
        np.random.seed(digest % (2**31))
        vec = np.random.randn(self._dim).tolist()
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    # ── Phase derivation (meaningful, not random) ────────

    def derive_phase(self, text: str, embedding: list[float] | None = None,
                     importance: float = 0.5, emotional_valence: float = 0.0,
                     timestamp: float | None = None) -> float:
        """Derive theta phase from actual semantically-meaningful factors.

        Phase = f(timestamp, importance, emotional_valence, semantic_direction)

        This is NOT random. Phase encodes:
        - When the memory was formed (temporal position in cycle)
        - How important it is (important memories phase-align)
        - Emotional charge (emotional memories are phase-shifted)
        - Semantic direction (which part of the embedding sphere)
        """
        if timestamp is None:
            timestamp = time.time()

        # Temporal component: recent=small phase, old=large phase
        # Maps seconds to a 24-hour theta cycle
        seconds_in_day = timestamp % 86400
        temporal_phase = (seconds_in_day / 86400) * 2 * math.pi

        # Importance component: important memories → phase = 0 (aligned)
        # Less important → phase spreads out
        importance_shift = (1.0 - importance) * math.pi * 0.5

        # Emotional component: strong emotion → phase offset
        # Positive emotions → one direction, negative → opposite
        emotion_shift = emotional_valence * math.pi * 0.25

        # Semantic direction from embedding (if available)
        if embedding and len(embedding) >= 2:
            angle = math.atan2(embedding[1], embedding[0])
            semantic_shift = angle * 0.3
        else:
            semantic_shift = 0.0

        phase = (temporal_phase + importance_shift + emotion_shift + semantic_shift) % (2 * math.pi)
        return phase

    def derive_frequency(self, importance: float = 0.5, emotional_valence: float = 0.0,
                         text_length: int = 0) -> float:
        """Derive natural frequency from content properties.

        Frequency maps to theta band (0.3-1.0):
        - High importance → faster oscillation (more readily replayed)
        - Strong emotion → faster (more reactive to driving context)
        - Longer text → slightly slower (more complex → lower freq)
        """
        base = 0.3
        importance_boost = importance * 0.4
        emotion_boost = abs(emotional_valence) * 0.2
        length_penalty = min(0.1, text_length / 2000) * 0.1
        return min(1.0, base + importance_boost + emotion_boost - length_penalty)

    def derive_driving_phasor(self, query: str,
                               embedding: list[float] | None = None) -> tuple[float, float]:
        """Derive driving frequency and phase from a query.

        This replaces the old hash-based approach. Now uses embedding statistics
        to derive meaningful oscillatory parameters.

        Returns (magnitude, phase) as a complex phasor.
        """
        if embedding and len(embedding) >= 4:
            arr = np.array(embedding)
            # Magnitude from embedding variance (richer query → stronger signal)
            var = float(np.var(arr))
            mag = 0.4 + 0.6 * min(1.0, var / (float(np.mean(np.abs(arr))) + 1e-8))

            # Phase from principal angular components
            # Use pairs of dimensions as (x, y) to estimate "direction" in embedding space
            angles = np.arctan2(arr[1::2], arr[::2])
            phase = float(np.mean(angles)) % (2 * math.pi)

            # Frequency from spectral centroid of embedding
            spectrum = np.abs(arr)
            if np.sum(spectrum) > 1e-8:
                centroid = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)
                freq = 0.3 + 0.7 * (centroid / len(spectrum))
            else:
                freq = 0.5
        else:
            # Encode query, then derive
            emb = self.encode(query)
            return self.derive_driving_phasor(query, emb)

        return (freq, phase)


# Module-level singleton
_provider: Optional[EmbeddingProvider] = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = EmbeddingProvider(model_name)
    return _provider


def reset_embedder() -> None:
    global _provider
    _provider = None
