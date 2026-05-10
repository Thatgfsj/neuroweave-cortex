"""Pluggable retrieval backends — v0.4 real mechanisms.

Two concrete retrievers:
- OscillationResonanceRetriever: phase-locking resonance with real embedding-derived phases
- VectorSimilarityRetriever: cosine similarity with ANN-indexed sub-linear lookup

Phase derivation is now meaningful:
  theta_phase = f(timestamp, importance, emotional_valence)
  driving_phase = f(embedding principal angles)

No more hash-based random phases.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .anchor import Anchor
from .graph import StarGraph, Constellation


@dataclass
class RetrievalResult:
    constellations: list[Constellation]
    latency_ms: float
    method: str
    top_score: float


class Retriever(ABC):
    def __init__(self, graph: StarGraph):
        self.graph = graph

    @abstractmethod
    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class VectorSimilarityRetriever(Retriever):
    """Cosine similarity with ANN-indexed sub-linear retrieval.

    Uses the graph's ANNIndex when embeddings are available, falling
    back to linear scan for cold start / small graphs.
    """

    @property
    def name(self) -> str:
        return "VectorSimilarity"

    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()

        if not embedding:
            from .embedding import get_embedder
            embedder = get_embedder()
            embedding = embedder.encode(query)

        ann = self.graph._get_ann_index() if self.graph._ann_index is not None else None

        if ann is not None and ann.size > 0:
            if not self.graph._ids_in_ann_sync():
                ann.clear()
                for a in self.graph.anchors.values():
                    if a.embedding:
                        ann.add(a.id, a.embedding)
                ann.rebuild()
            results = ann.query(embedding, k=top_k * 3)
            scored = []
            for aid, sim in results:
                if aid in self.graph.anchors:
                    anchor = self.graph.anchors[aid]
                    scored.append((sim * anchor.retention_score, anchor))
            scored.sort(key=lambda x: -x[0])
        else:
            # Fallback linear scan
            scored = []
            for anchor in self.graph.anchors.values():
                if anchor.embedding:
                    sim = self._cosine_sim(embedding, anchor.embedding)
                    scored.append((sim * anchor.retention_score, anchor))
                else:
                    sim = self._text_overlap(query, anchor.text)
                    scored.append((sim * anchor.retention_score, anchor))
            scored.sort(key=lambda x: -x[0])

        constellations = []
        for i in range(min(top_k, len(scored))):
            score, anchor = scored[i]
            c = Constellation(anchors=[anchor], edges=[])
            constellations.append(c)

        latency = (time.perf_counter() - t0) * 1000
        top_score = scored[0][0] if scored else 0.0

        return RetrievalResult(
            constellations=constellations,
            latency_ms=latency,
            method=self.name,
            top_score=top_score,
        )

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x**2 for x in a))
        nb = math.sqrt(sum(x**2 for x in b))
        return dot / (na * nb + 1e-8)

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        def bigrams(s):
            return {s[i:i+2] for i in range(len(s)-1)}
        ba, bb = bigrams(a), bigrams(b)
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)


class OscillationResonanceRetriever(Retriever):
    """Phase-locking resonance retrieval with meaningful phase derivation.

    Mathematical definition:
      Res(q, m) = (1-w) × |z_q · z_m| / (||z_q|| ||z_m||) + w × cos(Δφ)

    Phase is now derived from real embedding structure, not hash():
      - Query phase: principal angular components of embedding
      - Anchor phase: f(timestamp, importance, emotional_valence)
      - Frequency: f(importance, emotional_valence, text_length)

    This captures BOTH:
      - Semantic similarity (phasor magnitude alignment via embeddings)
      - Temporal/emotional context position (phase alignment via meaningful φ)
    """

    def __init__(self, graph: StarGraph, spread_steps: int = 3,
                 phase_weight: float = 0.3):
        super().__init__(graph)
        self.spread_steps = spread_steps
        self.phase_weight = phase_weight

    @property
    def name(self) -> str:
        return "OscillationResonance"

    def _to_phasor(self, anchor: Anchor) -> complex:
        mag = anchor.retention_score
        phase = anchor.oscillator.phase_offset
        return mag * complex(math.cos(phase), math.sin(phase))

    def _derive_driving_phasor(self, query: str,
                                embedding: list[float] | None = None) -> tuple[complex, float]:
        """Derive driving phasor from query using real embedding statistics.

        No more hash() — uses embedding principal angles for phase and
        spectral centroid for frequency.
        """
        if embedding and len(embedding) >= 4:
            arr = np.array(embedding)
            var = float(np.var(arr))
            mag = 0.4 + 0.6 * min(1.0, var / (float(np.mean(np.abs(arr))) + 1e-8))
            angles = np.arctan2(arr[1::2], arr[::2])
            phase = float(np.mean(angles)) % (2 * math.pi)
            spectrum = np.abs(arr)
            if np.sum(spectrum) > 1e-8:
                centroid = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)
                freq = 0.3 + 0.7 * (centroid / len(spectrum))
            else:
                freq = 0.5
        else:
            from .embedding import get_embedder
            embedder = get_embedder()
            emb = embedder.encode(query)
            return self._derive_driving_phasor(query, emb)

        phasor = mag * complex(math.cos(phase), math.sin(phase))
        return (phasor, freq)

    def _resonance_score(self, query_phasor: complex, anchor: Anchor) -> float:
        """Compute Res(q, m) = (1-w) × mag_sim + w × cos(Δφ)."""
        anchor_phasor = self._to_phasor(anchor)

        dot = abs(query_phasor.real * anchor_phasor.real +
                  query_phasor.imag * anchor_phasor.imag)
        mag_product = abs(query_phasor) * abs(anchor_phasor)
        if mag_product < 1e-8:
            return 0.0
        mag_sim = dot / mag_product

        query_phase = math.atan2(query_phasor.imag, query_phasor.real)
        anchor_phase = math.atan2(anchor_phasor.imag, anchor_phasor.real)
        phase_diff = abs(query_phase - anchor_phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        phase_sim = math.cos(phase_diff)

        return min(1.0, (1.0 - self.phase_weight) * mag_sim + self.phase_weight * max(0.0, phase_sim))

    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()

        if not embedding:
            from .embedding import get_embedder
            embedder = get_embedder()
            embedding = embedder.encode(query)

        driving_phasor, driving_freq = self._derive_driving_phasor(query, embedding)

        # Compute resonance for all anchors
        resonance_map: dict[str, float] = {}
        for anchor in self.graph.anchors.values():
            score = self._resonance_score(driving_phasor, anchor)
            if score > 0.05:
                resonance_map[anchor.id] = score

        # Also check cortical index
        cortical = self.graph.cortical_lookup(embedding, top_k=10)
        for aid, sim in cortical:
            if aid not in resonance_map:
                resonance_map[aid] = sim * 0.7

        # Spread activation from top seeds
        sorted_seeds = sorted(resonance_map.items(), key=lambda x: -x[1])[:top_k * 2]
        activation = self.graph.spread_activation(
            [aid for aid, _ in sorted_seeds],
            steps=self.spread_steps,
        )

        combined: dict[str, float] = {}
        for aid in set(list(resonance_map.keys()) + list(activation.keys())):
            r = resonance_map.get(aid, 0.0)
            a = activation.get(aid, 0.0)
            combined[aid] = 0.6 * r + 0.4 * a

        sorted_anchors = sorted(combined.items(), key=lambda x: -x[1])

        constellations = []
        seen: set[str] = set()
        for aid, score in sorted_anchors[:top_k * 5]:
            if aid not in seen and aid in self.graph.anchors:
                c = self.graph.find_constellation(aid, max_size=15)
                if c.anchors:
                    constellations.append(c)
                    for a in c.anchors:
                        seen.add(a.id)

        unique = []
        seen_c: set[str] = set()
        for c in constellations:
            key = tuple(sorted(a.id for a in c.anchors[:5]))
            if key not in seen_c:
                seen_c.add(key)
                unique.append(c)

        latency = (time.perf_counter() - t0) * 1000
        top_score = min(1.0, sorted_anchors[0][1] if sorted_anchors else 0.0)

        return RetrievalResult(
            constellations=unique[:top_k],
            latency_ms=latency,
            method=self.name,
            top_score=top_score,
        )


def compare_retrievers(graph: StarGraph, queries: list[str],
                       embeddings: list[list[float]] | None = None,
                       ground_truth: list[list[str]] | None = None
                       ) -> list[dict]:
    vec_ret = VectorSimilarityRetriever(graph)
    osc_ret = OscillationResonanceRetriever(graph)

    results = []
    for i, query in enumerate(queries):
        emb = embeddings[i] if embeddings else None
        gt = ground_truth[i] if ground_truth else None

        vec_result = vec_ret.retrieve(query, emb)
        osc_result = osc_ret.retrieve(query, emb)

        comparison = {
            "query": query[:60],
            "vector_similarity": {
                "latency_ms": vec_result.latency_ms,
                "top_score": vec_result.top_score,
                "num_results": len(vec_result.constellations),
            },
            "oscillation_resonance": {
                "latency_ms": osc_result.latency_ms,
                "top_score": osc_result.top_score,
                "num_results": len(osc_result.constellations),
            },
        }

        if gt:
            comparison["vector_similarity"]["recall@3"] = _recall_at_k(
                vec_result, gt, k=3)
            comparison["oscillation_resonance"]["recall@3"] = _recall_at_k(
                osc_result, gt, k=3)

        results.append(comparison)

    return results


def _recall_at_k(result: RetrievalResult, ground_truth_ids: list[str], k: int = 3) -> float:
    retrieved_ids = set()
    for c in result.constellations[:k]:
        for a in c.anchors:
            retrieved_ids.add(a.id)

    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 1.0
    return len(retrieved_ids & gt_set) / len(gt_set)
