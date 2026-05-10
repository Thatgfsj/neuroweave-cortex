"""Pluggable retrieval backends — v0.3 decoupled retrieval.

Abstract Retriever base with two concrete implementations:
- OscillationResonanceRetriever: phase-locking resonance (innovative)
- VectorSimilarityRetriever: cosine similarity (baseline)

This allows empirical validation: is oscillation resonance actually better?
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
    """Standardized retrieval result for comparison."""
    constellations: list[Constellation]
    latency_ms: float
    method: str
    top_score: float


class Retriever(ABC):
    """Abstract retrieval interface."""

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
    """Baseline: cosine similarity between query embedding and anchor embeddings.

    This is what Mem0, vector DBs, and most RAG systems use.
    """

    @property
    def name(self) -> str:
        return "VectorSimilarity"

    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()

        if not embedding:
            embedding = self._simple_embed(query)

        # Score all anchors by cosine similarity
        scored = []
        for anchor in self.graph.anchors.values():
            if anchor.embedding:
                sim = self._cosine_sim(embedding, anchor.embedding)
                scored.append((sim * anchor.retention_score, anchor))
            else:
                # Fallback: text overlap
                sim = self._text_overlap(query, anchor.text)
                scored.append((sim * anchor.retention_score, anchor))

        scored.sort(key=lambda x: -x[0])

        # Build pseudo-constellations from top anchors
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

    @staticmethod
    def _simple_embed(text: str, dim: int = 64) -> list[float]:
        """Deterministic pseudo-embedding from text (no model needed)."""
        h = hash(text)
        np.random.seed(abs(h) % (2**31))
        vec = np.random.randn(dim).tolist()
        norm = math.sqrt(sum(x**2 for x in vec))
        return [x / norm for x in vec]


class OscillationResonanceRetriever(Retriever):
    """Phase-locking resonance retrieval.

    Mathematical definition:
      Res(q, m) = |z_q · z_m| / (||z_q|| ||z_m||) × cos(Δφ)

    where z_q, z_m are complex phasors derived from query and anchor,
    and Δφ is the phase difference.

    This captures BOTH semantic similarity (phasor magnitude alignment)
    AND sequence/context position (phase alignment).
    """

    def __init__(self, graph: StarGraph, spread_steps: int = 3,
                 phase_weight: float = 0.3):
        super().__init__(graph)
        self.spread_steps = spread_steps
        self.phase_weight = phase_weight  # how much phase matters vs magnitude

    @property
    def name(self) -> str:
        return "OscillationResonance"

    def _to_phasor(self, anchor: Anchor) -> complex:
        """Convert anchor's oscillator state to a complex phasor.

        Magnitude = retention_score (how "loud" this memory is)
        Phase = oscillator's natural phase offset (where it is in the cycle)
        """
        mag = anchor.retention_score
        phase = anchor.oscillator.phase_offset
        return mag * complex(math.cos(phase), math.sin(phase))

    def _derive_driving_phasor(self, query: str,
                                embedding: list[float] | None = None) -> complex:
        """Derive the driving oscillation from the query."""
        if embedding:
            # Use embedding statistics
            arr = np.array(embedding)
            mag = float(np.std(arr)) + 0.3  # variance → magnitude
            angles = np.arctan2(arr[1::2], arr[::2])
            phase = float(np.mean(angles)) % (2 * math.pi)
        else:
            # Deterministic from text hash
            h = abs(hash(query))
            mag = 0.5 + 0.5 * ((h % 1000) / 1000.0)
            phase = ((h // 1000) % 6283) / 1000.0

        return mag * complex(math.cos(phase), math.sin(phase))

    def _resonance_score(self, query_phasor: complex, anchor: Anchor) -> float:
        """Compute Res(q, m) = |z_q · z_m| / (||z_q|| ||z_m||) × cos(Δφ)."""
        anchor_phasor = self._to_phasor(anchor)

        # Phasor dot product magnitude (normalized)
        dot = abs(query_phasor.real * anchor_phasor.real +
                  query_phasor.imag * anchor_phasor.imag)
        mag_product = abs(query_phasor) * abs(anchor_phasor)
        if mag_product < 1e-8:
            return 0.0
        mag_sim = dot / mag_product

        # Phase difference
        query_phase = math.atan2(query_phasor.imag, query_phasor.real)
        anchor_phase = math.atan2(anchor_phasor.imag, anchor_phasor.real)
        phase_diff = abs(query_phase - anchor_phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        phase_sim = math.cos(phase_diff)

        # Combined: magnitude similarity + phase alignment (clamped to [0,1])
        return min(1.0, (1.0 - self.phase_weight) * mag_sim + self.phase_weight * max(0.0, phase_sim))

    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()

        # Step 1: Derive driving phasor from query
        driving = self._derive_driving_phasor(query, embedding)

        # Step 2: Compute resonance for all anchors
        resonance_map: dict[str, float] = {}
        for anchor in self.graph.anchors.values():
            score = self._resonance_score(driving, anchor)
            if score > 0.05:
                resonance_map[anchor.id] = score

        # Step 3: Also check cortical index for consolidated memories
        if embedding:
            cortical = self.graph.cortical_lookup(embedding, top_k=10)
            for aid, sim in cortical:
                if aid not in resonance_map:
                    resonance_map[aid] = sim * 0.7  # cortical hits weighted lower

        # Step 4: Spread activation from top seeds
        sorted_seeds = sorted(resonance_map.items(), key=lambda x: -x[1])[:top_k * 2]
        activation = self.graph.spread_activation(
            [aid for aid, _ in sorted_seeds],
            steps=self.spread_steps,
        )

        # Step 5: Merge resonance + activation
        combined: dict[str, float] = {}
        for aid in set(list(resonance_map.keys()) + list(activation.keys())):
            r = resonance_map.get(aid, 0.0)
            a = activation.get(aid, 0.0)
            combined[aid] = 0.6 * r + 0.4 * a

        # Step 6: Build constellations from top activated anchors
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

        # Deduplicate and limit
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
    """Run both retrievers on the same queries and compare results.

    Returns comparison metrics for each query.
    """
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
            # Compute recall@k for each method
            comparison["vector_similarity"]["recall@3"] = _recall_at_k(
                vec_result, gt, k=3)
            comparison["oscillation_resonance"]["recall@3"] = _recall_at_k(
                osc_result, gt, k=3)

        results.append(comparison)

    return results


def _recall_at_k(result: RetrievalResult, ground_truth_ids: list[str], k: int = 3) -> float:
    """Compute recall@k: fraction of ground truth found in top k results."""
    retrieved_ids = set()
    for c in result.constellations[:k]:
        for a in c.anchors:
            retrieved_ids.add(a.id)

    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 1.0
    return len(retrieved_ids & gt_set) / len(gt_set)
