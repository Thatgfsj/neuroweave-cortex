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
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .anchor import Anchor
from .graph import StarGraph, Constellation
from .config import Config


@dataclass
class RetrievalTraceEntry:
    """Why a memory appeared in a retrieval result."""

    memory_id: str
    score: float
    reason: str
    pathway: str = ""
    matched_terms: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "memory_id": self.memory_id,
            "score": round(_clip01(self.score), 4),
            "reason": self.reason,
        }
        if self.pathway:
            item["pathway"] = self.pathway
        if self.matched_terms:
            item["matched_terms"] = self.matched_terms
        if self.components:
            item["components"] = {
                key: round(_clip01(value), 4)
                for key, value in self.components.items()
            }
        return item


@dataclass
class RetrievalTrace:
    """JSON-serializable retrieval explainability payload."""

    query: str
    method: str
    retrieved_memories: list[RetrievalTraceEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "method": self.method,
            "retrieved_memories": [
                entry.to_dict()
                for entry in self.retrieved_memories
            ],
        }


@dataclass
class RetrievalResult:
    constellations: list[Constellation]
    latency_ms: float
    method: str
    top_score: float
    retrieval_trace: dict[str, Any] = field(default_factory=dict)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "the", "to", "user", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}
_TEMPORAL_WORDS = {
    "after", "april", "august", "before", "date", "day", "december",
    "during", "february", "friday", "january", "july", "june", "march",
    "may", "monday", "month", "november", "october", "saturday", "september",
    "since", "sunday", "thursday", "time", "today", "tomorrow", "tuesday",
    "until", "wednesday", "week", "when", "year", "yesterday",
}
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4})\b",
    re.IGNORECASE,
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokenize_terms(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", text)
    }
    return {
        token
        for token in tokens
        if len(token) > 1 and token not in _STOPWORDS
    }


def _has_temporal_signal(text: str) -> bool:
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", text)
    }
    return bool(terms & _TEMPORAL_WORDS) or bool(_DATE_RE.search(text))


def _matched_terms(query: str, anchor: Anchor) -> list[str]:
    query_terms = _tokenize_terms(query)
    memory_terms = _tokenize_terms(anchor.text)
    tag_terms = _tokenize_terms(" ".join(anchor.tags))
    return sorted(query_terms & (memory_terms | tag_terms))[:8]


def _trace_reason(query: str, anchor: Anchor, components: dict[str, float],
                  matched_terms: list[str]) -> str:
    reasons: list[str] = []

    if _has_temporal_signal(query) and _has_temporal_signal(anchor.text):
        reasons.append("temporal_match")

    if matched_terms:
        reasons.append("entity_match")

    tag_terms = _tokenize_terms(" ".join(anchor.tags))
    if tag_terms and tag_terms & set(matched_terms):
        reasons.append("tag_match")

    if components.get("semantic_similarity", 0.0) >= 0.35:
        reasons.append("semantic_match")
    if components.get("text_overlap", 0.0) >= 0.10:
        reasons.append("lexical_match")
    if components.get("phase_similarity", 0.0) >= 0.75:
        reasons.append("phase_match")
    if components.get("frequency_similarity", 0.0) >= 0.75:
        reasons.append("frequency_match")
    if components.get("cortical_similarity", 0.0) >= 0.20:
        reasons.append("cortical_lookup")
    if components.get("activation", 0.0) >= 0.20:
        reasons.append("activation_spread")
    if components.get("graph_expansion", 0.0) > 0.0:
        reasons.append("graph_expansion")
    if anchor.retention_score >= 0.60:
        reasons.append("retention_boost")

    if not reasons:
        reasons.append("score_ranked")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return " + ".join(deduped)


def _build_retrieval_trace(query: str, method: str,
                           rows: list[tuple[Anchor, float, str, dict[str, float]]]
                           ) -> dict[str, Any]:
    entries: list[RetrievalTraceEntry] = []
    seen: set[str] = set()

    for anchor, score, pathway, components in rows:
        if anchor.id in seen:
            continue
        seen.add(anchor.id)
        matches = _matched_terms(query, anchor)
        entries.append(RetrievalTraceEntry(
            memory_id=anchor.id,
            score=score,
            reason=_trace_reason(query, anchor, components, matches),
            pathway=pathway,
            matched_terms=matches,
            components=components,
        ))

    entries.sort(key=lambda entry: entry.score, reverse=True)
    return RetrievalTrace(
        query=query,
        method=method,
        retrieved_memories=entries,
    ).to_dict()


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

        if not self.graph.anchors:
            latency = (time.perf_counter() - t0) * 1000
            return RetrievalResult(
                constellations=[],
                latency_ms=latency,
                method=self.name,
                top_score=0.0,
                retrieval_trace=_build_retrieval_trace(query, self.name, []),
            )

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
                    score = sim * anchor.retention_score
                    scored.append((
                        score,
                        anchor,
                        "ann_vector",
                        {
                            "semantic_similarity": sim,
                            "retention": anchor.retention_score,
                        },
                    ))
            scored.sort(key=lambda x: -x[0])
        else:
            # Fallback linear scan
            scored = []
            for anchor in self.graph.anchors.values():
                if anchor.embedding:
                    sim = self._cosine_sim(embedding, anchor.embedding)
                    score = sim * anchor.retention_score
                    scored.append((
                        score,
                        anchor,
                        "linear_vector",
                        {
                            "semantic_similarity": sim,
                            "retention": anchor.retention_score,
                        },
                    ))
                else:
                    sim = self._text_overlap(query, anchor.text)
                    score = sim * anchor.retention_score
                    scored.append((
                        score,
                        anchor,
                        "text_overlap",
                        {
                            "text_overlap": sim,
                            "retention": anchor.retention_score,
                        },
                    ))
            scored.sort(key=lambda x: -x[0])

        constellations = []
        for i in range(min(top_k, len(scored))):
            _, anchor, _, _ = scored[i]
            c = Constellation(anchors=[anchor], edges=[])
            constellations.append(c)

        latency = (time.perf_counter() - t0) * 1000
        top_score = scored[0][0] if scored else 0.0
        trace_rows = [
            (anchor, score, pathway, components)
            for score, anchor, pathway, components in scored[:top_k]
        ]

        return RetrievalResult(
            constellations=constellations,
            latency_ms=latency,
            method=self.name,
            top_score=top_score,
            retrieval_trace=_build_retrieval_trace(query, self.name, trace_rows),
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

    def __init__(self, graph: StarGraph, spread_steps: int | None = None,
                 phase_weight: float | None = None):
        super().__init__(graph)
        c = Config.get().retrieval
        self.spread_steps = spread_steps if spread_steps is not None else c.spreading.steps
        self.phase_weight = phase_weight if phase_weight is not None else c.oscillation.phase_weight

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
            c = Config.get().retrieval.oscillation
            arr = np.array(embedding)
            var = float(np.var(arr))
            mag = c.magnitude_base + c.magnitude_variance_factor * min(1.0, var / (float(np.mean(np.abs(arr))) + 1e-8))
            angles = np.arctan2(arr[1::2], arr[::2])
            phase = float(np.mean(angles)) % (2 * math.pi)
            spectrum = np.abs(arr)
            if np.sum(spectrum) > 1e-8:
                centroid = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)
                freq = c.frequency_base + c.frequency_centroid_factor * (centroid / len(spectrum))
            else:
                freq = c.default_frequency
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

    def _trace_components(self, query_phasor: complex, driving_freq: float,
                          anchor: Anchor) -> dict[str, float]:
        anchor_phasor = self._to_phasor(anchor)
        query_phase = math.atan2(query_phasor.imag, query_phasor.real)
        anchor_phase = math.atan2(anchor_phasor.imag, anchor_phasor.real)
        phase_diff = abs(query_phase - anchor_phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        phase_similarity = max(0.0, math.cos(phase_diff))

        freq_diff = abs(driving_freq - anchor.oscillator.natural_frequency)
        frequency_similarity = 1.0 - min(1.0, freq_diff)

        return {
            "resonance": self._resonance_score(query_phasor, anchor),
            "phase_similarity": phase_similarity,
            "frequency_similarity": frequency_similarity,
            "retention": anchor.retention_score,
        }

    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()

        if not self.graph.anchors:
            latency = (time.perf_counter() - t0) * 1000
            return RetrievalResult(
                constellations=[],
                latency_ms=latency,
                method=self.name,
                top_score=0.0,
                retrieval_trace=_build_retrieval_trace(query, self.name, []),
            )

        if not embedding:
            from .embedding import get_embedder
            embedder = get_embedder()
            embedding = embedder.encode(query)

        driving_phasor, driving_freq = self._derive_driving_phasor(query, embedding)

        rc = Config.get().retrieval
        # Compute resonance for all anchors
        resonance_map: dict[str, float] = {}
        trace_components: dict[str, dict[str, float]] = {}
        trace_pathways: dict[str, set[str]] = {}
        for anchor in self.graph.anchors.values():
            components = self._trace_components(driving_phasor, driving_freq, anchor)
            score = components["resonance"]
            if score > 0.05:
                resonance_map[anchor.id] = score
                trace_components[anchor.id] = components
                trace_pathways.setdefault(anchor.id, set()).add("oscillation_resonance")

        # Also check cortical index
        cortical = self.graph.cortical_lookup(embedding, top_k=rc.cortical.top_k)
        for aid, sim in cortical:
            if aid in self.graph.anchors:
                trace_components.setdefault(aid, {
                    "retention": self.graph.anchors[aid].retention_score,
                })
                trace_components[aid]["cortical_similarity"] = sim
                trace_pathways.setdefault(aid, set()).add("cortical_lookup")
            if aid not in resonance_map:
                resonance_map[aid] = sim * rc.spreading.resonance_map_bonus

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
            combined[aid] = rc.spreading.resonance_weight * r + rc.spreading.activation_weight * a
            if aid in self.graph.anchors:
                trace_components.setdefault(aid, {
                    "retention": self.graph.anchors[aid].retention_score,
                })
                trace_components[aid]["activation"] = a
                trace_components[aid]["combined_score"] = combined[aid]
                if a > 0:
                    trace_pathways.setdefault(aid, set()).add("activation_spread")

        sorted_anchors = sorted(combined.items(), key=lambda x: -x[1])

        constellations = []
        for aid, score in sorted_anchors[:top_k]:
            if aid in self.graph.anchors:
                a = self.graph.anchors[aid]
                constellations.append(Constellation(anchors=[a], edges=[]))

        latency = (time.perf_counter() - t0) * 1000
        top_score = min(1.0, sorted_anchors[0][1] if sorted_anchors else 0.0)
        trace_rows: list[tuple[Anchor, float, str, dict[str, float]]] = []
        seen_trace: set[str] = set()
        for c in constellations:
            for anchor in c.anchors:
                if anchor.id in seen_trace:
                    continue
                seen_trace.add(anchor.id)
                components = dict(trace_components.get(anchor.id, {}))
                score = combined.get(anchor.id, 0.0)
                pathways = set(trace_pathways.get(anchor.id, set()))
                trace_rows.append((
                    anchor,
                    score,
                    "+".join(sorted(pathways)) or "oscillation_resonance",
                    components,
                ))

        return RetrievalResult(
            constellations=constellations,
            latency_ms=latency,
            method=self.name,
            top_score=top_score,
            retrieval_trace=_build_retrieval_trace(query, self.name, trace_rows),
        )


def compare_retrievers(graph: StarGraph, queries: list[str],
                       embeddings: list[list[float]] | None = None,
                       ground_truth: list[list[str]] | None = None,
                       include_trace: bool = False,
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

        if include_trace:
            comparison["vector_similarity"]["retrieval_trace"] = vec_result.retrieval_trace
            comparison["oscillation_resonance"]["retrieval_trace"] = osc_result.retrieval_trace

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
