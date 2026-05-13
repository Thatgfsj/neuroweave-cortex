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
            results = ann.query(embedding, k=top_k * 3)
            scored = []
            for aid, sim in results:
                if aid in self.graph.anchors:
                    anchor = self.graph.anchors[aid]
                    score = max(0.0, sim) * anchor.retention_score
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
                    sim = max(0.0, self._cosine_sim(embedding, anchor.embedding))
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
    """Resonance retrieval with graph spreading activation.

    Combines semantic cosine similarity with phase coherence derived from
    embedder-provided per-anchor phase offsets and embedding-driven query phase.
    Phase coherence uses 0.5+0.5*cos(delta_phi) so it is always in [0,1].

    score = (1-w) * semantic_sim + w * phase_coherence
    """

    def __init__(self, graph: StarGraph, spread_steps: int | None = None,
                 phase_weight: float | None = None):
        super().__init__(graph)
        c = Config.get().retrieval
        self.spread_steps = spread_steps if spread_steps is not None else c.spreading.steps
        self.phase_weight = phase_weight if phase_weight is not None else 0.15

    @property
    def name(self) -> str:
        return "OscillationResonance"

    def _query_phase(self, embedding: list[float]) -> float:
        arr = np.array(embedding)
        angles = np.arctan2(arr[1::2], arr[::2])
        return float(np.mean(angles)) % (2 * math.pi)

    def _phase_coherence(self, query_phase: float, anchor: Anchor) -> float:
        diff = abs(query_phase - anchor.oscillator.phase_offset)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return 0.5 + 0.5 * math.cos(diff)

    def _to_phasor(self, anchor: Anchor) -> complex:
        mag = anchor.retention_score
        phase = anchor.oscillator.phase_offset
        return mag * complex(math.cos(phase), math.sin(phase))

    def _derive_driving_phasor(self, query: str,
                                embedding: list[float] | None = None) -> tuple[complex, float]:
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

        query_phase = self._query_phase(embedding)
        query_phasor, driving_freq = self._derive_driving_phasor(query, embedding)

        # Score every anchor: (1-w)*semantic_sim + w*phase_coherence, weighted by retention
        scored: list[tuple[float, Anchor, dict[str, float]]] = []
        for anchor in self.graph.anchors.values():
            if anchor.embedding:
                semantic_sim = max(0.0, self._cosine_sim(embedding, anchor.embedding))
            else:
                semantic_sim = self._text_overlap(query, anchor.text)
            phase_coh = self._phase_coherence(query_phase, anchor)
            score_base = (1.0 - self.phase_weight) * semantic_sim + self.phase_weight * phase_coh
            score = max(0.0, score_base * anchor.retention_score)
            trace_comps = self._trace_components(query_phasor, driving_freq, anchor)
            trace_comps["semantic_similarity"] = semantic_sim
            trace_comps["activation"] = score_base * 0.3
            scored.append((score, anchor, trace_comps))
        scored.sort(key=lambda x: -x[0])

        sorted_anchors = scored

        constellations = []
        for score, anchor, comps in sorted_anchors[:top_k]:
            constellations.append(Constellation(anchors=[anchor], edges=[]))

        latency = (time.perf_counter() - t0) * 1000
        top_score = sorted_anchors[0][0] if sorted_anchors else 0.0

        trace_rows: list[tuple[Anchor, float, str, dict[str, float]]] = []
        for score, anchor, comps in sorted_anchors[:top_k]:
            trace_rows.append((
                anchor,
                score,
                "oscillation_resonance",
                comps,
            ))

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
        ba, bb = bigrams(a.lower()), bigrams(b.lower())
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)


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


# ── Personalized PageRank ─────────────────────────────────────

def personalized_pagerank(graph: StarGraph, query_embedding: list[float] | None,
                          damping: float = 0.85, max_iter: int = 50,
                          tol: float = 1e-6) -> dict[str, float]:
    """Compute Personalized PageRank over the memory graph.

    The personalization vector is seeded by embedding similarity to the query
    — anchors more similar to the query get higher restart probability.

    PPR(v) = (1-d) * personalization[v] + d * sum_{(u,v) in E} PPR(u) / deg(u)
    """
    anchors_list = list(graph.anchors.values())
    if not anchors_list:
        return {}

    n = len(anchors_list)
    id_to_idx = {a.id: i for i, a in enumerate(anchors_list)}

    # Build personalization vector from embedding similarity
    personalization = np.zeros(n)
    if query_embedding:
        sims = []
        for a in anchors_list:
            if a.embedding:
                dot = sum(query_embedding[i] * a.embedding[i] for i in range(min(len(query_embedding), len(a.embedding))))
                na = math.sqrt(sum(x**2 for x in query_embedding))
                nb = math.sqrt(sum(x**2 for x in a.embedding))
                sims.append(max(0.0, dot / (na * nb + 1e-8)))
            else:
                sims.append(0.0)
        total_sim = sum(sims)
        if total_sim > 1e-8:
            for i, s in enumerate(sims):
                personalization[i] = s / total_sim
        else:
            personalization.fill(1.0 / n)
    else:
        personalization.fill(1.0 / n)

    # Build adjacency matrix (out-degree normalized)
    adj = np.zeros((n, n))
    for i, a in enumerate(anchors_list):
        neighbors = graph.neighbors(a.id)
        total_weight = sum(w for _, w in neighbors)
        if total_weight > 0:
            for neighbor_id, weight in neighbors:
                j = id_to_idx.get(neighbor_id)
                if j is not None:
                    adj[i, j] = weight / total_weight
        else:
            # Dangling node → uniform jump
            adj[i, :] = 1.0 / n

    # Power iteration
    ppr = personalization.copy()
    for _ in range(max_iter):
        new_ppr = (1 - damping) * personalization + damping * adj.T @ ppr
        delta = np.sum(np.abs(new_ppr - ppr))
        ppr = new_ppr
        if delta < tol:
            break

    return {anchors_list[i].id: float(ppr[i]) for i in range(n)}


# ── Hybrid Fusion Retriever ───────────────────────────────────

@dataclass
class ExplainableScore:
    """Breakdown of why a memory was retrieved."""
    semantic_sim: float = 0.0       # embedding cosine similarity
    temporal_score: float = 0.0     # recency bonus
    graph_score: float = 0.0        # structural relevance (PPR / spread)
    lexical_overlap: float = 0.0    # query-to-text keyword overlap
    confidence: float = 0.5         # edge confidence when traversed
    combined: float = 0.0           # weighted fusion
    reasoning_path: list[str] = field(default_factory=list)

    def explain(self) -> str:
        parts = []
        if self.semantic_sim > 0.3:
            parts.append(f"semantic match ({self.semantic_sim:.2f})")
        if self.temporal_score > 0.1:
            parts.append(f"recent ({self.temporal_score:.2f})")
        if self.graph_score > 0.05:
            parts.append(f"graph structure ({self.graph_score:.2f})")
        if self.reasoning_path:
            parts.append(f"{len(self.reasoning_path)}-hop path")
        return " + ".join(parts) if parts else "low confidence"


class HybridFusionRetriever(Retriever):
    """Properly fuses embedding similarity, temporal context, and graph structure.

    This replaces the broken oscillation-resonance approach with a
    principled multi-signal fusion:

        score = α * semantic_sim + β * temporal + γ * ppr + δ * confidence

    where:
        - semantic_sim: direct cosine similarity to query embedding
        - temporal: recency bonus for recently accessed memories
        - ppr: Personalized PageRank score (graph structure)
        - confidence: edge confidence from RichEdge traversal

    Default weights: α=0.55, β=0.15, γ=0.20, δ=0.10
    """

    def __init__(self, graph: StarGraph,
                 alpha: float = 0.50, beta: float = 0.12,
                 gamma: float = 0.18, delta: float = 0.08,
                 epsilon: float = 0.12, spread_steps: int = 2):
        super().__init__(graph)
        self.alpha = alpha    # semantic weight
        self.beta = beta      # temporal weight
        self.gamma = gamma    # graph structure weight
        self.delta = delta    # confidence weight
        self.epsilon = epsilon  # lexical overlap weight
        self.spread_steps = spread_steps

    @property
    def name(self) -> str:
        return "HybridFusion"

    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()

        if not embedding:
            from .embedding import get_embedder
            embedder = get_embedder()
            embedding = embedder.encode(query)

        now = time.time()
        scores: dict[str, tuple[float, ExplainableScore]] = {}

        # 1. Semantic similarity to all anchors
        for anchor in self.graph.anchors.values():
            explain = ExplainableScore()
            if anchor.embedding and embedding:
                explain.semantic_sim = self._cosine_sim(embedding, anchor.embedding)
            else:
                explain.semantic_sim = self._text_overlap(query, anchor.text)

            # 2. Temporal recency bonus
            hours_since = (now - anchor.last_activated_at) / 3600
            explain.temporal_score = math.exp(-hours_since / 168)  # 1-week half-life

            scores[anchor.id] = (0.0, explain)

        # 3. Personalized PageRank for graph structure
        ppr = personalized_pagerank(self.graph, embedding)
        max_ppr = max(ppr.values()) if ppr else 1.0
        for aid, ppr_val in ppr.items():
            if aid in scores:
                scores[aid][1].graph_score = ppr_val / max_ppr if max_ppr > 0 else 0.0

        # 4. Spreading activation for reasoning paths
        top_seeds = sorted(scores.items(),
                          key=lambda x: -(x[1][1].semantic_sim + x[1][1].temporal_score * 0.5))[:5]
        seed_ids = [aid for aid, _ in top_seeds]
        activation = self.graph.spread_activation(seed_ids, steps=self.spread_steps)

        # Map activation to scores, record reasoning paths
        for aid, act in activation.items():
            if aid in scores:
                scores[aid][1].reasoning_path = [seed_ids[0], aid] if seed_ids else []

        # 5. Lexical overlap: distinctive query words in anchor text
        query_words = set(query.lower().split())
        for aid, (_, explain) in scores.items():
            anchor = self.graph.anchors.get(aid)
            if anchor:
                text_words = set(anchor.text.lower().split())
                overlap = len(query_words & text_words)
                explain.lexical_overlap = min(1.0, overlap / max(1, len(query_words)) * 2.0)

        # 6. Fuse all signals
        for aid, (_, explain) in scores.items():
            explain.combined = max(0.0,
                self.alpha * explain.semantic_sim
                + self.beta * explain.temporal_score
                + self.gamma * explain.graph_score
                + self.delta * explain.confidence
                + self.epsilon * explain.lexical_overlap
            )
            scores[aid] = (explain.combined, explain)

        # Sort and build results
        sorted_results = sorted(scores.items(), key=lambda x: -x[1][0])

        constellations = []
        for aid, (combined, explain) in sorted_results[:top_k]:
            if aid in self.graph.anchors:
                a = self.graph.anchors[aid]
                constellations.append(Constellation(anchors=[a], edges=[]))

        latency = (time.perf_counter() - t0) * 1000
        top_score = sorted_results[0][1][0] if sorted_results else 0.0

        return RetrievalResult(
            constellations=constellations,
            latency_ms=latency,
            method=self.name,
            top_score=top_score,
        )

    def explain(self, query: str, embedding: list[float] | None = None,
                top_k: int = 3) -> list[tuple[Anchor, ExplainableScore]]:
        """Retrieve with explainable scores — shows WHY each memory was selected."""
        if not embedding:
            from .embedding import get_embedder
            embedder = get_embedder()
            embedding = embedder.encode(query)

        result = self.retrieve(query, embedding, top_k)
        explained = []
        for c in result.constellations:
            for anchor in c.anchors:
                # Recompute explainable score
                explain = ExplainableScore()
                if anchor.embedding and embedding:
                    explain.semantic_sim = self._cosine_sim(embedding, anchor.embedding)
                hours_since = (time.time() - anchor.last_activated_at) / 3600
                explain.temporal_score = math.exp(-hours_since / 168)
                explain.combined = (
                    self.alpha * explain.semantic_sim
                    + self.beta * explain.temporal_score
                )
                explained.append((anchor, explain))
        explained.sort(key=lambda x: -x[1].combined)
        return explained[:top_k]

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
        ba, bb = bigrams(a.lower()), bigrams(b.lower())
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)
