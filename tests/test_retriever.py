"""Tests for retriever module — dataclasses, helpers, and retrieval methods."""

import math
import time

import pytest

from star_graph.anchor import Anchor, AnchorVector
from star_graph.graph import StarGraph, Constellation
from star_graph.retriever import (
    _clip01,
    _tokenize_terms,
    _has_temporal_signal,
    _matched_terms,
    _trace_reason,
    _build_retrieval_trace,
    _recall_at_k,
    _cosine_sim,
    RetrievalTraceEntry,
    RetrievalTrace,
    RetrievalResult,
    ExplainableScore,
    VectorSimilarityRetriever,
    OscillationResonanceRetriever,
    HybridFusionRetriever,
    compare_retrievers,
    personalized_pagerank,
)


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                tags: list | None = None, importance: float = 0.5,
                emotional_valence: float = 0.0, created_at: float = 0.0) -> Anchor:
    a = Anchor(
        id=name, text=text or f"Memory {name}",
        vector=AnchorVector(importance=importance,
                           emotional_valence=emotional_valence,
                           success_feedback=0.5,
                           confidence=0.5,
                           frequency=0.3),
        tags=tags or [],
        created_at=created_at or time.time(),
    )
    if embedding:
        a.embedding = embedding
    return a


# ── Helper functions ──────────────────────────────────────────

class TestClip01:
    def test_positive_lt_1(self):
        assert _clip01(0.5) == 0.5

    def test_negative(self):
        assert _clip01(-0.3) == 0.0

    def test_above_1(self):
        assert _clip01(1.5) == 1.0

    def test_int(self):
        assert _clip01(42) == 1.0


class TestTokenizeTerms:
    def test_basic(self):
        tokens = _tokenize_terms("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stopwords_excluded(self):
        tokens = _tokenize_terms("the hello is happy")
        assert "the" not in tokens
        assert "hello" in tokens
        assert "happy" in tokens

    def test_single_char_excluded(self):
        tokens = _tokenize_terms("a x b hello")
        assert "hello" in tokens
        assert "x" not in tokens

    def test_case_insensitive(self):
        tokens = _tokenize_terms("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens


class TestHasTemporalSignal:
    def test_temporal_word(self):
        assert _has_temporal_signal("the meeting was on monday morning")

    def test_date_pattern(self):
        assert _has_temporal_signal("event on 2024-07-14 was great")

    def test_no_temporal(self):
        assert not _has_temporal_signal("hello world test function")


class TestMatchedTerms:
    def test_matches_in_text(self):
        a = make_anchor("a1", text="redis timeout debug", tags=["database"])
        terms = _matched_terms("redis timeout", a)
        assert "redis" in terms
        assert "timeout" in terms

    def test_matches_in_tags(self):
        a = make_anchor("a1", text="connection issues", tags=["redis", "timeout"])
        terms = _matched_terms("redis timeout", a)
        assert "redis" in terms

    def test_empty_returns_empty(self):
        a = make_anchor("a1", text="completely unrelated")
        terms = _matched_terms("redis timeout", a)
        assert len(terms) == 0


class TestTraceReason:
    def test_returns_reason_string(self):
        a = make_anchor("a1", text="redis timeout", tags=["database"])
        result = _trace_reason("redis timeout", a,
                              {"semantic_similarity": 0.8}, ["redis"])
        assert isinstance(result, str)
        assert "entity_match" in result or "semantic_match" in result

    def test_temporal_match_detected(self):
        a = make_anchor("a1", text="event on monday at 3pm", tags=["schedule"])
        result = _trace_reason("what happened on monday", a,
                              {"semantic_similarity": 0.5}, ["monday"])
        assert isinstance(result, str)


class TestBuildRetrievalTrace:
    def test_empty_rows(self):
        trace = _build_retrieval_trace("test query", "VectorTest", [])
        assert trace["query"] == "test query"
        assert trace["method"] == "VectorTest"
        assert trace["retrieved_memories"] == []

    def test_with_rows(self):
        a = make_anchor("a1", "redis timeout fix", tags=["redis"])
        a.embedding = [0.1, 0.2, 0.3]
        rows = [(a, 0.9, "hippocampal", {"semantic_similarity": 0.8})]
        trace = _build_retrieval_trace("redis timeout", "VectorTest", rows)
        assert len(trace["retrieved_memories"]) >= 1


class TestRecallAtK:
    def test_empty_result(self):
        g = StarGraph()
        result = RetrievalResult(constellations=[], latency_ms=10.0,
                                method="test", top_score=0.0)
        assert _recall_at_k(result, ["a1"], k=3) == 0.0

    def test_empty_ground_truth_returns_1(self):
        g = StarGraph()
        result = RetrievalResult(constellations=[], latency_ms=10.0,
                                method="test", top_score=0.0)
        assert _recall_at_k(result, [], k=3) == 1.0

    def test_perfect_match(self):
        a1 = make_anchor("gt1", "ground truth", embedding=[1.0, 0.0, 0.0])
        a2 = make_anchor("gt2", "also truth", embedding=[0.0, 1.0, 0.0])
        c1 = Constellation(anchors=[a1], edges=[])
        c2 = Constellation(anchors=[a2], edges=[])
        result = RetrievalResult(constellations=[c1, c2], latency_ms=10.0,
                                method="test", top_score=0.9)
        score = _recall_at_k(result, ["gt1", "gt2"], k=3)
        assert score == 1.0


class TestCosineSim:
    def test_identical(self):
        assert _cosine_sim([1.0], [1.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert _cosine_sim([], []) == 0.0


# ── Dataclass tests ──────────────────────────────────────────

class TestRetrievalTraceEntry:
    def test_to_dict_basic(self):
        entry = RetrievalTraceEntry(memory_id="a1", score=0.85,
                                    reason="semantic match")
        d = entry.to_dict()
        assert d["memory_id"] == "a1"
        assert d["score"] == 0.85
        assert d["reason"] == "semantic match"

    def test_to_dict_full(self):
        entry = RetrievalTraceEntry(
            memory_id="a1", score=0.9, reason="multi-match",
            pathway="oscillation", matched_terms=["redis", "timeout"],
            components={"semantic": 0.8, "phase": 0.6},
        )
        d = entry.to_dict()
        assert d["pathway"] == "oscillation"
        assert "redis" in d["matched_terms"]
        assert "semantic" in d["components"]


class TestRetrievalTrace:
    def test_to_dict_empty(self):
        trace = RetrievalTrace(query="test", method="VectorTest")
        d = trace.to_dict()
        assert d["query"] == "test"
        assert d["retrieved_memories"] == []

    def test_to_dict_with_entries(self):
        entry = RetrievalTraceEntry(memory_id="a1", score=0.9,
                                    reason="match")
        trace = RetrievalTrace(query="test", method="VectorTest",
                              retrieved_memories=[entry])
        d = trace.to_dict()
        assert len(d["retrieved_memories"]) == 1


class TestExplainableScore:
    def test_init_defaults(self):
        score = ExplainableScore()
        assert score.semantic_sim == 0.0
        assert score.temporal_score == 0.0
        assert score.confidence == 0.5

    def test_with_values(self):
        score = ExplainableScore(semantic_sim=0.8, temporal_score=0.3,
                                graph_score=0.1, combined=0.75,
                                reasoning_path=["a1", "a2"])
        assert score.semantic_sim == 0.8
        assert score.reasoning_path == ["a1", "a2"]

    def test_explain_basic(self):
        score = ExplainableScore(semantic_sim=0.5)
        result = score.explain()
        assert "semantic match" in result

    def test_explain_graph(self):
        score = ExplainableScore(graph_score=0.1)
        result = score.explain()
        assert "graph structure" in result

    def test_explain_reasoning_path(self):
        score = ExplainableScore(reasoning_path=["a1", "a2", "a3"])
        result = score.explain()
        assert "3-hop" in result

    def test_explain_low_confidence(self):
        score = ExplainableScore()
        result = score.explain()
        assert "low confidence" in result


# ── Retriever tests ──────────────────────────────────────────

class TestVectorSimilarityRetriever:
    def test_empty_graph(self):
        g = StarGraph()
        retriever = VectorSimilarityRetriever(g)
        result = retriever.retrieve("any query",
                                    embedding=[0.1, 0.2, 0.3])
        assert result.constellations == []
        assert result.latency_ms >= 0
        assert result.method == "VectorSimilarity"

    def test_single_anchor(self):
        g = StarGraph()
        a = make_anchor("a1", "redis timeout fix", embedding=[1.0, 0.0, 0.0],
                       tags=["redis"])
        g.add_anchor(a)
        retriever = VectorSimilarityRetriever(g)
        result = retriever.retrieve("redis timeout",
                                    embedding=[1.0, 0.0, 0.0])
        assert result.method == "VectorSimilarity"

    def test_retrieve_with_trace(self):
        g = StarGraph()
        a = make_anchor("a1", "redis timeout fix", embedding=[0.1, 0.2, 0.3],
                       tags=["redis", "timeout"])
        g.add_anchor(a)
        retriever = VectorSimilarityRetriever(g)
        result = retriever.retrieve("redis timeout",
                                    embedding=[0.1, 0.2, 0.3])
        assert result.method == "VectorSimilarity"


class TestOscillationResonanceRetriever:
    def test_empty_graph(self):
        g = StarGraph()
        retriever = OscillationResonanceRetriever(g)
        result = retriever.retrieve("any query",
                                    embedding=[0.1, 0.2, 0.3])
        assert result.constellations == []
        assert result.method == "OscillationResonance"

    def test_single_anchor(self):
        g = StarGraph()
        a = make_anchor("a1", "redis timeout fix", embedding=[0.1, 0.2, 0.3, 0.0, 0.0])
        a.oscillator.natural_frequency = 0.5
        a.oscillator.phase_offset = 0.0
        a.oscillator.coupling_strength = 1.0
        g.add_anchor(a)
        retriever = OscillationResonanceRetriever(g)
        result = retriever.retrieve("redis timeout",
                                    embedding=[0.1, 0.2, 0.3, 0.0, 0.0])
        assert result.method == "OscillationResonance"


class TestHybridFusionRetriever:
    def test_empty_graph(self):
        g = StarGraph()
        retriever = HybridFusionRetriever(g)
        result = retriever.retrieve("any query",
                                    embedding=[0.1, 0.2, 0.3])
        assert result.constellations == []
        assert result.method == "HybridFusion"

    def test_fusion_with_anchors(self):
        g = StarGraph()
        a = make_anchor("a1", "redis timeout fix", embedding=[0.1, 0.2, 0.3, 0.0, 0.0],
                       tags=["redis", "timeout"])
        a.oscillator.natural_frequency = 0.3
        a.oscillator.phase_offset = 0.0
        g.add_anchor(a)
        retriever = HybridFusionRetriever(g)
        result = retriever.retrieve("redis timeout",
                                    embedding=[0.1, 0.2, 0.3, 0.0, 0.0])
        assert result.method == "HybridFusion"


class TestCompareRetrievers:
    def test_single_query(self):
        g = StarGraph()
        a = make_anchor("a1", "redis timeout fix", embedding=[0.1, 0.2, 0.3, 0.0, 0.0])
        a.oscillator.natural_frequency = 0.3
        g.add_anchor(a)
        results = compare_retrievers(
            g, ["redis timeout"],
            embeddings=[[0.1, 0.2, 0.3, 0.0, 0.0]],
        )
        assert len(results) == 1
        assert "vector_similarity" in results[0]
        assert "oscillation_resonance" in results[0]

    def test_multiple_queries(self):
        g = StarGraph()
        a = make_anchor("a1", "redis timeout", embedding=[0.1, 0.2, 0.3, 0.0, 0.0])
        a.oscillator.natural_frequency = 0.3
        g.add_anchor(a)
        results = compare_retrievers(
            g, ["redis", "timeout", "database"],
            embeddings=[[0.1, 0.2, 0.3, 0.0, 0.0]] * 3,
        )
        assert len(results) == 3

    def test_empty_graph(self):
        g = StarGraph()
        results = compare_retrievers(g, ["test"], embeddings=[[0.1, 0.2]])
        assert len(results) == 1


class TestPersonalizedPagerank:
    def test_empty_graph(self):
        g = StarGraph()
        scores = personalized_pagerank(g, [0.1, 0.2])
        assert scores == {}

    def test_single_node(self):
        g = StarGraph()
        a = make_anchor("a1", embedding=[1.0, 0.0])
        g.add_anchor(a)
        scores = personalized_pagerank(g, [1.0, 0.0])
        assert "a1" in scores

    def test_with_edges(self):
        g = StarGraph()
        a1 = make_anchor("a1", "first", embedding=[1.0, 0.0])
        a2 = make_anchor("a2", "second", embedding=[0.8, 0.2])
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.5, edge_type="related")
        scores = personalized_pagerank(g, [1.0, 0.0], damping=0.85, max_iter=5)
        assert isinstance(scores, dict)

    def test_no_seeds(self):
        g = StarGraph()
        scores = personalized_pagerank(g, [0.1, 0.2], damping=0.85, max_iter=2)
        assert scores == {}

    def test_with_precomputed_cache(self):
        g = StarGraph()
        a1 = make_anchor("a1", "test", embedding=[0.5, 0.5])
        g.add_anchor(a1)
        precomputed = {"a1": {"a1": 1.0}}
        scores = personalized_pagerank(
            g, query_embedding=[0.5, 0.5], precomputed=precomputed)
        assert isinstance(scores, dict)

    def test_with_multiple_nodes(self):
        g = StarGraph()
        for i in range(5):
            a = make_anchor(f"a{i}", f"node {i}", embedding=[0.1 * i, 0.2 * i])
            a.oscillator.natural_frequency = 0.1 * i
            g.add_anchor(a)
        for i in range(4):
            g.add_edge(f"a{i}", f"a{i+1}", weight=0.5, edge_type="related")
        scores = personalized_pagerank(g, [0.5, 0.5], damping=0.85, max_iter=3)
        assert isinstance(scores, dict)


# ── OscillationResonanceRetriever extended ───────────────

class TestOscResonanceExtended:
    def test_retrieve_auto_embedding(self):
        g = StarGraph()
        a = make_anchor("a1", "test memory", embedding=[0.5] * 384)
        g.add_anchor(a)
        retriever = OscillationResonanceRetriever(g)
        result = retriever.retrieve("test memory", embedding=None, top_k=3)
        assert len(result.constellations) >= 1

    def test_retrieve_with_oscillator(self):
        g = StarGraph()
        a = make_anchor("a1", "oscillating memory", embedding=[0.5] * 384)
        a.oscillator.natural_frequency = 0.3
        g.add_anchor(a)
        retriever = OscillationResonanceRetriever(g)
        result = retriever.retrieve("oscillating", embedding=[0.5] * 384, top_k=3)
        assert isinstance(result.constellations, list)

    def test_retrieve_no_embedding_anchor(self):
        g = StarGraph()
        a = make_anchor("a1", "text only anchor")
        g.add_anchor(a)
        retriever = OscillationResonanceRetriever(g)
        result = retriever.retrieve("text", embedding=[0.5] * 384, top_k=3)
        assert isinstance(result.constellations, list)

    def test_trace_components(self):
        g = StarGraph()
        a = make_anchor("a1", "test", embedding=[0.5] * 384)
        a.oscillator.natural_frequency = 0.3
        g.add_anchor(a)
        retriever = OscillationResonanceRetriever(g)
        comps = retriever._trace_components(complex(0.1, 0.2), 0.5, a)
        assert "resonance" in comps
        assert "phase_similarity" in comps
        assert "frequency_similarity" in comps

    def test_resonance_score_edge(self):
        g = StarGraph()
        a = make_anchor("a1", "test", embedding=[0.5] * 384)
        g.add_anchor(a)
        retriever = OscillationResonanceRetriever(g)
        score = retriever._resonance_score(complex(0.0, 0.0), a)
        assert score == 0.0


# ── VectorSimilarityRetriever extended ────────────────────

class TestVectorSimExtended:
    def test_retrieve_auto_embedding(self):
        g = StarGraph()
        a = make_anchor("a1", "auto embed test", embedding=[0.5] * 384)
        g.add_anchor(a)
        retriever = VectorSimilarityRetriever(g)
        result = retriever.retrieve("auto embed", embedding=None, top_k=3)
        assert len(result.constellations) >= 1

    def test_retrieve_with_ann(self):
        g = StarGraph()
        a = make_anchor("a1", "ann test", embedding=[0.5] * 384)
        g.add_anchor(a)
        # Ensure ANN index exists
        g._get_ann_index()
        retriever = VectorSimilarityRetriever(g)
        result = retriever.retrieve("ann", embedding=[0.5] * 384, top_k=3)
        assert isinstance(result.constellations, list)

    def test_retrieve_text_overlap_fallback(self):
        g = StarGraph()
        a = make_anchor("a1", "no embedding text", embedding=None)
        g.add_anchor(a)
        retriever = VectorSimilarityRetriever(g)
        result = retriever.retrieve("no embedding", embedding=[0.5] * 384, top_k=3)
        assert isinstance(result.constellations, list)


# ── HybridFusionRetriever ─────────────────────────────────

class TestHybridFusion:
    def test_init(self):
        g = StarGraph()
        hf = HybridFusionRetriever(g)
        assert hf.graph is g

    def test_retrieve_empty(self):
        g = StarGraph()
        hf = HybridFusionRetriever(g)
        result = hf.retrieve("test", top_k=3)
        assert result.constellations == []

    def test_retrieve_with_anchor(self):
        g = StarGraph()
        a = make_anchor("a1", "hybrid test", embedding=[0.5] * 384)
        a.oscillator.natural_frequency = 0.3
        g.add_anchor(a)
        hf = HybridFusionRetriever(g)
        result = hf.retrieve("hybrid", embedding=[0.5] * 384, top_k=3)
        assert len(result.constellations) >= 1


# ── compare_retrievers extended ───────────────────────────

class TestCompareRetrieversExtended:
    def test_with_oscillator_anchors(self):
        g = StarGraph()
        a = make_anchor("a1", "osc test", embedding=[0.5] * 384)
        a.oscillator.natural_frequency = 0.3
        g.add_anchor(a)
        results = compare_retrievers(
            g, ["osc test"], embeddings=[[0.5] * 384])
        assert len(results) == 1
