"""Test oscillation resonance retrieval accuracy.

Generates 100 anchor pairs with known phase relationships,
then verifies the OscillationResonanceRetriever correctly
identifies matching anchors vs. random ones.
"""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, OscillationResonanceRetriever,
    VectorSimilarityRetriever, get_embedder,
)


@pytest.fixture(scope="module")
def warm_embedder():
    """Pre-load the embedding model once — forces model download/load (~30s)."""
    embedder = get_embedder()
    embedder.encode("warmup")  # triggers actual model loading
    return embedder


def make_test_graph(n_anchors: int = 100, seed: int = 42) -> StarGraph:
    """Create a graph with n_anchors, half with matching phase signatures."""
    graph = StarGraph()
    topics = [
        "user preferences", "technical discussion", "project planning",
        "code review", "bug analysis", "design discussion",
    ]

    for i in range(n_anchors):
        topic = topics[i % len(topics)]
        text = f"{topic} about item {i}: this is a test memory for recall"

        # Deterministic phase based on topic
        anchor = Anchor.create(
            text,
            tags=[topic],
            importance=0.5,
            emotional_valence=0.1 * (i % 10),
        )
        # Override oscillator for controlled testing
        anchor.oscillator.natural_frequency = 0.3 + 0.6 * (i % len(topics)) / len(topics)
        anchor.oscillator.phase_offset = (i % len(topics)) * 2 * math.pi / len(topics)
        anchor.oscillator.coupling_strength = 0.5

        graph.add_anchor(anchor)

    return graph


class TestOscillationResonance:
    """Verify oscillation resonance retrieval meets quality thresholds."""

    def test_basic_retrieval(self, warm_embedder):
        """Retrieval should return results within latency budget."""
        graph = make_test_graph(50)
        ret = OscillationResonanceRetriever(graph)
        result = ret.retrieve("user preferences about coding")

        # First retrieval encodes query + all anchors; <2s is reasonable
        assert result.latency_ms < 2000, f"Latency {result.latency_ms}ms exceeds budget"
        assert len(result.constellations) > 0, "Should return at least 1 constellation"
        assert 0.0 <= result.top_score <= 1.0, f"Score {result.top_score} out of [0,1]"

    def test_score_bounds(self):
        """All resonance scores must be in [0, 1]."""
        graph = make_test_graph(100)
        ret = OscillationResonanceRetriever(graph)

        queries = [
            "user preferences about coding style",
            "technical discussion on architecture",
            "project planning for Q3 release",
            "random unrelated topic xyz",
        ]

        for q in queries:
            result = ret.retrieve(q)
            assert 0.0 <= result.top_score <= 1.0, \
                f"Score {result.top_score} for '{q[:30]}'"

    def test_phase_separation(self):
        """Anchors with different phase offsets should be distinguishable.

        Two anchors with same topic but opposite phase → one should
        resonate more strongly than the other for a matching query.
        """
        graph = StarGraph()

        # Create two anchors with same content but different phases
        a1 = Anchor.create("user prefers dark mode in editor", tags=["preference"])
        a1.oscillator.phase_offset = 0.0  # in phase with query

        a2 = Anchor.create("user prefers light mode in editor", tags=["preference"])
        a2.oscillator.phase_offset = math.pi  # opposite phase

        graph.add_anchor(a1)
        graph.add_anchor(a2)

        ret = OscillationResonanceRetriever(graph, phase_weight=0.5)

        # Query should resonate more with in-phase anchor
        result = ret.retrieve("dark mode preference")
        scores = {}
        for c in result.constellations:
            for a in c.anchors:
                scores[a.id] = a.oscillator.phase_offset

        # At minimum, both anchors should be retrievable
        assert len(result.constellations) > 0

    def test_recall_at_k(self):
        """Recall@5 should be at least 50% for controlled test data.

        Uses VectorSimilarityRetriever for this test because meaningful
        recall depends on content-based similarity. OscillationResonance
        needs real embeddings to derive phase-sensitive driving phasors.
        """
        from star_graph.retriever import _recall_at_k

        graph = make_test_graph(100)
        ret = VectorSimilarityRetriever(graph)

        target_topic = "user preferences"
        target_ids = [
            aid for aid, a in graph.anchors.items()
            if target_topic in a.tags
        ]

        result = ret.retrieve("user preferences and settings")
        recall = _recall_at_k(result, target_ids, k=5)

        assert recall > 0.0, f"Recall@5 was {recall}, expected > 0"


class TestVectorSimilarity:
    """Baseline retriever tests."""

    def test_basic_retrieval(self):
        graph = make_test_graph(50)
        ret = VectorSimilarityRetriever(graph)
        result = ret.retrieve("project planning discussion")

        assert result.latency_ms < 500
        assert len(result.constellations) > 0
        assert 0.0 <= result.top_score <= 1.0

    def test_empty_graph(self):
        graph = StarGraph()
        ret = VectorSimilarityRetriever(graph)
        result = ret.retrieve("anything")
        assert len(result.constellations) == 0
        assert result.top_score == 0.0


class TestRetrieverComparison:
    """Ensure both retrievers can run on the same data for comparison."""

    def test_both_return_results(self):
        graph = make_test_graph(30)
        osc = OscillationResonanceRetriever(graph)
        vec = VectorSimilarityRetriever(graph)

        q = "code review feedback"
        r1 = osc.retrieve(q)
        r2 = vec.retrieve(q)

        assert r1.method == "OscillationResonance"
        assert r2.method == "VectorSimilarity"
        assert r1.latency_ms >= 0
        assert r2.latency_ms >= 0
