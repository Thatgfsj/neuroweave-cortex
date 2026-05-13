import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import Anchor, StarGraph
from star_graph.retriever import (
    OscillationResonanceRetriever,
    VectorSimilarityRetriever,
    compare_retrievers,
)


def make_anchor(anchor_id: str, text: str, embedding: list[float],
                tags: list[str] | None = None) -> Anchor:
    return Anchor(
        id=anchor_id,
        text=text,
        embedding=embedding,
        tags=tags or [],
    )


def test_vector_retriever_emits_locomo_style_trace():
    graph = StarGraph()
    target = make_anchor(
        "m_182",
        "Alice visited Hawaii on July 14, 2024.",
        [1.0, 0.0, 0.0, 0.0],
        tags=["Alice", "Hawaii", "travel"],
    )
    distractor = make_anchor(
        "m_999",
        "Bob discussed database pooling patterns.",
        [0.0, 1.0, 0.0, 0.0],
        tags=["Bob", "database"],
    )
    graph.add_anchor(target)
    graph.add_anchor(distractor)

    result = VectorSimilarityRetriever(graph).retrieve(
        "When did Alice visit Hawaii?",
        embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=1,
    )

    trace = result.retrieval_trace
    assert trace["query"] == "When did Alice visit Hawaii?"
    assert trace["method"] == "VectorSimilarity"
    assert trace["retrieved_memories"][0]["memory_id"] == "m_182"

    reason = trace["retrieved_memories"][0]["reason"]
    assert "temporal_match" in reason
    assert "entity_match" in reason
    assert "semantic_match" in reason
    assert trace["retrieved_memories"][0]["matched_terms"] == ["alice", "hawaii"]


def test_oscillation_retriever_trace_explains_phase_and_activation():
    graph = StarGraph()
    anchor = make_anchor(
        "m_phase",
        "Alice visited Hawaii on July 14, 2024.",
        [1.0, 0.0, 0.0, 0.0],
        tags=["Alice", "Hawaii"],
    )

    # Derive query driving phasor, then align anchor phase to it for resonance
    from star_graph.embedding import get_embedder
    embedder = get_embedder()
    query_emb = [1.0, 0.0, 0.0, 0.0]
    driving_freq, driving_phase = embedder.derive_driving_phasor(
        "When did Alice visit Hawaii?", query_emb)

    anchor.oscillator.natural_frequency = driving_freq
    anchor.oscillator.phase_offset = driving_phase
    anchor.oscillator.coupling_strength = 1.0
    graph.add_anchor(anchor)

    result = OscillationResonanceRetriever(graph, phase_weight=0.5).retrieve(
        "When did Alice visit Hawaii?",
        embedding=query_emb,
        top_k=1,
    )

    memory = result.retrieval_trace["retrieved_memories"][0]
    assert memory["memory_id"] == "m_phase"
    assert "phase_match" in memory["reason"]
    assert "frequency_match" in memory["reason"]
    assert "activation_spread" in memory["reason"]
    assert math.isclose(memory["components"]["phase_similarity"], 1.0)


def test_empty_graph_trace_is_present():
    graph = StarGraph()

    result = VectorSimilarityRetriever(graph).retrieve(
        "anything",
        embedding=[1.0, 0.0],
    )

    assert result.retrieval_trace == {
        "query": "anything",
        "method": "VectorSimilarity",
        "retrieved_memories": [],
    }


def test_compare_retrievers_can_include_trace():
    graph = StarGraph()
    anchor = make_anchor(
        "m_1",
        "Alice visited Hawaii on July 14, 2024.",
        [1.0, 0.0, 0.0, 0.0],
        tags=["Alice", "Hawaii"],
    )
    anchor.oscillator.natural_frequency = 0.3
    anchor.oscillator.phase_offset = 0.0
    anchor.oscillator.coupling_strength = 1.0
    graph.add_anchor(anchor)

    comparison = compare_retrievers(
        graph,
        ["When did Alice visit Hawaii?"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        include_trace=True,
    )[0]

    assert "retrieval_trace" in comparison["vector_similarity"]
    assert "retrieval_trace" in comparison["oscillation_resonance"]
