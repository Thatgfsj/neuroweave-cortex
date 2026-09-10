"""Serialization tests for the Qdrant storage backend.

These tests verify the payload serialization logic (anchors, ghosts, rich
edges) and the save/load round-trip using an in-memory fake client — no real
Qdrant service is contacted.
"""

import json
from types import SimpleNamespace

import pytest

from star_graph.anchor import Anchor, AnchorPrediction, AnchorVector, MemoryState, Oscillator
from star_graph.ghost import GhostNode
from star_graph.graph import StarGraph
from star_graph.memory_core.qdrant_storage import (
    QdrantStorage,
    _anchor_to_point,
    _graph_to_payloads,
    _point_to_anchor,
)


class FakePoint:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeCountResult:
    def __init__(self, count):
        self.count = count


class _FakeCollectionsResult:
    def __init__(self, names):
        self.collections = [SimpleNamespace(name=n) for n in names]


class FakeClient:
    """In-memory stand-in for QdrantClient — never touches the network."""

    def __init__(self):
        self.points = {}  # collection_name -> {point_id: FakePoint}

    def get_collections(self):
        return _FakeCollectionsResult(list(self.points.keys()))

    def create_collection(self, collection_name, vectors_config=None):
        self.points.setdefault(collection_name, {})

    def upsert(self, collection_name, points):
        bucket = self.points.setdefault(collection_name, {})
        for p in points:
            bucket[p.id] = FakePoint(p.id, p.vector, p.payload)

    def count(self, collection_name):
        return _FakeCountResult(len(self.points.get(collection_name, {})))

    def scroll(self, collection_name, limit=100, offset=None,
               with_payload=True, with_vectors=False):
        bucket = self.points.get(collection_name, {})
        return list(bucket.values()), None

    def close(self):
        pass


def make_point(payload, vector=None, point_id="p1"):
    return SimpleNamespace(id=point_id, vector=vector or [], payload=payload)


def make_anchor(aid: str = "a1", embedding: list | None = None) -> Anchor:
    return Anchor(
        id=aid,
        text="Qdrant round trip anchor",
        vector=AnchorVector(importance=0.7, emotional_valence=-0.2,
                            hippocampal_dependency=0.2),
        embedding=embedding,
        prediction=AnchorPrediction(emotional_tone=0.4, expected_duration=12.5,
                                    confidence=0.8,
                                    next_topic_embedding=[0.1, 0.2, 0.3]),
        oscillator=Oscillator(natural_frequency=0.42, phase_offset=1.25,
                              coupling_strength=0.6, damping=0.2),
        created_at=1000.0,
        last_activated_at=2000.0,
        source_session="sess-1",
        tags=["alpha", "beta"],
        replay_count=3,
        state=MemoryState.CONSOLIDATING,
        source_attribution="self_reported",
        source_trust=0.9,
    )


def make_storage() -> QdrantStorage:
    """Build a QdrantStorage backed by a fake in-memory client."""
    store = QdrantStorage.__new__(QdrantStorage)
    store._collection = "test_graph"
    store._vector_size = 8
    store._client = FakeClient()
    return store


class TestAnchorPointSerialization:
    def test_prediction_and_oscillator_roundtrip(self):
        anchor = make_anchor()
        payload = _anchor_to_point(anchor)

        # Payload must contain valid JSON dicts — no to_dict() on dataclasses
        pred_data = json.loads(payload["prediction_json"])
        assert pred_data["emotional_tone"] == pytest.approx(0.4)
        assert pred_data["expected_duration"] == pytest.approx(12.5)
        assert pred_data["confidence"] == pytest.approx(0.8)
        assert pred_data["next_topic_embedding"] == [0.1, 0.2, 0.3]

        osc_data = json.loads(payload["oscillator_json"])
        assert osc_data["natural_frequency"] == pytest.approx(0.42)
        assert osc_data["phase_offset"] == pytest.approx(1.25)
        assert osc_data["coupling_strength"] == pytest.approx(0.6)
        assert osc_data["damping"] == pytest.approx(0.2)

    def test_point_to_anchor_roundtrip(self):
        anchor = make_anchor(embedding=[0.5, 0.25, 0.125])
        payload = _anchor_to_point(anchor)
        restored = _point_to_anchor(make_point(payload, vector=[0.5, 0.25, 0.125],
                                               point_id="a1"))

        assert restored.id == "a1"
        assert restored.text == anchor.text
        # tags must round-trip as a list (Anchor.tags is list[str], not set)
        assert restored.tags == ["alpha", "beta"]
        assert isinstance(restored.tags, list)
        assert restored.prediction == anchor.prediction
        assert restored.oscillator == anchor.oscillator
        assert restored.vector.to_list() == pytest.approx(anchor.vector.to_list())
        assert restored.state == MemoryState.CONSOLIDATING
        assert restored.source_attribution == "self_reported"
        assert restored.source_trust == pytest.approx(0.9)
        assert restored.replay_count == 3

    def test_legacy_ghost_state_maps_to_dormant(self):
        payload = _anchor_to_point(make_anchor())
        payload["state"] = "ghost"
        restored = _point_to_anchor(make_point(payload, point_id="a1"))
        assert restored.state == MemoryState.DORMANT


class TestGraphToPayloadsGhosts:
    def test_ghosts_read_from_ghost_subsystem(self):
        g = StarGraph()
        ghost = GhostNode(
            id="gh1",
            compressed_embedding=[0.1, 0.2, 0.3],
            residual_edges={"a1": 0.4},
            emotion_trace=-0.5,
            pruned_at=1234.0,
            original_tags=["t1"],
            original_importance=0.8,
            semantic_shadow="...shadow...",
            revival_count=2,
        )
        g._ghost_subsystem.ghosts[ghost.id] = ghost

        _, _, _, ghost_points = _graph_to_payloads(g)
        assert len(ghost_points) == 1
        p = ghost_points[0]
        assert p["id"] == "g:gh1"
        assert p["payload"]["original_tags"] == ["t1"]
        assert p["payload"]["revival_count"] == 2
        assert p["payload"]["original_importance"] == pytest.approx(0.8)
        assert json.loads(p["payload"]["compressed_embedding_json"]) == [0.1, 0.2, 0.3]
        assert json.loads(p["payload"]["residual_edges_json"]) == {"a1": 0.4}
        assert p["payload"]["emotion_trace"] == pytest.approx(-0.5)
        assert p["payload"]["semantic_shadow"] == "...shadow..."

    def test_no_ghosts_yields_no_points(self):
        g = StarGraph()
        _, _, _, ghost_points = _graph_to_payloads(g)
        assert ghost_points == []


class TestQdrantSaveLoadRoundTrip:
    def test_full_roundtrip(self):
        g = StarGraph()
        cortical = make_anchor("cortical", embedding=[0.1, 0.2, 0.3])
        hippocampal = make_anchor("hippo")
        hippocampal.embedding = [0.4, 0.5, 0.6]
        hippocampal.vector.hippocampal_dependency = 0.9  # not cortical
        g.add_anchor(cortical)
        g.add_anchor(hippocampal)

        from star_graph.graph import RichEdge
        rich = RichEdge(source="cortical", target="hippo", weight=0.6,
                        edge_type="causal", relation="causes",
                        confidence=0.9, source_type="explicit",
                        temporal_order="before", causal_strength=0.7)
        g.edges[("cortical", "hippo")] = rich

        ghost = GhostNode(
            id="gh1",
            compressed_embedding=[0.1, 0.2],
            residual_edges={"cortical": 0.3},
            emotion_trace=0.25,
            pruned_at=1111.0,
            original_tags=["memory"],
            original_importance=0.6,
            semantic_shadow="...gist...",
            revival_count=1,
        )
        g._ghost_subsystem.ghosts[ghost.id] = ghost

        store = make_storage()
        store.save(g)
        loaded = store.load()

        # Anchors
        assert set(loaded.anchors) == {"cortical", "hippo"}
        a = loaded.anchors["cortical"]
        assert a.tags == ["alpha", "beta"]
        assert a.prediction.emotional_tone == pytest.approx(0.4)
        assert a.prediction.next_topic_embedding == [0.1, 0.2, 0.3]
        assert a.oscillator.natural_frequency == pytest.approx(0.42)
        assert a.state == MemoryState.CONSOLIDATING

        # Rich edge restored (not downgraded to a plain Edge)
        edge = loaded.edges[loaded._key("cortical", "hippo")]
        assert isinstance(edge, RichEdge)
        assert edge.confidence == pytest.approx(0.9)
        assert edge.temporal_order == "before"
        assert edge.causal_strength == pytest.approx(0.7)
        assert edge.relation == "causes"

        # Adjacency rebuilt
        assert loaded.node_degree("cortical") == 1
        assert loaded._adjacency["hippo"] == {"cortical"}

        # Cortical index rebuilt (only cortical anchors with embeddings)
        index_ids = [aid for _, aid in loaded.cortical_index]
        assert index_ids == ["cortical"]

        # Ghosts restored into the ghost subsystem
        assert "gh1" in loaded._ghost_subsystem.ghosts
        lg = loaded._ghost_subsystem.ghosts["gh1"]
        assert lg.semantic_shadow == "...gist..."
        assert lg.compressed_embedding == pytest.approx([0.1, 0.2])
        assert lg.residual_edges == {"cortical": 0.3}
        assert lg.emotion_trace == pytest.approx(0.25)
        assert lg.original_tags == ["memory"]
        assert lg.original_importance == pytest.approx(0.6)
        assert lg.revival_count == 1

    def test_empty_graph_roundtrip(self):
        store = make_storage()
        store.save(StarGraph())
        loaded = store.load()
        assert len(loaded.anchors) == 0
        assert len(loaded.edges) == 0
