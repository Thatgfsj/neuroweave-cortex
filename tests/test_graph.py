"""Tests for graph module — Edge, RichEdge, StarGraph, Constellation, Schema."""

import time

import pytest

from star_graph.graph import (
    Edge,
    RichEdge,
    Constellation,
    Schema,
    StarGraph,
    ReflectionNode,
    EDGE_TRAVERSAL_WEIGHTS,
    EXPLICABLE_RELATIONS,
    STRONG_RELATIONS,
    LEGACY_EDGE_TYPES,
    _cosine_sim_simple,
)
from star_graph.anchor import Anchor, AnchorVector, MemoryState


def make_anchor(name: str, text: str = "", embedding: list | None = None) -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}")
    if embedding:
        a.embedding = embedding
    return a


# ── Edge ──────────────────────────────────────────────────

class TestEdge:
    def test_default_values(self):
        e = Edge(source="a", target="b")
        assert e.source == "a"
        assert e.target == "b"
        assert e.weight == 0.5
        assert e.edge_type == "topical"
        assert e.co_activation_count == 0

    def test_strengthen(self):
        e = Edge(source="a", target="b", weight=0.5)
        e.strengthen(0.1)
        assert e.weight == pytest.approx(0.6)
        assert e.co_activation_count == 1

    def test_strengthen_clamped_at_1(self):
        e = Edge(source="a", target="b", weight=0.95)
        e.strengthen(0.1)
        assert e.weight == 1.0

    def test_weaken(self):
        e = Edge(source="a", target="b", weight=0.5)
        e.weaken(0.1)
        assert e.weight == pytest.approx(0.4)

    def test_weaken_clamped_at_0(self):
        e = Edge(source="a", target="b", weight=0.01)
        e.weaken(0.1)
        assert e.weight == 0.0

    def test_success_rate_untested(self):
        e = Edge(source="a", target="b")
        assert e.success_rate == 0.5

    def test_success_rate_perfect(self):
        e = Edge(source="a", target="b", success_count=5)
        assert e.success_rate == 1.0

    def test_record_success(self):
        e = Edge(source="a", target="b")
        e.record_success()
        assert e.success_count == 1
        assert e.weight > 0.5

    def test_record_failure(self):
        e = Edge(source="a", target="b")
        e.record_failure()
        assert e.failure_count == 1
        assert e.weight < 0.5

    def test_traversal_weight(self):
        e = Edge(source="a", target="b", weight=0.5, edge_type="causes")
        assert e.traversal_weight == pytest.approx(0.5 * 1.5)

    def test_is_active(self):
        e = Edge(source="a", target="b", weight=0.5)
        assert e.is_active

    def test_not_active(self):
        e = Edge(source="a", target="b", weight=0.05)
        assert not e.is_active


# ── RichEdge ──────────────────────────────────────────────

class TestRichEdge:
    def test_default_values(self):
        e = RichEdge(source="a", target="b")
        assert e.source == "a"
        assert e.target == "b"
        assert e.confidence == 0.5
        assert e.source_type == "implicit"

    def test_explicit_factory(self):
        e = RichEdge.explicit("a", "b", relation="uses", weight=0.8, session="s1")
        assert e.confidence == 0.95
        assert e.source_type == "explicit"
        assert e.relation == "uses"

    def test_implicit_factory(self):
        e = RichEdge.implicit("a", "b", relation="co-occurs")
        assert e.confidence == 0.42
        assert e.source_type == "implicit"

    def test_inferred_factory(self):
        e = RichEdge.inferred("a", "b", weight=0.4)
        assert e.confidence == 0.42
        assert e.source_type == "inferred"

    def test_temporal_factory(self):
        e = RichEdge.temporal("a", "b", order="before", weight=0.6, session="s1")
        assert e.edge_type == "temporal"
        assert e.temporal_order == "before"
        assert e.confidence == 0.85

    def test_causal_factory(self):
        e = RichEdge.causal("a", "b", strength=0.8)
        assert e.edge_type == "causal"
        assert e.causal_strength == 0.8

    def test_supersedes_factory(self):
        e = RichEdge.supersedes("old_key", "new_a", "new_b")
        assert e.edge_type == "superseded_by"
        assert e.confidence == 0.7

    def test_contradicts_factory(self):
        e = RichEdge.contradicts("a", "b", confidence=0.8)
        assert e.edge_type == "contradicts"

    def test_derived_from_factory(self):
        e = RichEdge.derived_from("a", "b")
        assert e.edge_type == "derived_from"

    def test_from_edge(self):
        simple = Edge(source="a", target="b", weight=0.7, edge_type="topical")
        e = RichEdge.from_edge(simple)
        assert e.source == "a"
        assert e.weight == 0.7

    def test_reinforce(self):
        e = RichEdge(source="a", target="b")
        old_conf = e.confidence
        e.reinforce("s1")
        assert e.confidence > old_conf
        assert e.stability > 0
        assert e.reinforcement_count == 1

    def test_mark_stale(self):
        e = RichEdge(source="a", target="b")
        e.mark_stale("replacement_key")
        assert e.is_stale
        assert e.replaced_by == "replacement_key"

    def test_apply_decay(self):
        e = RichEdge(source="a", target="b", weight=0.8, decay_rate=0.1)
        e.apply_decay(10.0)
        assert e.weight < 0.8

    def test_stale_decays_faster(self):
        e_stale = RichEdge(source="a", target="b", weight=0.8, decay_rate=0.1)
        e_stale.is_stale = True
        e_fresh = RichEdge(source="c", target="d", weight=0.8, decay_rate=0.1)
        e_stale.apply_decay(10.0)
        e_fresh.apply_decay(10.0)
        assert e_stale.weight < e_fresh.weight

    def test_is_expired(self):
        e = RichEdge(source="a", target="b", valid_until=1.0)  # in the past
        assert e.is_expired

    def test_not_expired(self):
        e = RichEdge(source="a", target="b", valid_until=time.time() + 86400)
        assert not e.is_expired

    def test_retrieval_score(self):
        e = RichEdge(source="a", target="b", weight=0.7, confidence=0.8)
        score = e.retrieval_score
        assert score > 0.0

    def test_expired_edge_retrieval_score_zero(self):
        e = RichEdge(source="a", target="b", valid_until=1.0)
        assert e.retrieval_score == 0.0

    def test_record_success(self):
        e = RichEdge(source="a", target="b")
        e.record_success()
        assert e.success_count == 1

    def test_record_failure(self):
        e = RichEdge(source="a", target="b")
        e.record_failure()
        assert e.failure_count == 1

    def test_version_history_trimmed(self):
        e = RichEdge(source="a", target="b")
        for i in range(25):
            e.strengthen(0.01)
        assert len(e.version_history) <= 20


# ── ReflectionNode ────────────────────────────────────────

class TestReflectionNode:
    def test_from_failure(self):
        r = ReflectionNode.from_failure("Don't deploy on Friday", ["a1", "a2"])
        assert r.reflection_type == "failure_analysis"
        assert r.strength == 0.6
        assert len(r.id) == 16

    def test_from_success(self):
        r = ReflectionNode.from_success("CI/CD catches bugs early", ["a1"])
        assert r.reflection_type == "success_pattern"
        assert r.confidence == 0.8

    def test_from_lesson(self):
        r = ReflectionNode.from_lesson("Always write tests first", ["a1"])
        assert r.reflection_type == "lesson_learned"
        assert r.strength == 0.5

    def test_is_relevant_new(self):
        r = ReflectionNode.from_lesson("test", ["a1"])
        assert r.is_relevant

    def test_reinforce(self):
        r = ReflectionNode.from_lesson("test", ["a1"])
        old_strength = r.strength
        r.reinforce()
        assert r.strength > old_strength

    def test_weaken(self):
        r = ReflectionNode.from_lesson("test", ["a1"])
        old_strength = r.strength
        r.weaken()
        assert r.strength < old_strength


# ── Constellation ─────────────────────────────────────────

class TestConstellation:
    def test_empty_constellation(self):
        c = Constellation(anchors=[], edges=[])
        assert c.centroid_vector.importance == 0.5  # default
        assert c.total_weight == 0.0
        freq, phase = c.dominant_oscillation
        assert freq == 0.5
        assert phase == 0.0

    def test_single_anchor_centroid(self):
        a = make_anchor("a1", "test")
        a.vector = AnchorVector(importance=0.8, frequency=0.4)
        c = Constellation(anchors=[a], edges=[])
        assert c.centroid_vector.importance == 0.8

    def test_dominant_oscillation(self):
        a = make_anchor("a1", "test")
        a.oscillator.natural_frequency = 0.7
        a.oscillator.phase_offset = 1.5
        c = Constellation(anchors=[a], edges=[])
        freq, phase = c.dominant_oscillation
        assert freq == 0.7
        assert phase == 1.5

    def test_total_weight(self):
        e1 = Edge(source="a", target="b", weight=0.5)
        e2 = Edge(source="b", target="c", weight=0.3)
        c = Constellation(anchors=[], edges=[e1, e2])
        assert c.total_weight == 0.8

    def test_label(self):
        c = Constellation(anchors=[], edges=[], label="test_constellation")
        assert c.label == "test_constellation"


# ── Schema ────────────────────────────────────────────────

class TestSchema:
    def test_default_values(self):
        s = Schema(id="s1", template="user likes X", slots={"X": "thing"}, instance_ids=[])
        assert s.id == "s1"
        assert s.confidence == 0.0
        assert s.instance_ids == []

    def test_match_positive(self):
        s = Schema(id="s1", template="user likes python programming",
                    slots={"language": "python"}, instance_ids=[])
        score, slots = s.match("python programming user likes")
        assert score > 0.0

    def test_match_negative(self):
        s = Schema(id="s1", template="user likes python programming",
                    slots={"language": "python"}, instance_ids=[])
        score, slots = s.match("completely unrelated text here")
        assert score < 0.5


# ── StarGraph ─────────────────────────────────────────────

class TestStarGraphCRUD:
    def test_empty_graph(self):
        g = StarGraph()
        assert len(g.anchors) == 0
        assert len(g.edges) == 0

    def test_add_anchor(self):
        g = StarGraph()
        a = make_anchor("a1", "test", embedding=[0.1, 0.2, 0.3])
        result = g.add_anchor(a)
        assert result == "a1"
        assert "a1" in g.anchors
        assert len(g.cortical_index) == 1

    def test_add_anchor_without_embedding(self):
        g = StarGraph()
        a = make_anchor("a1", "test")
        g.add_anchor(a)
        assert "a1" in g.anchors
        assert len(g.cortical_index) == 0

    def test_add_edge_between_existing_anchors(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "test1"))
        g.add_anchor(make_anchor("a2", "test2"))
        edge = g.add_edge("a1", "a2", weight=0.7, edge_type="causes")
        assert edge is not None
        assert edge.weight == pytest.approx(0.7 * 1.1)  # strong relation boost
        assert g._key("a1", "a2") in g.edges

    def test_add_edge_nonexistent_anchor(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "test"))
        edge = g.add_edge("a1", "nonexistent")
        assert edge is None

    def test_add_edge_reinforces_existing(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        e1 = g.add_edge("a1", "a2", weight=0.5, edge_type="causes",
                       source_type="explicit", confidence=0.8)
        old_weight = e1.weight
        g.add_edge("a1", "a2", weight=0.5, edge_type="causes",
                  source_type="explicit", confidence=0.8)
        # Second add should strengthen the existing edge
        assert e1.weight > old_weight

    def test_add_edge_with_confidence_creates_rich(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        edge = g.add_edge("a1", "a2", weight=0.5, confidence=0.9, source_type="explicit")
        assert isinstance(edge, RichEdge)

    def test_remove_anchor(self):
        g = StarGraph()
        a = make_anchor("a1", "test", embedding=[0.1, 0.2])
        g.add_anchor(a)
        g.remove_anchor("a1")
        assert "a1" not in g.anchors

    def test_remove_anchor_cleans_edges(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        g.add_edge("a1", "a2", weight=0.5)
        g.remove_anchor("a1")
        assert len(g.edges) == 0

    def test_node_degree(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        g.add_anchor(make_anchor("a3", "t3"))
        g.add_edge("a1", "a2")
        g.add_edge("a1", "a3")
        assert g.node_degree("a1") == 2


class TestStarGraphNeighbors:
    def test_neighbors(self):
        g = StarGraph()
        for n in ["a1", "a2", "a3"]:
            g.add_anchor(make_anchor(n, f"text {n}"))
        g.add_edge("a1", "a2", weight=0.8)
        g.add_edge("a1", "a3", weight=0.3)
        neighbors = g.neighbors("a1")
        assert len(neighbors) == 2
        assert neighbors[0][0] == "a2"  # higher weight first

    def test_neighbors_min_weight(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        g.add_edge("a1", "a2", weight=0.3)
        neighbors = g.neighbors("a1", min_weight=0.5)
        assert len(neighbors) == 0

    def test_neighbors_nonexistent(self):
        g = StarGraph()
        assert g.neighbors("nonexistent") == []


class TestStarGraphSpreading:
    def test_spread_activation(self):
        g = StarGraph()
        for n in ["a1", "a2", "a3"]:
            g.add_anchor(make_anchor(n, f"text {n}"))
        g.add_edge("a1", "a2", weight=0.8)
        g.add_edge("a2", "a3", weight=0.6)
        activation = g.spread_activation(["a1"], steps=2)
        assert "a1" in activation
        assert activation["a1"] == 1.0

    def test_spread_empty_graph(self):
        g = StarGraph()
        activation = g.spread_activation(["nonexistent"])
        assert activation == {}


class TestStarGraphConstellation:
    def test_find_constellation(self):
        g = StarGraph()
        for n in ["a1", "a2", "a3"]:
            g.add_anchor(make_anchor(n, f"text {n}"))
        g.add_edge("a1", "a2", weight=0.8)
        g.add_edge("a2", "a3", weight=0.6)
        c = g.find_constellation("a1")
        assert len(c.anchors) >= 1

    def test_find_constellation_nonexistent(self):
        g = StarGraph()
        c = g.find_constellation("nonexistent")
        assert len(c.anchors) == 0


class TestStarGraphResonance:
    def test_oscillatory_resonance(self):
        g = StarGraph()
        a = make_anchor("a1", "test")
        a.oscillator.natural_frequency = 0.5
        a.oscillator.phase_offset = 0.0
        a.oscillator.coupling_strength = 1.0
        g.add_anchor(a)
        resonance = g.oscillatory_resonance(0.5, 0.0)
        assert "a1" in resonance

    def test_oscillatory_resonance_below_threshold(self):
        g = StarGraph()
        a = make_anchor("a1", "test")
        a.oscillator.natural_frequency = 0.2
        a.oscillator.coupling_strength = 0.1
        g.add_anchor(a)
        resonance = g.oscillatory_resonance(0.9, 0.0, min_strength=0.5)
        assert len(resonance) == 0


class TestStarGraphCorticalLoopup:
    def test_cortical_lookup_empty(self):
        g = StarGraph()
        assert g.cortical_lookup([0.1, 0.2]) == []

    def test_cortical_lookup_returns_results(self):
        g = StarGraph()
        a = make_anchor("a1", "test", embedding=[1.0, 0.0, 0.0, 0.0])
        g.add_anchor(a)
        results = g.cortical_lookup([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert len(results) == 1
        assert results[0][0] == "a1"


class TestStarGraphAnalysis:
    def test_get_prune_candidates(self):
        g = StarGraph()
        a = make_anchor("a1", "test")
        a.vector.recency = 0.01
        a.vector.frequency = 0.0
        a.vector.importance = 0.1
        a.vector.success_feedback = 0.5
        a.vector.confidence = 0.5
        g.add_anchor(a)
        candidates = g.get_prune_candidates(threshold=0.5)
        assert "a1" in candidates

    def test_get_dormant_edges(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        g.add_edge("a1", "a2", weight=0.05)
        dormant = g.get_dormant_edges(threshold=0.1)
        assert len(dormant) == 1

    def test_stats(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1", embedding=[0.1, 0.2, 0.3]))
        g.add_anchor(make_anchor("a2", "t2"))
        g.add_edge("a1", "a2", weight=0.5)
        s = g.stats()
        assert s["anchors"] == 2
        assert s["edges"] == 1

    def test_count_constellations(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        g.add_edge("a1", "a2", weight=0.5)
        # connected → one constellation
        assert g.count_constellations() == 1

    def test_count_constellations_disconnected(self):
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "t1"))
        g.add_anchor(make_anchor("a2", "t2"))
        # no edge → two constellations
        assert g.count_constellations() == 2


class TestStarGraphReflections:
    def test_add_reflection(self):
        g = StarGraph()
        r = ReflectionNode.from_lesson("test lesson", ["a1"])
        result = g.add_reflection(r)
        assert result == r.id
        assert r.id in g.reflections

    def test_find_reflections(self):
        g = StarGraph()
        r = ReflectionNode.from_lesson("test lesson", ["a1", "a2"])
        g.add_reflection(r)
        results = g.find_reflections(["a1"])
        assert len(results) == 1

    def test_find_reflections_type_filter(self):
        g = StarGraph()
        r1 = ReflectionNode.from_lesson("lesson", ["a1"])
        r2 = ReflectionNode.from_failure("failure", ["a1"])
        g.add_reflection(r1)
        g.add_reflection(r2)
        results = g.find_reflections(["a1"], types=["lesson_learned"])
        assert len(results) == 1
        assert results[0].reflection_type == "lesson_learned"


class TestStarGraphChainRecording:
    def test_record_chain_success(self):
        g = StarGraph()
        for n in ["a1", "a2", "a3"]:
            g.add_anchor(make_anchor(n, f"text {n}"))
        g.add_edge("a1", "a2", weight=0.5)
        g.add_edge("a2", "a3", weight=0.5)
        updated = g.record_chain_success(["a1", "a2", "a3"])
        assert updated >= 1

    def test_record_chain_failure(self):
        g = StarGraph()
        for n in ["a1", "a2"]:
            g.add_anchor(make_anchor(n, f"text {n}"))
        g.add_edge("a1", "a2", weight=0.5)
        updated = g.record_chain_failure(["a1", "a2"])
        assert updated >= 1


class TestStarGraphEviction:
    def test_evict_expired_edges(self):
        g = StarGraph()
        for n in ["a1", "a2"]:
            g.add_anchor(make_anchor(n, f"text {n}"))
        e = g.add_edge("a1", "a2", weight=0.5, valid_until=1.0)
        assert e is not None
        removed = g.evict_expired_edges()
        assert removed == 1

    def test_evict_anchors_lowest_retention(self):
        g = StarGraph()
        a1 = make_anchor("a1", "text1")
        a1.created_at = 0.0  # old enough to evict
        g.add_anchor(a1)
        a2 = make_anchor("a2", "text2")
        a2.created_at = 0.0
        g.add_anchor(a2)
        evicted = g._evict_anchors(1, policy="lowest_retention")
        assert len(evicted) == 1

    def test_evict_anchors_fifo(self):
        g = StarGraph()
        a1 = make_anchor("a1", "text1")
        a1.created_at = 100.0
        g.add_anchor(a1)
        a2 = make_anchor("a2", "text2")
        a2.created_at = 200.0
        g.add_anchor(a2)
        evicted = g._evict_anchors(1, policy="fifo")
        assert len(evicted) == 1


class TestStarGraphCommunity:
    def test_get_community_anchors(self):
        g = StarGraph()
        a1 = make_anchor("a1", "text1")
        a1.community_id = "com1"
        a2 = make_anchor("a2", "text2")
        a2.community_id = "com2"
        g.add_anchor(a1)
        g.add_anchor(a2)
        anchors = g.get_community_anchors("com1")
        assert len(anchors) == 1
        assert anchors[0].id == "a1"

    def test_anchors_by_community(self):
        g = StarGraph()
        a1 = make_anchor("a1", "text1")
        a1.community_id = "com1"
        g.add_anchor(a1)
        result = g.anchors_by_community()
        assert "com1" in result

    def test_get_bridge_neighbors(self):
        g = StarGraph()
        a1 = make_anchor("a1", "text1")
        a1.community_id = "com1"
        a2 = make_anchor("a2", "text2")
        a2.community_id = "com2"
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.5)
        bridges = g.get_bridge_neighbors("a1")
        assert len(bridges) == 1
        assert bridges[0][0] == "a2"


class TestStarGraphTemporalSlice:
    def test_temporal_slice_returns_dict(self):
        g = StarGraph()
        for i in range(5):
            a = make_anchor(f"a{i}", f"text {i}")
            a.embedding = [float(i), 0.0, 0.0, 0.0]
            g.add_anchor(a)
        ts = g.temporal_slice(max_core=2, max_active=5)
        assert "core_ids" in ts
        assert "active_ids" in ts
        assert "background_count" in ts
        assert ts["total_anchors"] == 5


class TestConstants:
    def test_edgetraversal_weights_has_known_types(self):
        assert "causes" in EDGE_TRAVERSAL_WEIGHTS
        assert "fixes" in EDGE_TRAVERSAL_WEIGHTS
        assert "contradicts" in EDGE_TRAVERSAL_WEIGHTS

    def test_explicable_relations(self):
        assert "causes" in EXPLICABLE_RELATIONS
        assert "fixes" in EXPLICABLE_RELATIONS

    def test_strong_relations(self):
        assert "causes" in STRONG_RELATIONS
        assert "fixes" in STRONG_RELATIONS

    def test_legacy_edge_types(self):
        assert "topical" in LEGACY_EDGE_TYPES
        assert "semantic" in LEGACY_EDGE_TYPES


def test_cosine_sim_simple():
    assert _cosine_sim_simple([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_sim_simple([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
