"""Tests for abstraction module — AbstractNode, AbstractionEngine,
PatternMemory, AbstractiveMemoryEngine."""

import math

import pytest

from star_graph.abstraction import (
    AbstractNode,
    AbstractionEngine,
    PatternMemory,
    AbstractiveMemoryEngine,
    _cosine_sim_abstractive,
)
from star_graph.anchor import Anchor, AnchorVector
from star_graph.graph import StarGraph


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                tags: list | None = None, source_session: str = "s1",
                stability: float = 0.5) -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}", tags=tags or [],
              source_session=source_session)
    a.vector.stability = stability
    if embedding:
        a.embedding = embedding
    return a


class TestAbstractNode:
    def test_create(self):
        an = AbstractNode(
            id="an1", label="Test Concept",
            description="A concept emerging from memories",
            source_anchor_ids=["a1", "a2"],
            centroid_embedding=[0.1, 0.2, 0.3],
            confidence=0.8,
        )
        assert an.id == "an1"
        assert an.label == "Test Concept"
        assert len(an.source_anchor_ids) == 2
        assert an.confidence == 0.8
        assert an.level == 1

    def test_is_stable_below_threshold(self):
        an = AbstractNode(
            id="an1", label="Test", description="",
            source_anchor_ids=[], centroid_embedding=[],
            confidence=0.3,
        )
        assert an.is_stable is False

    def test_is_stable_above_threshold(self):
        an = AbstractNode(
            id="an1", label="Test", description="",
            source_anchor_ids=[], centroid_embedding=[],
            confidence=0.95,
        )
        assert an.is_stable is True

    def test_defaults(self):
        an = AbstractNode(
            id="an1", label="Test", description="test",
            source_anchor_ids=[], centroid_embedding=[],
            confidence=0.5,
        )
        assert an.tags == []
        assert an.created_at > 0


class TestCosineSimAbstractive:
    def test_identical(self):
        assert _cosine_sim_abstractive([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_sim_abstractive([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_different_lengths(self):
        # zip stops at shortest; norms computed over full vectors
        # dot=5, na=sqrt(14)=3.74, nb=sqrt(5)=2.24 ⇒ 5/8.37=0.597
        result = _cosine_sim_abstractive([1.0, 2.0, 3.0], [1.0, 2.0])
        assert 0.5 < result < 0.7


class TestAbstractionEngine:
    def test_init_defaults(self):
        ae = AbstractionEngine()
        assert ae.min_cluster_size >= 2
        assert ae.similarity_threshold > 0
        assert ae.abstracts == {}

    def test_init_custom(self):
        ae = AbstractionEngine(min_cluster_size=5, similarity_threshold=0.8)
        assert ae.min_cluster_size == 5
        assert ae.similarity_threshold == 0.8

    def test_discover_too_few_anchors(self):
        ae = AbstractionEngine(min_cluster_size=5)
        anchors = {
            "a1": make_anchor("a1", "test 1", embedding=[1.0, 0.0]),
            "a2": make_anchor("a2", "test 2", embedding=[1.0, 0.0]),
        }
        embeddings = {"a1": [1.0, 0.0], "a2": [1.0, 0.0]}
        result = ae.discover(anchors, embeddings)
        assert result == []

    def test_discover_with_cluster(self):
        ae = AbstractionEngine(min_cluster_size=2, similarity_threshold=0.5)
        anchors = {
            "a1": make_anchor("a1", "python coding", embedding=[1.0, 0.0, 0.0, 0.0],
                           tags=["dev", "python"]),
            "a2": make_anchor("a2", "flask api", embedding=[1.0, 0.0, 0.0, 0.0],
                           tags=["dev", "python"]),
        }
        embeddings = {
            "a1": [1.0, 0.0, 0.0, 0.0],
            "a2": [1.0, 0.0, 0.0, 0.0],
        }
        result = ae.discover(anchors, embeddings)
        assert len(result) >= 1
        assert result[0].label != ""

    def test_discover_dissimilar_anchors(self):
        ae = AbstractionEngine(min_cluster_size=2, similarity_threshold=0.99)
        anchors = {
            "a1": make_anchor("a1", "test 1", embedding=[1.0, 0.0, 0.0, 0.0]),
            "a2": make_anchor("a2", "test 2", embedding=[0.0, 1.0, 0.0, 0.0]),
            "a3": make_anchor("a3", "test 3", embedding=[0.0, 0.0, 1.0, 0.0]),
        }
        embeddings = {
            "a1": [1.0, 0.0, 0.0, 0.0],
            "a2": [0.0, 1.0, 0.0, 0.0],
            "a3": [0.0, 0.0, 1.0, 0.0],
        }
        result = ae.discover(anchors, embeddings)
        assert result == []

    def test_discover_already_covered(self):
        ae = AbstractionEngine(min_cluster_size=2, similarity_threshold=0.5)
        anchors = {
            "a1": make_anchor("a1", "test 1", embedding=[1.0, 0.0, 0.0, 0.0]),
            "a2": make_anchor("a2", "test 2", embedding=[1.0, 0.0, 0.0, 0.0]),
            "a3": make_anchor("a3", "test 3", embedding=[1.0, 0.0, 0.0, 0.0]),
        }
        embeddings = {
            "a1": [1.0, 0.0, 0.0, 0.0],
            "a2": [1.0, 0.0, 0.0, 0.0],
            "a3": [1.0, 0.0, 0.0, 0.0],
        }
        result = ae.discover(anchors, embeddings)
        # All 3 anchors in same cluster, one abstract created
        assert len(result) == 1

    def test_match_empty(self):
        ae = AbstractionEngine()
        result = ae.match([1.0, 0.0])
        assert result == []

    def test_match_with_abstracts(self):
        ae = AbstractionEngine(min_cluster_size=2, similarity_threshold=0.5)
        anchors = {
            "a1": make_anchor("a1", "test 1", embedding=[1.0, 0.0, 0.0, 0.0],
                           tags=["dev"]),
            "a2": make_anchor("a2", "test 2", embedding=[1.0, 0.0, 0.0, 0.0],
                           tags=["dev"]),
        }
        embeddings = {"a1": [1.0, 0.0, 0.0, 0.0],
                     "a2": [1.0, 0.0, 0.0, 0.0]}
        ae.discover(anchors, embeddings)
        result = ae.match([1.0, 0.0, 0.0, 0.0])
        assert len(result) >= 1

    def test_match_no_centroid(self):
        ae = AbstractionEngine()
        ae.abstracts["empty"] = AbstractNode(
            id="empty", label="test", description="",
            source_anchor_ids=[], centroid_embedding=[], confidence=0.5,
        )
        result = ae.match([1.0, 0.0])
        assert result == []  # empty centroid skipped

    def test_stats_empty(self):
        ae = AbstractionEngine()
        s = ae.stats
        assert s["total_abstracts"] == 0

    def test_stats_with_abstracts(self):
        ae = AbstractionEngine(min_cluster_size=2, similarity_threshold=0.5)
        anchors = {
            "a1": make_anchor("a1", "test", embedding=[1.0, 0.0, 0.0, 0.0], tags=["t1"]),
            "a2": make_anchor("a2", "test", embedding=[1.0, 0.0, 0.0, 0.0], tags=["t1"]),
        }
        embeddings = {"a1": [1.0, 0.0, 0.0, 0.0],
                     "a2": [1.0, 0.0, 0.0, 0.0]}
        ae.discover(anchors, embeddings)
        s = ae.stats
        assert s["total_abstracts"] >= 1
        assert s["avg_confidence"] > 0

    def test_generate_label_uses_tags(self):
        ae = AbstractionEngine()
        a1 = make_anchor("a1", "python code", tags=["dev", "python"])
        a2 = make_anchor("a2", "python script", tags=["dev", "python"])
        label = ae._generate_label([a1, a2])
        assert "dev" in label or "python" in label

    def test_generate_label_fallback(self):
        ae = AbstractionEngine()
        a = make_anchor("a1", "deploying Docker containers", tags=[])
        label = ae._generate_label([a])
        assert isinstance(label, str)
        assert len(label) > 0

    def test_extract_common_tags(self):
        tags = AbstractionEngine._extract_common_tags([
            make_anchor("a1", tags=["dev", "python"]),
            make_anchor("a2", tags=["dev", "docker"]),
            make_anchor("a3", tags=["dev", "python"]),
        ])
        assert "dev" in tags

    def test_extract_common_tags_no_repeats(self):
        tags = AbstractionEngine._extract_common_tags([
            make_anchor("a1", tags=["unique_a"]),
            make_anchor("a2", tags=["unique_b"]),
        ])
        assert len(tags) == 0  # no tag appears in 2+ anchors


class TestPatternMemory:
    def test_defaults(self):
        pm = PatternMemory(
            id="pm1", pattern_text="test pattern",
            centroid_embedding=[0.1, 0.2],
        )
        assert pm.occurrence_count == 1
        assert pm.stability == 0.3
        assert pm.promoted is False
        assert pm.is_recurring is False
        assert pm.is_stable is False

    def test_is_recurring_by_occurrences(self):
        pm = PatternMemory(
            id="pm1", pattern_text="test",
            centroid_embedding=[0.1, 0.2],
            occurrence_count=5,
        )
        assert pm.is_recurring is True

    def test_is_recurring_by_sessions(self):
        pm = PatternMemory(
            id="pm1", pattern_text="test",
            centroid_embedding=[0.1, 0.2],
            source_session_ids=["s1", "s2", "s3"],
        )
        assert pm.is_recurring is True

    def test_is_stable(self):
        pm = PatternMemory(
            id="pm1", pattern_text="test",
            centroid_embedding=[0.1, 0.2],
            stability=0.8,
        )
        assert pm.is_stable is True


class TestAbstractiveMemoryEngine:
    def test_init_defaults(self):
        ame = AbstractiveMemoryEngine()
        assert ame.min_occurrences == 3
        assert ame.similarity_threshold == 0.75
        assert ame.cross_session_bonus == 1.3
        assert ame.patterns == {}

    def test_init_custom(self):
        ame = AbstractiveMemoryEngine(
            min_occurrences=5, similarity_threshold=0.9, cross_session_bonus=1.5,
        )
        assert ame.min_occurrences == 5

    def test_abstraction_engine_lazy_init(self):
        ame = AbstractiveMemoryEngine()
        ae = ame.abstraction_engine
        assert isinstance(ae, AbstractionEngine)
        # Should be cached
        assert ame.abstraction_engine is ae

    def test_extract_patterns_not_enough_sessions(self):
        ame = AbstractiveMemoryEngine()
        g = StarGraph()
        a = make_anchor("a1", "test", embedding=[0.1] * 384, source_session="s1")
        g.add_anchor(a)
        patterns = ame.extract_patterns(g)
        assert patterns == []

    def test_extract_patterns_cross_session(self):
        ame = AbstractiveMemoryEngine(similarity_threshold=0.5)
        g = StarGraph()
        a1 = make_anchor("a1", "python deployment", embedding=[1.0] * 10,
                       source_session="s1")
        a2 = make_anchor("a2", "python deployment fix", embedding=[1.0] * 10,
                       source_session="s2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        patterns = ame.extract_patterns(g)
        assert len(patterns) >= 1
        assert patterns[0].occurrence_count >= 2

    def test_extract_patterns_no_embedding(self):
        ame = AbstractiveMemoryEngine()
        g = StarGraph()
        a1 = make_anchor("a1", "test", source_session="s1")
        a2 = make_anchor("a2", "test", source_session="s2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        patterns = ame.extract_patterns(g)
        assert patterns == []

    def test_extract_patterns_dissimilar(self):
        ame = AbstractiveMemoryEngine(similarity_threshold=0.99)
        g = StarGraph()
        a1 = make_anchor("a1", "python", embedding=[1.0] + [0.0] * 9,
                       source_session="s1")
        a2 = make_anchor("a2", "finance", embedding=[0.0] * 5 + [1.0] + [0.0] * 4,
                       source_session="s2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        patterns = ame.extract_patterns(g)
        assert patterns == []

    def test_promote_stable_patterns(self):
        ame = AbstractiveMemoryEngine(similarity_threshold=0.5)
        g = StarGraph()
        a1 = make_anchor("a1", "python deployment", embedding=[1.0] * 10,
                       source_session="s1", stability=0.5)
        a2 = make_anchor("a2", "python deploy", embedding=[1.0] * 10,
                       source_session="s2", stability=0.5)
        g.add_anchor(a1)
        g.add_anchor(a2)
        ame.extract_patterns(g)
        # Manually boost a pattern to recurring
        for p in ame.patterns.values():
            p.occurrence_count = 5
            p.source_session_ids = ["s1", "s2", "s3"]
        promoted = ame.promote_stable_patterns(g)
        assert len(promoted) >= 1

    def test_consolidate_existing_patterns(self):
        ame = AbstractiveMemoryEngine(similarity_threshold=0.5)
        g = StarGraph()
        a1 = make_anchor("a1", "python", embedding=[1.0] * 10,
                       source_session="s1")
        a2 = make_anchor("a2", "python", embedding=[1.0] * 10,
                       source_session="s2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        ame.extract_patterns(g)
        # Add a new matching anchor
        a3 = make_anchor("a3", "python new", embedding=[1.0] * 10,
                       source_session="s3")
        g.add_anchor(a3)
        stats = ame.consolidate_existing_patterns(g)
        assert stats["new_matches"] >= 1

    def test_stats_empty(self):
        ame = AbstractiveMemoryEngine()
        s = ame.stats
        assert s["total_patterns"] == 0

    def test_stats_with_patterns(self):
        ame = AbstractiveMemoryEngine(similarity_threshold=0.5)
        g = StarGraph()
        a1 = make_anchor("a1", "test", embedding=[1.0] * 10, source_session="s1")
        a2 = make_anchor("a2", "test", embedding=[1.0] * 10, source_session="s2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        ame.extract_patterns(g)
        s = ame.stats
        assert s["total_patterns"] >= 1
