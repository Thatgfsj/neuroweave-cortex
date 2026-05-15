"""Tests for MemoryCompetition — interference-based forgetting and competitive retrieval."""

import time

import pytest

from star_graph.anchor import Anchor, AnchorVector
from star_graph.config import Config
from star_graph.competition import MemoryCompetition


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                retention: float = 0.5, emotional_valence: float = 0.0,
                tags: list | None = None, importance: float = 0.5,
                stability: float = 0.5, recency: float = 1.0) -> Anchor:
    a = Anchor(
        id=name, text=text or f"Memory {name}",
        vector=AnchorVector(
            importance=importance,
            stability=stability,
            recency=recency,
            emotional_valence=emotional_valence,
        ),
    )
    if embedding:
        a.embedding = embedding
    if tags:
        a.tags = list(tags)
    return a


@pytest.fixture
def competition_cfg():
    cfg = Config.defaults()
    return cfg


class TestMemoryCompetitionInit:
    def test_initializes_with_config(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        mc = MemoryCompetition(g, config=competition_cfg)
        assert mc.graph is g
        assert mc.cfg is competition_cfg

    def test_initializes_without_config(self):
        from star_graph.graph import StarGraph
        g = StarGraph()
        mc = MemoryCompetition(g)
        assert mc.graph is g
        assert mc.cfg is not None

    def test_cosine_sim_identical(self):
        v = [1.0, 2.0, 3.0]
        assert MemoryCompetition._cosine_sim(v, v) == pytest.approx(1.0, abs=1e-4)

    def test_cosine_sim_orthogonal(self):
        assert MemoryCompetition._cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_sim_empty(self):
        assert MemoryCompetition._cosine_sim([], [1.0]) == 0.0


class TestMemoryCompetitionApply:
    def test_nonexistent_anchor_returns_empty(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        mc = MemoryCompetition(g, config=competition_cfg)
        result = mc.apply_competition("nonexistent")
        assert result == {}

    def test_no_competitors_returns_empty(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        anchor = make_anchor("a1", embedding=[0.1, 0.2, 0.3])
        g.add_anchor(anchor)
        mc = MemoryCompetition(g, config=competition_cfg)
        result = mc.apply_competition("a1")
        assert result == {}

    def test_similar_anchor_suppressed(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        activated = make_anchor("a1", embedding=[0.1, 0.2, 0.3], retention=0.9,
                                emotional_valence=0.8)
        similar = make_anchor("a2", embedding=[0.11, 0.21, 0.31])
        g.add_anchor(activated)
        g.add_anchor(similar)
        g.add_edge("a1", "a2", weight=0.5)

        mc = MemoryCompetition(g, config=competition_cfg)
        result = mc.apply_competition("a1")
        # a2 should be suppressed since it's similar and connected
        assert "a2" in result
        assert result["a2"] > 0

    def test_self_not_suppressed(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        anchor = make_anchor("a1", embedding=[0.1, 0.2, 0.3])
        g.add_anchor(anchor)
        mc = MemoryCompetition(g, config=competition_cfg)
        result = mc.apply_competition("a1")
        assert "a1" not in result

    def test_tag_overlap_increases_competition(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        activated = make_anchor("a1", embedding=[0.1, 0.2, 0.3], tags=["redis", "timeout"],
                                retention=0.9)
        similar = make_anchor("a2", embedding=[0.11, 0.21, 0.31],
                              tags=["redis", "timeout", "debug"])
        g.add_anchor(activated)
        g.add_anchor(similar)

        mc = MemoryCompetition(g, config=competition_cfg)
        result = mc.apply_competition("a1")
        assert "a2" in result


class TestMemoryCompetitionInterference:
    def test_new_anchor_interferes_with_old(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        old = make_anchor("old", embedding=[0.1, 0.2, 0.3, 0.4, 0.5,
                                             0.6, 0.7, 0.8, 0.9, 1.0])
        old.created_at = 1000.0
        g.add_anchor(old)

        mc = MemoryCompetition(g, config=competition_cfg)
        new = make_anchor("new", embedding=[0.11, 0.21, 0.31, 0.41, 0.51,
                                             0.61, 0.71, 0.81, 0.91, 1.01])
        new.created_at = 2000.0  # newer

        suppressed = mc.interference_forget(new)
        assert "old" in suppressed

    def test_interference_without_embedding_skipped(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        old = make_anchor("old")  # no embedding
        g.add_anchor(old)

        mc = MemoryCompetition(g, config=competition_cfg)
        new = make_anchor("new", embedding=[0.1, 0.2, 0.3, 0.4, 0.5,
                                             0.6, 0.7, 0.8, 0.9, 1.0])
        new.created_at = 2000.0

        suppressed = mc.interference_forget(new)
        assert len(suppressed) == 0

    def test_interference_adds_contradicted_tag(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        old = make_anchor("old", embedding=[0.1, 0.2, 0.3, 0.4, 0.5,
                                             0.6, 0.7, 0.8, 0.9, 1.0])
        old.created_at = 1000.0
        g.add_anchor(old)

        mc = MemoryCompetition(g, config=competition_cfg)
        new = make_anchor("new", embedding=[0.105, 0.205, 0.305, 0.405, 0.505,
                                             0.605, 0.705, 0.805, 0.905, 1.005])
        new.created_at = 2000.0

        mc.interference_forget(new)
        assert "contradicted" in g.anchors["old"].tags


class TestMemoryCompetitionRIF:
    def test_nonexistent_retrieved_no_error(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        mc = MemoryCompetition(g, config=competition_cfg)
        mc.retrieval_induced_forgetting("nonexistent", ["a2"])

    def test_retrieval_weakens_competitor(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        retrieved = make_anchor("r1", embedding=[0.1, 0.2, 0.3, 0.4, 0.5,
                                                   0.6, 0.7, 0.8, 0.9, 1.0])
        competitor = make_anchor("c1", embedding=[0.11, 0.21, 0.31, 0.41, 0.51,
                                                    0.61, 0.71, 0.81, 0.91, 1.01])
        g.add_anchor(retrieved)
        g.add_anchor(competitor)

        old_importance = competitor.vector.importance
        mc = MemoryCompetition(g, config=competition_cfg)
        mc.retrieval_induced_forgetting("r1", ["c1"])
        assert competitor.vector.importance < old_importance

    def test_rif_requires_similarity(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        retrieved = make_anchor("r1", embedding=[1.0, 0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0, 0.0])
        # Dissimilar embedding
        competitor = make_anchor("c1", embedding=[0.0, 0.0, 0.0, 0.0, 0.0,
                                                    0.0, 0.0, 0.0, 0.0, 1.0])
        g.add_anchor(retrieved)
        g.add_anchor(competitor)

        old_importance = competitor.vector.importance
        mc = MemoryCompetition(g, config=competition_cfg)
        mc.retrieval_induced_forgetting("r1", ["c1"])
        # Should not be weakened since similarity < threshold
        assert competitor.vector.importance == old_importance


class TestMemoryCompetitionStats:
    def test_empty_graph_stats(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        mc = MemoryCompetition(g, config=competition_cfg)
        stats = mc.competition_stats
        assert stats["contradicted_anchors"] == 0
        assert stats["revision_edges"] == 0
        assert not stats["competition_active"]

    def test_stats_with_contradicted(self, competition_cfg):
        from star_graph.graph import StarGraph
        g = StarGraph()
        a = make_anchor("a1", tags=["contradicted"])
        b = make_anchor("a2", tags=["test"])
        g.add_anchor(a)
        g.add_anchor(b)
        mc = MemoryCompetition(g, config=competition_cfg)
        stats = mc.competition_stats
        assert stats["contradicted_anchors"] == 1
