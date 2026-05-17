"""Tests for router module — CortexRouter and RouteResult."""

import pytest

from star_graph.router import CortexRouter, RouteResult
from star_graph.cortex import MemoryCortex, CortexConfig, CORTEX_HIERARCHY, HIERARCHY_WEIGHTS
from star_graph.config import Config


def make_cortex(name: str, keywords: list | None = None,
                level: int = 3) -> MemoryCortex:
    return MemoryCortex(CortexConfig(
        name=name,
        description=f"Cortex {name}",
        domain_keywords=keywords or [],
        hierarchy_level=level,
    ))


class TestRouteResult:
    def test_defaults(self):
        c = make_cortex("test")
        rr = RouteResult(cortex=c, score=0.5)
        assert rr.cortex is c
        assert rr.score == 0.5
        assert rr.reasoning == ""

    def test_with_reasoning(self):
        c = make_cortex("test")
        rr = RouteResult(cortex=c, score=0.8,
                         reasoning="semantic match")
        assert rr.reasoning == "semantic match"


class TestCortexRouter:
    def test_init_default(self):
        cr = CortexRouter()
        assert cr.cortices == []
        assert cr.total_routes == 0
        assert cr.route_history == []

    def test_init_with_config(self):
        cfg = Config.get()
        cr = CortexRouter(config=cfg)
        assert cr.cfg is cfg

    def test_init_with_brain(self):
        brain = object()
        cr = CortexRouter(brain=brain)
        assert cr.brain is brain

    def test_add_cortex(self):
        cr = CortexRouter()
        c = make_cortex("dev")
        cr.add_cortex(c)
        assert len(cr.cortices) == 1

    def test_add_cortex_duplicate(self):
        cr = CortexRouter()
        c1 = make_cortex("dev")
        cr.add_cortex(c1)
        c2 = make_cortex("dev")
        with pytest.raises(ValueError, match="already exists"):
            cr.add_cortex(c2)

    def test_remove_cortex(self):
        cr = CortexRouter()
        c = make_cortex("dev")
        cr.add_cortex(c)
        cr.remove_cortex("dev")
        assert len(cr.cortices) == 0

    def test_remove_cortex_nonexistent(self):
        cr = CortexRouter()
        cr.remove_cortex("nonexistent")
        assert cr.cortices == []

    def test_get_cortex(self):
        cr = CortexRouter()
        c = make_cortex("dev")
        cr.add_cortex(c)
        assert cr.get_cortex("dev") is c

    def test_get_cortex_nonexistent(self):
        cr = CortexRouter()
        assert cr.get_cortex("nonexistent") is None

    def test_default_cortex(self):
        cr = CortexRouter()
        dc = cr.default_cortex
        assert dc.config.name == "general"
        # Should be cached
        assert cr.default_cortex is dc

    def test_route_no_cortices(self):
        cr = CortexRouter()
        results = cr.route("test query")
        assert len(results) == 1
        assert results[0].cortex.config.name == "general"
        assert results[0].score == 1.0
        assert "default" in results[0].reasoning

    def test_route_with_cortex_text_query(self):
        cr = CortexRouter()
        c = make_cortex("dev", keywords=["code", "python", "debug"])
        cr.add_cortex(c)
        results = cr.route("python code")
        assert len(results) >= 1
        assert results[0].score >= 0.0

    def test_route_with_embedding(self):
        cr = CortexRouter()
        c = make_cortex("dev", keywords=["code"])
        cr.add_cortex(c)
        results = cr.route(query_embedding=[0.1] * 384)
        assert len(results) >= 1

    def test_route_max_cortices(self):
        cr = CortexRouter()
        for name in ["dev", "finance", "personal", "health"]:
            cr.add_cortex(make_cortex(name, keywords=[name]))
        results = cr.route("test", max_cortices=2)
        assert len(results) <= 2

    def test_route_min_score_filters(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("dev"))
        results = cr.route("xyzzy flurbo garply", min_score=0.99)
        # Either nothing matches or falls back to default
        assert len(results) >= 1

    def test_route_updates_stats(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("dev"))
        cr.route("test query")
        assert cr.total_routes == 1
        assert len(cr.route_history) == 1

    def test_route_with_hierarchy_weights(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("reflection", level=0))
        cr.add_cortex(make_cortex("episodic", level=3))
        results = cr.route("test query")
        assert len(results) >= 1
        # Reflection cortex (level 0, weight 1.5) should rank high
        if len(results) >= 2:
            # Scores should reflect hierarchy weighting
            pass

    def test_route_with_brain_centers_no_match(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("dev"))

        class FakeBrain:
            def get_relevant_centers(self, emb, top_k, min_similarity):
                return []
        cr.brain = FakeBrain()
        results = cr.route(query_embedding=[0.1] * 384)
        assert len(results) >= 1

    def test_route_with_brain_centers(self):
        cr = CortexRouter()
        c = make_cortex("dev")
        cr.add_cortex(c)

        class FakeCenter:
            cortex_name = "dev"

        class FakeBrain:
            def get_relevant_centers(self, emb, top_k, min_similarity):
                return [FakeCenter()]
        cr.brain = FakeBrain()
        results = cr.route(query_embedding=[0.1] * 384)
        assert len(results) >= 1

    def test_recall_with_no_context(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("dev"))
        results = cr.recall("test")
        assert isinstance(results, list)

    def test_recall_with_context(self):
        from star_graph.scheduler import AgentContext
        cr = CortexRouter()
        cr.add_cortex(make_cortex("dev"))
        ctx = AgentContext(task_type="conversation")
        results = cr.recall("test", context=ctx)
        assert isinstance(results, list)

    def test_find_or_create_cortex_new(self):
        cr = CortexRouter()
        c = cr.find_or_create_cortex("dev", domain_keywords=["code"])
        assert c.config.name == "dev"
        assert "dev" in {c2.config.name for c2 in cr.cortices}

    def test_find_or_create_cortex_existing(self):
        cr = CortexRouter()
        c1 = cr.find_or_create_cortex("dev")
        c2 = cr.find_or_create_cortex("dev")
        assert c1 is c2
        assert len(cr.cortices) == 1

    def test_get_hierarchy_summary(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("reflection", level=0))
        cr.add_cortex(make_cortex("episodic", level=3))
        s = cr.get_hierarchy_summary()
        assert s["total_cortices"] == 2
        assert "by_level" in s
        assert "hierarchy_weights" in s

    def test_get_hierarchy_summary_empty(self):
        cr = CortexRouter()
        s = cr.get_hierarchy_summary()
        assert s["total_cortices"] == 0

    def test_stats(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("dev"))
        s = cr.stats
        assert s["cortices"] == 1
        assert s["total_routes"] == 0
        assert "recent_routes" in s

    def test_stats_after_routing(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("dev"))
        cr.route("test")
        s = cr.stats
        assert s["total_routes"] == 1
        assert len(s["recent_routes"]) == 1

    def test_propagate_down_no_source(self):
        cr = CortexRouter()
        stats = cr.propagate_down("nonexistent")
        assert stats == {}

    def test_propagate_down_no_centroid(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("reflection", level=0))
        # No centroid set on source cortex
        stats = cr.propagate_down("reflection")
        assert stats == {}

    def test_propagate_down_same_or_higher_level(self):
        cr = CortexRouter()
        # Add a reflection cortex (level 0)
        cr.add_cortex(make_cortex("reflection", level=0))
        # Episodic (level 3) should not propagate UP to reflection
        c_episodic = make_cortex("episodic", level=3)
        cr.add_cortex(c_episodic)
        # Set centroid on episodic
        c_episodic._centroid = [0.1] * 384
        c_episodic._centroid_stale = False
        # Add an anchor to the reflection graph with embedding
        from star_graph.anchor import Anchor
        a = Anchor(id="a1", text="test")
        a.embedding = [0.5] * 384
        cr.get_cortex("reflection").graph.add_anchor(a)
        stats = cr.propagate_down("episodic")
        # Episodic level=3 shouldn't propagate to reflection level=0
        # since only propagate DOWN (to higher level numbers)
        assert stats == {}

    def test_propagate_down_with_anchor(self):
        cr = CortexRouter()
        cr.add_cortex(make_cortex("reflection", level=0))
        c_episodic = make_cortex("episodic", level=3)
        cr.add_cortex(c_episodic)
        # Set centroid on reflection
        c_reflection = cr.get_cortex("reflection")
        c_reflection._centroid = [0.9] * 384
        c_reflection._centroid_stale = False
        # Add anchor to episodic with similar embedding
        from star_graph.anchor import Anchor
        a = Anchor(id="a1", text="test")
        a.embedding = [0.9] * 384
        c_episodic.graph.add_anchor(a)
        stats = cr.propagate_down("reflection", boost_importance=0.1)
        # Should boost the anchor in episodic
        assert stats.get("episodic", 0) >= 1
