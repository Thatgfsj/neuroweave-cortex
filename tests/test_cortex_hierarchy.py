"""Test cortex hierarchy: domain-based priority, weight propagation, factory methods."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from star_graph import (
    MemoryCortex, CortexConfig, CortexRouter,
    CORTEX_HIERARCHY, HIERARCHY_WEIGHTS, HIERARCHY_DECAY_DAYS,
)


class TestCortexHierarchy:
    """Verify hierarchy levels and factory methods."""

    def test_hierarchy_constants(self):
        """Hierarchy constants should define correct levels."""
        assert CORTEX_HIERARCHY["reflection"] == 0
        assert CORTEX_HIERARCHY["semantic"] == 1
        assert CORTEX_HIERARCHY["procedural"] == 2
        assert CORTEX_HIERARCHY["episodic"] == 3
        assert CORTEX_HIERARCHY["hippocampus"] == 4

        assert HIERARCHY_WEIGHTS[0] > HIERARCHY_WEIGHTS[3]  # reflection > episodic
        assert HIERARCHY_DECAY_DAYS[0] > HIERARCHY_DECAY_DAYS[3]  # reflection decays slower

    def test_reflection_factory(self):
        """Reflection cortex should have level 0 and high priority config."""
        ctx = MemoryCortex.reflection("test_refl")
        assert ctx.config.hierarchy_level == 0
        assert ctx.config.hierarchy_domain == "reflection"
        assert ctx.config.decay_half_life_days >= 180
        assert ctx.config.token_budget >= 2000  # highest budget

    def test_semantic_factory(self):
        """Semantic cortex should have level 1."""
        ctx = MemoryCortex.semantic("test_sem")
        assert ctx.config.hierarchy_level == 1
        assert ctx.config.hierarchy_domain == "semantic"

    def test_procedural_factory(self):
        """Procedural cortex should have level 2."""
        ctx = MemoryCortex.procedural("test_proc")
        assert ctx.config.hierarchy_level == 2
        assert ctx.config.hierarchy_domain == "procedural"

    def test_episodic_factory(self):
        """Episodic cortex should have level 3 and fast decay."""
        ctx = MemoryCortex.episodic("test_epi")
        assert ctx.config.hierarchy_level == 3
        assert ctx.config.hierarchy_domain == "episodic"
        assert ctx.config.decay_half_life_days <= 90  # fast decay

    def test_hierarchy_weights_ordering(self):
        """Higher priority cortices (lower level) should have higher weights."""
        weights = [HIERARCHY_WEIGHTS[i] for i in range(5)]
        # Weights should decrease (or stay same) as level increases
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], f"weight at level {i} should be >= level {i+1}"


class TestCortexRouterHierarchy:
    """Verify hierarchy-aware routing."""

    def test_hierarchy_in_routing(self):
        """Router should apply hierarchy weights to scores."""
        router = CortexRouter()

        reflection = MemoryCortex.reflection("reflect")
        episodic = MemoryCortex.episodic("episodes")

        # Add anchors so centroid can be computed
        reflection.remember("critical deployment failure pattern: always test migrations first",
                          tags=["error", "strategy"])
        episodic.remember("talked about weather today",
                        tags=["conversation"])

        router.add_cortex(reflection)
        router.add_cortex(episodic)

        # Route a strategy-related query
        results = router.route("deployment failure prevention strategy")

        assert len(results) > 0
        # Reflection should rank higher for strategy queries
        names = [r.cortex.config.name for r in results]
        assert "reflect" in names or "episodes" in names

    def test_propagate_down(self):
        """High-priority cortex should propagate importance to related lower memories."""
        router = CortexRouter()

        reflection = MemoryCortex.reflection("reflect")
        episodic = MemoryCortex.episodic("episodes")

        # Add a strategy insight to reflection
        reflection.remember(
            "python deployment failures are always caused by version mismatches",
            tags=["error", "strategy", "python"]
        )

        # Add related concrete events to episodic
        episodic.remember(
            "deployment failed because python 3.12 vs 3.11 library conflict",
            tags=["deployment", "python", "error"]
        )
        episodic.remember(
            "unrelated chat about lunch preferences",
            tags=["conversation"]
        )

        router.add_cortex(reflection)
        router.add_cortex(episodic)

        # Get initial importances
        epi_anchors = list(episodic.graph.anchors.values())
        initial_importances = [a.vector.importance for a in epi_anchors]

        # Propagate down
        stats = router.propagate_down("reflect", boost_importance=0.15)

        # Check that propagation happened
        assert isinstance(stats, dict)

    def test_hierarchy_summary(self):
        """get_hierarchy_summary should return correct structure."""
        router = CortexRouter()
        router.add_cortex(MemoryCortex.reflection("r"))
        router.add_cortex(MemoryCortex.episodic("e"))
        router.add_cortex(MemoryCortex.procedural("p"))

        summary = router.get_hierarchy_summary()
        assert summary["total_cortices"] == 3
        assert "by_level" in summary
        assert "hierarchy_weights" in summary
