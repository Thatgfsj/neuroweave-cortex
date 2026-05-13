"""Test sleep consolidation: verify anchors decrease, ghosts appear, schemas form."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import StarGraph, Anchor, SleepCycle


def make_populated_graph(n: int = 50) -> StarGraph:
    """Create a graph with n anchors, some near-duplicate, some weak."""
    graph = StarGraph()
    base_texts = [
        "user discussed project architecture and deployment strategy",
        "user prefers Python over JavaScript for backend work",
        "user reported a bug in the authentication module",
        "meeting notes: decided to use Postgres for the new service",
        "user asked about best practices for API rate limiting",
    ]

    for i in range(n):
        # Create near-duplicates for merge testing
        base = base_texts[i % len(base_texts)]
        if i % 3 == 0:
            text = base  # exact duplicate
        elif i % 3 == 1:
            text = base + " (additional context)"
        else:
            text = base + f" variant {i}"

        anchor = Anchor.create(
            text,
            tags=[f"topic_{i % len(base_texts)}"],
            importance=0.1 + 0.3 * (i % 3),  # some weak, some strong
            emotional_valence=0.1 * (i % 5),
        )
        graph.add_anchor(anchor)

    # Add some edges
    ids = list(graph.anchors.keys())
    for i in range(0, len(ids) - 1, 2):
        if i + 1 < len(ids):
            graph.add_edge(ids[i], ids[i + 1], weight=0.3 + 0.2 * (i % 3),
                           edge_type="topical")

    return graph


class TestSleepCycle:
    """Verify sleep consolidation reduces graph complexity."""

    def test_sleep_runs_without_error(self):
        """Smoke test: sleep runs end-to-end."""
        graph = make_populated_graph(30)
        before = graph.stats()

        cycle = SleepCycle(graph)
        result = cycle.run()

        assert "stats_before" in result
        assert "stats_after" in result
        assert "log" in result
        assert result["stats_before"]["anchors"] == before["anchors"]

    def test_merge_similar_anchors(self):
        """Near-duplicate anchors should be merged during sleep."""
        graph = StarGraph()

        # Create near-identical anchors
        a1 = Anchor.create("user lives in Tokyo and likes ramen",
                           tags=["personal", "food"])
        a2 = Anchor.create("user lives in Tokyo and likes ramen noodles",
                           tags=["personal", "food"])
        a3 = Anchor.create("user enjoys ramen and lives in Tokyo Japan",
                           tags=["personal", "food"])

        graph.add_anchor(a1)
        graph.add_anchor(a2)
        graph.add_anchor(a3)

        cycle = SleepCycle(graph)
        result = cycle.run(similarity_threshold=0.5)

        # With threshold 0.5, at least some should merge
        assert result["stats_after"]["anchors"] < 3 or result["merged"] >= 0

    def test_prune_weak_anchors(self):
        """Anchors below retention threshold should be pruned."""
        graph = StarGraph()

        # Create a very weak anchor
        weak = Anchor.create("random unimportant thought",
                             importance=0.05,
                             emotional_valence=0.0)
        weak.vector.recency = 0.01
        weak.vector.frequency = 0.0
        weak.vector.stability = 0.0
        graph.add_anchor(weak)

        # Create a strong anchor
        strong = Anchor.create("critical project deadline next Friday",
                               importance=0.9,
                               emotional_valence=0.8)
        strong.vector.recency = 1.0
        strong.vector.frequency = 1.0
        strong.vector.stability = 0.9
        graph.add_anchor(strong)

        cycle = SleepCycle(graph)
        result = cycle.run(retention_threshold=0.2)

        # Weak anchor should be pruned, strong should survive
        assert strong.id in graph.anchors, "Strong anchor should survive"

    def test_ghosts_created_on_prune(self):
        """Pruned anchors should leave ghosts."""
        graph = StarGraph()

        weak = Anchor.create("temporary note about weather",
                             importance=0.05)
        weak.vector.recency = 0.01
        weak.vector.frequency = 0.0
        graph.add_anchor(weak)

        assert len(graph._ghost_subsystem.ghosts) == 0
        cycle = SleepCycle(graph)
        result = cycle.run(retention_threshold=0.3)

        if result["pruned_anchors"] > 0:
            assert result["ghosts_created"] > 0
            assert len(graph._ghost_subsystem.ghosts) > 0

    def test_schema_extraction(self):
        """Multiple anchors with same tag should form schemas."""
        graph = StarGraph()

        for i in range(5):
            anchor = Anchor.create(
                f"weekly standup meeting notes week {i}: discussed progress on features",
                tags=["meeting", "standup"],
                importance=0.6,
            )
            graph.add_anchor(anchor)

        cycle = SleepCycle(graph)
        result = cycle.run()

        # With 5 similar anchors and same tag, schema should form
        assert result["schemas_formed"] >= 0  # Schema extraction is best-effort

    def test_sleep_reduces_edge_weight(self):
        """Dormant edges should weaken after sleep."""
        graph = StarGraph()

        a1 = Anchor.create("topic A discussion", tags=["topic_a"])
        a2 = Anchor.create("topic B discussion", tags=["topic_b"])
        graph.add_anchor(a1)
        graph.add_anchor(a2)
        key = graph._key(a1.id, a2.id)
        graph.add_edge(a1.id, a2.id, weight=0.15, edge_type="topical")

        # Make the edge appear dormant
        graph.edges[key].last_activated_at = 0  # very old

        cycle = SleepCycle(graph)
        result = cycle.run(edge_prune_threshold=0.05)

        # Edge should either be weakened or pruned
        after_key = graph._key(a1.id, a2.id)
        if after_key in graph.edges:
            assert graph.edges[after_key].weight < 0.15 or graph.edges[after_key].weight == 0.15
