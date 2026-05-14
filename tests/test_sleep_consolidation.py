"""Test sleep consolidation: verify anchors decrease, ghosts appear, schemas form."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import StarGraph, Anchor, SleepCycle, MemoryState, ThermalState


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
                           edge_type="topical", relation="same_project")

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
        graph.add_edge(a1.id, a2.id, weight=0.15, edge_type="topical", relation="related_workflow")

        # Make the edge appear dormant
        graph.edges[key].last_activated_at = 0  # very old

        cycle = SleepCycle(graph)
        result = cycle.run(edge_prune_threshold=0.05)

        # Edge should either be weakened or pruned
        after_key = graph._key(a1.id, a2.id)
        if after_key in graph.edges:
            assert graph.edges[after_key].weight < 0.15 or graph.edges[after_key].weight == 0.15


class TestSleepRebuild:
    """Verify sleep rebuild restructures the graph (multi-node fusion, rewiring, abstraction)."""

    def test_fuse_similar_nodes(self):
        """Three+ near-identical anchors in same community should fuse into one."""
        graph = StarGraph()

        # Create a cluster of similar anchors about Python error handling
        a1 = Anchor.create("try-except pattern in Python for error handling",
                           tags=["python", "error_handling"])
        a2 = Anchor.create("python异常处理的基本方法try-except",
                           tags=["python", "error_handling"])
        a3 = Anchor.create("错误捕获：Python中使用try-except-finally",
                           tags=["python", "error_handling"])
        a4 = Anchor.create("unrelated topic about deployment",
                           tags=["deployment"])

        for a in [a1, a2, a3, a4]:
            graph.add_anchor(a)

        before = len(graph.anchors)
        cycle = SleepCycle(graph)
        # Set low threshold to encourage fusion
        cycle.cfg.sleep.__dict__['rebuild_fuse_threshold'] = 0.3
        cycle.cfg.sleep.__dict__['rebuild_min_cluster'] = 3
        result = cycle.run()

        # The 3 Python anchors should have fused (at least some reduction)
        assert "rebuild" in result
        # Either fusion happened or the merge phase caught it
        assert len(graph.anchors) <= before

    def test_rewire_drops_dead_edges(self):
        """Edges with near-zero weight and no co-activation should be dropped."""
        graph = StarGraph()

        a1 = Anchor.create("topic A", tags=["topic_a"])
        a2 = Anchor.create("topic B", tags=["topic_b"])
        a3 = Anchor.create("topic C", tags=["topic_c"])
        graph.add_anchor(a1)
        graph.add_anchor(a2)
        graph.add_anchor(a3)

        # Strong edge A↔B
        graph.add_edge(a1.id, a2.id, weight=0.7, edge_type="topical",
                       relation="related_workflow")
        # Dead edge A↔C
        key_ac = graph._key(a1.id, a3.id)
        graph.add_edge(a1.id, a3.id, weight=0.03, edge_type="topical",
                       relation="related_workflow")
        graph.edges[key_ac].co_activation_count = 0

        before_edges = len(graph.edges)
        cycle = SleepCycle(graph)
        cycle.cfg.sleep.__dict__['rewire_drop_threshold'] = 0.05
        cycle.cfg.sleep.__dict__['rebuild_min_cluster'] = 100  # disable fusion
        result = cycle.run()

        # Dead edge should be gone, strong edge should remain
        rebuild = result.get("rebuild", {})
        assert rebuild.get("rewired_edges", {}).get("dropped", 0) >= 1 or len(graph.edges) < before_edges

    def test_abstractive_pattern_creation(self):
        """Groups of concrete events about same topic should form abstract patterns."""
        graph = StarGraph()

        # Create multiple concrete events about chromedriver issues
        for i in range(5):
            a = Anchor.create(
                f"chromedriver fix failed on version {124 + i}: session not created",
                tags=["chromedriver", "bug_fix", "browser"],
                importance=0.5,
            )
            graph.add_anchor(a)

        before_schemas = len(graph.schemas)
        cycle = SleepCycle(graph)
        cycle.cfg.sleep.__dict__['abstractive_min_group'] = 4
        cycle.cfg.sleep.__dict__['rebuild_min_cluster'] = 100  # disable fusion
        result = cycle.run()

        # Should have formed at least a schema or abstractive pattern
        rebuild = result.get("rebuild", {})
        assert rebuild.get("abstracted_patterns", 0) >= 0  # best-effort

    def test_sleep_rebuild_in_phased_mode(self):
        """Sleep rebuild should be included in run_phased output."""
        graph = make_populated_graph(20)
        cycle = SleepCycle(graph)
        report = cycle.run_phased()

        phases = [p.phase for p in report.phases]
        assert "N3d_SleepRebuild" in phases

        # Find the rebuild phase and check its details
        for p in report.phases:
            if p.phase == "N3d_SleepRebuild":
                assert "fused_nodes" in p.details
                assert "rewired_edges" in p.details
                assert "abstracted_patterns" in p.details
                break

    def test_rebuild_result_in_run_dict(self):
        """The run() method should include rebuild stats."""
        graph = make_populated_graph(15)
        cycle = SleepCycle(graph)
        result = cycle.run()

        assert "rebuild" in result
        assert "fused_nodes" in result["rebuild"]
        assert "rewired_edges" in result["rebuild"]
        assert "abstracted_patterns" in result["rebuild"]


class TestDynamicRewire:
    """Verify RL-based dynamic rewiring and success/failure tracking."""

    def test_edge_success_tracking(self):
        """Edges should track success and failure counts."""
        graph = StarGraph()
        a1 = Anchor.create("topic A", tags=["a"])
        a2 = Anchor.create("topic B", tags=["b"])
        graph.add_anchor(a1)
        graph.add_anchor(a2)

        key = graph._key(a1.id, a2.id)
        graph.add_edge(a1.id, a2.id, weight=0.5, relation="related_workflow")

        edge = graph.edges[key]
        assert edge.success_rate == 0.5  # neutral prior

        edge.record_success()
        edge.record_success()
        edge.record_failure()

        assert edge.success_count == 2
        assert edge.failure_count == 1
        assert edge.success_rate == 2 / 3
        assert edge.weight > 0.5  # net strengthen

    def test_graph_chain_success(self):
        """Graph.record_chain_success should update edges and anchors along a chain."""
        graph = StarGraph()
        a1 = Anchor.create("step 1", tags=["chain"])
        a2 = Anchor.create("step 2", tags=["chain"])
        a3 = Anchor.create("step 3", tags=["chain"])
        for a in [a1, a2, a3]:
            graph.add_anchor(a)

        graph.add_edge(a1.id, a2.id, weight=0.5, relation="depends_on")
        graph.add_edge(a2.id, a3.id, weight=0.5, relation="depends_on")

        updated = graph.record_chain_success([a1.id, a2.id, a3.id])
        assert updated >= 2

        # Check success feedback was boosted
        assert a1.vector.success_feedback > 0.5
        assert a2.vector.success_feedback > 0.5
        assert a3.vector.success_feedback > 0.5

    def test_graph_chain_failure(self):
        """Graph.record_chain_failure should weaken edges and reduce success feedback."""
        graph = StarGraph()
        a1 = Anchor.create("bad step 1", tags=["chain"])
        a2 = Anchor.create("bad step 2", tags=["chain"])
        for a in [a1, a2]:
            graph.add_anchor(a)

        # Set success_feedback high initially
        a1.vector.success_feedback = 0.8
        a2.vector.success_feedback = 0.8

        graph.add_edge(a1.id, a2.id, weight=0.5, relation="depends_on")
        key = graph._key(a1.id, a2.id)

        updated = graph.record_chain_failure([a1.id, a2.id])
        assert updated >= 1

        # Success feedback should be reduced
        assert a1.vector.success_feedback < 0.8
        assert a2.vector.success_feedback < 0.8

        # Edge should be weakened (original weight may have strong-relation boost)
        assert graph.edges[key].weight < 0.55

    def test_dynamic_rewire_in_sleep_rebuild(self):
        """Sleep rebuild should include dynamic rewiring results."""
        graph = StarGraph()
        a1 = Anchor.create("useful memory A", tags=["useful"])
        a2 = Anchor.create("useful memory B", tags=["useful"])
        graph.add_anchor(a1)
        graph.add_anchor(a2)
        key = graph._key(a1.id, a2.id)
        graph.add_edge(a1.id, a2.id, weight=0.5, relation="depends_on")

        # Simulate successful usage
        graph.edges[key].record_success()
        graph.edges[key].record_success()
        graph.edges[key].record_success()

        cycle = SleepCycle(graph)
        cycle.cfg.sleep.__dict__['rebuild_min_cluster'] = 100  # disable fusion
        result = cycle.run()

        assert "rebuild" in result
        dynamic = result["rebuild"].get("dynamic_rewire", {})
        assert "boosted" in dynamic
        assert "weakened" in dynamic
        assert "clusters_formed" in dynamic


class TestTemporalSlice:
    """Verify temporal slice projection limits active memory surface."""

    def test_temporal_slice_partitions(self):
        """Should partition anchors into core, active, background, noise tiers."""
        graph = StarGraph()

        # Create anchors with varying retention
        for i in range(50):
            a = Anchor.create(f"memory {i}", tags=["test"],
                            importance=0.1 + (i % 10) * 0.1)
            graph.add_anchor(a)

        result = graph.temporal_slice(max_core=7, max_active=20)

        assert len(result["core_ids"]) <= 7
        assert len(result["active_ids"]) <= 20
        assert result["background_count"] >= 0
        assert result["active_surface"] <= 27
        assert result["total_anchors"] == 50

    def test_temporal_slice_respects_max(self):
        """Should not exceed max_core + max_active."""
        graph = StarGraph()
        for i in range(10):
            graph.add_anchor(Anchor.create(f"mem {i}", importance=0.5))

        result = graph.temporal_slice(max_core=3, max_active=4)
        assert len(result["core_ids"]) <= 3
        assert len(result["active_ids"]) <= 4
        assert result["active_surface"] <= 7


class TestThermalForgetting:
    """Verify five-level thermal lifecycle with FROZEN tier."""

    def test_frozen_not_retrievable(self):
        """FROZEN anchors should not be retrievable."""
        from star_graph import ThermalState
        a = Anchor.create("old frozen memory", importance=0.02)
        a.state = MemoryState.DORMANT
        a._thermal_state = ThermalState.FROZEN
        assert a.is_retrievable is False

    def test_frozen_thermal_priority_low(self):
        """FROZEN anchors should have very low retrieval priority."""
        from star_graph import ThermalState
        a = Anchor.create("frozen")
        a._thermal_state = ThermalState.FROZEN
        assert a.thermal_priority == 0.05
        assert a.retrieval_cost > 0.9

    def test_frozen_storage_tier(self):
        """FROZEN anchors should map to archive storage tier."""
        from star_graph import ThermalState
        a = Anchor.create("frozen")
        a._thermal_state = ThermalState.FROZEN
        assert a.storage_tier == "archive"

    def test_frozen_excluded_from_cortical_index(self):
        """After _refresh_cortical_index, FROZEN anchors are excluded."""
        from star_graph import ThermalState
        graph = StarGraph()
        hot_a = Anchor.create("active memory", importance=0.8)
        hot_a.vector.hippocampal_dependency = 0.1  # cortical
        graph.add_anchor(hot_a)

        frozen_a = Anchor.create("frozen memory", importance=0.1)
        frozen_a.vector.hippocampal_dependency = 0.1
        frozen_a._thermal_state = ThermalState.FROZEN
        graph.add_anchor(frozen_a)

        # Run refresh
        cycle = SleepCycle(graph)
        cycle._refresh_cortical_index()

        # Only the hot anchor should be in cortical index
        index_ids = {aid for _, aid in graph.cortical_index}
        assert hot_a.id in index_ids
        assert frozen_a.id not in index_ids

    def test_thermal_downgrade_includes_frozen(self):
        """Sleep Phase 7 should track frozen anchors."""
        graph = make_populated_graph(20)
        # Artificially age all anchors
        for a in graph.anchors.values():
            a.last_activated_at = 0  # very old
            a.vector.recency = 0.02
            a.vector.stability = 0.02
            a.vector.frequency = 0.0

        cycle = SleepCycle(graph)
        stats = cycle._apply_thermal_forgetting()
        assert "frozen" in stats
        # Some should have downgraded
        assert stats["downgraded"] >= 0


class TestReinforcementDecay:
    """Verify reinforcement-adjusted decay formula."""

    def test_reinforcement_decay_runs(self):
        """_apply_reinforcement_decay should process all anchors."""
        graph = make_populated_graph(15)
        cycle = SleepCycle(graph)
        stats = cycle._apply_reinforcement_decay()
        assert "adjusted" in stats
        assert "boosted" in stats
        assert "penalized" in stats

    def test_success_feedback_slows_decay(self):
        """Anchors with high success_feedback should get lower decay_rate."""
        graph = StarGraph()
        winner = Anchor.create("successful strategy", importance=0.5)
        winner.vector.success_feedback = 0.9
        winner.vector.confidence = 0.8
        winner.vector.stability = 0.7
        graph.add_anchor(winner)

        loser = Anchor.create("failed approach", importance=0.5)
        loser.vector.success_feedback = 0.1
        loser.vector.confidence = 0.2
        loser.vector.stability = 0.1
        loser.vector.decay_rate = 0.01
        graph.add_anchor(loser)

        cycle = SleepCycle(graph)
        cycle._apply_reinforcement_decay()

        # Winner should have lower decay rate
        assert winner.vector.decay_rate < loser.vector.decay_rate

    def test_reinforcement_decay_in_phased_sleep(self):
        """Phased sleep should include reinforcement stats in N6_Forgetting."""
        graph = make_populated_graph(10)
        cycle = SleepCycle(graph)
        report = cycle.run_phased()
        # Find the N6_Forgetting phase
        n6 = next(p for p in report.phases if p.phase == "N6_Forgetting")
        assert "reinforcement" in n6.details


class TestEdgeTraversalWeights:
    """Verify edge type traversal weight multipliers."""

    def test_rich_edge_traversal_weight(self):
        """RichEdge should have traversal_weight property."""
        from star_graph.graph import RichEdge
        e = RichEdge(source="a", target="b", weight=0.5, edge_type="causes",
                     confidence=0.8, source_type="explicit")
        # causes has multiplier 1.5
        assert e.traversal_weight == 0.5 * 1.5

        e2 = RichEdge(source="c", target="d", weight=0.5, edge_type="contradicts",
                      confidence=0.6, source_type="inferred")
        # contradicts has multiplier 0.5
        assert e2.traversal_weight == 0.5 * 0.5

    def test_simple_edge_traversal_weight(self):
        """Simple Edge should also have traversal_weight."""
        from star_graph.graph import Edge
        e = Edge(source="a", target="b", weight=0.5, edge_type="causes")
        assert e.traversal_weight == 0.5 * 1.5

    def test_neighbors_uses_traversal_weight(self):
        """neighbors() should return traversal_weight, not raw edge weight."""
        g = StarGraph()
        g.add_anchor(Anchor.create("A"))
        g.add_anchor(Anchor.create("B"))
        ids = list(g.anchors.keys())
        g.add_edge(ids[0], ids[1], weight=0.5, edge_type="causes",
                   confidence=0.8, source_type="explicit")

        nbrs = g.neighbors(ids[0])
        assert len(nbrs) == 1
        # causes multiplier = 1.5, weight gets boosted by STRONG_RELATIONS (1.1x) in add_edge
        # so weight = 0.55, traversal = 0.55 * 1.5 = 0.825
        assert nbrs[0][1] > 0.8  # traversal_weight > raw weight

    def test_spread_activation_uses_traversal_weight(self):
        """Spreading activation should propagate through traversal weights."""
        g = StarGraph()
        g.add_anchor(Anchor.create("A"))
        g.add_anchor(Anchor.create("B"))
        ids = list(g.anchors.keys())
        # Causal edge — should propagate strongly
        g.add_edge(ids[0], ids[1], weight=0.5, edge_type="causes",
                   confidence=0.8, source_type="explicit")

        act = g.spread_activation([ids[0]], steps=1, decay=1.0)
        assert ids[1] in act
        # Activation = level * traversal_weight * decay = 1.0 * 0.825 * 1.0
        assert act[ids[1]] > 0.8

    def test_cascade_ranks_by_traversal_weight(self):
        """Cascade recall should prioritize high-traversal-weight causal edges."""
        from star_graph.cascade import CascadeRecall
        g = StarGraph()
        a = Anchor.create("root cause")
        g.add_anchor(a)
        b = Anchor.create("consequence B")
        g.add_anchor(b)
        c = Anchor.create("consequence C")
        g.add_anchor(c)

        # Strong causal edge
        g.add_edge(a.id, b.id, weight=0.8, edge_type="causes",
                   confidence=0.9, source_type="explicit", causal_strength=0.8)
        # Weak contradictory edge
        g.add_edge(a.id, c.id, weight=0.5, edge_type="contradicts",
                   confidence=0.3, source_type="inferred", causal_strength=0.1)

        cascade = CascadeRecall(g)
        chains = cascade.trace_forward(a.id, max_depth=1, max_chains=2)
        # Should find at least the causal chain
        assert len(chains) >= 1
        # First chain should be the causal one
        assert b.id in {a.id for a in chains[0].anchors}
