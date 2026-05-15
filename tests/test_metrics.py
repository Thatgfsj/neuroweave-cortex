"""Tests for CognitiveMetrics — system health beyond retrieval precision."""

import time

import pytest

from star_graph.anchor import Anchor, AnchorVector
from star_graph.config import Config
from star_graph.graph import StarGraph
from star_graph.metrics import CognitiveMetrics


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                retention: float = 0.5, importance: float = 0.5,
                stability: float = 0.5, created_at: float = 0.0,
                state=None) -> Anchor:
    a = Anchor(
        id=name, text=text or f"Memory {name}",
        vector=AnchorVector(importance=importance, stability=stability),
        created_at=created_at or time.time(),
    )
    if embedding:
        a.embedding = embedding
    if state:
        a.state = state
    return a


class TestCognitiveMetricsInit:
    def test_initializes_with_graph(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        assert cm.graph is g
        assert cm._snapshots == []


class TestCognitiveMetricsSnapshot:
    def test_empty_graph(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        snap = cm.snapshot()
        assert snap["anchors"] == 0
        assert "timestamp" in snap

    def test_snapshot_with_anchors(self):
        g = StarGraph()
        for i in range(5):
            a = make_anchor(f"a{i}", embedding=[0.1 * j for j in range(10)])
            g.add_anchor(a)

        cm = CognitiveMetrics(g)
        snap = cm.snapshot()
        assert snap["anchors"] == 5
        assert snap["edges"] == 0
        assert "memory_stability" in snap
        assert "recall_plasticity" in snap
        assert "compression_ratio" in snap
        assert "semantic_drift_resistance" in snap
        assert "abstraction_emergence_rate" in snap
        assert "ghost_reactivation_accuracy" in snap

    def test_snapshot_stored_in_history(self):
        g = StarGraph()
        a = make_anchor("a1")
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        cm.snapshot()
        assert len(cm._snapshots) == 1

    def test_multiple_snapshots(self):
        g = StarGraph()
        a = make_anchor("a1")
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        cm.snapshot()
        cm.snapshot()
        assert len(cm._snapshots) == 2


class TestCognitiveMetricsStability:
    def test_empty_graph_stability(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        anchors = list(g.anchors.values())
        assert cm._memory_stability(anchors) == 0.0

    def test_single_anchor_stability(self):
        g = StarGraph()
        a = make_anchor("a1", stability=0.8)
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        anchors = list(g.anchors.values())
        stability = cm._memory_stability(anchors)
        assert 0.0 < stability <= 1.0


class TestCognitiveMetricsPlasticity:
    def test_empty_graph_plasticity(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        anchors = list(g.anchors.values())
        assert cm._recall_plasticity(anchors) == 0.0

    def test_recent_anchors_plasticity(self):
        g = StarGraph()
        a = make_anchor("a1", created_at=time.time())  # recent
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        anchors = list(g.anchors.values())
        plasticity = cm._recall_plasticity(anchors)
        assert 0.0 <= plasticity <= 1.0


class TestCognitiveMetricsCompression:
    def test_empty_graph_compression(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        assert cm._compression_ratio() == 0.0

    def test_with_anchors_only(self):
        g = StarGraph()
        a = make_anchor("a1")
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        assert cm._compression_ratio() == 0.0


class TestCognitiveMetricsDrift:
    def test_empty_anchors_drift(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        assert cm._semantic_drift_resistance([]) == 1.0

    def test_no_old_anchors_drift(self):
        g = StarGraph()
        a = make_anchor("a1", created_at=time.time(), stability=0.7)
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        anchors = list(g.anchors.values())
        assert cm._semantic_drift_resistance(anchors) == 1.0


class TestCognitiveMetricsAbstraction:
    def test_no_snapshots_zero(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        assert cm._abstraction_emergence_rate() == 0.0


class TestCognitiveMetricsGhostAccuracy:
    def test_no_ghosts_returns_one(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        assert cm._ghost_reactivation_accuracy() == 1.0

    def test_with_ghosts_no_revivals(self):
        from star_graph.ghost import GhostNode
        g = StarGraph()
        ghost = GhostNode(
            id="g1",
            compressed_embedding=[0.1, 0.2],
            residual_edges={},
            emotion_trace=0.0,
            pruned_at=time.time(),
            original_tags=[],
            original_importance=0.5,
            semantic_shadow="test shadow",
        )
        g._ghost_subsystem.ghosts["g1"] = ghost
        cm = CognitiveMetrics(g)
        assert cm._ghost_reactivation_accuracy() == 0.0

    def test_with_revived_ghosts(self):
        from star_graph.ghost import GhostNode
        g = StarGraph()
        ghost = GhostNode(
            id="g1",
            compressed_embedding=[0.1, 0.2],
            residual_edges={},
            emotion_trace=0.0,
            pruned_at=time.time(),
            original_tags=[],
            original_importance=0.5,
            semantic_shadow="test shadow",
            revival_count=3,
        )
        g._ghost_subsystem.ghosts["g1"] = ghost
        cm = CognitiveMetrics(g)
        accuracy = cm._ghost_reactivation_accuracy()
        assert accuracy > 0.0


class TestCognitiveMetricsCompare:
    def test_insufficient_snapshots(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        assert cm.compare() == {}

    def test_compare_with_two_snapshots(self):
        g = StarGraph()
        a = make_anchor("a1")
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        cm.snapshot()
        b = make_anchor("a2")
        g.add_anchor(b)
        cm.snapshot()
        result = cm.compare()
        assert "anchor_change" in result
        assert result["snapshots_collected"] == 2


class TestCognitiveMetricsReport:
    def test_report_no_snapshots(self):
        g = StarGraph()
        cm = CognitiveMetrics(g)
        result = cm.report()
        assert "No data" in result

    def test_report_with_snapshot(self):
        g = StarGraph()
        a = make_anchor("a1")
        g.add_anchor(a)
        cm = CognitiveMetrics(g)
        cm.snapshot()
        result = cm.report()
        assert "Cognitive Health Report" in result
        assert "Memory Stability" in result
        assert "Recall Plasticity" in result
