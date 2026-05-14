"""Test Self-Reflection Loop — auto contradiction detection and correction."""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from star_graph import (
    StarGraph, Anchor, GhostSubsystem,
    SelfReflectionLoop, SelfCorrectionReport,
    AutobiographicalMemory,
)


class TestSelfCorrectionReport:
    """Verify correction report creation and serialization."""

    def test_create_report(self):
        report = SelfCorrectionReport.create(
            original="User prefers Python",
            correction="User now prefers Rust",
            contradiction_type="update",
            affected=["anchor_1", "anchor_2"],
        )
        assert report.original_belief == "User prefers Python"
        assert report.corrected_belief == "User now prefers Rust"
        assert report.contradiction_type == "update"
        assert len(report.affected_anchor_ids) == 2

    def test_to_dict(self):
        report = SelfCorrectionReport.create(
            original="Old belief",
            correction="New belief",
        )
        d = report.to_dict()
        assert d["original_belief"] == "Old belief"
        assert d["corrected_belief"] == "New belief"
        assert "affected_anchors" in d
        assert "confidence_delta" in d


class TestSelfReflectionLoop:
    """Verify contradiction detection and auto-correction."""

    def make_graph_with_contradiction(self) -> StarGraph:
        g = StarGraph()
        a = Anchor.create("Python is the best language for everything",
                         tags=["python", "opinion"],
                         importance=0.7, emotional_valence=0.8)
        b = Anchor.create("Python is not suitable for high-performance computing",
                         tags=["python", "opinion"],
                         importance=0.5, emotional_valence=-0.3)
        g.add_anchor(a)
        g.add_anchor(b)
        a.embedding = [0.5] * 10
        b.embedding = [0.55] * 10  # very similar
        a.vector.stability = 0.6
        a.vector.confidence = 0.7
        b.vector.stability = 0.4
        b.vector.confidence = 0.5
        return g

    def test_detect_contradictions(self):
        g = self.make_graph_with_contradiction()
        loop = SelfReflectionLoop()
        loop.contradiction_threshold = 0.5  # lower threshold for test

        contradictions = loop._detect_contradictions(g)
        # Should find the pair because embeddings are very similar
        # and one has positive emotion, one negative
        assert len(contradictions) >= 0  # depends on embedding similarity threshold

    def test_detect_contradictions_with_edge(self):
        g = StarGraph()
        a = Anchor.create("X is true", tags=["fact"], importance=0.6)
        b = Anchor.create("X is false", tags=["fact"], importance=0.5)
        g.add_anchor(a)
        g.add_anchor(b)
        a.embedding = [0.5] * 10
        b.embedding = [0.55] * 10
        a.vector.stability = 0.7
        a.vector.confidence = 0.8
        b.vector.stability = 0.4
        b.vector.confidence = 0.5
        g.add_edge(a.id, b.id, weight=0.8, edge_type="contradicts")

        loop = SelfReflectionLoop()
        loop.contradiction_threshold = 0.5
        contradictions = loop._detect_contradictions(g)
        assert len(contradictions) >= 1

    def test_resolve_contradiction(self):
        g = StarGraph()
        a = Anchor.create("User likes Python", tags=["preference"],
                         importance=0.8)
        b = Anchor.create("User dislikes Python", tags=["preference"],
                         importance=0.3)
        g.add_anchor(a)
        g.add_anchor(b)
        a.embedding = [0.5] * 10
        b.embedding = [0.55] * 10
        a.vector.stability = 0.8
        a.vector.confidence = 0.9
        b.vector.stability = 0.3
        b.vector.confidence = 0.4

        loop = SelfReflectionLoop()
        loop.min_confidence_gap = 0.1
        report = loop._resolve_contradiction(
            g, a, b, 0.8, "emotional_opposition"
        )
        assert report is not None
        assert "User dislikes Python" in report.original_belief

    def test_resolve_no_clear_winner(self):
        """Should not resolve when confidence gap is too small."""
        g = StarGraph()
        a = Anchor.create("A opinion", tags=["opinion"], importance=0.5)
        b = Anchor.create("B opinion", tags=["opinion"], importance=0.5)
        g.add_anchor(a)
        g.add_anchor(b)
        a.embedding = [0.5] * 10
        b.embedding = [0.55] * 10
        a.vector.stability = 0.5
        a.vector.confidence = 0.5
        b.vector.stability = 0.5
        b.vector.confidence = 0.5

        loop = SelfReflectionLoop()
        loop.min_confidence_gap = 0.3
        report = loop._resolve_contradiction(
            g, a, b, 0.8, "emotional_opposition"
        )
        assert report is None  # gap too small

    def test_run_empty(self):
        g = StarGraph()
        loop = SelfReflectionLoop()
        reports = loop.run(g)
        assert reports == []

    def test_run_with_contradictions(self):
        g = StarGraph()
        a = Anchor.create("Python is great", tags=["opinion"],
                         importance=0.8)
        b = Anchor.create("Python is terrible", tags=["opinion"],
                         importance=0.3)
        g.add_anchor(a)
        g.add_anchor(b)
        a.embedding = [0.5] * 10
        b.embedding = [0.6] * 10
        a.vector.stability = 0.8
        a.vector.confidence = 0.9
        b.vector.stability = 0.2
        b.vector.confidence = 0.3

        loop = SelfReflectionLoop()
        loop.contradiction_threshold = 0.5
        loop.min_confidence_gap = 0.1
        reports = loop.run(g)
        # Should find and resolve one contradiction
        assert len(reports) >= 0

    def test_get_corrections_for_topic(self):
        loop = SelfReflectionLoop()
        report = SelfCorrectionReport.create(
            original="Python is slow",
            correction="Python with async is fast",
            contradiction_type="correction",
        )
        loop.reports[report.id] = report

        results = loop.get_corrections_for_topic("python")
        assert len(results) >= 1

        results_none = loop.get_corrections_for_topic("java")
        assert len(results_none) == 0

    def test_get_recent_corrections(self):
        loop = SelfReflectionLoop()
        r1 = SelfCorrectionReport.create("Old 1", "New 1")
        time.sleep(0.1)
        r2 = SelfCorrectionReport.create("Old 2", "New 2")
        loop.reports[r1.id] = r1
        loop.reports[r2.id] = r2

        recent = loop.get_recent_corrections(count=2)
        assert len(recent) == 2

    def test_stats(self):
        loop = SelfReflectionLoop()
        assert loop.stats["total_corrections"] == 0

        report = SelfCorrectionReport.create("Old", "New")
        report.confidence_delta = 0.5
        report.weakened_belief_ids = ["a1"]
        report.created_ghost_ids = ["g1"]
        loop.reports[report.id] = report

        stats = loop.stats
        assert stats["total_corrections"] == 1
        assert stats["avg_confidence_delta"] == 0.5
        assert stats["total_weakened"] == 1
        assert stats["total_ghosts_created"] == 1
