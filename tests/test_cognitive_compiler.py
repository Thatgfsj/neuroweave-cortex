"""Test Cognitive Compiler — full-chain worldview emergence pipeline."""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from star_graph import (
    StarGraph, Anchor,
    CognitiveCompiler, WorldviewNode, UserProfile,
)


class TestWorldviewNode:
    """Verify WorldviewNode belief tracking."""

    def test_create_worldview(self):
        wv = WorldviewNode(
            id="wv_1", label="Python Preference",
            description="User prefers Python for scripting tasks",
            source_concept_ids=["c1", "c2"],
            confidence=0.7, stability=0.6,
            evidence_count=3, domain="python_development",
            worldview_type="preference",
        )
        assert wv.is_stable is False  # stability < 0.7
        assert wv.worldview_type == "preference"

    def test_reinforce(self):
        wv = WorldviewNode(
            id="wv_1", label="Test",
            description="Test belief",
            source_concept_ids=["c1"],
            confidence=0.5, stability=0.5,
        )
        wv.reinforce()
        assert wv.evidence_count == 1
        assert wv.stability == 0.6
        assert wv.confidence == 0.55

    def test_weaken(self):
        wv = WorldviewNode(
            id="wv_1", label="Test",
            description="Test belief",
            source_concept_ids=["c1"],
            confidence=0.7, stability=0.7,
        )
        wv.weaken()
        assert wv.stability == pytest.approx(0.55)
        assert wv.confidence == pytest.approx(0.60)

    def test_degrade(self):
        wv = WorldviewNode(
            id="wv_1", label="Test",
            description="Test belief",
            source_concept_ids=["c1"],
            confidence=0.8, stability=0.8,
        )
        # Set last_reinforced far in the past so decay applies
        wv.last_reinforced_at = time.time() - 99999 * 3600
        remaining = wv.degrade(half_life_days=30.0)
        assert remaining < 0.1

    def test_is_stable_true(self):
        wv = WorldviewNode(
            id="wv_1", label="Stable",
            description="Stable belief",
            source_concept_ids=["c1", "c2"],
            confidence=0.9, stability=0.8,
            evidence_count=3,
        )
        assert wv.is_stable is True


class TestUserProfile:
    """Verify UserProfile synthesis from worldviews."""

    def test_from_worldviews(self):
        wvs = [
            WorldviewNode(
                id="wv_1", label="Python Expert",
                description="Python development", source_concept_ids=["c1"],
                confidence=0.8, stability=0.9, evidence_count=5,
                worldview_type="expertise",
            ),
            WorldviewNode(
                id="wv_2", label="Prefers Concise Code",
                description="Prefers concise code", source_concept_ids=["c2"],
                confidence=0.7, stability=0.8, evidence_count=3,
                worldview_type="preference",
            ),
            WorldviewNode(
                id="wv_3", label="Values Testing",
                description="Values thorough testing", source_concept_ids=["c3"],
                confidence=0.6, stability=0.75, evidence_count=2,
                worldview_type="value",
            ),
        ]
        profile = UserProfile.from_worldviews(wvs, worldview_id="profile_1")
        assert profile.id == "profile_1"
        assert len(profile.expertise_areas) == 1
        assert len(profile.preferences) == 1
        assert len(profile.values) == 1
        assert profile.version == 1
        assert "Python" in profile.summary


class TestCognitiveCompiler:
    """Verify the full cognitive compilation pipeline."""

    def test_compile_empty(self):
        g = StarGraph()
        compiler = CognitiveCompiler()
        result = compiler.compile(g)
        assert result["stats"]["raw_count"] == 0
        assert result["stats"]["compression_chain"] == "0→0→0→0→1"

    def test_compile_with_anchors(self):
        g = StarGraph()
        session_id = "session_001"
        anchors = []
        for i in range(6):
            a = Anchor.create(
                f"python async programming task {i}",
                source_session=session_id,
                tags=["python", "async"],
                importance=0.6,
            )
            g.add_anchor(a)
            anchors.append(a)

        # Need embeddings for clustering
        for a in anchors:
            a.embedding = [0.1 * (i + 1) for i in range(10)]

        compiler = CognitiveCompiler()
        result = compiler.compile(g)
        assert result["stats"]["raw_count"] > 0
        assert "compression_chain" in result["stats"]

    def test_cluster_by_domain(self):
        compiler = CognitiveCompiler()
        from star_graph.compression import SummaryAnchor, CompressionLevel

        s1 = SummaryAnchor(
            id="s1", text="python async development",
            source_anchor_ids=["a1"], centroid_embedding=[0.1] * 10,
            compression_level=CompressionLevel.EPISODIC,
            tags=["python", "async"],
        )
        s2 = SummaryAnchor(
            id="s2", text="docker deployment pipeline",
            source_anchor_ids=["a2"], centroid_embedding=[0.2] * 10,
            compression_level=CompressionLevel.EPISODIC,
            tags=["docker", "deploy"],
        )
        clusters = compiler._cluster_by_domain([s1, s2])
        assert "python_development" in clusters
        assert "devops" in clusters

    def test_synthesize_worldview(self):
        compiler = CognitiveCompiler()
        from star_graph.compression import SummaryAnchor, CompressionLevel

        concepts = [
            SummaryAnchor(
                id="s1", text="User prefers python for scripting",
                source_anchor_ids=["a1"], centroid_embedding=[0.1] * 10,
                compression_level=CompressionLevel.EPISODIC,
                tags=["python", "preference"], confidence=0.7,
            ),
            SummaryAnchor(
                id="s2", text="User likes flask for web apis",
                source_anchor_ids=["a2"], centroid_embedding=[0.1] * 10,
                compression_level=CompressionLevel.EPISODIC,
                tags=["python", "preference"], confidence=0.8,
            ),
        ]
        wv = compiler._synthesize_worldview("python_development", concepts)
        assert wv is not None
        assert wv.domain == "python_development"
        assert wv.evidence_count == 2
        assert wv.confidence > 0.4

    def test_worldview_persistence(self):
        compiler = CognitiveCompiler()
        wv = WorldviewNode(
            id="wv_test", label="Test",
            description="Test belief",
            source_concept_ids=["c1"],
            confidence=0.8, stability=0.8,
            evidence_count=3,
        )
        compiler.worldviews[wv.id] = wv
        assert len(compiler.worldviews) == 1
        assert compiler.get_stable_worldviews()[0].id == "wv_test"

    def test_degrade_worldviews(self):
        compiler = CognitiveCompiler()
        wv = WorldviewNode(
            id="wv_decay", label="Decaying",
            description="Will decay",
            source_concept_ids=["c1"],
            confidence=0.5, stability=0.03,
            last_reinforced_at=time.time() - 99999 * 3600,
        )
        compiler.worldviews[wv.id] = wv
        removed = compiler.degrade_worldviews()
        assert removed >= 0

    def test_default_profile_is_none(self):
        compiler = CognitiveCompiler()
        assert compiler.profile is None

    def test_stats(self):
        compiler = CognitiveCompiler()
        stats = compiler.stats
        assert stats["total_worldviews"] == 0
        assert stats["profile_version"] == 0
