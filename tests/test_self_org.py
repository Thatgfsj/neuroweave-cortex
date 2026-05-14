"""Tests for SelfOrganization — auto-cluster, merge similar, detect emergent topics."""

import math
import pytest
from star_graph.self_org import SelfOrganization, EmergentTopic
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


def _make_graph(n: int = 10, connected: bool = False) -> StarGraph:
    g = StarGraph()
    for i in range(n):
        a = Anchor.create(text=f"memory item number {i} about topic {i % 3}", tags=[f"topic_{i % 3}"])
        g.add_anchor(a)
    if connected:
        ids = list(g.anchors.keys())
        for i in range(len(ids) - 1):
            g.add_edge(ids[i], ids[i + 1], weight=0.6, edge_type="topical")
    return g


class TestSelfOrganizationInit:
    def test_default_params(self):
        so = SelfOrganization()
        assert so.merge_threshold == 0.88
        assert so.cluster_similarity == 0.55
        assert so.min_cluster_size == 3
        assert so.max_topics == 30

    def test_custom_params(self):
        so = SelfOrganization(merge_threshold=0.9, cluster_similarity=0.6, min_cluster_size=5, max_topics=10)
        assert so.merge_threshold == 0.9
        assert so.cluster_similarity == 0.6
        assert so.min_cluster_size == 5
        assert so.max_topics == 10

    def test_initial_state(self):
        so = SelfOrganization()
        assert so._total_merged == 0
        assert so._total_clustered == 0
        assert len(so._topics) == 0


class TestOrganize:
    def test_empty_graph(self):
        so = SelfOrganization()
        g = StarGraph()
        report = so.organize(g)
        assert report["communities_assigned"] == 0
        assert report["topics_detected"] == 0
        assert report["merges"] == 0

    def test_organize_disconnected_anchors(self):
        so = SelfOrganization()
        g = _make_graph(10, connected=False)
        report = so.organize(g)
        assert isinstance(report, dict)
        assert "communities_assigned" in report
        assert "topics_detected" in report
        assert "merges" in report

    def test_organize_connected_anchors(self):
        so = SelfOrganization()
        g = _make_graph(5, connected=True)
        report = so.organize(g)
        assert report["communities_assigned"] >= 0

    def test_organize_returns_dict(self):
        so = SelfOrganization()
        g = _make_graph(4, connected=False)
        report = so.organize(g)
        assert set(report.keys()) == {"topics_detected", "anchors_clustered", "merges", "communities_assigned"}


class TestCommunityAssignment:
    def test_disconnected_no_communities(self):
        so = SelfOrganization()
        g = StarGraph()
        for i in range(3):
            a = Anchor.create(text=f"item {i}")
            g.add_anchor(a)
        assigned = so._auto_assign_communities(g)
        assert assigned == 0

    def test_connected_gets_communities(self):
        so = SelfOrganization()
        g = StarGraph()
        ids = []
        for i in range(4):
            a = Anchor.create(text=f"connected item {i}")
            g.add_anchor(a)
            ids.append(a.id)
        for i in range(len(ids) - 1):
            g.add_edge(ids[i], ids[i + 1], weight=0.5)
        assigned = so._auto_assign_communities(g)
        assert assigned >= 2

    def test_singleton_not_assigned(self):
        so = SelfOrganization()
        g = StarGraph()
        a1 = Anchor.create(text="connected a")
        a2 = Anchor.create(text="connected b")
        a3 = Anchor.create(text="loner")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_anchor(a3)
        g.add_edge(a1.id, a2.id, weight=0.5)
        assigned = so._auto_assign_communities(g)
        assert assigned == 2  # only the connected pair


class TestTopicDetection:
    def test_too_few_anchors(self):
        so = SelfOrganization()
        g = _make_graph(2, connected=False)
        result = so._detect_topics(g)
        assert result["topics"] == 0
        assert result["anchors"] == 0

    def test_detect_topics_from_similar(self):
        so = SelfOrganization(cluster_similarity=0.3)
        g = StarGraph()
        import numpy as np
        base = np.random.default_rng(42).random(16).tolist()
        for i in range(6):
            emb = [v + np.random.default_rng(i).random() * 0.001 for v in base]
            text = f"python programming language coding development loops functions #{i}"
            a = Anchor.create(text=text)
            a.embedding = emb
            g.add_anchor(a)
        result = so._detect_topics(g)
        assert result["topics"] >= 1

    def test_max_topics_limit(self):
        so = SelfOrganization(cluster_similarity=0.99, max_topics=1, min_cluster_size=1)
        g = StarGraph()
        import numpy as np
        for i in range(10):
            a = Anchor.create(text=f"unique content item number {i}")
            a.embedding = np.random.default_rng(i).random(16).tolist()
            g.add_anchor(a)
        result = so._detect_topics(g)
        assert result["topics"] <= 1

    def test_topic_has_keywords(self):
        so = SelfOrganization(cluster_similarity=0.3, min_cluster_size=3)
        g = StarGraph()
        import numpy as np
        base = np.random.default_rng(99).random(16).tolist()
        for i in range(5):
            emb = [v + np.random.default_rng(i + 100).random() * 0.001 for v in base]
            a = Anchor.create(text=f"machine learning neural network deep training epoch gradient #{i}")
            a.embedding = emb
            g.add_anchor(a)
        so._detect_topics(g)
        topics = so.get_topics()
        assert len(topics) >= 1
        assert len(topics[0].keywords) >= 1


class TestMergeNearDuplicates:
    def test_no_merge_below_threshold(self):
        so = SelfOrganization(merge_threshold=0.99)
        g = StarGraph()
        a1 = Anchor.create(text="different content about apples")
        a2 = Anchor.create(text="completely unrelated about zebras")
        g.add_anchor(a1)
        g.add_anchor(a2)
        merges = so._merge_near_duplicates(g)
        assert merges == 0

    def test_merge_above_threshold_with_tag_overlap(self):
        so = SelfOrganization(merge_threshold=0.3)
        g = StarGraph()
        a1 = Anchor.create(text="similar content about python", tags=["programming", "python"])
        a2 = Anchor.create(text="similar content about python coding", tags=["programming", "python"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        merges = so._merge_near_duplicates(g)
        assert merges >= 1
        assert a2.id not in g.anchors  # a2 merged into a1

    def test_no_merge_without_tag_overlap(self):
        so = SelfOrganization(merge_threshold=0.3)
        g = StarGraph()
        a1 = Anchor.create(text="similar content about python", tags=["programming"])
        a2 = Anchor.create(text="similar content about python coding", tags=["cooking"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        merges = so._merge_near_duplicates(g)
        assert merges == 0
        assert len(g.anchors) == 2


class TestQueryAPI:
    def test_get_topics_empty(self):
        so = SelfOrganization()
        assert so.get_topics() == []

    def test_get_topics_min_coherence(self):
        so = SelfOrganization()
        topic = EmergentTopic(name="test", keywords=["a", "b"], coherence=0.5)
        so._topics["test"] = topic
        assert len(so.get_topics(min_coherence=0.6)) == 0
        assert len(so.get_topics(min_coherence=0.4)) == 1

    def test_get_topic_anchors(self):
        so = SelfOrganization()
        topic = EmergentTopic(name="test", keywords=["a"], anchor_ids={"a1", "a2"})
        so._topics["test"] = topic
        assert so.get_topic_anchors("test") == {"a1", "a2"}
        assert so.get_topic_anchors("nonexistent") == set()


class TestStats:
    def test_initial_stats(self):
        so = SelfOrganization()
        s = so.stats
        assert s["topics_detected"] == 0
        assert s["total_merged"] == 0
        assert s["total_clustered"] == 0
        assert s["topics"] == {}

    def test_stats_after_organize(self):
        so = SelfOrganization(cluster_similarity=0.3, min_cluster_size=3)
        g = StarGraph()
        for i in range(5):
            a = Anchor.create(text="deep learning neural network training ai model")
            g.add_anchor(a)
        so.organize(g)
        s = so.stats
        assert s["topics_detected"] >= 0
