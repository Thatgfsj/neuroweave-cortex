"""Tests for Cluster Memory — retrieval-integrated community pre-filtering."""

import pytest
import numpy as np
from star_graph.cluster_memory import ClusterRouter, ClusterCentroid
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


def _make_graph_with_communities(n_communities: int = 3, per_community: int = 5) -> StarGraph:
    g = StarGraph()
    rng = np.random.default_rng(42)
    for c in range(n_communities):
        base = rng.random(16).tolist()
        community_id = f"comm_{c}"
        for i in range(per_community):
            emb = [v + rng.random() * 0.01 for v in base]
            a = Anchor.create(text=f"community {c} item {i}")
            a.embedding = emb
            a.community_id = community_id
            g.add_anchor(a)
    return g


class TestClusterRouterInit:
    def test_defaults(self):
        cr = ClusterRouter()
        assert cr.min_cluster_size == 5
        assert cr.max_clusters == 20
        assert cr.similarity_threshold == 0.4

    def test_custom(self):
        cr = ClusterRouter(min_cluster_size=3, max_clusters=10, similarity_threshold=0.5)
        assert cr.min_cluster_size == 3
        assert cr.max_clusters == 10


class TestBuildIndex:
    def test_empty_graph(self):
        cr = ClusterRouter(min_cluster_size=2)
        g = StarGraph()
        count = cr.build_index(g)
        assert count == 0

    def test_build_from_communities(self):
        cr = ClusterRouter(min_cluster_size=2)
        g = _make_graph_with_communities(3, 5)
        count = cr.build_index(g)
        assert count >= 1

    def test_anchor_to_cluster_mapping(self):
        cr = ClusterRouter(min_cluster_size=2)
        g = _make_graph_with_communities(2, 5)
        cr.build_index(g)
        for aid in g.anchors:
            cluster_id = cr.get_anchor_cluster(aid)
            if cluster_id:
                scope = cr.get_cluster_scope(cluster_id)
                assert aid in scope


class TestRouting:
    def test_route_empty(self):
        cr = ClusterRouter()
        result = cr.route([0.1] * 16)
        assert result == []

    def test_route_finds_nearest(self):
        cr = ClusterRouter(min_cluster_size=2, similarity_threshold=0.0)
        g = _make_graph_with_communities(3, 5)
        cr.build_index(g)
        # Query near the first community's centroid
        result = cr.route([0.5] * 16)
        assert len(result) >= 1
        cluster_id, score = result[0]
        assert isinstance(cluster_id, str)
        assert 0.0 <= score <= 1.0


class TestClusterInfo:
    def test_get_cluster_info(self):
        cr = ClusterRouter(min_cluster_size=2)
        g = _make_graph_with_communities(1, 5)
        cr.build_index(g)
        clusters = list(cr._clusters.keys())
        if clusters:
            info = cr.get_cluster_info(clusters[0])
            assert "label" in info
            assert "size" in info
            assert info["size"] == 5
            assert "keywords" in info

    def test_get_nonexistent(self):
        cr = ClusterRouter()
        assert cr.get_cluster_info("nonexistent") == {}


class TestStats:
    def test_initial_stats(self):
        cr = ClusterRouter()
        s = cr.stats
        assert s["total_clusters"] == 0
        assert "clusters" in s

    def test_stats_after_build(self):
        cr = ClusterRouter(min_cluster_size=2)
        g = _make_graph_with_communities(2, 5)
        cr.build_index(g)
        s = cr.stats
        assert s["total_clusters"] >= 1
        assert s["total_anchors_indexed"] >= 5
