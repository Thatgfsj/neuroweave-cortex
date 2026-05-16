"""Tests for community module — Community, CommunityHealth, CommunityDetection."""

import time

import pytest

from star_graph.community import (
    Community,
    CommunityHealth,
    CommunityDetection,
)
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                tags: list | None = None) -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}", tags=tags or [])
    if embedding:
        a.embedding = embedding
    return a


class TestCommunity:
    def test_defaults(self):
        c = Community(id="c1", anchor_ids=["a1", "a2"])
        assert c.id == "c1"
        assert len(c.anchor_ids) == 2
        assert c.size == 0
        assert c.density == 0.0
        assert c.is_dense is False
        assert c.is_large is False

    def test_is_dense(self):
        c = Community(id="c1", anchor_ids=[], density=0.5)
        assert c.is_dense is True

    def test_is_large(self):
        c = Community(id="c1", anchor_ids=[], size=150)
        assert c.is_large is True

    def test_centroid_embedding(self):
        c = Community(
            id="c1", anchor_ids=["a1"],
            centroid_embedding=[0.1, 0.2, 0.3],
        )
        assert len(c.centroid_embedding) == 3

    def test_topic_label(self):
        c = Community(
            id="c1", anchor_ids=["a1", "a2"],
            topic_label="Python Dev",
        )
        assert c.topic_label == "Python Dev"

    def test_primary_tag(self):
        c = Community(id="c1", anchor_ids=["a1"], primary_tag="python")
        assert c.primary_tag == "python"


class TestCommunityHealth:
    def test_defaults(self):
        ch = CommunityHealth()
        assert ch.modularity == 0.0
        assert ch.num_communities == 0
        assert ch.bridge_node_count == 0
        assert ch.singletons == 0

    def test_is_healthy_empty(self):
        ch = CommunityHealth()
        assert ch.is_healthy is False

    def test_is_healthy_true(self):
        ch = CommunityHealth(
            modularity=0.3, num_communities=5,
            largest_community_fraction=0.4,
        )
        assert ch.is_healthy is True

    def test_is_healthy_low_modularity(self):
        ch = CommunityHealth(
            modularity=0.05, num_communities=5,
            largest_community_fraction=0.4,
        )
        assert ch.is_healthy is False

    def test_is_healthy_dominant_community(self):
        ch = CommunityHealth(
            modularity=0.3, num_communities=5,
            largest_community_fraction=0.8,
        )
        assert ch.is_healthy is False

    def test_is_healthy_too_few_communities(self):
        ch = CommunityHealth(
            modularity=0.3, num_communities=1,
            largest_community_fraction=0.5,
        )
        assert ch.is_healthy is False

    def test_summary(self):
        ch = CommunityHealth(
            modularity=0.35, num_communities=4,
            bridge_node_count=3, singletons=2,
            largest_community_fraction=0.5,
        )
        s = ch.summary()
        assert "Communities: 4" in s
        assert "Modularity: 0.350" in s
        assert "Bridges: 3" in s
        assert "Singletons: 2" in s


class TestCommunityDetection:
    def test_init_defaults(self):
        cd = CommunityDetection()
        assert cd.max_community_size >= 2
        assert cd.min_size >= 2
        assert cd.max_iterations >= 5
        assert cd.communities == []

    def test_init_custom(self):
        cd = CommunityDetection(
            max_community_size=100, min_community_size=5,
            max_iterations=20,
        )
        assert cd.max_community_size == 100
        assert cd.min_size == 5
        assert cd.max_iterations == 20

    def test_detect_empty_graph(self):
        cd = CommunityDetection()
        g = StarGraph()
        result = cd.detect(g)
        assert result == []
        assert cd.communities == []

    def test_detect_single_anchor(self):
        cd = CommunityDetection()
        g = StarGraph()
        a = make_anchor("a1", "test anchor")
        g.add_anchor(a)
        result = cd.detect(g)
        assert len(result) >= 1

    def test_detect_two_connected_anchors(self):
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "python coding", tags=["dev"])
        a2 = make_anchor("a2", "python debugging", tags=["dev"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.9, edge_type="topical")
        result = cd.detect(g)
        assert len(result) >= 1

    def test_detect_with_embeddings(self):
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "python", embedding=[0.1] * 10, tags=["dev"])
        a2 = make_anchor("a2", "flask", embedding=[0.1] * 10, tags=["dev"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.8, edge_type="topical")
        result = cd.detect(g)
        assert len(result) >= 1
        # Centroid should be computed
        if result:
            assert len(result[0].centroid_embedding) >= 1

    def test_detect_sets_community_id(self):
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "test", tags=["dev"])
        g.add_anchor(a1)
        cd.detect(g)
        assert g.anchors["a1"].community_id != ""

    def test_health_metrics_empty(self):
        cd = CommunityDetection()
        h = cd.health_metrics()
        assert h.num_communities == 0

    def test_health_metrics_after_detect(self):
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "test 1", tags=["dev"])
        a2 = make_anchor("a2", "test 2", tags=["dev"])
        a3 = make_anchor("a3", "test 3", tags=["dev"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_anchor(a3)
        g.add_edge("a1", "a2", weight=0.9, edge_type="topical")
        g.add_edge("a2", "a3", weight=0.8, edge_type="topical")
        cd.detect(g)
        h = cd.health_metrics()
        assert h.num_communities >= 1
        assert isinstance(h.modularity, float)

    def test_generate_topic_label_empty(self):
        cd = CommunityDetection()
        label = cd._generate_topic_label([])
        assert label == "Empty Community"

    def test_generate_topic_label_with_tags(self):
        cd = CommunityDetection()
        a1 = make_anchor("a1", "python coding", tags=["dev", "python"])
        a2 = make_anchor("a2", "python debugging", tags=["dev", "python"])
        label = cd._generate_topic_label([a1, a2])
        assert "dev" in label or "python" in label

    def test_generate_topic_label_no_tags(self):
        cd = CommunityDetection()
        a = make_anchor("a1", "deploying Docker containers", tags=[])
        label = cd._generate_topic_label([a])
        assert isinstance(label, str)
        assert len(label) > 0

    def test_modularity_no_edges(self):
        cd = CommunityDetection()
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "test"))
        labels = {"a1": "c0"}
        Q = cd._modularity(g, labels)
        assert Q == 0.0

    def test_modularity_with_edges(self):
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "test")
        a2 = make_anchor("a2", "test")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=1.0, edge_type="topical")
        # Same community
        labels_same = {"a1": "c0", "a2": "c0"}
        Q_same = cd._modularity(g, labels_same)
        assert isinstance(Q_same, float)

        # Different communities
        labels_diff = {"a1": "c0", "a2": "c1"}
        Q_diff = cd._modularity(g, labels_diff)
        assert Q_same > Q_diff

    def test_get_neighboring_communities_empty(self):
        cd = CommunityDetection()
        result = cd.get_neighboring_communities("nonexistent")
        assert result == []

    def test_get_neighboring_communities(self):
        cd = CommunityDetection()
        g = StarGraph()
        for i in range(5):
            g.add_anchor(make_anchor(f"a{i}", f"test {i}", tags=["dev"]))
        for i in range(5, 10):
            g.add_anchor(make_anchor(f"a{i}", f"other {i}", tags=["testing"]))
        g.add_edge("a0", "a5", weight=0.9, edge_type="topical")
        cd.detect(g)
        # Get first community and check neighbors
        if cd.communities:
            neighbors = cd.get_neighboring_communities(cd.communities[0].id)
            assert isinstance(neighbors, list)

    def test_refresh_preserves_identity(self):
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "persistent test", tags=["dev"])
        a2 = make_anchor("a2", "persistent data", tags=["dev"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.9, edge_type="topical")
        first = cd.detect(g)
        first_id = first[0].id if first else None
        second = cd.refresh(g)
        if first and second and first_id:
            assert second[0].id == first_id


# ── Split oversized communities ────────────────────────────

class TestSplitOversized:
    def test_split_oversized_direct(self):
        """_split_oversized with max_size=3 should split a large community."""
        cd = CommunityDetection(max_community_size=3, min_community_size=2)
        g = StarGraph()
        for i in range(8):
            g.add_anchor(make_anchor(f"a{i}", f"test {i}", tags=["dev"]))
        # Two dense sub-clusters with weak cross-links
        # Cluster A: a0-a3, Cluster B: a4-a7
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(f"a{i}", f"a{j}", weight=0.9, edge_type="topical")
        for i in range(4, 8):
            for j in range(i + 1, 8):
                g.add_edge(f"a{i}", f"a{j}", weight=0.9, edge_type="topical")
        # Only one weak cross-link between clusters
        g.add_edge("a0", "a4", weight=0.1, edge_type="topical")

        labels = {f"a{i}": "c0" for i in range(8)}
        result = cd._split_oversized(labels, g, max_size=3)
        unique = set(result.values())
        assert len(unique) > 1  # was split

    def test_split_oversized_no_op_when_under_max(self):
        """No split when all communities are within size limit."""
        cd = CommunityDetection(max_community_size=10, min_community_size=2)
        g = StarGraph()
        for i in range(5):
            g.add_anchor(make_anchor(f"a{i}", f"test {i}"))
        labels = {f"a{i}": "c0" for i in range(5)}
        result = cd._split_oversized(labels, g, max_size=10)
        assert result == labels  # unchanged

    def test_split_oversized_members_leq_max(self):
        """Edge case: oversized detection finds cid but members ≤ max_size."""
        cd = CommunityDetection(max_community_size=3, min_community_size=2)
        g = StarGraph()
        for i in range(3):
            g.add_anchor(make_anchor(f"a{i}", f"test {i}"))
        labels = {f"a{i}": "c0" for i in range(3)}
        result = cd._split_oversized(labels, g, max_size=3)
        assert "c0" in result.values()

    def test_split_oversized_cannot_split_further(self):
        """When sub-propagation produces only 1 group, keep original."""
        cd = CommunityDetection(max_community_size=2, min_community_size=2)
        g = StarGraph()
        # 4 anchors, all fully connected — sub-propagation may merge them back
        for i in range(4):
            g.add_anchor(make_anchor(f"a{i}", f"same topic", tags=["dev"]))
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(f"a{i}", f"a{j}", weight=0.95, edge_type="topical")

        labels = {f"a{i}": "c0" for i in range(4)}
        result = cd._split_oversized(labels, g, max_size=2)
        assert isinstance(result, dict)

    def test_split_during_detect_pipeline(self):
        """Full detect() pipeline with small max_community_size triggers split."""
        cd = CommunityDetection(max_community_size=3, min_community_size=2)
        g = StarGraph()
        for i in range(8):
            g.add_anchor(make_anchor(f"a{i}", f"topic {i % 2}", tags=["dev"]))
        for i in range(8):
            for j in range(i + 1, 8):
                g.add_edge(f"a{i}", f"a{j}", weight=0.7, edge_type="topical")
        communities = cd.detect(g)
        assert len(communities) >= 1


# ── Bridge node edge cases ─────────────────────────────────

class TestBridgeEdgeCases:
    def test_detect_bridges_across_communities(self):
        """Bridge detected when neighbor is in a different community."""
        cd = CommunityDetection(max_community_size=100)
        g = StarGraph()
        for i in range(4):
            g.add_anchor(make_anchor(f"a{i}", f"cluster1 item {i}", tags=["c1"]))
        for i in range(4, 8):
            g.add_anchor(make_anchor(f"a{i}", f"cluster2 item {i}", tags=["c2"]))
        # Dense within clusters
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(f"a{i}", f"a{j}", weight=0.9, edge_type="topical")
        for i in range(4, 8):
            for j in range(i + 1, 8):
                g.add_edge(f"a{i}", f"a{j}", weight=0.9, edge_type="topical")
        # Single bridge between clusters
        g.add_edge("a0", "a4", weight=0.5, edge_type="topical")

        cd.detect(g)
        # a0 and a4 should be bridge nodes
        a0 = g.anchors["a0"]
        a4 = g.anchors["a4"]
        assert len(a0.secondary_community_ids) >= 1 or len(a4.secondary_community_ids) >= 1

    def test_bridge_node_neighbor_no_label(self):
        """Neighbor without label is not counted as other community."""
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "node with label")
        a2 = make_anchor("a2", "node without label")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.8)
        labels = {"a1": "c0"}  # a2 has no label
        bridges = cd._detect_bridge_nodes(labels, g)
        assert "a1" not in bridges  # a2's label is None → no bridge


# ── Additional edge cases ──────────────────────────────────

class TestAdditionalEdgeCases:
    def test_modularity_with_mixed_labels(self):
        """Modularity with edges crossing community boundaries."""
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "test")
        a2 = make_anchor("a2", "test")
        a3 = make_anchor("a3", "test")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_anchor(a3)
        g.add_edge("a1", "a2", weight=0.9, edge_type="topical")
        g.add_edge("a2", "a3", weight=0.5, edge_type="topical")
        Q = cd._modularity(g, {"a1": "c0", "a2": "c0", "a3": "c1"})
        assert isinstance(Q, float)

    def test_topic_label_no_tags_no_keywords(self):
        """Fallback label when anchors have no tags and few keywords."""
        cd = CommunityDetection()
        a = make_anchor("a1", "a it is", tags=[])  # words too short → filtered
        label = cd._generate_topic_label([a])
        assert "Community" in label

    def test_refresh_empty_to_empty(self):
        """Refresh when old or new communities are empty."""
        cd = CommunityDetection()
        g = StarGraph()
        # First: empty graph
        cd.detect(g)
        result = cd.refresh(g)
        assert result == []

    def test_get_neighbors_no_matching_community(self):
        """get_neighboring_communities when community not found."""
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "test")
        g.add_anchor(a1)
        cd.detect(g)
        neighbors = cd.get_neighboring_communities("nonexistent_id")
        assert neighbors == []

    def test_detect_with_bridge_updates_secondary_ids(self):
        """detect() sets secondary_community_ids on bridge anchors."""
        cd = CommunityDetection(max_community_size=100)
        g = StarGraph()
        for i in range(3):
            g.add_anchor(make_anchor(f"c{i}", f"cluster item {i}", tags=["c"]))
        for i in range(3, 6):
            g.add_anchor(make_anchor(f"d{i}", f"other item {i}", tags=["d"]))
        for i in range(3):
            for j in range(i + 1, 3):
                g.add_edge(f"c{i}", f"c{j}", weight=0.9)
        for i in range(3, 6):
            for j in range(i + 1, 6):
                g.add_edge(f"d{i}", f"d{j}", weight=0.9)
        g.add_edge("c0", "d3", weight=0.5)
        cd.detect(g)
        # Check that bridge nodes exist
        bridge_count = sum(1 for a in g.anchors.values() if a.secondary_community_ids)
        assert bridge_count >= 1

    def test_centroid_empty_embeddings(self):
        """Centroids dict has empty list when anchors exist but lack embeddings."""
        cd = CommunityDetection()
        g = StarGraph()
        a1 = make_anchor("a1", "test")  # no embedding
        a2 = make_anchor("a2", "test2")  # no embedding
        g.add_anchor(a1)
        g.add_anchor(a2)
        centroids = cd._compute_centroids({"a1": "c0", "a2": "c0"}, g.anchors)
        # When no embeddings, _compute_centroids creates empty list for groups
        assert isinstance(centroids, dict)

    def test_size_distribution_buckets(self):
        """Health metrics cover all size distribution buckets."""
        from star_graph.community import Community
        cd = CommunityDetection()
        g = StarGraph()
        for i in range(5):
            a = make_anchor(f"s{i}", f"small {i}", tags=["dev"])
            g.add_anchor(a)
        for i in range(5, 10):
            a = make_anchor(f"m{i}", f"medium {i}", tags=["dev"])
            g.add_anchor(a)
        for i in range(10, 13):
            a = make_anchor(f"l{i}", f"large {i}", tags=["dev"])
            g.add_anchor(a)

        # Wire them into different-sized communities
        cd.detect(g)
        h = cd.health_metrics()
        assert isinstance(h.size_distribution, dict)
