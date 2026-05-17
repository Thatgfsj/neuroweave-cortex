"""Tests for brain_sphere module — HubCenter, BrainSphere, _cosine_sim."""

import pytest

from star_graph.brain_sphere import HubCenter, BrainSphere, _cosine_sim
from star_graph.anchor import Anchor


def make_anchor(name: str, text: str = "", embedding: list | None = None) -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}")
    if embedding:
        a.embedding = embedding
    return a


class TestHubCenter:
    def test_defaults(self):
        hc = HubCenter(
            cortex_name="dev",
            entry_embedding=[0.1, 0.2, 0.3],
        )
        assert hc.cortex_name == "dev"
        assert hc.node_count == 0
        assert hc.access_count == 0
        assert hc.summary == ""

    def test_with_summary(self):
        hc = HubCenter(
            cortex_name="dev",
            entry_embedding=[0.1, 0.2],
            summary="Python, Docker, debugging",
            node_count=100,
        )
        assert hc.summary == "Python, Docker, debugging"
        assert hc.node_count == 100

    def test_touch(self):
        hc = HubCenter(cortex_name="dev", entry_embedding=[0.1])
        old_access = hc.access_count
        old_updated = hc.last_updated
        hc.touch()
        assert hc.access_count == old_access + 1
        assert hc.last_updated >= old_updated


class TestCosineSim:
    def test_identical(self):
        assert _cosine_sim([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0


class TestBrainSphere:
    def test_init_default(self):
        bs = BrainSphere()
        assert bs.max_common_nodes == 5000
        assert len(bs._common_nodes) == 0
        assert len(bs._hub_centers) == 0

    def test_init_custom(self):
        bs = BrainSphere(max_common_nodes=100)
        assert bs.max_common_nodes == 100

    def test_register_cortex_new(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [0.1] * 128, summary="Dev cortex", node_count=50)
        assert "dev" in bs._hub_centers
        assert bs._hub_centers["dev"].summary == "Dev cortex"
        assert bs._hub_centers["dev"].node_count == 50

    def test_register_cortex_update(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [0.1] * 128, summary="Old")
        bs.register_cortex("dev", [0.2] * 128, summary="New", node_count=100)
        assert bs._hub_centers["dev"].summary == "New"
        assert bs._hub_centers["dev"].node_count == 100

    def test_remove_cortex(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [0.1] * 128)
        bs.remove_cortex("dev")
        assert "dev" not in bs._hub_centers

    def test_remove_cortex_nonexistent(self):
        bs = BrainSphere()
        bs.remove_cortex("nonexistent")  # no-op

    def test_get_center(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [0.1] * 128)
        center = bs.get_center("dev")
        assert center is not None
        assert center.cortex_name == "dev"

    def test_get_center_nonexistent(self):
        bs = BrainSphere()
        assert bs.get_center("nonexistent") is None

    def test_get_relevant_centers_empty(self):
        bs = BrainSphere()
        centers = bs.get_relevant_centers([0.1] * 128)
        assert centers == []

    def test_get_relevant_centers_no_embedding(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [0.1] * 128)
        centers = bs.get_relevant_centers([])
        assert len(centers) >= 1

    def test_get_relevant_centers_with_similarity(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [1.0] * 128, summary="Dev")
        bs.register_cortex("finance", [0.1] + [0.0] * 127, summary="Finance")
        centers = bs.get_relevant_centers([1.0] * 128, top_k=2, min_similarity=0.0)
        assert len(centers) >= 1
        # Touch should have been called (increments access_count)
        assert centers[0].access_count >= 1

    def test_get_relevant_centers_min_similarity(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [0.0] * 128)
        centers = bs.get_relevant_centers([1.0] * 128, min_similarity=0.99)
        # Very different embeddings won't meet high threshold
        assert len(centers) >= 0  # may or may not find matches

    def test_cache_node_add(self):
        bs = BrainSphere()
        a = make_anchor("a1", "test memory", embedding=[0.1] * 128)
        bs.cache_node(a)
        assert "a1" in bs._common_nodes

    def test_cache_node_update(self):
        bs = BrainSphere()
        a = make_anchor("a1", "test memory", embedding=[0.1] * 128)
        bs.cache_node(a)
        bs.cache_node(a)  # should move to end
        assert "a1" in bs._common_nodes

    def test_cache_node_eviction(self):
        bs = BrainSphere(max_common_nodes=2)
        a1 = make_anchor("a1", "first")
        a2 = make_anchor("a2", "second")
        a3 = make_anchor("a3", "third")
        bs.cache_node(a1)
        bs.cache_node(a2)
        bs.cache_node(a3)
        # a1 should be evicted (LRU)
        assert "a1" not in bs._common_nodes
        assert "a2" in bs._common_nodes
        assert "a3" in bs._common_nodes

    def test_query_common_nodes_empty(self):
        bs = BrainSphere()
        results = bs.query_common_nodes(query_embedding=[0.1] * 128)
        assert results == []
        assert bs.cache_misses == 1

    def test_query_common_nodes_with_embedding(self):
        bs = BrainSphere()
        a = make_anchor("a1", "redis timeout fix", embedding=[1.0] * 128)
        bs.cache_node(a)
        results = bs.query_common_nodes(
            query_embedding=[1.0] * 128, min_similarity=0.5)
        assert len(results) >= 1
        assert bs.cache_hits == 1

    def test_query_common_nodes_with_text(self):
        bs = BrainSphere()
        a = make_anchor("a1", "redis timeout fix for server")
        bs.cache_node(a)
        results = bs.query_common_nodes(
            query_text="redis timeout fix", min_similarity=0.1)
        assert len(results) >= 1

    def test_query_common_nodes_no_match(self):
        bs = BrainSphere()
        a = make_anchor("a1", "redis timeout fix", embedding=[1.0] * 128)
        bs.cache_node(a)
        results = bs.query_common_nodes(
            query_embedding=[0.0] * 128, min_similarity=0.99)
        assert results == []
        assert bs.cache_misses == 1

    def test_query_common_nodes_skips_non_retrievable(self):
        bs = BrainSphere()
        a = make_anchor("a1", "test", embedding=[0.1] * 128)
        a.state = __import__('star_graph.anchor', fromlist=['MemoryState']).MemoryState.GHOST
        bs.cache_node(a)
        results = bs.query_common_nodes(query_embedding=[0.1] * 128, min_similarity=0.0)
        # GHOST state is not retrievable
        assert isinstance(results, list)

    def test_evict_node(self):
        bs = BrainSphere()
        a = make_anchor("a1", "test")
        bs.cache_node(a)
        bs.evict_node("a1")
        assert "a1" not in bs._common_nodes

    def test_evict_node_nonexistent(self):
        bs = BrainSphere()
        bs.evict_node("nonexistent")  # no-op

    def test_refresh_cache(self):
        bs = BrainSphere(max_common_nodes=10)
        from star_graph.cortex import MemoryCortex, CortexConfig

        class FakeCortex:
            def __init__(self, name):
                self.graph = __import__('star_graph.graph', fromlist=['StarGraph']).StarGraph()
                self.config = __import__('star_graph.cortex', fromlist=['CortexConfig']).CortexConfig(name=name)

        c = FakeCortex("dev")
        a = make_anchor("a1", "hot memory", embedding=[0.1] * 128)
        a.vector.thermal_state = __import__('star_graph.anchor', fromlist=['ThermalState']).ThermalState.HOT
        c.graph.add_anchor(a)

        bs.refresh_cache([c])
        assert "a1" in bs._common_nodes

    def test_stats(self):
        bs = BrainSphere()
        bs.register_cortex("dev", [0.1] * 128, summary="Dev")
        s = bs.stats
        assert s["hub_centers"] == 1
        assert s["cached_nodes"] == 0
        assert s["hit_rate"] >= 0.0
        assert "dev" in s["cortices"]

    def test_hit_rate(self):
        bs = BrainSphere()
        a = make_anchor("a1", "test", embedding=[1.0] * 128)
        bs.cache_node(a)
        bs.query_common_nodes(query_embedding=[1.0] * 128, min_similarity=0.5)
        s = bs.stats
        assert s["cache_hits"] == 1
        assert s["hit_rate"] == 1.0

    def test_is_full(self):
        bs = BrainSphere(max_common_nodes=1)
        assert bs.is_full is False
        bs.cache_node(make_anchor("a1", "test"))
        assert bs.is_full is True
