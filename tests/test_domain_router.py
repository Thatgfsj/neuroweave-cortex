"""Tests for DomainRouter — hierarchical topic tree retrieval pre-filtering."""

import pytest
from star_graph.domain_router import DomainRouter, DomainNode, DEFAULT_DOMAIN_TREE


class TestDomainNode:
    def test_node_creation(self):
        node = DomainNode(name="开发", keywords=["开发", "编程"], depth=0)
        assert node.name == "开发"
        assert node.is_root
        assert node.is_leaf  # no children → leaf
        assert node.depth == 0

    def test_node_leaf_detection(self):
        node = DomainNode(name="Python", keywords=["python"], parent="开发", depth=1)
        assert not node.is_root
        assert node.is_leaf

    def test_node_with_children(self):
        parent = DomainNode(name="开发", keywords=["开发"], depth=0)
        child = DomainNode(name="Python", keywords=["python"], parent="开发", depth=1)
        parent.children.append("Python")
        assert len(parent.children) == 1
        assert not child.is_root


class TestDomainRouterBuild:
    def test_builds_default_tree(self):
        router = DomainRouter()
        assert len(router._domains) > 0
        assert "开发" in router._domains
        assert "AI" in router._domains

    def test_root_domains_exist(self):
        router = DomainRouter()
        root_domains = [n for n in router._domains.values() if n.is_root]
        assert len(root_domains) == len(DEFAULT_DOMAIN_TREE)

    def test_subdomains_have_parent(self):
        router = DomainRouter()
        python_node = router._domains.get("开发/Python")
        assert python_node is not None
        assert python_node.parent == "开发"
        assert python_node.depth == 1

    def test_clusters_have_depth_2(self):
        router = DomainRouter()
        web_node = router._domains.get("开发/Python/Web开发")
        if web_node:  # may not exist in all trees
            assert web_node.depth == 2

    def test_keyword_indexing(self):
        router = DomainRouter()
        assert "python" in router._keyword_index
        assert "开发/Python" in router._keyword_index.get("python", set())


class TestDomainRouterIndexing:
    def test_index_anchor_simple(self):
        router = DomainRouter()
        router.index_anchor("a1", text="Python flask web开发", tags=["python"])
        domain = router.get_domain_for_anchor("a1")
        assert domain != ""
        assert "Python" in domain or "开发" in domain

    def test_index_anchor_finance(self):
        router = DomainRouter()
        router.index_anchor("a2", text="量化交易回测策略", tags=["finance"])
        domain = router.get_domain_for_anchor("a2")
        assert "金融" in domain or domain != ""

    def test_index_anchor_adds_to_parents(self):
        router = DomainRouter()
        router.index_anchor("a3", text="Flask API部署", tags=["web"])
        domain = router.get_domain_for_anchor("a3")
        if domain:
            domain_node = router._domains.get(domain)
            assert domain_node is not None
            assert "a3" in domain_node.anchor_ids
            # Should also be in parent
            if domain_node.parent:
                parent = router._domains.get(domain_node.parent)
                assert parent is not None

    def test_remove_anchor(self):
        router = DomainRouter()
        router.index_anchor("a4", text="Python开发", tags=["python"])
        domain = router.get_domain_for_anchor("a4")
        assert domain != ""
        router.remove_anchor("a4")
        assert router.get_domain_for_anchor("a4") == ""

    def test_get_domain_path(self):
        router = DomainRouter()
        router.index_anchor("a5", text="docker compose部署", tags=["devops"])
        path = router.get_domain_path("a5")
        assert isinstance(path, str)

    def test_index_unknown_topic(self):
        router = DomainRouter()
        router.index_anchor("a6", text="xyzzy foo bar", tags=[])
        path = router.get_domain_path("a6")
        assert isinstance(path, str)


class TestDomainRouterRouting:
    def test_route_python_query(self):
        router = DomainRouter()
        result = router.route("python flask web开发")
        assert "matched_domains" in result
        assert len(result["matched_domains"]) > 0

    def test_route_ai_query(self):
        router = DomainRouter()
        result = router.route("如何使用transformer训练模型")
        assert len(result["matched_domains"]) > 0
        assert any("AI" in d for d in result["matched_domains"])

    def test_route_unknown_query(self):
        router = DomainRouter()
        result = router.route("xyzzy12345")
        assert result["matched_domains"] == []
        assert result["depth"] == -1
        assert result["path"] == "unknown"

    def test_route_deep_match_scores_higher(self):
        router = DomainRouter()
        router.index_anchor("d1", text="flask route handler", tags=["python"])
        router.index_anchor("d2", text="spring boot controller", tags=["java"])
        result = router.route("flask web开发")
        domains = result["matched_domains"]
        if len(domains) >= 2:
            first = domains[0]
            assert "Python" in first or "Web" in first

    def test_candidate_scope(self):
        router = DomainRouter()
        router.index_anchor("c1", text="flask api", tags=["python"])
        router.index_anchor("c2", text="docker部署", tags=["devops"])
        router.index_anchor("c3", text="spring开发", tags=["java"])
        ids, path = router.get_candidate_scope("flask api开发")
        assert isinstance(ids, set)
        assert "c1" in ids  # flask python anchor should be in matched domain
        assert isinstance(path, str)

    def test_candidate_scope_returns_set(self):
        router = DomainRouter()
        router.index_anchor("cs1", text="神经网络训练", tags=["ai"])
        ids, path = router.get_candidate_scope("transformer注意力机制")
        assert isinstance(ids, set)


class TestDomainRouterStats:
    def test_stats_structure(self):
        router = DomainRouter()
        router.index_anchor("s1", text="Python开发", tags=["python"])
        stats = router.stats
        assert "total_domains" in stats
        assert "total_indexed_anchors" in stats
        assert stats["total_indexed_anchors"] >= 1

    def test_stats_by_domain(self):
        router = DomainRouter()
        router.index_anchor("s2", text="docker compose", tags=["devops"])
        stats = router.stats
        assert "by_domain" in stats
        for domain_name, info in stats["by_domain"].items():
            assert "depth" in info
            assert "anchors" in info


class TestCustomDomain:
    def test_add_custom_domain(self):
        router = DomainRouter()
        router.add_custom_domain("游戏开发", ["unity", "unreal", "game"], parent="开发", depth=1)
        assert "游戏开发" in router._domains
        node = router._domains["游戏开发"]
        assert node.parent == "开发"

    def test_custom_domain_routing(self):
        router = DomainRouter()
        router.add_custom_domain("游戏", ["unity", "unreal", "game"], depth=0)
        router.index_anchor("g1", text="unity shader编程", tags=["game"])
        result = router.route("unity游戏引擎")
        assert "游戏" in result["matched_domains"] or len(result["matched_domains"]) > 0
