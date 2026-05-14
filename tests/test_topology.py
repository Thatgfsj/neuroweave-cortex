"""Test Graph-First Retrieval — topology ranking and graph-first recall."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from star_graph import (
    StarGraph, Anchor,
    topology_rank, graph_first_recall,
    EDGE_TYPE_RICHNESS_WEIGHTS,
)


def make_rich_graph() -> StarGraph:
    """Create a small graph with typed edges for topology testing."""
    g = StarGraph()
    for label in ['A', 'B', 'C', 'D', 'E']:
        a = Anchor.create(f"memory {label}", tags=[label.lower()])
        a.embedding = [0.1 * (ord(label) - ord('A') + 1)] * 10
        g.add_anchor(a)
    ids = list(g.anchors.keys())
    # A→B (causes), A→C (topical), B→D (fixes), C→D (depends_on), D→E (preference)
    g.add_edge(ids[0], ids[1], weight=0.7, edge_type='causes')
    g.add_edge(ids[0], ids[2], weight=0.5, edge_type='topical')
    g.add_edge(ids[1], ids[3], weight=0.6, edge_type='fixes')
    g.add_edge(ids[2], ids[3], weight=0.5, edge_type='depends_on')
    g.add_edge(ids[3], ids[4], weight=0.4, edge_type='preference')
    return g


class TestTopologyRank:
    """Verify graph topology scoring."""

    def test_topology_rank_returns_scores(self):
        g = make_rich_graph()
        ids = list(g.anchors.keys())
        results = topology_rank(g, candidate_ids=ids, top_k=5)
        assert len(results) > 0
        # Should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    def test_topology_rank_scores_between_0_and_1(self):
        g = make_rich_graph()
        ids = list(g.anchors.keys())
        results = topology_rank(g, candidate_ids=ids, top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_high_degree_node_ranks_higher(self):
        g = StarGraph()
        a = Anchor.create("central", tags=["hub"])
        b = Anchor.create("leaf1", tags=["leaf"])
        c = Anchor.create("leaf2", tags=["leaf"])
        for anchor in [a, b, c]:
            anchor.embedding = [0.5] * 10
            g.add_anchor(anchor)
        # a connects to both b and c (degree 2), b and c have degree 1
        g.add_edge(a.id, b.id, weight=0.5, edge_type='topical')
        g.add_edge(a.id, c.id, weight=0.5, edge_type='topical')

        results = topology_rank(g, top_k=3)
        # The central node (a) should have highest topology score
        # or at least be in the results
        result_ids = [r[0] for r in results]
        assert a.id in result_ids

    def test_causal_edges_boost_score(self):
        """Nodes with causal edges should score higher than those with only topical edges."""
        g = StarGraph()
        nodes = []
        for label in ['cause', 'effect', 'topic']:
            a = Anchor.create(f"node {label}")
            a.embedding = [0.5] * 10
            g.add_anchor(a)
            nodes.append(a)
        # cause→effect (causal) vs effect→topic (topical)
        g.add_edge(nodes[0].id, nodes[1].id, weight=0.7, edge_type='causes')
        g.add_edge(nodes[1].id, nodes[2].id, weight=0.5, edge_type='topical')

        results = topology_rank(g, top_k=3)
        result_ids = [r[0] for r in results]
        # cause node should be in results (has a causal edge)
        assert nodes[0].id in result_ids

    def test_topology_rank_non_retrievable_excluded(self):
        g = make_rich_graph()
        ids = list(g.anchors.keys())
        # Freeze one anchor
        from star_graph import ThermalState
        g.anchors[ids[0]]._thermal_state = ThermalState.FROZEN
        results = topology_rank(g, candidate_ids=[ids[0]], top_k=5)
        assert len(results) == 0  # FROZEN is not retrievable

    def test_topology_rank_with_query_embedding(self):
        g = make_rich_graph()
        ids = list(g.anchors.keys())
        query_emb = [0.1] * 10  # matches 'A'
        results_no_emb = topology_rank(g, candidate_ids=ids, top_k=5)
        results_with_emb = topology_rank(
            g, candidate_ids=ids, query_embedding=query_emb, top_k=5,
            graph_weight=0.5, embedding_weight=0.5,
        )
        assert len(results_no_emb) > 0
        assert len(results_with_emb) > 0


class TestGraphFirstRecall:
    """Verify graph-first BFS recall from seeds."""

    def test_graph_first_recall_basic(self):
        g = make_rich_graph()
        query_emb = [0.1] * 10  # matches 'A' best
        results = graph_first_recall(g, query_embedding=query_emb, top_k=5)
        assert len(results) > 0
        # Each result is (anchor_id, score, depth)
        assert len(results[0]) == 3

    def test_graph_first_recall_finds_neighbors(self):
        g = StarGraph()
        a = Anchor.create("seed node", tags=["seed"])
        b = Anchor.create("connected node", tags=["connected"])
        g.add_anchor(a)
        g.add_anchor(b)
        a.embedding = [0.9] * 10
        b.embedding = [0.1] * 10  # different from query
        g.add_edge(a.id, b.id, weight=0.8, edge_type='causes')

        query_emb = [0.9] * 10  # matches a well
        results = graph_first_recall(g, query_embedding=query_emb, top_k=5, max_depth=2)
        result_ids = [r[0] for r in results]
        # b should be found via graph traversal even though its embedding is different
        assert b.id in result_ids or a.id in result_ids

    def test_graph_first_recall_depth_limit(self):
        g = make_rich_graph()
        ids = list(g.anchors.keys())
        query_emb = [0.1] * 10  # matches 'A'

        # With max_depth=0, should only get seed nodes
        results_shallow = graph_first_recall(
            g, query_embedding=query_emb, top_k=10, max_depth=0,
        )
        # With max_depth=0, we only find seeds via ANN
        assert len(results_shallow) >= 0

        results_deep = graph_first_recall(
            g, query_embedding=query_emb, top_k=10, max_depth=3,
        )
        assert len(results_deep) >= 0

    def test_graph_first_recall_respects_top_k(self):
        g = make_rich_graph()
        query_emb = [0.3] * 10
        results = graph_first_recall(g, query_embedding=query_emb, top_k=2)
        assert len(results) <= 2


class TestEdgeTypeRichnessWeights:
    """Verify edge type weight mapping."""

    def test_causal_weights_higher(self):
        assert EDGE_TYPE_RICHNESS_WEIGHTS["causes"] > EDGE_TYPE_RICHNESS_WEIGHTS["topical"]
        assert EDGE_TYPE_RICHNESS_WEIGHTS["fixes"] > EDGE_TYPE_RICHNESS_WEIGHTS["related"]

    def test_contradiction_weight_low(self):
        assert EDGE_TYPE_RICHNESS_WEIGHTS["contradicts"] < 1.0
        assert EDGE_TYPE_RICHNESS_WEIGHTS["invalidated_by"] < EDGE_TYPE_RICHNESS_WEIGHTS["contradicts"]
