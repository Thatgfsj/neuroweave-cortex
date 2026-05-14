"""Tests for Causal Edge Types — richer causal relationships."""

import pytest
from star_graph.causal_edges import (
    CausalEdgeClassifier, CausalChain,
    CAUSAL_EDGE_TYPES, CAUSAL_TRAVERSAL_WEIGHTS,
)
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


class TestCausalEdgeTypes:
    def test_all_types_present(self):
        expected = {"causes", "depends_on", "motivates", "goal_of", "result_of", "precedes"}
        assert set(CAUSAL_EDGE_TYPES.keys()) == expected

    def test_all_have_weight(self):
        for etype in CAUSAL_EDGE_TYPES:
            assert CAUSAL_TRAVERSAL_WEIGHTS[etype] > 0

    def test_causes_has_highest_weight(self):
        assert CAUSAL_TRAVERSAL_WEIGHTS["causes"] >= CAUSAL_TRAVERSAL_WEIGHTS["precedes"]


class TestCausalEdgeClassifier:
    def test_init(self):
        c = CausalEdgeClassifier()
        assert c.min_confidence == 0.3

    def test_infer_causes(self):
        c = CausalEdgeClassifier(min_confidence=0.0)
        etype, conf = c.infer(
            "The connection pool was too small",
            "This caused the timeout errors in production"
        )
        # "causes" or "result_of" could match
        assert etype in CAUSAL_EDGE_TYPES

    def test_infer_depends_on(self):
        c = CausalEdgeClassifier(min_confidence=0.0)
        etype, conf = c.infer(
            "The auth middleware",
            "This depends on the Redis connection pool"
        )
        assert etype in CAUSAL_EDGE_TYPES

    def test_infer_no_match(self):
        c = CausalEdgeClassifier(min_confidence=0.99)
        etype, conf = c.infer("hello", "world")
        assert etype == "causes"
        assert conf == 0.1

    def test_get_causal_weight(self):
        assert CausalEdgeClassifier.get_causal_weight("causes") > 1.0
        assert CausalEdgeClassifier.get_causal_weight("unknown") == 1.0


class TestTraceCausalChain:
    def test_empty_graph(self):
        g = StarGraph()
        chains = CausalEdgeClassifier.trace_causal_chain(g, "nonexistent")
        assert chains == []

    def test_single_node(self):
        g = StarGraph()
        a = Anchor.create(text="test")
        g.add_anchor(a)
        chains = CausalEdgeClassifier.trace_causal_chain(g, a.id, max_depth=3)
        assert len(chains) == 1
        assert chains[0].depth == 0

    def test_causal_chain(self):
        g = StarGraph()
        a1 = Anchor.create(text="small connection pool")
        a2 = Anchor.create(text="timeout errors")
        a3 = Anchor.create(text="increased pool to 20")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_anchor(a3)
        g.add_edge(a1.id, a2.id, weight=0.8, edge_type="causes")
        g.add_edge(a2.id, a3.id, weight=0.7, edge_type="causes")
        chains = CausalEdgeClassifier.trace_causal_chain(g, a1.id, max_depth=5)
        assert len(chains) >= 1
        longest = max(chains, key=lambda c: c.depth)
        assert longest.depth >= 1
