"""Tests for cascade module — CausalChain and CascadeRecall."""

import pytest

from star_graph.cascade import CausalChain, CascadeRecall
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


def make_anchor(name: str, text: str = "", embedding: list | None = None) -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}")
    if embedding:
        a.embedding = embedding
    return a


class TestCausalChain:
    def test_empty_chain(self):
        chain = CausalChain()
        assert chain.narrative == ""
        assert not chain.is_valid
        assert chain.depth == 0

    def test_single_anchor(self):
        a = make_anchor("a1", "first event")
        chain = CausalChain(anchors=[a], chain_type="causal")
        assert "first event" in chain.narrative
        assert not chain.is_valid  # need at least 2

    def test_two_anchors_causal(self):
        a1 = make_anchor("a1", "bug found")
        a2 = make_anchor("a2", "bug fixed")
        chain = CausalChain(anchors=[a1, a2], chain_type="causal",
                           total_confidence=0.5)
        assert chain.is_valid
        assert "→" in chain.narrative

    def test_two_anchors_temporal(self):
        a1 = make_anchor("a1", "login page")
        a2 = make_anchor("a2", "dashboard")
        chain = CausalChain(anchors=[a1, a2], chain_type="temporal",
                           total_confidence=0.5)
        assert chain.is_valid
        assert " then " in chain.narrative

    def test_derived_chain(self):
        a1 = make_anchor("a1", "user data")
        a2 = make_anchor("a2", "analytics report")
        chain = CausalChain(anchors=[a1, a2], chain_type="derived",
                           total_confidence=0.5)
        assert "→" in chain.narrative


class TestCascadeRecall:
    def test_init(self):
        g = StarGraph()
        cr = CascadeRecall(g)
        assert cr.graph is g

    def test_trace_backward_empty_graph(self):
        g = StarGraph()
        cr = CascadeRecall(g)
        chains = cr.trace_backward("nonexistent", max_depth=3)
        assert chains == []

    def test_trace_forward_empty_graph(self):
        g = StarGraph()
        cr = CascadeRecall(g)
        chains = cr.trace_forward("nonexistent", max_depth=3)
        assert chains == []

    def test_trace_backward_with_anchor(self):
        g = StarGraph()
        a1 = make_anchor("a1", "first cause")
        g.add_anchor(a1)
        cr = CascadeRecall(g)
        chains = cr.trace_backward("a1", max_depth=3)
        assert isinstance(chains, list)

    def test_trace_forward_with_anchor(self):
        g = StarGraph()
        a1 = make_anchor("a1", "first effect")
        g.add_anchor(a1)
        cr = CascadeRecall(g)
        chains = cr.trace_forward("a1", max_depth=3)
        assert isinstance(chains, list)
