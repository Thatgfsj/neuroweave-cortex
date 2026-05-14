"""Test Spreading Activation — local subgraph BFS with edge-type-weighted traversal."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from star_graph import (
    StarGraph, Anchor, SpreadingActivation, ActivatedNode,
)


def make_graph() -> StarGraph:
    """Create a small graph for spreading tests."""
    g = StarGraph()
    for label in ['A', 'B', 'C', 'D', 'E', 'F']:
        g.add_anchor(Anchor.create(f"memory {label}", tags=[label.lower()]))
    ids = list(g.anchors.keys())
    # A→B (causes), A→C (causes), B→D (topical), C→E (contradicts), D→F (preference)
    g.add_edge(ids[0], ids[1], weight=0.7, edge_type='causes',
               confidence=0.8, source_type='explicit')
    g.add_edge(ids[0], ids[2], weight=0.6, edge_type='causes',
               confidence=0.7, source_type='explicit')
    g.add_edge(ids[1], ids[3], weight=0.5, edge_type='topical',
               confidence=0.5, source_type='implicit')
    g.add_edge(ids[2], ids[4], weight=0.5, edge_type='contradicts',
               confidence=0.6, source_type='inferred')
    g.add_edge(ids[3], ids[5], weight=0.4, edge_type='preference',
               confidence=0.5, source_type='implicit')
    return g


class TestSpreadingActivation:
    """Verify local subgraph activation spreading."""

    def test_activate_from_seeds(self):
        """Should spread activation from explicit seed IDs."""
        g = make_graph()
        ids = list(g.anchors.keys())
        sa = SpreadingActivation(g)

        result = sa.activate(seed_ids=[ids[0]], top_k=10)
        assert len(result) >= 2  # at least A and immediate neighbors

        # Seed node should have highest activation
        assert result[0].anchor_id == ids[0]
        assert result[0].activation_depth == 0

    def test_activate_empty_seeds(self):
        """Should return empty list when no seeds match."""
        g = StarGraph()
        sa = SpreadingActivation(g)
        result = sa.activate(seed_ids=["nonexistent"])
        assert result == []

    def test_activate_no_seeds(self):
        """Should return empty list with no seeds and no embedding."""
        g = StarGraph()
        sa = SpreadingActivation(g)
        result = sa.activate()
        assert result == []

    def test_causal_edges_propagate_more(self):
        """Causal edges (×1.5) should propagate more activation than topical (×1.0)."""
        g = StarGraph()
        a = Anchor.create("a")
        b = Anchor.create("b")
        c = Anchor.create("c")
        g.add_anchor(a)
        g.add_anchor(b)
        g.add_anchor(c)

        # Causal edge: a→b
        g.add_edge(a.id, b.id, weight=0.5, edge_type='causes',
                   confidence=0.8, source_type='explicit')
        # Topical edge: a→c
        g.add_edge(a.id, c.id, weight=0.5, edge_type='topical',
                   confidence=0.5, source_type='implicit')

        sa = SpreadingActivation(g)
        sa.configure(decay=1.0)  # no decay for clean test
        result = sa.activate(seed_ids=[a.id], top_k=10)

        # Find b and c
        b_node = next(n for n in result if n.anchor_id == b.id)
        c_node = next(n for n in result if n.anchor_id == c.id)
        # Causal should accumulate more activation
        assert b_node.accumulated_activation > c_node.accumulated_activation

    def test_max_depth_limits_spread(self):
        """Activation should not spread beyond max_depth."""
        g = make_graph()
        ids = list(g.anchors.keys())
        sa = SpreadingActivation(g)
        sa.configure(max_depth=1)

        result = sa.activate(seed_ids=[ids[0]], top_k=10)
        depths = {n.anchor_id: n.activation_depth for n in result}

        # With max_depth=1, F (depth 3 from A) should not appear
        assert depths.get(ids[5], -1) <= 1 or ids[5] not in depths

    def test_configure_overrides_defaults(self):
        """configure() should update hyperparameters."""
        sa = SpreadingActivation(StarGraph())
        sa.configure(decay=0.5, max_depth=2, top_k_seeds=10, top_k_results=20)
        assert sa._decay == 0.5
        assert sa._max_depth == 2
        assert sa._top_k_seeds == 10
        assert sa._top_k_results == 20

    def test_activate_from_text(self):
        """activate_from_text should encode query and run activation."""
        g = make_graph()
        sa = SpreadingActivation(g)
        result = sa.activate_from_text("memory A", top_k=5)
        # Should find at least the matching anchor
        assert len(result) >= 1

    def test_get_subgraph(self):
        """get_subgraph should return activated subgraph structure."""
        g = make_graph()
        ids = list(g.anchors.keys())
        sa = SpreadingActivation(g)
        result = sa.activate(seed_ids=[ids[0]], top_k=5)
        subgraph = sa.get_subgraph(result)
        assert ids[0] in subgraph
        # Should have edges among activated nodes
        assert isinstance(subgraph[ids[0]], list)


class TestActivatedNode:
    """Verify ActivatedNode dataclass."""

    def test_fields(self):
        from star_graph import Anchor
        a = Anchor.create("test")
        node = ActivatedNode(
            anchor_id=a.id,
            anchor=a,
            accumulated_activation=0.75,
            activation_depth=2,
            source_seeds=["seed_1"],
            path=["seed_1", "mid", a.id],
        )
        assert node.text == a.text
        assert node.tags == a.tags
        assert node.accumulated_activation == 0.75
        assert node.activation_depth == 2
