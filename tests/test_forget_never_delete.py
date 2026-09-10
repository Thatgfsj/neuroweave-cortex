"""Tests for the Never-Delete philosophy (v1.3.2): forget() is logical, never physical.

Policy: forgetting NEVER removes a memory from the graph. It
  1. demotes the anchor to DORMANT via the state machine,
  2. lowers its weight (activation_level → 0.01 hard floor, importance × 0.3),
  3. optionally records a compressed ghost residual trace,
  4. with invalid=True stamps invalid_at — the only retrieval blocker.

The anchor stays in graph.anchors with its edges, so activation propagation
can still reach and reactivate it.
"""

import pytest

from star_graph import MemoryManager
from star_graph.anchor import MemoryState


FORGET_IMPORTANCE = 0.8
FORGET_EXPECTED_IMPORTANCE = 0.8 * 0.3  # ×0.3 discount applied by forget()


@pytest.fixture()
def mgr() -> MemoryManager:
    return MemoryManager()


def _make_connected_pair(mgr: MemoryManager):
    """Store two memories joined by a strong explicit edge (weight 0.8).

    The explicit edge guarantees propagation can flow keeper → forgotten
    regardless of embedding similarity.
    """
    forgotten = mgr.remember(
        "User once lived in Oslo and spoke Norwegian",
        tags=["fact"], importance=FORGET_IMPORTANCE, skip_gate=True,
    )
    keeper = mgr.remember(
        "User loves hiking in the mountains every spring",
        tags=["preference"], importance=FORGET_IMPORTANCE, skip_gate=True,
    )
    edge = mgr.graph.add_edge(forgotten.id, keeper.id, weight=0.8, edge_type="topical")
    assert edge is not None
    return forgotten, keeper


class TestForgetNeverDelete:
    def test_forget_keeps_anchor_in_graph(self, mgr):
        """(a) forget() must NOT remove the anchor from graph.anchors."""
        forgotten, _ = _make_connected_pair(mgr)
        result = mgr.forget(forgotten.id)
        assert result is not None            # backward compat: returns the anchor
        assert result.id == forgotten.id
        assert forgotten.id in mgr.graph.anchors
        assert mgr.graph.anchors[forgotten.id] is result

    def test_forget_demotes_state_to_dormant(self, mgr):
        """(b) forget() walks the state machine down to DORMANT."""
        forgotten, _ = _make_connected_pair(mgr)
        mgr.forget(forgotten.id)
        assert forgotten.state == MemoryState.DORMANT
        # The demotion path is recorded in the state history.
        assert any(s is MemoryState.DORMANT for s, _ in forgotten.state_history)

    def test_forget_lowers_weight_to_floor(self, mgr):
        """(c) activation_level sits at the 0.01 hard floor, importance ×0.3."""
        forgotten, _ = _make_connected_pair(mgr)
        mgr.forget(forgotten.id)
        assert forgotten.activation_level == pytest.approx(0.01)
        assert forgotten.vector.importance == pytest.approx(FORGET_EXPECTED_IMPORTANCE)

    def test_forgotten_memory_still_retrievable(self, mgr):
        """(d1) Without invalid_at, a forgotten memory stays retrievable."""
        forgotten, _ = _make_connected_pair(mgr)
        mgr.forget(forgotten.id)
        assert forgotten.is_retrievable is True

    def test_forgotten_memory_reachable_via_propagation(self, mgr):
        """(d2) graph.spread_activation still reaches the forgotten anchor."""
        forgotten, keeper = _make_connected_pair(mgr)
        mgr.forget(forgotten.id)
        activation = mgr.graph.spread_activation([keeper.id])
        assert forgotten.id in activation
        assert activation[forgotten.id] > 0.0

    def test_forget_records_ghost_residual(self, mgr):
        """With create_ghost=True a compressed residual trace is recorded."""
        forgotten, _ = _make_connected_pair(mgr)
        mgr.forget(forgotten.id, create_ghost=True)
        assert forgotten.id in mgr.ghosts.ghosts

    def test_forget_invalid_stamps_invalid_at(self, mgr):
        """invalid=True marks user-repudiated content (retrieval blocker only)."""
        forgotten, _ = _make_connected_pair(mgr)
        assert forgotten.invalid_at is None
        mgr.forget(forgotten.id, invalid=True)
        # Still never deleted…
        assert forgotten.id in mgr.graph.anchors
        assert forgotten.state == MemoryState.DORMANT
        # …but invalid_at now blocks retrieval.
        assert forgotten.invalid_at is not None
        assert forgotten.is_retrievable is False

    def test_forget_is_idempotent(self, mgr):
        """Forgetting twice keeps the anchor DORMANT and intact (no data loss)."""
        forgotten, _ = _make_connected_pair(mgr)
        mgr.forget(forgotten.id)
        mgr.forget(forgotten.id)
        assert forgotten.id in mgr.graph.anchors
        assert forgotten.state == MemoryState.DORMANT
        assert forgotten.activation_level == pytest.approx(0.01)
        assert forgotten.vector.importance == pytest.approx(FORGET_EXPECTED_IMPORTANCE)
        assert forgotten.text  # content untouched

    def test_forget_nonexistent_returns_none(self, mgr):
        assert mgr.forget("no-such-anchor-id") is None

    def test_edges_survive_forget(self, mgr):
        """Never-Delete keeps the neighborhood intact — edges are not removed."""
        forgotten, keeper = _make_connected_pair(mgr)
        mgr.forget(forgotten.id)
        assert keeper.id in mgr.graph._adjacency.get(forgotten.id, set())
