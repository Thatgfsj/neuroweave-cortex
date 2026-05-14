"""Tests for RetrievalBudget — hop/node/token limits for retrieval."""

import pytest
from star_graph.retrieval_budget import RetrievalBudget, BudgetState


class TestRetrievalBudgetInit:
    def test_defaults(self):
        rb = RetrievalBudget()
        assert rb.max_hops == 3
        assert rb.max_nodes == 24
        assert rb.max_tokens == 6000

    def test_custom(self):
        rb = RetrievalBudget(max_hops=2, max_nodes=10, max_tokens=1000)
        assert rb.max_hops == 2
        assert rb.max_nodes == 10
        assert rb.max_tokens == 1000

    def test_begin_creates_state(self):
        rb = RetrievalBudget()
        state = rb.begin()
        assert isinstance(state, BudgetState)
        assert state.hops_used == 0
        assert state.nodes_activated == 0
        assert state.tokens_used == 0


class TestHopBudget:
    def test_allows_within_limit(self):
        rb = RetrievalBudget(max_hops=3)
        state = rb.begin()
        assert rb.allow_hop(state, depth=0) is True
        assert rb.allow_hop(state, depth=1) is True
        assert rb.allow_hop(state, depth=2) is True

    def test_rejects_at_limit(self):
        rb = RetrievalBudget(max_hops=3)
        state = rb.begin()
        assert rb.allow_hop(state, depth=3) is False
        assert state.truncated is True

    def test_rejects_beyond_limit(self):
        rb = RetrievalBudget(max_hops=2)
        state = rb.begin()
        assert rb.allow_hop(state, depth=2) is False


class TestNodeBudget:
    def test_allow_within_limit(self):
        rb = RetrievalBudget(max_nodes=5)
        state = rb.begin()
        for _ in range(5):
            assert rb.allow_node(state) is True
        assert state.nodes_activated == 5

    def test_rejects_beyond_limit(self):
        rb = RetrievalBudget(max_nodes=3)
        state = rb.begin()
        for _ in range(3):
            rb.allow_node(state)
        assert rb.allow_node(state) is False

    def test_enforce_nodes_truncates(self):
        rb = RetrievalBudget(max_nodes=3)
        items = list(range(10))
        truncated = rb.enforce_nodes(items)
        assert len(truncated) == 3
        assert truncated == [0, 1, 2]


class TestTokenBudget:
    def test_count_tokens(self):
        rb = RetrievalBudget()
        assert rb.count_tokens("hello world") >= 2
        assert rb.count_tokens("") == 0
        assert rb.count_tokens("a" * 400) == 100  # char/4

    def test_allow_tokens_within_budget(self):
        rb = RetrievalBudget(max_tokens=100)
        state = rb.begin()
        assert rb.allow_tokens(state, "short text") is True
        assert state.tokens_used > 0

    def test_allow_tokens_exceeds_budget(self):
        rb = RetrievalBudget(max_tokens=10)
        state = rb.begin()
        assert rb.allow_tokens(state, "a" * 200) is False  # 50 tokens > 10

    def test_enforce_tokens_truncates(self):
        rb = RetrievalBudget(max_tokens=50)
        items = ["a" * 80, "b" * 80, "c" * 80, "d" * 80]  # ~20 tokens each
        truncated = rb.enforce_tokens(items)
        assert len(truncated) <= 3  # fits within 50 tokens


class TestStats:
    def test_stats(self):
        rb = RetrievalBudget(max_hops=2, max_nodes=10, max_tokens=500)
        s = rb.stats
        assert s["max_hops"] == 2
        assert s["max_nodes"] == 10
        assert s["max_tokens"] == 500
