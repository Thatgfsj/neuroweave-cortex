"""Tests for cost_estimator module — SleepCostEstimator and CostEstimate."""

import pytest

from star_graph.cost_estimator import (
    CostEstimate,
    SleepCostEstimator,
    PRICING,
    TOKEN_ESTIMATES,
    TIME_ESTIMATES,
)
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor, MemoryState


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                state=MemoryState.ACTIVE, source_session: str = "s1",
                tags: list | None = None) -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}", tags=tags or [],
              source_session=source_session)
    a.state = state
    if embedding:
        a.embedding = embedding
    return a


class TestPricingConstants:
    def test_has_gpt4_mini(self):
        assert "gpt-4o-mini" in PRICING
        assert PRICING["gpt-4o-mini"]["input"] == 0.00015

    def test_template_is_free(self):
        assert PRICING["template"]["input"] == 0.0
        assert PRICING["template"]["output"] == 0.0

    def test_has_claude_haiku(self):
        assert "claude-3-haiku" in PRICING


class TestTokenEstimates:
    def test_atom_facts_per_anchor(self):
        assert TOKEN_ESTIMATES["atom_facts_per_anchor"] == 250

    def test_compression_per_cluster(self):
        assert TOKEN_ESTIMATES["compression_per_cluster"] == 400


class TestTimeEstimates:
    def test_swr_replay(self):
        assert "swr_replay_per_100_anchors" in TIME_ESTIMATES

    def test_llm_call_overhead(self):
        assert TIME_ESTIMATES["llm_call_overhead"] == 2.0


class TestCostEstimate:
    def test_defaults(self):
        ce = CostEstimate()
        assert ce.estimated_cost_usd == 0.0
        assert ce.total_anchors == 0
        assert ce.provider == "template"
        assert ce.dry_run is False
        assert ce.is_free is True

    def test_is_free_with_llm_cost(self):
        ce = CostEstimate(llm_cost_usd=0.01)
        assert ce.is_free is False

    def test_summary_no_llm_calls(self):
        ce = CostEstimate(total_anchors=100, total_edges=200)
        s = ce.summary()
        assert "100 anchors" in s
        assert "200 edges" in s
        assert "offline" in s

    def test_summary_with_llm_calls(self):
        ce = CostEstimate(
            total_anchors=100, total_edges=200,
            estimated_llm_calls=5, estimated_input_tokens=5000,
            estimated_output_tokens=1000, llm_cost_usd=0.003,
            total_duration_seconds=5.0,
        )
        s = ce.summary()
        assert "5 LLM calls" in s
        assert "6000 tokens" in s

    def test_detailed(self):
        ce = CostEstimate(
            total_anchors=100, total_edges=200,
            estimated_llm_calls=3, total_duration_seconds=5.0,
            provider="template", model="gpt-4o-mini",
        )
        d = ce.detailed()
        assert "Sleep Cost Estimate" in d
        assert "100" in d

    def test_detailed_dry_run(self):
        ce = CostEstimate(dry_run=True)
        d = ce.detailed()
        assert "DRY RUN" in d

    def test_detailed_with_phase_estimates(self):
        ce = CostEstimate(
            phase_estimates={
                "N1 Replay": {"duration": 1.0, "items": 50},
            },
        )
        d = ce.detailed()
        assert "Phase breakdown" in d
        assert "N1 Replay" in d


class TestSleepCostEstimator:
    def test_init(self):
        sce = SleepCostEstimator()
        assert sce is not None

    def test_estimate_empty_graph(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        est = sce.estimate(g)
        assert isinstance(est, CostEstimate)
        assert est.total_anchors == 0
        assert est.total_edges == 0

    def test_estimate_with_anchors(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        for i in range(10):
            a = make_anchor(f"a{i}", f"text {i}", source_session="s1")
            a.state = MemoryState.ACTIVE
            g.add_anchor(a)
        est = sce.estimate(g)
        assert est.total_anchors == 10

    def test_estimate_with_dormant_anchors(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        for i in range(5):
            a = make_anchor(f"a{i}", f"text {i}",
                          embedding=[0.1] * 384, source_session="s1")
            a.state = MemoryState.DORMANT
            g.add_anchor(a)
        for i in range(5, 8):
            a = make_anchor(f"a{i}", f"text {i}")
            a.state = MemoryState.CONSOLIDATING
            g.add_anchor(a)
        est = sce.estimate(g)
        assert est.dormant_anchors == 5
        assert est.consolidating_anchors == 3

    def test_estimate_dry_run(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        est = sce.estimate(g, dry_run=True)
        assert est.dry_run is True

    def test_estimate_with_compression_clusters(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        # Create enough dormant anchors in same session to form a cluster
        for i in range(5):
            a = make_anchor(f"a{i}", f"session memory {i}",
                          embedding=[0.1] * 384, source_session="s1")
            a.state = MemoryState.DORMANT
            g.add_anchor(a)
        est = sce.estimate(g)
        assert est.compression_clusters >= 1
        assert est.compression_summaries >= 1

    def test_estimate_without_embeddings(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        for i in range(5):
            a = make_anchor(f"a{i}", f"text {i}", source_session="s1")
            a.state = MemoryState.DORMANT
            g.add_anchor(a)
        est = sce.estimate(g)
        # Without embeddings, no clusters should form
        assert est.compression_clusters == 0

    def test_estimate_phase_estimates_present(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "test"))
        est = sce.estimate(g)
        assert "N1 Replay" in est.phase_estimates
        assert "N2 Merge" in est.phase_estimates
        assert "5b Compression" in est.phase_estimates
        assert "5c Atom Facts" in est.phase_estimates
        assert "7 Prune" in est.phase_estimates
        assert "8 Index" in est.phase_estimates

    def test_estimate_total_duration(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        for i in range(50):
            g.add_anchor(make_anchor(f"a{i}", f"text {i}"))
        est = sce.estimate(g)
        assert est.total_duration_seconds >= 0

    def test_estimate_from_sleep(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "test"))

        class FakeSleep:
            graph = g
            cfg = None
        est = sce.estimate_from_sleep(FakeSleep())
        assert isinstance(est, CostEstimate)

    def test_estimate_with_manager_object(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        g.add_anchor(make_anchor("a1", "test"))

        class FakeManager:
            def __init__(self):
                self.graph = g
                self.cfg = None
        est = sce.estimate(FakeManager())
        assert isinstance(est, CostEstimate)

    def test_estimate_with_edges(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        a1 = make_anchor("a1", "text 1")
        a2 = make_anchor("a2", "text 2")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.5, edge_type="topical")
        est = sce.estimate(g)
        assert est.total_edges == 1

    def test_estimate_default_provider(self):
        sce = SleepCostEstimator()
        g = StarGraph()
        est = sce.estimate(g)
        assert est.provider == "template"
