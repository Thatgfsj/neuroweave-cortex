"""Sleep Cost Estimator — predict resource usage before running sleep.

Analyzes the current graph state and estimates:
- SWR replay duration (based on anchor count)
- Compression LLM calls and token consumption
- Atom fact extraction LLM calls and tokens
- Total estimated cost (in USD) and wall-clock time
- Supports dry-run mode (estimate only, no execution)

Usage:
    estimator = SleepCostEstimator()
    estimate = estimator.estimate(graph, config)
    print(f"Estimated cost: ${estimate.estimated_cost_usd:.4f}")
    print(f"Estimated time: {estimate.total_duration_seconds:.1f}s")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Pricing constants (USD per 1K tokens) ─────────────────────

# GPT-4o-mini pricing
PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},    # per 1K tokens
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "qwen-2.5-3b": {"input": 0.0, "output": 0.0},           # local Ollama = free
    "template": {"input": 0.0, "output": 0.0},               # offline = free
}

# Default token estimates per operation
TOKEN_ESTIMATES = {
    "atom_facts_per_anchor": 250,     # input tokens per anchor in atom fact extraction
    "atom_facts_output_per_fact": 80, # output tokens per extracted fact
    "compression_per_cluster": 400,   # input tokens per cluster for compression
    "compression_output_per_summary": 120,  # output tokens per compressed summary
}

# Time estimates per operation (seconds)
TIME_ESTIMATES = {
    "swr_replay_per_100_anchors": 0.5,
    "merge_per_100_anchors": 0.3,
    "prune_per_100_anchors": 0.1,
    "hebbian_per_100_edges": 0.05,
    "index_rebuild_per_100_anchors": 0.2,
    "llm_call_overhead": 2.0,         # network latency per API call
}


@dataclass
class CostEstimate:
    """Estimated resource consumption for one sleep cycle."""

    # Costs
    estimated_cost_usd: float = 0.0          # total estimated cost
    llm_cost_usd: float = 0.0                # LLM API cost component
    compute_cost_usd: float = 0.0            # compute-only cost (negligible)

    # Duration
    total_duration_seconds: float = 0.0      # wall-clock estimate
    llm_duration_seconds: float = 0.0        # time spent waiting for LLM
    compute_duration_seconds: float = 0.0    # time spent on local compute

    # Token counts
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_llm_calls: int = 0

    # Anchor/edge counts
    total_anchors: int = 0
    total_edges: int = 0
    dormant_anchors: int = 0                 # will be processed
    consolidating_anchors: int = 0
    clusters_found: int = 0

    # Compression details
    compression_clusters: int = 0            # clusters eligible for compression
    compression_summaries: int = 0           # estimated summaries produced

    # Atom facts details
    fact_clusters: int = 0                   # clusters eligible for fact extraction
    fact_anchors: int = 0                    # anchors in fact-eligible clusters
    estimated_facts: int = 0                 # estimated facts extracted

    # Phase breakdown
    phase_estimates: dict = field(default_factory=dict)

    # Meta
    provider: str = "template"               # active LLM provider
    model: str = ""                          # active LLM model
    dry_run: bool = False                    # was this a dry-run estimate?

    @property
    def is_free(self) -> bool:
        return self.llm_cost_usd == 0.0

    def summary(self) -> str:
        """One-line summary of the estimate."""
        parts = [f"{self.total_anchors} anchors, {self.total_edges} edges"]
        if self.estimated_llm_calls > 0:
            parts.append(f"{self.estimated_llm_calls} LLM calls")
            parts.append(f"{self.estimated_input_tokens + self.estimated_output_tokens} tokens")
            parts.append(f"${self.llm_cost_usd:.4f}")
        else:
            parts.append("0 LLM calls (offline)")
        parts.append(f"~{self.total_duration_seconds:.1f}s")
        return " | ".join(parts)

    def detailed(self) -> str:
        """Multi-line detailed estimate."""
        lines = [
            "╔═══════════════════════════════════════╗",
            "║  Sleep Cost Estimate                 ║",
            "╠═══════════════════════════════════════╣",
        ]
        if self.dry_run:
            lines.append("║  MODE: DRY RUN (no execution)        ║")
        lines += [
            f"║  Anchors: {self.total_anchors:>5} total, "
            f"{self.dormant_anchors:>4} dormant, {self.consolidating_anchors:>4} consolidating",
            f"║  Edges:   {self.total_edges:>5} total",
            f"╠═══════════════════════════════════════╣",
            f"║  LLM Provider: {self.provider:<22} ║",
            f"║  LLM Model:    {self.model:<22} ║",
            f"║  LLM Calls:    {self.estimated_llm_calls:>4} calls                   ║",
            f"║  Input Tokens:  {self.estimated_input_tokens:>6}                    ║",
            f"║  Output Tokens: {self.estimated_output_tokens:>6}                    ║",
            f"║  LLM Cost:     ${self.llm_cost_usd:<8.4f}               ║",
            f"╠═══════════════════════════════════════╣",
            f"║  Compute:      {self.compute_duration_seconds:>6.1f}s                   ║",
            f"║  LLM Wait:     {self.llm_duration_seconds:>6.1f}s                   ║",
            f"║  TOTAL TIME:   {self.total_duration_seconds:>6.1f}s                   ║",
            f"║  TOTAL COST:   ${self.estimated_cost_usd:<8.4f}               ║",
            "╚═══════════════════════════════════════╝",
        ]
        if self.phase_estimates:
            lines.append("  Phase breakdown:")
            for phase, est in self.phase_estimates.items():
                lines.append(f"    {phase}: ~{est.get('duration', 0):.1f}s, "
                           f"{est.get('items', 0)} items")
        return "\n".join(lines)


class SleepCostEstimator:
    """Estimate resource consumption of a sleep cycle without executing it.

    Usage:
        estimator = SleepCostEstimator()
        estimate = estimator.estimate(graph)  # or estimator.estimate(manager)
        print(estimate.detailed())

        # Dry-run mode
        estimate = estimator.estimate(graph, dry_run=True)
    """

    def __init__(self):
        pass

    def estimate(self, graph_or_manager,
                 config=None,
                 dry_run: bool = False) -> CostEstimate:
        """Estimate the cost of running one full sleep cycle.

        Args:
            graph_or_manager: StarGraph or MemoryManager instance.
            config: Optional Config object.
            dry_run: If True, marks the estimate as dry-run.

        Returns:
            CostEstimate with all predictions.
        """
        from .graph import StarGraph
        from .anchor import Anchor, MemoryState

        # Accept either a StarGraph or a manager
        if hasattr(graph_or_manager, 'graph') and hasattr(graph_or_manager, 'cfg'):
            manager = graph_or_manager
            graph = manager.graph
            if config is None:
                config = manager.cfg
        else:
            graph = graph_or_manager

        if config is None:
            from .config import Config
            config = Config.get()

        estimate = CostEstimate(dry_run=dry_run)
        estimate.phase_estimates = {}

        # ── Count anchors and edges ──
        all_anchors = list(graph.anchors.values()) if hasattr(graph, 'anchors') else []
        estimate.total_anchors = len(all_anchors)
        estimate.total_edges = len(graph.edges) if hasattr(graph, 'edges') else 0

        dormant = [a for a in all_anchors if a.state == MemoryState.DORMANT]
        consolidating = [a for a in all_anchors if a.state == MemoryState.CONSOLIDATING]
        estimate.dormant_anchors = len(dormant)
        estimate.consolidating_anchors = len(consolidating)

        # ── Determine provider ──
        af_cfg = getattr(config, 'atom_facts', None)
        provider = getattr(af_cfg, 'provider', 'template') if af_cfg else 'template'
        model = getattr(af_cfg, 'model', 'gpt-4o-mini') if af_cfg else ''
        estimate.provider = provider
        estimate.model = model

        pricing = PRICING.get(model, PRICING.get("gpt-4o-mini", {"input": 0.0, "output": 0.0}))

        # ── Phase 1: SWR Replay ──
        replay_count = len(dormant) + len(consolidating)
        replay_time = (replay_count / 100) * TIME_ESTIMATES["swr_replay_per_100_anchors"]
        estimate.phase_estimates["N1 Replay"] = {
            "duration": replay_time, "items": replay_count,
        }

        # ── Phase 2: Merge + Bridge ──
        merge_time = (estimate.total_anchors / 100) * TIME_ESTIMATES["merge_per_100_anchors"]
        estimate.phase_estimates["N2 Merge"] = {
            "duration": merge_time, "items": estimate.total_anchors,
        }

        # ── Phase 5b: Compression ──
        # Group eligible anchors by session
        eligible = dormant + consolidating
        sessions: dict[str, list] = {}
        for a in eligible:
            if a.embedding and a.source_session:
                sessions.setdefault(a.source_session, []).append(a)

        compression_cfg = getattr(config, 'compression', None)
        min_cluster = getattr(compression_cfg, 'min_cluster_size', 3) if compression_cfg else 3
        clusters = {k: v for k, v in sessions.items() if len(v) >= min_cluster}
        estimate.compression_clusters = len(clusters)
        estimate.compression_summaries = len(clusters) * 2  # rough: episodic + strategic per cluster

        # Compression is template-based (offline), so no LLM cost
        comp_time = len(clusters) * 0.1
        estimate.phase_estimates["5b Compression"] = {
            "duration": comp_time, "items": sum(len(v) for v in clusters.values()),
        }

        # ── Phase 5c: Atom Facts ──
        fact_cfg = getattr(config, 'atom_facts', None)
        fact_min_cluster = getattr(fact_cfg, 'min_cluster_size', 3) if fact_cfg else 3
        max_batch = getattr(fact_cfg, 'max_anchors_per_batch', 15) if fact_cfg else 15

        # Group by session AND by tag for fact extraction
        fact_clusters = {k: v for k, v in sessions.items() if len(v) >= fact_min_cluster}
        estimate.fact_clusters = len(fact_clusters)
        estimate.fact_anchors = sum(len(v) for v in fact_clusters.values())

        # Estimate fact outputs
        avg_facts_per_anchor = 2
        estimate.estimated_facts = estimate.fact_anchors * avg_facts_per_anchor

        # LLM cost estimation
        if provider == "template":
            # Offline template mode — zero API cost
            fact_input_tokens = 0
            fact_output_tokens = 0
            fact_calls = 0
            fact_llm_time = 0.0
        else:
            # API-based LLM
            batches = max(1, math.ceil(estimate.fact_anchors / max_batch))
            fact_calls = batches
            fact_input_tokens = estimate.fact_anchors * TOKEN_ESTIMATES["atom_facts_per_anchor"]
            fact_output_tokens = estimate.estimated_facts * TOKEN_ESTIMATES["atom_facts_output_per_fact"]
            fact_llm_time = batches * TIME_ESTIMATES["llm_call_overhead"]

        estimate.estimated_input_tokens = fact_input_tokens
        estimate.estimated_output_tokens = fact_output_tokens
        estimate.estimated_llm_calls = fact_calls
        estimate.llm_cost_usd = (
            (fact_input_tokens / 1000) * pricing["input"]
            + (fact_output_tokens / 1000) * pricing["output"]
        )
        estimate.llm_duration_seconds = fact_llm_time

        estimate.phase_estimates["5c Atom Facts"] = {
            "duration": fact_llm_time, "items": estimate.estimated_facts,
            "llm_calls": fact_calls, "input_tokens": fact_input_tokens,
            "output_tokens": fact_output_tokens,
        }

        # ── Phase 7: Prune ──
        prune_time = (estimate.total_anchors / 100) * TIME_ESTIMATES["prune_per_100_anchors"]
        estimate.phase_estimates["7 Prune"] = {
            "duration": prune_time, "items": estimate.total_anchors,
        }

        # ── Phase 8: Index Rebuild ──
        index_time = (estimate.total_anchors / 100) * TIME_ESTIMATES["index_rebuild_per_100_anchors"]
        estimate.phase_estimates["8 Index"] = {
            "duration": index_time, "items": estimate.total_anchors,
        }

        # ── Phase 3/4/6: fast operations ──
        estimate.phase_estimates["N3 Compress"] = {"duration": 0.1, "items": 0}
        estimate.phase_estimates["REM Emotional"] = {"duration": 0.05, "items": estimate.total_anchors}
        estimate.phase_estimates["6 Hub"] = {"duration": 0.1, "items": 0}

        # ── Totals ──
        estimate.compute_duration_seconds = sum(
            p.get("duration", 0) for p in estimate.phase_estimates.values()
        )
        estimate.total_duration_seconds = (
            estimate.compute_duration_seconds + estimate.llm_duration_seconds
        )
        estimate.compute_cost_usd = 0.0  # local compute is free
        estimate.estimated_cost_usd = estimate.llm_cost_usd

        return estimate

    def estimate_from_sleep(self, sleep_cycle,
                           dry_run: bool = False) -> CostEstimate:
        """Estimate cost from an existing SleepCycle instance."""
        return self.estimate(sleep_cycle.graph, getattr(sleep_cycle, 'cfg', None),
                           dry_run=dry_run)
