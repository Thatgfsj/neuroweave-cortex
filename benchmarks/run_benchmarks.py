"""Public Benchmark Suite — standardized evaluation for NWC memory system.

Three core benchmarks aligned with industry standards:
  - LongMemEval-style: long-term retention over 100+ turns
  - LoCoMo-style: multi-turn memory consistency
  - BEAM-style: large-scale retrieval precision (1K–10K anchors)

Usage:
    python benchmarks/run_benchmarks.py --quick          # ~30s smoke test
    python benchmarks/run_benchmarks.py --standard       # full suite (~5 min)
    python benchmarks/run_benchmarks.py --large          # 10K anchor stress test

Output: JSON report + terminal dashboard of Recall@K, MRR, precision metrics.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure star_graph is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from star_graph import MemoryManager, Anchor, get_embedder, seed_everything, config
from star_graph.bm25 import BM25Index


# ── Metrics ───────────────────────────────────────────────────

@dataclass
class BenchmarkMetrics:
    """Standard retrieval metrics comparable to published results."""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0                     # Mean Reciprocal Rank
    ndcg_at_10: float = 0.0              # Normalized Discounted Cumulative Gain
    precision_at_10: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    queries_run: int = 0
    total_memories: int = 0

    def to_dict(self) -> dict:
        return {
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "precision@10": round(self.precision_at_10, 4),
            "latency_ms_mean": round(self.latency_ms_mean, 2),
            "latency_ms_p95": round(self.latency_ms_p95, 2),
            "queries_run": self.queries_run,
            "total_memories": self.total_memories,
        }

    def report(self) -> str:
        return (
            f"Recall@1={self.recall_at_1:.3f} @3={self.recall_at_3:.3f} "
            f"@5={self.recall_at_5:.3f} @10={self.recall_at_10:.3f}  "
            f"MRR={self.mrr:.3f}  NDCG@10={self.ndcg_at_10:.3f}  "
            f"Latency(mean/p95)={self.latency_ms_mean:.1f}/{self.latency_ms_p95:.1f}ms"
        )


# ── Helpers ───────────────────────────────────────────────────

def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
    return dcg / max(0.001, ideal)


# ── LongMemEval-style: Long-term Retention ────────────────────

def run_long_term_benchmark(n_sessions: int = 8, turns_per_session: int = 15,
                            probes: int = 20) -> BenchmarkMetrics:
    """Simulate multi-session conversations with planted memory probes.

    Probes are inserted in early sessions, then queried from late sessions.
    Measures how well the system retains information over long contexts.
    """
    seed_everything(42)
    mgr = MemoryManager()
    embedder = get_embedder()

    probe_ids: list[str] = []
    probe_texts: dict[str, str] = {}
    probe_sessions: dict[str, int] = {}

    # Plant distinctive facts in early sessions (first 60%)
    early_sessions = max(1, int(n_sessions * 0.6))
    for i in range(probes):
        session = random.randint(0, early_sessions - 1)
        text = (
            f"IMPORTANT FACT #{i}: User's preferred database for project "
            f"alpha_{i} is {random.choice(['PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'SQLite'])} "
            f"with {random.choice(['replication', 'sharding', 'caching', 'backup', 'monitoring'])} "
            f"configured at {random.randint(1000, 9999)} port."
        )
        tags = [f"project_alpha_{i}", "database", "preference"]
        anchor = mgr.remember(text, tags=tags, importance=0.85 + random.random() * 0.15)
        probe_ids.append(anchor.id)
        probe_texts[anchor.id] = text
        probe_sessions[anchor.id] = session

    # Fill remaining sessions with background noise
    topics = ["debugging", "deployment", "testing", "refactoring", "monitoring",
              "performance", "security", "docs", "ci-cd", "code-review"]
    for s in range(n_sessions):
        for t in range(turns_per_session):
            topic = random.choice(topics)
            text = (
                f"Session {s} turn {t}: Discussed {topic} — encountered {random.choice(['error', 'warning', 'info'])} "
                f"in {random.choice(['auth', 'api', 'db', 'cache', 'queue'])} module. "
                f"Resolution: {random.choice(['fixed', 'deferred', 'documented', 'monitored'])}."
            )
            mgr.remember(text, tags=[topic, f"session_{s}"], importance=0.2 + random.random() * 0.3)

    # Sleep consolidation after ingestion
    mgr.sleep()

    # Query probes from late sessions
    latencies: list[float] = []
    recall_1 = recall_3 = recall_5 = recall_10 = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    precision_10_sum = 0.0

    for pid in probe_ids:
        expected_text = probe_texts[pid]
        # Extract distinguishing keywords from the probe text
        keywords = [w for w in expected_text.split()
                    if w.isalpha() and len(w) > 5 and w.isupper()][:2]
        if not keywords:
            keywords = expected_text.split()[2:5]
        query = " ".join(keywords) if keywords else f"project alpha_{pid}"

        t0 = time.time()
        result = mgr.recall(query, max_items=10)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = [item.anchor.id for item in result.items if item.anchor]
        relevant = {pid}

        recall_1 += _recall_at_k(retrieved_ids, relevant, 1)
        recall_3 += _recall_at_k(retrieved_ids, relevant, 3)
        recall_5 += _recall_at_k(retrieved_ids, relevant, 5)
        recall_10 += _recall_at_k(retrieved_ids, relevant, 10)
        mrr_sum += _mrr(retrieved_ids, relevant)
        ndcg_sum += _ndcg_at_k(retrieved_ids, relevant, 10)
        precision_10_sum += len(set(retrieved_ids[:10]) & relevant) / max(1, len(retrieved_ids[:10]))

    n = probes
    latencies.sort()
    return BenchmarkMetrics(
        recall_at_1=round(recall_1 / n, 4),
        recall_at_3=round(recall_3 / n, 4),
        recall_at_5=round(recall_5 / n, 4),
        recall_at_10=round(recall_10 / n, 4),
        mrr=round(mrr_sum / n, 4),
        ndcg_at_10=round(ndcg_sum / n, 4),
        precision_at_10=round(precision_10_sum / n, 4),
        latency_ms_mean=round(statistics.mean(latencies), 2),
        latency_ms_p95=round(latencies[int(n * 0.95)] if n > 1 else latencies[0], 2),
        queries_run=n,
        total_memories=n_sessions * turns_per_session + probes,
    )


# ── LoCoMo-style: Multi-turn Consistency ─────────────────────

def run_consistency_benchmark(n_rounds: int = 5, turns_per_round: int = 12) -> BenchmarkMetrics:
    """Test multi-turn memory consistency — evolving facts over time.

    Plants a fact, then updates it across conversation rounds. Tests whether
    the latest version is retrieved (no stale memory interference).
    """
    seed_everything(123)
    mgr = MemoryManager()
    embedder = get_embedder()

    tracked_facts: dict[str, str] = {}  # fact_name → latest_anchor_id
    fact_versions: dict[str, list[str]] = defaultdict(list)  # fact_name → [anchor_ids]

    fact_names = [
        "user_preferred_language", "project_deadline", "server_endpoint",
        "db_connection_string", "api_key_name", "primary_branch",
        "deployment_target", "monitoring_tool",
    ]

    # Plant and evolve facts across rounds
    for r in range(n_rounds):
        for fn in fact_names:
            if r == 0:
                text = f"FACT [{fn}]: Initial value set to v{r}_{fn}_initial"
            else:
                text = f"FACT [{fn}]: UPDATED to v{r}_{fn}_revised (replaces previous value)"
            anchor = mgr.remember(
                text, tags=[fn, "fact", f"round_{r}"],
                importance=0.7 + random.random() * 0.3,
            )
            tracked_facts[fn] = anchor.id
            fact_versions[fn].append(anchor.id)

        # Fill with background noise
        for t in range(turns_per_round):
            topic = random.choice(["debug", "chat", "review", "planning", "analysis"])
            mgr.remember(
                f"Round {r} turn {t}: {topic} discussion about {random.choice(fact_names)}",
                tags=[topic, f"round_{r}"], importance=0.2,
            )

        mgr.micro_consolidate()

    mgr.sleep()

    # Query: should return LATEST version, not stale versions
    latencies: list[float] = []
    recall_1 = recall_3 = recall_5 = recall_10 = 0.0
    mrr_sum = ndcg_sum = precision_10_sum = 0.0

    for fn in fact_names:
        latest_id = tracked_facts[fn]
        query = f"what is the current value of {fn.replace('_', ' ')}"

        t0 = time.time()
        result = mgr.recall(query, max_items=10)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = [item.anchor.id for item in result.items if item.anchor]
        relevant = {latest_id}

        recall_1 += _recall_at_k(retrieved_ids, relevant, 1)
        recall_3 += _recall_at_k(retrieved_ids, relevant, 3)
        recall_5 += _recall_at_k(retrieved_ids, relevant, 5)
        recall_10 += _recall_at_k(retrieved_ids, relevant, 10)
        mrr_sum += _mrr(retrieved_ids, relevant)
        ndcg_sum += _ndcg_at_k(retrieved_ids, relevant, 10)
        precision_10_sum += len(set(retrieved_ids[:10]) & relevant) / max(1, len(retrieved_ids[:10]))

    n = len(fact_names)
    latencies.sort()
    return BenchmarkMetrics(
        recall_at_1=round(recall_1 / n, 4),
        recall_at_3=round(recall_3 / n, 4),
        recall_at_5=round(recall_5 / n, 4),
        recall_at_10=round(recall_10 / n, 4),
        mrr=round(mrr_sum / n, 4),
        ndcg_at_10=round(ndcg_sum / n, 4),
        precision_at_10=round(precision_10_sum / n, 4),
        latency_ms_mean=round(statistics.mean(latencies), 2),
        latency_ms_p95=round(latencies[int(n * 0.95)] if n > 1 else latencies[0], 2),
        queries_run=n,
        total_memories=n_rounds * (len(fact_names) + turns_per_round),
    )


# ── BEAM-style: Large-scale Retrieval Precision ──────────────

def run_large_scale_benchmark(n_anchors: int = 1000) -> BenchmarkMetrics:
    """BEAM-style large-scale retrieval test.

    Ingests N anchors with controlled ground-truth queries, then measures
    retrieval precision under increasing scale.
    """
    seed_everything(777)
    mgr = MemoryManager()

    # Categories with distinct vocabularies
    categories = {
        "medical": ["diagnosis", "treatment", "symptom", "patient", "prescription",
                     "surgery", "therapy", "clinical", "prognosis", "pathology"],
        "legal": ["statute", "plaintiff", "defendant", "jurisdiction", "appeal",
                   "testimony", "verdict", "affidavit", "litigation", "arbitration"],
        "engineering": ["pipeline", "throughput", "latency", "scalability", "redundancy",
                        "fault-tolerance", "load-balancer", "microservice", "endpoint", "protocol"],
        "finance": ["portfolio", "dividend", "volatility", "liquidity", "arbitrage",
                     "hedge", "derivative", "equity", "bond", "leverage"],
        "cooking": ["braise", "marinate", "simmer", "caramelize", "emulsify",
                     "deglaze", "blanch", "julienne", "proof", "temper"],
    }

    # Plant ground-truth anchors with distinctive queries
    gt_anchors: dict[str, str] = {}  # anchor_id → query
    for i in range(min(100, n_anchors // 10)):
        cat, words = random.choice(list(categories.items()))
        w1, w2 = random.sample(words, 2)
        text = f"GROUND TRUTH #{i}: The {cat} procedure for {w1} involves specialized {w2} techniques "
        text += f"documented in reference manual GT-{i:04d}. Unique identifier: gt_{w1}_{w2}_{i}"
        anchor = mgr.remember(
            text, tags=[cat, f"gt_{i}", w1, w2],
            importance=0.5 + random.random() * 0.5,
        )
        gt_anchors[anchor.id] = f"{w1} {w2} reference manual"

    # Fill with background noise
    for i in range(n_anchors - len(gt_anchors)):
        cat, words = random.choice(list(categories.items()))
        w1, w2, w3 = random.sample(words, 3)
        text = f"Noise #{i}: Discussion about {w1}, {w2}, and {w3} in the context of {cat} operations."
        mgr.remember(text, tags=[cat, "noise", w1], importance=0.1 + random.random() * 0.2)

    mgr.sleep()

    # Query all ground-truth anchors
    latencies: list[float] = []
    recall_1 = recall_3 = recall_5 = recall_10 = 0.0
    mrr_sum = ndcg_sum = precision_10_sum = 0.0

    for aid, query in gt_anchors.items():
        t0 = time.time()
        result = mgr.recall(query, max_items=10)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = [item.anchor.id for item in result.items if item.anchor]
        relevant = {aid}

        recall_1 += _recall_at_k(retrieved_ids, relevant, 1)
        recall_3 += _recall_at_k(retrieved_ids, relevant, 3)
        recall_5 += _recall_at_k(retrieved_ids, relevant, 5)
        recall_10 += _recall_at_k(retrieved_ids, relevant, 10)
        mrr_sum += _mrr(retrieved_ids, relevant)
        ndcg_sum += _ndcg_at_k(retrieved_ids, relevant, 10)
        precision_10_sum += len(set(retrieved_ids[:10]) & relevant) / max(1, len(retrieved_ids[:10]))

    n = len(gt_anchors)
    latencies.sort()
    return BenchmarkMetrics(
        recall_at_1=round(recall_1 / n, 4),
        recall_at_3=round(recall_3 / n, 4),
        recall_at_5=round(recall_5 / n, 4),
        recall_at_10=round(recall_10 / n, 4),
        mrr=round(mrr_sum / n, 4),
        ndcg_at_10=round(ndcg_sum / n, 4),
        precision_at_10=round(precision_10_sum / n, 4),
        latency_ms_mean=round(statistics.mean(latencies), 2),
        latency_ms_p95=round(latencies[int(n * 0.95)] if n > 1 else latencies[0], 2),
        queries_run=n,
        total_memories=n_anchors,
    )


# ── CLI ───────────────────────────────────────────────────────

def run_all(scale: str = "quick") -> dict:
    """Run all three benchmarks and return combined report.

    Args:
        scale: "quick" (~30s), "standard" (~5min), "large" (10K anchors)
    """
    if scale == "quick":
        lt = run_long_term_benchmark(n_sessions=4, turns_per_session=8, probes=10)
        con = run_consistency_benchmark(n_rounds=3, turns_per_round=6)
        ls = run_large_scale_benchmark(n_anchors=200)
    elif scale == "large":
        lt = run_long_term_benchmark(n_sessions=12, turns_per_session=20, probes=50)
        con = run_consistency_benchmark(n_rounds=8, turns_per_round=20)
        ls = run_large_scale_benchmark(n_anchors=10000)
    else:  # standard
        lt = run_long_term_benchmark(n_sessions=8, turns_per_session=15, probes=20)
        con = run_consistency_benchmark(n_rounds=5, turns_per_round=12)
        ls = run_large_scale_benchmark(n_anchors=1000)

    report = {
        "benchmark": "neuroweave-cortex",
        "version": "1.0.4-dev",
        "scale": scale,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "long_term_retention": lt.to_dict(),
        "multi_turn_consistency": con.to_dict(),
        "large_scale_precision": ls.to_dict(),
        "aggregate": {
            "mean_recall_at_5": round(
                (lt.recall_at_5 + con.recall_at_5 + ls.recall_at_5) / 3, 4),
            "mean_mrr": round(
                (lt.mrr + con.mrr + ls.mrr) / 3, 4),
            "mean_latency_ms": round(
                (lt.latency_ms_mean + con.latency_ms_mean + ls.latency_ms_mean) / 3, 2),
            "total_queries": lt.queries_run + con.queries_run + ls.queries_run,
            "max_memories": max(lt.total_memories, con.total_memories, ls.total_memories),
        },
    }
    return report


def print_dashboard(report: dict) -> None:
    """Print a formatted terminal dashboard."""
    agg = report["aggregate"]
    lt = report["long_term_retention"]
    con = report["multi_turn_consistency"]
    ls = report["large_scale_precision"]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     NeuroWeave Cortex — Public Benchmark Dashboard           ║
║     v{report['version']}  |  Scale: {report['scale']:<10}  |  {report['timestamp']}    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Long-Term Retention (LongMemEval-style)                     ║
║    Recall@1: {lt['recall@1']:<7}  @3: {lt['recall@3']:<7}  @5: {lt['recall@5']:<7}  @10: {lt['recall@10']:<7}  ║
║    MRR: {lt['mrr']:<8}  NDCG@10: {lt['ndcg@10']:<8}                              ║
║    Latency: mean={lt['latency_ms_mean']}ms  p95={lt['latency_ms_p95']}ms                              ║
║                                                              ║
║  Multi-Turn Consistency (LoCoMo-style)                       ║
║    Recall@1: {con['recall@1']:<7}  @3: {con['recall@3']:<7}  @5: {con['recall@5']:<7}  @10: {con['recall@10']:<7}  ║
║    MRR: {con['mrr']:<8}  NDCG@10: {con['ndcg@10']:<8}                              ║
║    Latency: mean={con['latency_ms_mean']}ms  p95={con['latency_ms_p95']}ms                              ║
║                                                              ║
║  Large-Scale Precision (BEAM-style)                          ║
║    Recall@1: {ls['recall@1']:<7}  @3: {ls['recall@3']:<7}  @5: {ls['recall@5']:<7}  @10: {ls['recall@10']:<7}  ║
║    MRR: {ls['mrr']:<8}  NDCG@10: {ls['ndcg@10']:<8}                              ║
║    Latency: mean={ls['latency_ms_mean']}ms  p95={ls['latency_ms_p95']}ms                              ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  AGGREGATE                                                   ║
║    Mean Recall@5: {agg['mean_recall_at_5']:<7}  Mean MRR: {agg['mean_mrr']:<7}                       ║
║    Mean Latency: {agg['mean_latency_ms']}ms  |  {agg['total_queries']} queries over {agg['max_memories']} memories     ║
║    Target: Recall@5 ≥ 0.90 (Mem0 2026 benchmark)             ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="NWC Public Benchmark Suite")
    ap.add_argument("--quick", action="store_true", help="~30s smoke test")
    ap.add_argument("--standard", action="store_true", help="Full suite (~5 min)")
    ap.add_argument("--large", action="store_true", help="10K anchor stress test")
    ap.add_argument("--output", type=str, help="Save JSON report to file")
    args = ap.parse_args()

    scale = "quick"
    if args.standard:
        scale = "standard"
    elif args.large:
        scale = "large"

    print(f"\nRunning NWC benchmark suite ({scale} scale)...\n")
    report = run_all(scale)
    print_dashboard(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")

    # Exit code based on target
    agg = report["aggregate"]
    if agg["mean_recall_at_5"] >= 0.90:
        print("PASS: Recall@5 >= 0.90 (Mem0 2026 target)")
        sys.exit(0)
    elif agg["mean_recall_at_5"] >= 0.75:
        print("WARN: Recall@5 >= 0.75 (below Mem0 target of 0.90)")
        sys.exit(0)
    else:
        print("NOTE: Recall@5 < 0.75 — consider tuning retrieval weights")
        sys.exit(0)
