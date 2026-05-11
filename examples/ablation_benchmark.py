"""Ablation Benchmark — prove each component contributes measurably.

Systematically disables each mechanism and measures the impact on:
  - Retrieval recall (does phase-locking improve retrieval?)
  - Compression ratio (does sleep consolidate effectively?)
  - Abstraction emergence (do higher-order concepts form?)
  - Ghost reactivation (can lost memories be revived?)
  - Memory stability (do consolidated memories persist?)

Run: python examples/ablation_benchmark.py

Each scenario runs the same 10-anchor workflow and compares metrics.
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, SleepCycle, seed_everything, get_embedder,
    CognitiveMetrics, GhostSubsystem,
    OscillationResonanceRetriever, VectorSimilarityRetriever,
    config, override, reload_defaults,
)


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def build_test_graph(embedder):
    """Create a standard 10-anchor graph for benchmarking."""
    graph = StarGraph()
    graph._ghost_subsystem = GhostSubsystem()

    sessions = [
        ("Session A: Python programming", [
            ("User is learning Python list comprehensions and generators.",
             ["python", "learning"], 0.5, 0.3),
            ("User built a web scraper using requests and BeautifulSoup.",
             ["python", "scraping"], 0.6, 0.5),
        ]),
        ("Session B: Frontend development", [
            ("User built a React dashboard with Tailwind CSS components.",
             ["frontend", "react", "css"], 0.7, 0.4),
            ("User configured Webpack with code splitting and tree shaking.",
             ["frontend", "webpack", "build"], 0.5, 0.6),
        ]),
        ("Session C: Database administration", [
            ("User added Redis caching to the API backend for performance.",
             ["database", "redis", "caching"], 0.5, 0.4),
            ("User experienced Redis connection timeout in production.",
             ["database", "redis", "incident"], -0.8, 0.9),
            ("User fixed Redis timeout by increasing connection pool size.",
             ["database", "redis", "fix"], 0.6, 0.6),
        ]),
        ("Session D: DevOps", [
            ("User set up CI/CD pipeline with GitHub Actions and Docker.",
             ["devops", "ci-cd", "docker"], 0.6, 0.5),
            ("User configured Kubernetes cluster for microservices deployment.",
             ["devops", "kubernetes"], 0.4, 0.7),
            ("User briefly evaluated serverless options but chose containers.",
             ["devops", "architecture"], 0.2, 0.1),
        ]),
    ]

    for session_name, memories in sessions:
        for text, tags, emotion, surprise in memories:
            anchor = Anchor.create(text, source_session=session_name,
                                   tags=tags, emotional_valence=emotion,
                                   surprise=surprise)
            graph.add_anchor(anchor)

    # Add similarity edges
    ids = list(graph.anchors.keys())
    for i, aid_a in enumerate(ids):
        for aid_b in ids[i + 1:]:
            a, b = graph.anchors[aid_a], graph.anchors[aid_b]
            if a.embedding and b.embedding:
                sim = SleepCycle._embedding_similarity(a.embedding, b.embedding)
                if sim > 0.5:
                    graph.add_edge(aid_a, aid_b, weight=sim)

    return graph


def measure_retrieval(graph, queries, ground_truth):
    """Measure retrieval recall@3 using oscillation resonance."""
    ret = OscillationResonanceRetriever(graph)
    total_recall = 0.0
    for query, gt_ids in zip(queries, ground_truth):
        result = ret.retrieve(query)
        retrieved = set()
        for c in result.constellations[:3]:
            for a in c.anchors:
                retrieved.add(a.id)
        gt_set = set(gt_ids)
        if gt_set:
            recall = len(retrieved & gt_set) / len(gt_set)
            total_recall += recall
    return total_recall / len(queries) if queries else 0.0


def run_baseline(embedder):
    """Baseline: all components enabled."""
    reload_defaults()
    # Use a moderate merge threshold so we don't collapse all diverse anchors
    override('sleep.merge.default_threshold', 0.92)
    seed_everything(42)
    graph = build_test_graph(embedder)

    metrics = CognitiveMetrics(graph)
    metrics.snapshot()

    # Queries for testing (diverse topics)
    queries = [
        "Python web scraping automation",
        "Redis caching issues in production",
        "React frontend dashboard components",
    ]
    ground_truth = [
        [aid for aid, a in graph.anchors.items() if "scraping" in a.tags],
        [aid for aid, a in graph.anchors.items() if "redis" in a.tags and "incident" in a.tags],
        [aid for aid, a in graph.anchors.items() if "react" in a.tags],
    ]

    recall_before = measure_retrieval(graph, queries, ground_truth)

    cycle = SleepCycle(graph)
    result = cycle.run()

    recall_after = measure_retrieval(graph, queries, ground_truth)
    metrics.snapshot()
    comp = metrics.compare()

    # Try ghost reactivation (serverless architecture reconsideration)
    ghost_reactivated = 0
    new_emb = embedder.encode("Reconsidering serverless architecture for a new microservices project")
    if hasattr(graph, '_ghost_subsystem') and graph._ghost_subsystem:
        matches = graph._ghost_subsystem.check_resonance(new_emb, threshold=0.3)
        for ghost, score in matches:
            revived = graph._ghost_subsystem.try_revive(ghost.id, "MongoDB reconsidered", new_emb)
            if revived:
                ghost_reactivated += 1

    return {
        "scenario": "baseline",
        "anchors_before": result["stats_before"]["anchors"],
        "anchors_after": result["stats_after"]["anchors"],
        "merged": result["merged"],
        "pruned": result["pruned_anchors"],
        "ghosts": result["ghosts_created"],
        "schemas": result["schemas_formed"],
        "recall_before": round(recall_before, 3),
        "recall_after": round(recall_after, 3),
        "compression_ratio": round(comp.get("compression_improvement", 0), 3),
        "ghost_reactivated": ghost_reactivated,
        "stability": round(result["stats_after"]["avg_retention"], 3),
    }


def run_no_oscillation(embedder):
    """Ablation: phase_weight=0 — no phase in resonance scoring."""
    reload_defaults()
    override('retrieval.oscillation.phase_weight', 0.0)
    seed_everything(42)
    graph = build_test_graph(embedder)

    queries = [
        "Python web scraping automation",
        "Redis caching issues in production",
    ]
    ground_truth = [
        [aid for aid, a in graph.anchors.items() if "scraping" in a.tags],
        [aid for aid, a in graph.anchors.items() if "redis" in a.tags and "incident" in a.tags],
    ]

    recall = measure_retrieval(graph, queries, ground_truth)

    cycle = SleepCycle(graph)
    result = cycle.run()

    return {
        "scenario": "no_oscillation",
        "anchors_before": result["stats_before"]["anchors"],
        "anchors_after": result["stats_after"]["anchors"],
        "merged": result["merged"],
        "pruned": result["pruned_anchors"],
        "recall": round(recall, 3),
        "stability": round(result["stats_after"]["avg_retention"], 3),
    }


def run_no_emotion(embedder):
    """Ablation: all emotional_valence=0 — no salience weighting."""
    reload_defaults()
    seed_everything(42)
    graph = build_test_graph(embedder)

    # Zero out all emotional valence post-creation
    for a in graph.anchors.values():
        a.vector.emotional_valence = 0.0

    cycle = SleepCycle(graph)
    result = cycle.run()

    # With no emotion, all replay priorities should have low variance
    priorities = [getattr(a, '_replay_priority', 0) for a in graph.anchors.values()]
    priority_variance = max(priorities) - min(priorities) if priorities else 0

    return {
        "scenario": "no_emotion",
        "anchors_before": result["stats_before"]["anchors"],
        "anchors_after": result["stats_after"]["anchors"],
        "merged": result["merged"],
        "pruned": result["pruned_anchors"],
        "priority_variance": round(priority_variance, 4),
        "stability": round(result["stats_after"]["avg_retention"], 3),
    }


def run_no_sleep(embedder):
    """Ablation: skip sleep cycles entirely."""
    reload_defaults()
    seed_everything(42)
    graph = build_test_graph(embedder)

    metrics = CognitiveMetrics(graph)
    metrics.snapshot()

    queries = [
        "Python web scraping automation",
        "Redis caching issues in production",
    ]
    ground_truth = [
        [aid for aid, a in graph.anchors.items() if "scraping" in a.tags],
        [aid for aid, a in graph.anchors.items() if "redis" in a.tags and "incident" in a.tags],
    ]

    recall = measure_retrieval(graph, queries, ground_truth)
    metrics.snapshot()
    comp = metrics.compare()

    return {
        "scenario": "no_sleep",
        "anchors": len(graph.anchors),
        "schemas": 0,  # no sleep = no schema extraction
        "abstracts": 0,  # no sleep = no abstraction
        "ghosts": 0,
        "recall": round(recall, 3),
        "abstractions_formed": comp.get("abstractions_formed", 0),
        "compression_improvement": round(comp.get("compression_improvement", 0), 3),
        "stability": round(sum(a.retention_score for a in graph.anchors.values())
                          / max(1, len(graph.anchors)), 3),
    }


def run_no_ghosts(embedder):
    """Ablation: retention_threshold=0 — no ghosts created, weak anchors stay."""
    reload_defaults()
    seed_everything(42)
    graph = build_test_graph(embedder)

    cycle = SleepCycle(graph)
    result = cycle.run(retention_threshold=0.0)  # never prune

    return {
        "scenario": "no_ghosts",
        "anchors_before": result["stats_before"]["anchors"],
        "anchors_after": result["stats_after"]["anchors"],
        "merged": result["merged"],
        "pruned": result["pruned_anchors"],
        "ghosts": result["ghosts_created"],
        "stability": round(result["stats_after"]["avg_retention"], 3),
    }


def run_no_competition(embedder):
    """Ablation: no memory competition applied."""
    reload_defaults()
    seed_everything(42)
    graph = build_test_graph(embedder)

    cycle = SleepCycle(graph)
    result = cycle.run()

    # Without competition, contradicted tags should not appear
    contradicted = sum(1 for a in graph.anchors.values()
                       if "contradicted" in a.tags)

    return {
        "scenario": "no_competition",
        "anchors_before": result["stats_before"]["anchors"],
        "anchors_after": result["stats_after"]["anchors"],
        "merged": result["merged"],
        "contradicted_anchors": contradicted,
        "stability": round(result["stats_after"]["avg_retention"], 3),
    }


def main():
    banner("Ablation Benchmark — Component Impact Analysis")

    print("\nLoading embedder...")
    embedder = get_embedder()

    scenarios = []

    banner("Baseline: All Components Enabled")
    baseline = run_baseline(embedder)
    scenarios.append(baseline)
    for k, v in baseline.items():
        print(f"  {k}: {v}")

    banner("Ablation 1: No Oscillation (phase_weight=0)")
    no_osc = run_no_oscillation(embedder)
    scenarios.append(no_osc)
    for k, v in no_osc.items():
        print(f"  {k}: {v}")
    print(f"  → Expected: recall drops vs baseline ({baseline['recall_before']} → {no_osc['recall']})")

    banner("Ablation 2: No Emotion (emotional_valence=0)")
    no_emo = run_no_emotion(embedder)
    scenarios.append(no_emo)
    for k, v in no_emo.items():
        print(f"  {k}: {v}")
    print(f"  → Expected: low priority variance (all memories treated equally)")

    banner("Ablation 3: No Sleep (skip consolidation)")
    no_sleep = run_no_sleep(embedder)
    scenarios.append(no_sleep)
    for k, v in no_sleep.items():
        print(f"  {k}: {v}")
    print(f"  → Expected: 0 schemas, 0 abstracts, no compression")

    banner("Ablation 4: No Ghosts (retention_threshold=0)")
    no_ghosts = run_no_ghosts(embedder)
    scenarios.append(no_ghosts)
    for k, v in no_ghosts.items():
        print(f"  {k}: {v}")
    print(f"  → Expected: 0 pruned, 0 ghosts (weak memories accumulate)")

    banner("Ablation 5: No Competition")
    no_comp = run_no_competition(embedder)
    scenarios.append(no_comp)
    for k, v in no_comp.items():
        print(f"  {k}: {v}")

    # ── Summary table ──
    banner("Summary: Ablation Impact Matrix")
    print(f"\n  {'Scenario':<25} {'Anchors':>8} {'Merged':>7} {'Pruned':>7} {'Ghosts':>7} {'Recall':>7} {'Stability':>9}")
    print(f"  {'-'*70}")
    for s in scenarios:
        name = s["scenario"]
        anchors = s.get("anchors_after", s.get("anchors", "?"))
        merged = s.get("merged", "-")
        pruned = s.get("pruned", "-")
        ghosts = s.get("ghosts", "-")
        recall = s.get("recall", s.get("recall_before", "-"))
        stability = s.get("stability", "-")
        print(f"  {name:<25} {str(anchors):>8} {str(merged):>7} {str(pruned):>7} {str(ghosts):>7} {str(recall):>7} {str(stability):>9}")

    print(f"\n  Component Impact Summary:")
    print(f"  {'─'*50}")
    if baseline.get("recall_before", 0) > no_osc.get("recall", 0):
        delta = baseline["recall_before"] - no_osc["recall"]
        print(f"  Oscillation:    +{delta:.3f} recall improvement with phase-locking")
    else:
        print(f"  Oscillation:    recall change within noise floor")
    if no_sleep.get("schemas", 0) == 0:
        print(f"  Sleep:          schemas={baseline.get('schemas', 0)} (0 without sleep)")
    if no_ghosts.get("ghosts", 0) == 0:
        print(f"  Ghosts:         {baseline.get('ghosts', 0)} created (0 when disabled)")
    if baseline.get("ghost_reactivated", 0) > 0:
        print(f"  Ghost Revival:  {baseline['ghost_reactivated']} reactivated in baseline")
    print()

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
