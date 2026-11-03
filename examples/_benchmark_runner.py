"""Benchmark orchestration for the memory benchmark.

Extracted verbatim from ``examples/memory_benchmark.py`` (issue #10) to keep the
CLI entry point small. Function bodies are unchanged.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from star_graph import (
    StarGraph, Anchor, SleepCycle, seed_everything, get_embedder,
    OscillationResonanceRetriever, VectorSimilarityRetriever,
    HybridFusionRetriever, GhostSubsystem, reload_defaults,
)

if __package__:
    from ._benchmark_support import (
        ConversationTurn, TestQuery, SyntheticDataGenerator,
        RawHistoryBaseline, TFIDFBaseline,
        recall_at_k, content_recall_at_k, content_interference_score, mrr,
        banner, _estimate_graph_tokens, _print_summary,
    )
else:
    from _benchmark_support import (
        ConversationTurn, TestQuery, SyntheticDataGenerator,
        RawHistoryBaseline, TFIDFBaseline,
        recall_at_k, content_recall_at_k, content_interference_score, mrr,
        banner, _estimate_graph_tokens, _print_summary,
    )


# ═══════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

def run_benchmark(quick: bool = False) -> dict:
    """Run the full 5-category benchmark suite."""
    reload_defaults()
    seed_everything(42)

    num_sessions = 6 if quick else 12
    turns_per_session = 80 if quick else 420

    banner("Star Graph Memory — Comprehensive Benchmark Suite")
    print(f"  Sessions: {num_sessions} | Turns/session: ~{turns_per_session}")
    print(f"  Total turns: ~{num_sessions * turns_per_session}")

    # ── Generate data ──
    banner("Phase 1: Generating Synthetic Dataset")
    generator = SyntheticDataGenerator(seed=42)
    turns, queries = generator.generate(num_sessions, turns_per_session)
    print(f"  Generated {len(turns)} conversation turns across {num_sessions} sessions")
    print(f"  Generated {len(queries)} test queries across 5 categories")

    # ── Build baselines ──
    banner("Phase 2: Building Baselines")
    raw = RawHistoryBaseline(turns)
    tfidf = TFIDFBaseline(turns)
    print(f"  Raw history: {raw.token_count} tokens")
    print(f"  TF-IDF: {tfidf.token_count} tokens (no compression)")

    # ── Build star graph ──
    banner("Phase 3: Building Star Graph Memory")
    embedder = get_embedder()
    graph = StarGraph()
    graph._ghost_subsystem = GhostSubsystem()

    for turn in turns:
        anchor = Anchor.create(
            turn.text,
            source_session=turn.session_id,
            tags=turn.tags,
            emotional_valence=turn.emotional_valence,
            importance=turn.importance,
        )
        # Set session-relative timestamps so phase derivation differentiates sessions
        session_num = int(turn.session_id[1:])
        # Spread sessions across recent hours (1-12 hours ago for 12 sessions)
        hours_ago = (num_sessions - session_num + 1) * 4 + random.uniform(0, 2)
        anchor.created_at = time.time() - hours_ago * 3600
        anchor.last_activated_at = anchor.created_at
        graph.add_anchor(anchor)

    # Build similarity edges with tag-aware boosting for cross-session connectivity.
    # Use ANN-indexed neighbor lookup for sub-quadratic edge construction.
    from star_graph.index import ANNIndex
    ann = ANNIndex(dim=len(graph.anchors[list(graph.anchors.keys())[0]].embedding))
    for aid, anchor in graph.anchors.items():
        if anchor.embedding:
            ann.add(aid, anchor.embedding)

    ids = list(graph.anchors.keys())
    # Group anchors by tag for guaranteed tag-aware connectivity
    tag_to_aids: dict[str, list[str]] = {}
    for aid in ids:
        a = graph.anchors[aid]
        for tag in a.tags:
            tag_to_aids.setdefault(tag, []).append(aid)

    # 1. Build edges within tag groups (guaranteed cross-session connectivity)
    tag_edges_built = 0
    for tag, tag_aids in tag_to_aids.items():
        for i, aid_a in enumerate(tag_aids):
            for aid_b in tag_aids[i + 1:]:
                a, b = graph.anchors[aid_a], graph.anchors[aid_b]
                if a.embedding and b.embedding:
                    key = graph._key(aid_a, aid_b)
                    if key in graph.edges:
                        continue
                    sim = _cosine_sim(a.embedding, b.embedding)
                    tag_ov = len(set(a.tags) & set(b.tags))
                    sim = min(1.0, sim * 1.25 + tag_ov * 0.08)
                    if sim > 0.48:
                        graph.add_edge(aid_a, aid_b, weight=sim, edge_type="semantic")
                        tag_edges_built += 1

    # 2. Build edges via ANN for broader connectivity (cosine > 0.6, no tag overlap)
    ann_edges_built = 0
    for aid_a in ids:
        a = graph.anchors[aid_a]
        if not a.embedding:
            continue
        neighbors = ann.query(a.embedding, k=30)
        for aid_b, score in neighbors:
            if aid_b <= aid_a:  # avoid duplicates (ANN returns self + both directions)
                continue
            key = graph._key(aid_a, aid_b)
            if key in graph.edges:
                continue
            b = graph.anchors.get(aid_b)
            if not b or not b.embedding:
                continue
            sim = _cosine_sim(a.embedding, b.embedding)
            tag_ov = len(set(a.tags) & set(b.tags))
            threshold = 0.48 if tag_ov > 0 else 0.6
            if sim > threshold:
                graph.add_edge(aid_a, aid_b, weight=sim, edge_type="topical")
                ann_edges_built += 1

    print(f"  Anchors: {len(graph.anchors)}")
    print(f"  Edges: {len(graph.edges)} (tag-group: {tag_edges_built}, ANN: {ann_edges_built})")

    # ── Run sleep consolidation ──
    banner("Phase 4: Sleep Consolidation")
    cycle = SleepCycle(graph)
    sleep_result = cycle.run()
    print(f"  Before: {sleep_result['stats_before']['anchors']} anchors")
    print(f"  Merged: {sleep_result['merged']}")
    print(f"  Pruned: {sleep_result['pruned_anchors']} → {sleep_result['ghosts_created']} ghosts")
    print(f"  Schemas: {sleep_result['schemas_formed']}")
    print(f"  After: {sleep_result['stats_after']['anchors']} anchors")

    # ── Run benchmarks ──
    banner("Phase 5: Running Benchmarks")

    results = {
        "config": {"sessions": num_sessions, "turns_per_session": turns_per_session,
                    "total_turns": len(turns)},
        "sleep_stats": {
            "anchors_before": sleep_result["stats_before"]["anchors"],
            "anchors_after": sleep_result["stats_after"]["anchors"],
            "merged": sleep_result["merged"],
            "pruned": sleep_result["pruned_anchors"],
            "ghosts": sleep_result["ghosts_created"],
            "schemas": sleep_result["schemas_formed"],
            "compression_ratio": round(
                sleep_result["stats_after"]["anchors"] / max(1, sleep_result["stats_before"]["anchors"]), 3
            ),
        },
        "categories": {},
        "baselines": {},
    }

    # Run each category
    categories = defaultdict(list)
    for q in queries:
        categories[q.category].append(q)

    for cat_name, cat_queries in categories.items():
        cat_results = _benchmark_category(graph, raw, tfidf, cat_queries, turns)
        results["categories"][cat_name] = cat_results

    # Compute aggregate content-recall scores
    all_cr3 = []
    all_cr5 = []
    all_vec_cr3 = []
    all_vec_cr5 = []
    all_hyb_cr3 = []
    all_hyb_cr5 = []
    all_interf = []
    all_hyb_interf = []
    for cat_name, cat_data in results["categories"].items():
        sg = cat_data.get("star_graph", {})
        vec = cat_data.get("vector_only", {})
        hyb = cat_data.get("hybrid_fusion", {})
        all_cr3.append(sg.get("content_recall@3", 0))
        all_cr5.append(sg.get("content_recall@5", 0))
        all_vec_cr3.append(vec.get("content_recall@3", 0))
        all_vec_cr5.append(vec.get("content_recall@5", 0))
        all_hyb_cr3.append(hyb.get("content_recall@3", 0))
        all_hyb_cr5.append(hyb.get("content_recall@5", 0))
        if sg.get("interference_score", 0) > 0:
            all_interf.append(sg["interference_score"])
        if hyb.get("interference_score", 0) > 0:
            all_hyb_interf.append(hyb["interference_score"])

    results["aggregate"] = {
        "star_graph": {
            "content_recall@3": round(sum(all_cr3) / len(all_cr3), 3) if all_cr3 else 0,
            "content_recall@5": round(sum(all_cr5) / len(all_cr5), 3) if all_cr5 else 0,
            "interference_score": round(sum(all_interf) / len(all_interf), 3) if all_interf else 0,
            "tokens": _estimate_graph_tokens(graph),
        },
        "vector_only": {
            "content_recall@3": round(sum(all_vec_cr3) / len(all_vec_cr3), 3) if all_vec_cr3 else 0,
            "content_recall@5": round(sum(all_vec_cr5) / len(all_vec_cr5), 3) if all_vec_cr5 else 0,
        },
        "hybrid_fusion": {
            "content_recall@3": round(sum(all_hyb_cr3) / len(all_hyb_cr3), 3) if all_hyb_cr3 else 0,
            "content_recall@5": round(sum(all_hyb_cr5) / len(all_hyb_cr5), 3) if all_hyb_cr5 else 0,
            "interference_score": round(sum(all_hyb_interf) / len(all_hyb_interf), 3) if all_hyb_interf else 0,
        },
        "raw_history": {
            "tokens": raw.token_count,
        },
        "tfidf": {
            "tokens": tfidf.token_count,
        },
    }

    # ── Print summary ──
    _print_summary(results)

    # Save results
    results_path = Path(__file__).parent / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    return results


def _benchmark_category(graph: StarGraph, raw: RawHistoryBaseline,
                         tfidf: TFIDFBaseline, queries: list[TestQuery],
                         turns: list[ConversationTurn]) -> dict:
    """Run one category of benchmarks against all baselines."""
    osc_ret = OscillationResonanceRetriever(graph)
    vec_ret = VectorSimilarityRetriever(graph)
    hyb_ret = HybridFusionRetriever(graph)

    cat_results = {
        "num_queries": len(queries),
        "description": queries[0].category if queries else "",
        "star_graph": {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
                        "content_recall@3": 0.0, "content_recall@5": 0.0,
                        "mrr": 0.0, "interference_score": 0.0},
        "vector_only": {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
                         "content_recall@3": 0.0},
        "hybrid_fusion": {"content_recall@3": 0.0, "content_recall@5": 0.0,
                          "interference_score": 0.0},
    }

    sg_retrieved = []     # anchor IDs
    sg_texts = []         # anchor texts
    vec_retrieved = []
    vec_texts = []
    hyb_texts = []
    gt_list = []
    content_scores_sg = []
    content_scores_vec = []
    content_scores_hyb = []
    interference_scores = []
    interference_scores_hyb = []

    for q in queries:
        # Star Graph oscillation resonance
        result = osc_ret.retrieve(q.query)
        sg_ids, sg_t = [], []
        for c in result.constellations:
            for a in c.anchors:
                sg_ids.append(a.id)
                sg_t.append(a.text)
        sg_retrieved.append(sg_ids)
        sg_texts.append(sg_t)

        # Vector only
        vresult = vec_ret.retrieve(q.query)
        vec_ids, vec_t = [], []
        for c in vresult.constellations:
            for a in c.anchors:
                vec_ids.append(a.id)
                vec_t.append(a.text)
        vec_retrieved.append(vec_ids)
        vec_texts.append(vec_t)

        # Hybrid fusion
        hresult = hyb_ret.retrieve(q.query)
        hyb_t = []
        for c in hresult.constellations:
            for a in c.anchors:
                hyb_t.append(a.text)
        hyb_texts.append(hyb_t)

        gt_list.append(q.ground_truth_ids if q.ground_truth_ids else [])

        # Content-based recall
        if q.required_keywords:
            content_scores_sg.append(
                content_recall_at_k(sg_t, q.required_keywords, 3))
            content_scores_vec.append(
                content_recall_at_k(vec_t, q.required_keywords, 3))
            content_scores_hyb.append(
                content_recall_at_k(hyb_t, q.required_keywords, 3))

        # Interference score
        if q.forbidden_keywords:
            interference_scores.append(
                content_interference_score(sg_t, q.required_keywords, q.forbidden_keywords))
            interference_scores_hyb.append(
                content_interference_score(hyb_t, q.required_keywords, q.forbidden_keywords))

    # ID-based metrics (strict)
    if any(gt_list):
        cat_results["star_graph"]["recall@1"] = round(
            sum(recall_at_k(r, g, 1) for r, g in zip(sg_retrieved, gt_list) if g) / max(1, sum(1 for g in gt_list if g)), 3)
        cat_results["star_graph"]["recall@3"] = round(
            sum(recall_at_k(r, g, 3) for r, g in zip(sg_retrieved, gt_list) if g) / max(1, sum(1 for g in gt_list if g)), 3)
        cat_results["star_graph"]["recall@5"] = round(
            sum(recall_at_k(r, g, 5) for r, g in zip(sg_retrieved, gt_list) if g) / max(1, sum(1 for g in gt_list if g)), 3)
        cat_results["star_graph"]["mrr"] = round(
            mrr([r for r, g in zip(sg_retrieved, gt_list) if g],
                [g for r, g in zip(sg_retrieved, gt_list) if g]), 3)

        cat_results["vector_only"]["recall@1"] = round(
            sum(recall_at_k(r, g, 1) for r, g in zip(vec_retrieved, gt_list) if g) / max(1, sum(1 for g in gt_list if g)), 3)
        cat_results["vector_only"]["recall@3"] = round(
            sum(recall_at_k(r, g, 3) for r, g in zip(vec_retrieved, gt_list) if g) / max(1, sum(1 for g in gt_list if g)), 3)
        cat_results["vector_only"]["recall@5"] = round(
            sum(recall_at_k(r, g, 5) for r, g in zip(vec_retrieved, gt_list) if g) / max(1, sum(1 for g in gt_list if g)), 3)

    # Content-based metrics (semantically meaningful)
    if content_scores_sg:
        cat_results["star_graph"]["content_recall@3"] = round(
            sum(content_scores_sg) / len(content_scores_sg), 3)
        cat_results["star_graph"]["content_recall@5"] = round(
            sum(content_recall_at_k(sg_t, q.required_keywords, 5)
                for sg_t, q in zip(sg_texts, queries) if q.required_keywords)
            / max(1, sum(1 for q in queries if q.required_keywords)), 3)
    if content_scores_vec:
        cat_results["vector_only"]["content_recall@3"] = round(
            sum(content_scores_vec) / len(content_scores_vec), 3)
        cat_results["vector_only"]["content_recall@5"] = round(
            sum(content_recall_at_k(vec_t, q.required_keywords, 5)
                for vec_t, q in zip(vec_texts, queries) if q.required_keywords)
            / max(1, sum(1 for q in queries if q.required_keywords)), 3)
    if content_scores_hyb:
        cat_results["hybrid_fusion"]["content_recall@3"] = round(
            sum(content_scores_hyb) / len(content_scores_hyb), 3)
        cat_results["hybrid_fusion"]["content_recall@5"] = round(
            sum(content_recall_at_k(hyb_t, q.required_keywords, 5)
                for hyb_t, q in zip(hyb_texts, queries) if q.required_keywords)
            / max(1, sum(1 for q in queries if q.required_keywords)), 3)
    if interference_scores:
        cat_results["star_graph"]["interference_score"] = round(
            sum(interference_scores) / len(interference_scores), 3)
    if interference_scores_hyb:
        cat_results["hybrid_fusion"]["interference_score"] = round(
            sum(interference_scores_hyb) / len(interference_scores_hyb), 3)

    return cat_results


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
