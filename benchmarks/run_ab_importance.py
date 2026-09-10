"""Experiment 1 — importance weight A/B (W-code vs W-plan vs W-code+nov).

The arm's ImportanceEngine is the ONLY write gate (score → anchor importance,
should_store → store/drop). Evaluation retrieval is a FIXED default
HybridFusionRetriever for every arm, so ranking differences are attributable
to the write-side weight config alone.

Usage:
    python benchmarks/run_ab_importance.py                # auto: LoCoMo if local, else synthetic 200+100
    python benchmarks/run_ab_importance.py --smoke        # 20 facts + 10 noise, 10 queries (~1 min)
    python benchmarks/run_ab_importance.py --mode synthetic --facts 200 --noise 100 --queries 50
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_experiment import (  # noqa: E402
    ArmMetrics,
    importance_arms,
    load_locomo_corpus,
    metrics_table,
    run_importance_arm,
    build_synthetic_corpus,
    save_report,
    weights_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B: importance weight configs")
    p.add_argument("--mode", choices=["auto", "synthetic", "locomo"], default="auto",
                   help="dataset mode; auto = LoCoMo if locally available else synthetic")
    p.add_argument("--facts", type=int, default=200, help="synthetic fact count")
    p.add_argument("--noise", type=int, default=100, help="synthetic noise count")
    p.add_argument("--queries", type=int, default=50, help="query count")
    p.add_argument("--seed", type=int, default=42, help="corpus/query random seed")
    p.add_argument("--top-k", type=int, default=10, help="retrieval depth for metrics")
    p.add_argument("--smoke", action="store_true",
                   help="smoke test: 20 facts + 10 noise, 10 queries (pipeline check only)")
    p.add_argument("--arms", default="", help="comma-separated arm filter, e.g. 'W-code,W-plan'")
    p.add_argument("--out", default="", help="output JSON path (default benchmarks/ab_results/…)")
    p.add_argument("--locomo-sessions", type=int, default=3,
                   help="LoCoMo conversations to use when available")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_facts, n_noise, n_queries = args.facts, args.noise, args.queries
    if args.smoke:
        n_facts, n_noise, n_queries = 20, 10, 10

    mode = args.mode
    if mode == "auto":
        mode = "locomo" if load_locomo_corpus(max_sessions=1) is not None else "synthetic"

    if mode == "locomo":
        loaded = load_locomo_corpus(max_sessions=args.locomo_sessions)
        if loaded is None:
            print("LoCoMo 数据不可用，自动降级为合成评测 (synthetic)。")
            mode = "synthetic"
        else:
            records, queries = loaded
            queries = queries[:n_queries]
            print(f"数据集: LoCoMo ({args.locomo_sessions} 会话) — "
                  f"{len(records)} 条记忆, {len(queries)} 条查询")
    if mode == "synthetic":
        records, queries = build_synthetic_corpus(n_facts=n_facts, n_noise=n_noise,
                                                  seed=args.seed)
        queries = queries[:n_queries]
        print(f"数据集: 合成 (seed={args.seed}) — {len(records)} 条记忆 "
              f"({n_facts} 事实 + {n_noise} 噪声), {len(queries)} 条查询")

    arms = importance_arms()
    if args.arms:
        wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
        arms = [a for a in arms if a.name in wanted]

    print(f"实验一：重要性权重 A/B — {len(arms)} 组配置, 每组独立 clean 环境")
    print(weights_table(arms))
    print()

    results: list[ArmMetrics] = []
    for arm in arms:
        print(f"── 运行 {arm.name}: {arm.description}", flush=True)
        m = run_importance_arm(arm, records, queries, top_k=args.top_k)
        results.append(m)
        print(f"   存储 {m.n_gate_stored}/{m.n_written}, 门控错误 {m.n_gate_errors}, "
              f"hit@3={m.hit_at_3:.3f}, MRR={m.mrr:.3f}", flush=True)

    title = (f"实验一：重要性权重 A/B（mode={mode}, seed={args.seed}, "
             f"记忆={len(records)}, 查询={len(queries)}, top_k={args.top_k}）"
             f"{' [SMOKE]' if args.smoke else ''}")
    table = metrics_table(results, title)
    print()
    print(table)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "ab_results", f"importance_ab_{stamp}.json")
    payload = {
        "experiment": "importance_weights_ab",
        "title": title,
        "mode": mode,
        "seed": args.seed,
        "smoke": args.smoke,
        "n_records": len(records),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "timestamp": stamp,
        "arms": [r.to_dict() for r in results],
        "markdown": table,
    }
    save_report(out, payload)
    print(f"\n结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
