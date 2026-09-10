"""Experiment 2 — retrieval formula A/B (R-hybrid vs R-plan).

The WRITE phase is identical for every arm (one shared graph, code-default
importance engine). Both retrievers are read-only over the graph, so both arms
rank the exact same stored memories — any metric difference is attributable to
the fusion formula alone.

Usage:
    python benchmarks/run_ab_retrieval.py                 # auto: LoCoMo if local, else synthetic 200+100
    python benchmarks/run_ab_retrieval.py --smoke         # 20 facts + 10 noise, 10 queries (~1 min)
    python benchmarks/run_ab_retrieval.py --mode synthetic --facts 200 --noise 100 --queries 50
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_experiment import (  # noqa: E402
    ABArm,
    ArmMetrics,
    W_CODE_WEIGHTS,
    _reset_embedder_state,
    _teardown_env,
    _write_phase,
    build_synthetic_corpus,
    load_locomo_corpus,
    metrics_table,
    retrieval_arms,
    run_retrieval_arm,
    save_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B: retrieval fusion formulas")
    p.add_argument("--mode", choices=["auto", "synthetic", "locomo"], default="auto",
                   help="dataset mode; auto = LoCoMo if locally available else synthetic")
    p.add_argument("--facts", type=int, default=200, help="synthetic fact count")
    p.add_argument("--noise", type=int, default=100, help="synthetic noise count")
    p.add_argument("--queries", type=int, default=50, help="query count")
    p.add_argument("--seed", type=int, default=42, help="corpus/query random seed")
    p.add_argument("--top-k", type=int, default=10, help="retrieval depth for metrics")
    p.add_argument("--smoke", action="store_true",
                   help="smoke test: 20 facts + 10 noise, 10 queries (pipeline check only)")
    p.add_argument("--arms", default="", help="comma-separated arm filter, e.g. 'R-plan'")
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

    arms = retrieval_arms()
    if args.arms:
        wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
        arms = [a for a in arms if a.name in wanted]

    print(f"实验二：检索公式 A/B — {len(arms)} 组配置, 共享同一写入图（写入阶段完全一致）")

    # Shared write phase: ONE clean env, ONE graph — identical storage for all arms.
    # Both retrievers are read-only over the graph, so sharing is side-effect free.
    tmpdir = tempfile.mkdtemp(prefix="nwc_ab_ret_shared_")
    _reset_embedder_state()
    write_arm = ABArm(name="fixed-write",
                      description="所有检索臂共用的写入配置（代码默认重要性引擎）",
                      importance_weights=dict(W_CODE_WEIGHTS))
    print(f"── 写入阶段（全部臂共享, 代码默认重要性门控）", flush=True)
    shared_env = _write_phase(write_arm, records, tmpdir, verbose=True)
    print(f"   图中锚点: {len(shared_env.manager.graph.anchors)}, "
          f"事实锚点: {len(shared_env.fact_anchor_ids)}, "
          f"门控错误: {shared_env.gate_errors}", flush=True)

    try:
        results: list[ArmMetrics] = []
        for arm in arms:
            print(f"── 检索 {arm.name}: {arm.description}", flush=True)
            m = run_retrieval_arm(arm, records, queries, top_k=args.top_k,
                                  shared_env=shared_env)
            results.append(m)
            print(f"   hit@1={m.hit_at_1:.3f}, hit@3={m.hit_at_3:.3f}, "
                  f"MRR={m.mrr:.3f}, has_answer={m.has_answer:.3f}", flush=True)
    finally:
        _teardown_env(shared_env)

    title = (f"实验二：检索公式 A/B（mode={mode}, seed={args.seed}, "
             f"记忆={len(records)}, 查询={len(queries)}, top_k={args.top_k}）"
             f"{' [SMOKE]' if args.smoke else ''}")
    table = metrics_table(results, title)
    print()
    print(table)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "ab_results", f"retrieval_ab_{stamp}.json")
    payload = {
        "experiment": "retrieval_formula_ab",
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
