"""Ablation: systematic evaluation of each cognitive module's contribution.

Runs LoCoMo benchmark with 5 configurations:
  1. full      — Complete system (RRF pipeline + sleep + all modules)
  2. no_sleep  — Skip sleep consolidation
  3. no_retrieval_fusion — RRF off, vector similarity only
  4. no_spreading — Spreading activation off
  5. no_cache  — Exact cache + cognitive cache off

Each config runs 3 times (different seeds) and reports mean ± std.

Usage:
    python -m benchmarks.ablation [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, get_embedder, seed_everything, MemoryRuntime,
    RetrievalPipeline, Config,
)
from benchmarks.run_locomo_full import (
    load_locomo, extract_all_turns, has_answer, f1_score,
)


CONFIGS = {
    "full": {
        "sleep.enabled": True,
        "retrieval.rrf_enabled": True,
        "spreading.enabled": True,
        "exact_cache.enabled": True,
        "cognitive_cache.enabled": True,
    },
    "no_sleep": {
        "sleep.enabled": False,
        "retrieval.rrf_enabled": True,
        "spreading.enabled": True,
        "exact_cache.enabled": True,
        "cognitive_cache.enabled": True,
    },
    "no_rrf": {
        "sleep.enabled": True,
        "retrieval.rrf_enabled": False,
        "spreading.enabled": True,
        "exact_cache.enabled": True,
        "cognitive_cache.enabled": True,
    },
    "no_spreading": {
        "sleep.enabled": True,
        "retrieval.rrf_enabled": True,
        "spreading.enabled": False,
        "exact_cache.enabled": True,
        "cognitive_cache.enabled": True,
    },
    "no_cache": {
        "sleep.enabled": True,
        "retrieval.rrf_enabled": True,
        "spreading.enabled": True,
        "exact_cache.enabled": False,
        "cognitive_cache.enabled": False,
    },
}

SEEDS = [42, 123, 256]


def run_config(config_name: str, overrides: dict, conversations, embedder, num_conv: int) -> dict:
    """Run LoCoMo with a specific config override. Returns {hits, total, f1}."""
    total_hits = total_qa = 0
    total_f1 = 0.0

    for conv in conversations[:num_conv]:
        turns, session_keys = extract_all_turns(conv)
        graph = StarGraph()
        cfg = Config.get()

        # Apply config overrides
        for k, v in overrides.items():
            parts = k.split(".")
            target = cfg
            for p in parts[:-1]:
                target = getattr(target, p, target)
            setattr(target, parts[-1], v)

        rt = MemoryRuntime(graph=graph, config=cfg)
        rp = RetrievalPipeline(rt)

        for turn in turns:
            text = turn['text']
            if not text.strip():
                continue
            embedding = embedder.encode(text)
            anchor = Anchor.create(text=text, embedding=embedding, importance=0.5)
            graph.add_anchor(anchor)
            rt.raw_buffer.add(text=text, embedding=embedding, importance=0.5, anchor_id=anchor.id)

        for qa in conv['qa']:
            question = qa['question']
            answer = qa.get('answer') or qa.get('adversarial_answer', '')
            ctx = rp.recall(query=question, max_items=10)
            combined = " ".join(
                (item.compressed_text or (item.anchor.text[:200] if item.anchor else ""))[:200]
                for item in ctx.items
            )
            hit = has_answer(answer, combined)
            f1 = f1_score(combined[:2000], answer)
            if hit:
                total_hits += 1
            total_qa += 1
            total_f1 += f1

    return {
        "hits": total_hits,
        "total": total_qa,
        "f1": total_f1 / max(1, total_qa),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--quick", action="store_true", help="3 conversations only")
    parser.add_argument("--locomo-path", type=str,
                        default="C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json")
    args = parser.parse_args()

    dataset = load_locomo(args.locomo_path)
    num_conv = 3 if args.quick else len(dataset)
    print(f"Ablation Study — {num_conv} conversations, {len(SEEDS)} seeds per config\n")

    results = {}
    for config_name, overrides in CONFIGS.items():
        seed_results = []
        for seed in SEEDS:
            seed_everything(seed)
            res = run_config(config_name, overrides, dataset, get_embedder(), num_conv)
            seed_results.append(res)
            print(f"  {config_name:>12s} seed={seed}: {res['hits']}/{res['total']} = {res['hits']/res['total']*100:.1f}%")

        means = {k: sum(r[k] for r in seed_results) / len(seed_results) for k in ["hits", "total"]}
        stds = {k: (sum((r[k] - means[k])**2 for r in seed_results) / len(seed_results))**0.5
                for k in ["hits"]}
        rate = means["hits"] / means["total"] * 100
        rate_std = stds["hits"] / means["total"] * 100 if means["total"] > 0 else 0
        results[config_name] = {
            "has_answer": f"{rate:.1f}±{rate_std:.1f}",
            "has_answer_mean": round(rate, 1),
            "has_answer_std": round(rate_std, 1),
        }
        print(f"  {'─'*40}")
        print(f"  {config_name:>12s} mean: {rate:.1f}% ± {rate_std:.1f}%\n")

    print(f"{'='*50}")
    print("  Ablation Summary:")
    print(f"  {'Config':>12s}  {'has_answer':>12s}")
    print(f"  {'─'*30}")
    for name, r in results.items():
        print(f"  {name:>12s}  {r['has_answer']:>12s}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
