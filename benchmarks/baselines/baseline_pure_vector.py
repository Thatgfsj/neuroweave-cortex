"""Baseline: Pure Vector Similarity (no graph, no sleep, no cognitive modules).

Used as the simplest baseline for LoCoMo comparison.
All baselines use the same embedding model (all-MiniLM-L6-v2).

Usage:
    python -m benchmarks.baselines.baseline_pure_vector [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from star_graph import StarGraph, Anchor, get_embedder, seed_everything
from star_graph.math_utils import cosine_sim
from benchmarks.run_locomo_full import (
    load_locomo, extract_all_turns, has_answer, f1_score, normalize_answer,
)


def evaluate_baseline(graph, qa_pairs, embedder, top_k=10):
    """Pure vector similarity — simplest baseline."""
    anchors = list(graph.anchors.values())
    results = []
    for qa in qa_pairs:
        question = qa['question']
        answer = qa.get('answer') or qa.get('adversarial_answer', '')
        q_emb = embedder.encode(question)

        scored = []
        for a in anchors:
            if a.embedding:
                sim = cosine_sim(q_emb, a.embedding)
                scored.append((sim, a))
        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        combined = " ".join(a.text[:200] for _, a in top)
        hit = has_answer(answer, combined)
        f1 = f1_score(combined[:2000], answer)
        results.append({"hit": hit, "f1": f1, "category": qa.get('category', '4')})

    hits = sum(1 for r in results if r['hit'])
    f1_avg = sum(r['f1'] for r in results) / max(1, len(results))
    return hits, len(results), f1_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--locomo-path', type=str,
                        default='C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json')
    args = parser.parse_args()

    seed_everything(42)
    dataset = load_locomo(args.locomo_path)
    num = 3 if args.quick else len(dataset)
    print(f"Baseline: Pure Vector Similarity — {num} conversations")
    print(f"{'─'*50}")

    embedder = get_embedder()
    total_hits = total_qa = 0
    total_f1 = 0.0

    for ci, conv in enumerate(dataset[:num]):
        turns, _ = extract_all_turns(conv)
        graph = StarGraph()
        for turn in turns:
            text = turn['text']
            if not text.strip():
                continue
            emb = embedder.encode(text)
            anchor = Anchor.create(text=text, embedding=emb, importance=0.5)
            graph.add_anchor(anchor)

        hits, n, f1 = evaluate_baseline(graph, conv['qa'], embedder)
        total_hits += hits; total_qa += n; total_f1 += f1 * n
        print(f"  [{ci+1}/{num}] {conv.get('sample_id','?')}: {hits}/{n} = {hits/n*100:.1f}%")

    print(f"{'─'*50}")
    print(f"  Overall: {total_hits}/{total_qa} = {total_hits/total_qa*100:.1f}%  F1={total_f1/total_qa:.4f}")


if __name__ == "__main__":
    main()
