"""Revision v2.0 experiment batch — baselines + component ablations on LoCoMo-10.

Runs on the native HybridFusion pipeline (the same path that produced the
44.2% headline result) so every row is comparable. The graph for each
conversation is built once and reused across all configurations.

Configurations:
  vanilla_rag  — cosine similarity, top-10            (verify 31.2)
  bm25         — Okapi BM25, top-10                   (verify 26.5)
  hybrid_rrf   — ANN top-10 + BM25 top-40, RRF k=60   (verify 33.1)
  nwc_full     — HybridFusionRetriever defaults       (verify 44.2)
  a1_nolife    — beta=0    (no temporal/lifecycle signal)
  a2_noact     — gamma=0, spread_steps=0 (no graph activation)
  a3_nolex     — epsilon=0 (no BM25/lexical signal)
  a4_noconf    — delta=0   (no confidence signal)
  a5_single    — alpha only (pure vector store)

Each QA hit/miss is saved per configuration for bootstrap statistics.

Usage:
    python benchmarks/run_revision_exp.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import get_embedder, seed_everything
from star_graph.bm25 import BM25Index
from star_graph.index import ANNIndex
from star_graph.retriever import HybridFusionRetriever
from benchmarks.run_locomo_full import (
    load_locomo, extract_all_turns, build_graph, build_edges, has_answer,
)

DATASET = 'E:/locomo-10/data/locomo10.json'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'revision_results')


def rrf_fuse(ann_ranked, bm25_ranked, k=60):
    """Standard Reciprocal Rank Fusion: score(d) = sum 1/(k + rank)."""
    scores = {}
    for rank, doc_id in enumerate(ann_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve_cosine(graph, q_emb, top_k=10):
    ranked = sorted(
        graph.anchors.values(),
        key=lambda a: _cos(a.embedding, q_emb) if a.embedding else -1.0,
        reverse=True,
    )
    return [a.id for a in ranked[:top_k]]


def _cos(a, b):
    if a is None or b is None:
        return -1.0
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


def run_config(name, config_fn, graph, qa_pairs, embedder, bm25, extra_texts):
    hits = []
    for qa in qa_pairs:
        question = qa['question']
        answer = str(qa.get('answer') or qa.get('adversarial_answer', ''))
        q_emb = embedder.encode(question)
        texts = config_fn(graph, question, q_emb, bm25, extra_texts)
        combined = ' '.join(texts)
        hits.append(1 if has_answer(answer, combined) else 0)
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    seed_everything(42)
    dataset = load_locomo(DATASET)
    num_conv = 1 if args.quick else len(dataset)
    os.makedirs(OUT_DIR, exist_ok=True)

    embedder = get_embedder()

    # ---- build graphs once ----
    graphs, conv_qa, conv_bm25 = [], [], []
    t0 = time.time()
    for ci, conv in enumerate(dataset[:num_conv]):
        turns, session_keys = extract_all_turns(conv)
        graph = build_graph(turns, session_keys)
        build_edges(graph)
        bm25 = BM25Index()
        for aid, anchor in graph.anchors.items():
            bm25.add(aid, anchor.text)
        graphs.append(graph)
        conv_qa.append(conv['qa'])
        conv_bm25.append(bm25)
        print(f'[build {ci+1}/{num_conv}] {len(graph.anchors)} anchors',
              flush=True)
    print(f'graphs built in {time.time()-t0:.0f}s', flush=True)

    # ---- configuration factory: retriever weight variants ----
    def make_retriever(graph, bm25, **kw):
        defaults = dict(alpha=0.50, beta=0.12, gamma=0.18, delta=0.08,
                        epsilon=0.12, spread_steps=2)
        defaults.update(kw)
        return HybridFusionRetriever(graph, bm25_index=bm25, **defaults)

    def cfg_nwc_full(graph, q, q_emb, bm25, extra):
        ret = make_retriever(graph, bm25)
        res = ret.retrieve(q, q_emb, top_k=10)
        texts = [a.text[:200] for c in res.constellations for a in c.anchors]
        seen = set()
        for aid, _ in bm25.search(q, top_k=40):
            if aid not in seen:
                a = graph.anchors.get(aid)
                if a:
                    seen.add(aid)
                    texts.append(a.text[:200])
        return texts[:60]

    def cfg_ablated(ret_kw):
        def fn(graph, q, q_emb, bm25, extra):
            ret = make_retriever(graph, bm25, **ret_kw)
            res = ret.retrieve(q, q_emb, top_k=10)
            return [a.text[:200] for c in res.constellations for a in c.anchors]
        return fn

    def cfg_vanilla(graph, q, q_emb, bm25, extra):
        ids = retrieve_cosine(graph, q_emb, top_k=10)
        return [graph.anchors[i].text[:200] for i in ids]

    def cfg_bm25(graph, q, q_emb, bm25, extra):
        ids = [aid for aid, _ in bm25.search(q, top_k=10)]
        return [graph.anchors[i].text[:200] for i in ids]

    def cfg_hybrid(graph, q, q_emb, bm25, extra):
        ann_ids = retrieve_cosine(graph, q_emb, top_k=10)
        bm_ids = [aid for aid, _ in bm25.search(q, top_k=40)]
        fused = rrf_fuse(ann_ids, bm_ids, k=60)
        return [graph.anchors[i].text[:200] for i, _ in fused[:10]]

    CONFIGS = [
        ('vanilla_rag', cfg_vanilla),
        ('bm25', cfg_bm25),
        ('hybrid_rrf', cfg_hybrid),
        ('nwc_full', cfg_nwc_full),
        ('a1_no_lifecycle', cfg_ablated(dict(beta=0.0))),
        ('a2_no_activation', cfg_ablated(dict(gamma=0.0, spread_steps=0))),
        ('a3_no_lexical', cfg_ablated(dict(epsilon=0.0))),
        ('a4_no_confidence', cfg_ablated(dict(delta=0.0))),
        ('a5_single_layer', cfg_vanilla),  # pure vector store ≡ vanilla RAG
    ]

    cat_names = {'1': 'Temporal', '2': 'Short', '3': 'Long',
                 '4': 'Composite', '5': 'Adversarial'}

    summary = {}
    for name, fn in CONFIGS:
        t0 = time.time()
        all_hits, all_cats = [], []
        for ci in range(num_conv):
            hits = run_config(name, fn, graphs[ci], conv_qa[ci], embedder,
                              conv_bm25[ci], None)
            cats = [str(qa.get('category', '4')) for qa in conv_qa[ci]]
            all_hits.extend(hits)
            all_cats.extend(cats)
            print(f'  [{name}] conv {ci+1}/{num_conv}: '
                  f'{sum(hits)}/{len(hits)}={sum(hits)/len(hits)*100:.1f}%',
                  flush=True)
        by_cat = {}
        for c in sorted(set(all_cats)):
            idx = [i for i, cc in enumerate(all_cats) if cc == c]
            hits_c = [all_hits[i] for i in idx]
            by_cat[cat_names.get(c, c)] = {
                'hits': sum(hits_c), 'total': len(hits_c),
                'acc': round(sum(hits_c) / len(hits_c) * 100, 1),
            }
        summary[name] = {
            'hits': sum(all_hits), 'total': len(all_hits),
            'acc': round(sum(all_hits) / len(all_hits) * 100, 1),
            'by_category': by_cat,
            'elapsed_min': round((time.time() - t0) / 60, 1),
        }
        # per-QA array for bootstrap
        with open(os.path.join(OUT_DIR, f'{name}_qa.json'), 'w') as f:
            json.dump({'hits': all_hits, 'cats': all_cats}, f)
        print(f'[{name}] done: {summary[name]}', flush=True)

    with open(os.path.join(OUT_DIR, 'revision_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
