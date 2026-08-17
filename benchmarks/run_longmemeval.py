"""LongMemEval runner — NWC vs baselines (revision v2.0).

Data: xiaowu0162/longmemeval-cleaned (500 entries, per-entry haystack of
sessions). For each QA entry we build a memory graph from its haystack
sessions, then evaluate the same retrieval configurations used on LoCoMo-10.

Usage:
    python benchmarks/run_longmemeval.py [--quick] [--data PATH]

Output:
    benchmarks/revision_results/longmemeval_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import StarGraph, Anchor, get_embedder, seed_everything
from star_graph.bm25 import BM25Index
from star_graph.retriever import HybridFusionRetriever
from benchmarks.run_locomo_full import has_answer
from benchmarks.run_revision_exp import retrieve_cosine, rrf_fuse, OUT_DIR

DEFAULT_DATA = r'E:\locomo-10\longmemeval\longmemeval_s_cleaned.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA)
    args = parser.parse_args()

    seed_everything(42)
    with open(args.data, encoding='utf-8') as f:
        data = json.load(f)
    print(f'LongMemEval: {len(data)} QA entries', flush=True)
    if args.quick:
        data = data[:20]

    embedder = get_embedder()
    t0 = time.time()

    def make_fns(graph, bm25):
        def cfg_vanilla(graph, q, q_emb, bm25):
            ids = retrieve_cosine(graph, q_emb, top_k=10)
            return [graph.anchors[i].text[:200] for i in ids]

        def cfg_bm25(graph, q, q_emb, bm25):
            ids = [aid for aid, _ in bm25.search(q, top_k=10)]
            return [graph.anchors[i].text[:200] for i in ids]

        def cfg_hybrid(graph, q, q_emb, bm25):
            ann_ids = retrieve_cosine(graph, q_emb, top_k=10)
            bm_ids = [aid for aid, _ in bm25.search(q, top_k=40)]
            fused = rrf_fuse(ann_ids, bm_ids, k=60)
            return [graph.anchors[i].text[:200] for i, _ in fused[:10]]

        def cfg_nwc(graph, q, q_emb, bm25):
            ret = HybridFusionRetriever(graph, bm25_index=bm25)
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

        return {'vanilla_rag': cfg_vanilla, 'bm25': cfg_bm25,
                'hybrid_rrf': cfg_hybrid, 'nwc_full': cfg_nwc}

    names = ['vanilla_rag', 'bm25', 'hybrid_rrf', 'nwc_full']
    totals = {k: {'hits': 0, 'total': 0} for k in names}
    by_task = {}

    # --- checkpoint / resume ---
    ckpt_path = os.path.join(OUT_DIR, 'longmemeval_ckpt.json')
    start = 0
    if os.path.exists(ckpt_path) and not args.quick:
        with open(ckpt_path, encoding='utf-8') as f:
            ck = json.load(f)
        totals = ck['totals']
        by_task = ck['by_task']
        start = ck['entries_done']
        print(f'resuming from entry {start}', flush=True)

    for ei in range(start, len(data)):
        e = data[ei]
        try:
            q = str(e.get('question', ''))
            a = e.get('answer')
            if isinstance(a, list):
                a = [str(x) for x in a]
            else:
                a = str(a)
            qtype = str(e.get('question_type') or e.get('category') or 'other')
            if not q or not a:
                continue

            # build graph from haystack sessions
            graph = StarGraph()
            for sess in e.get('haystack_sessions', []):
                for t in sess:
                    text = t.get('content') if isinstance(t, dict) else str(t)
                    if not text or not text.strip():
                        continue
                    emb = embedder.encode(text)
                    graph.add_anchor(Anchor.create(text=text, embedding=emb,
                                                   importance=0.5))
            bm25 = BM25Index()
            for aid, anchor in graph.anchors.items():
                bm25.add(aid, anchor.text)
            fns = make_fns(graph, bm25)
            q_emb = embedder.encode(q)

            for name in names:
                texts = fns[name](graph, q, q_emb, bm25)
                hit = has_answer(a, ' '.join(texts))
                totals[name]['hits'] += 1 if hit else 0
                totals[name]['total'] += 1
            by_task.setdefault(qtype, {k: {'hits': 0, 'total': 0} for k in names})
            for name in names:
                texts = fns[name](graph, q, q_emb, bm25)
                hit = has_answer(a, ' '.join(texts))
                by_task[qtype][name]['hits'] += 1 if hit else 0
                by_task[qtype][name]['total'] += 1
        except Exception as ex:
            print(f'[err] entry {ei}: {str(ex)[:150]}', flush=True)

        if (ei + 1) % 50 == 0:
            print(f'[{ei+1}/{len(data)}] '
                  + ' '.join(f'{k}:{v["hits"]}/{v["total"]}'
                             for k, v in totals.items()), flush=True)
        if (ei + 1) % 50 == 0:
            with open(ckpt_path, 'w', encoding='utf-8') as f:
                json.dump({'totals': totals, 'by_task': by_task,
                           'entries_done': ei + 1}, f)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    summary = {}
    for name in names:
        v = totals[name]
        summary[name] = {
            'has_answer_pct': round(v['hits'] / v['total'] * 100, 1)
            if v['total'] else 0,
            'hits': v['hits'], 'total': v['total'],
        }
    task_summary = {}
    for task, m in by_task.items():
        task_summary[task] = {
            k: round(v['hits'] / v['total'] * 100, 1) if v['total'] else 0
            for k, v in m.items()
        }

    out = {'dataset': 'LongMemEval-small', 'entries': len(data),
           'elapsed_min': round((time.time() - t0) / 60, 1),
           'overall': summary, 'by_task': task_summary}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'longmemeval_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
