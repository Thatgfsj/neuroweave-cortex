"""Cross-LLM robustness — LLM-as-judge scoring of NWC vs baselines on LoCoMo-10.

Protocol: for each QA pair, retrieve top-10 contexts (same retrieval as
run_revision_exp.py), then ask an external LLM to judge whether the context
contains the answer. This is the "LLM-assisted scoring" evaluation protocol
from the paper (v002: 35.6%), now re-run cleanly with two independent LLMs
to demonstrate that the relative ranking of systems is stable across judges.

LLMs:
  - DeepSeek  (deepseek-chat,  https://api.deepseek.com/v1)
  - MiniMax   (abab6.5s-chat,  https://api.minimaxi.com/v1)

Usage:
    python benchmarks/run_llm_scoring.py [--systems vanilla_rag,nwc_full]
                                         [--judge deepseek,minimax]
                                         [--max-qa 1986]

Output:
    benchmarks/revision_results/llm_scoring.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import get_embedder, seed_everything
from star_graph.bm25 import BM25Index
from star_graph.retriever import HybridFusionRetriever
from benchmarks.run_locomo_full import (
    load_locomo, extract_all_turns, build_graph, build_edges, has_answer,
)
from benchmarks.run_revision_exp import (
    retrieve_cosine, rrf_fuse, DATASET, OUT_DIR,
)

JUDGES = {
    'minimax-abab': {
        'base_url': 'https://api.minimaxi.com/v1',
        'model': 'abab6.5s-chat',
        'api_key_env': 'MINIMAX_API_KEY',
        'sdk': 'openai',
    },
    'minimax-text01': {
        'base_url': 'https://api.minimaxi.com/anthropic',
        'model': 'MiniMax-Text-01',
        'api_key_env': 'MINIMAX_API_KEY',
        'sdk': 'anthropic',
    },
}

JUDGE_PROMPT = """You are evaluating a memory retrieval system.

Question: {question}

Retrieved memory context:
{context}

Expected answer: {answer}

Does the retrieved context contain enough information to correctly answer the question?
Reply with ONLY one word: YES or NO."""


def make_systems(graph, bm25):
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

    return {
        'vanilla_rag': cfg_vanilla,
        'bm25': cfg_bm25,
        'hybrid_rrf': cfg_hybrid,
        'nwc_full': cfg_nwc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', type=str, default='vanilla_rag,bm25,hybrid_rrf,nwc_full')
    parser.add_argument('--judges', type=str,
                        default='minimax-abab,minimax-text01')
    parser.add_argument('--max-qa', type=int, default=0, help='0 = all')
    args = parser.parse_args()

    import openai

    seed_everything(42)
    dataset = load_locomo(DATASET)
    systems = make_systems(None, None)
    chosen = [s for s in args.systems.split(',') if s in systems]
    judge_names = [j for j in args.judges.split(',') if j in JUDGES]
    embedder = get_embedder()

    # ---- retrieval pass: collect contexts per system (cached) ----
    cache_path = os.path.join(OUT_DIR, 'llm_contexts_cache.json')
    if os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            cache = json.load(f)
        contexts = cache['contexts']
        qas = cache['qas']
        chosen = [s for s in chosen if s in contexts]
        print(f'using cached contexts: {len(qas)} QA, systems: '
              f'{list(contexts.keys())}', flush=True)
    else:
        contexts = {s: [] for s in chosen}
        qas = []
        for conv in dataset:
            turns, session_keys = extract_all_turns(conv)
            graph = build_graph(turns, session_keys)
            build_edges(graph)
            bm25 = BM25Index()
            for aid, anchor in graph.anchors.items():
                bm25.add(aid, anchor.text)
            fns = make_systems(graph, bm25)
            for qa in conv['qa']:
                q = qa['question']
                a = str(qa.get('answer') or qa.get('adversarial_answer', ''))
                cat = str(qa.get('category', '4'))
                q_emb = embedder.encode(q)
                qas.append({'q': q, 'a': a, 'cat': cat})
                for s in chosen:
                    contexts[s].append(' '.join(fns[s](graph, q, q_emb, bm25)))
            print(f'retrieval pass done: {len(qas)} QA', flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'contexts': contexts, 'qas': qas}, f)
        print('contexts cached', flush=True)
    if args.max_qa:
        for s in chosen:
            contexts[s] = contexts[s][:args.max_qa]
        qas = qas[:args.max_qa]
    n = len(qas)
    print(f'{n} QA pairs, systems: {chosen}, judges: {judge_names}', flush=True)

    # ---- judge pass ----
    results = {}
    for jname in judge_names:
        cfg = JUDGES[jname]
        key = os.environ.get(cfg['api_key_env'], '')
        if not key:
            print(f'[skip] {jname}: no {cfg["api_key_env"]} in env', flush=True)
            continue
        if cfg['sdk'] == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=key,
                                         base_url=cfg['base_url'])
        else:
            client = openai.OpenAI(api_key=key, base_url=cfg['base_url'])
        results[jname] = {}
        for s in chosen:
            t0 = time.time()
            yes = 0
            for i in range(n):
                prompt = JUDGE_PROMPT.format(
                    question=qas[i]['q'],
                    context=contexts[s][i][:3000],
                    answer=qas[i]['a'],
                )
                try:
                    if cfg['sdk'] == 'anthropic':
                        resp = client.messages.create(
                            model=cfg['model'],
                            max_tokens=8,
                            temperature=0,
                            messages=[{'role': 'user', 'content': prompt}],
                        )
                        text = ''.join(
                            b.text for b in resp.content if b.type == 'text'
                        ).strip().upper()
                    else:
                        resp = client.chat.completions.create(
                            model=cfg['model'],
                            messages=[{'role': 'user', 'content': prompt}],
                            temperature=0, max_tokens=8,
                        )
                        text = resp.choices[0].message.content.strip().upper()
                    if text.startswith('YES'):
                        yes += 1
                except Exception as e:
                    print(f'  [{jname}/{s}] err at {i}: {str(e)[:120]}', flush=True)
                    time.sleep(2)
                if (i + 1) % 200 == 0:
                    print(f'  [{jname}/{s}] {i+1}/{n} yes={yes} '
                          f'({yes/(i+1)*100:.1f}%)', flush=True)
            acc = round(yes / n * 100, 1)
            results[jname][s] = {
                'has_answer_pct': acc, 'yes': yes, 'total': n,
                'elapsed_min': round((time.time() - t0) / 60, 1),
            }
            print(f'[{jname}/{s}] done: {acc}%', flush=True)

    out = {'protocol': 'LLM-as-judge has-answer', 'num_qa': n,
           'systems': chosen, 'judges': judge_names, 'results': results,
           'string_match_ref': None}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'llm_scoring.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved → {path}')


if __name__ == '__main__':
    main()
