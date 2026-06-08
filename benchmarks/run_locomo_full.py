"""LoCoMo-10 全量评估 — 论文基准 + RRF 对比

Usage:
    python benchmarks/run_locomo_full.py

Outputs:
    benchmarks/locomo_results.json    — 完整结果
    benchmarks/locomo_results.csv     — 图表数据
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, get_embedder, seed_everything, MemoryRuntime,
    RetrievalPipeline, AgentContext, Config,
)
from star_graph.math_utils import cosine_sim
from star_graph.retriever import HybridFusionRetriever
from star_graph.bm25 import BM25Index
from star_graph.index import ANNIndex


# ── Metrics ──

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    s = s.replace(",", "")
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|and)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def has_answer(answers, text):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    text_tokens = [ps.stem(w) for w in normalize_answer(text).split()]
    if not isinstance(answers, list):
        answers = [answers]
    for answer in answers:
        answer = str(answer)
        answer = _normalize(answer)
        answer_tokens = [ps.stem(w) for w in normalize_answer(answer).split()]
        if not answer_tokens:
            continue
        for i in range(len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[i:i + len(answer_tokens)]:
                return True
        window = max(len(answer_tokens) + 6, 10)
        for i in range(len(text_tokens) - min(window, len(text_tokens)) + 1):
            window_tokens = set(text_tokens[i:i + window])
            overlap = sum(1 for t in answer_tokens if t in window_tokens)
            if overlap / len(answer_tokens) >= 0.75:
                return True
        ans_bigrams = set(zip(answer_tokens, answer_tokens[1:])) if len(answer_tokens) > 1 else set()
        text_bigrams = set(zip(text_tokens, text_tokens[1:])) if len(text_tokens) > 1 else set()
        if ans_bigrams and text_bigrams:
            bigram_jaccard = len(ans_bigrams & text_bigrams) / len(ans_bigrams | text_bigrams)
            if bigram_jaccard > 0.5:
                return True
        if len(answer_tokens) <= 3:
            if normalize_answer(answer).lower() in normalize_answer(text).lower():
                return True
    return False

def f1_score(prediction, ground_truth):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    pred_tokens = [ps.stem(w) for w in normalize_answer(str(prediction)).split()]
    gt_tokens = [ps.stem(w) for w in normalize_answer(str(ground_truth)).split()]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# ── Data ──

def load_locomo(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def extract_all_turns(conversation):
    turns = []
    conv = conversation['conversation']
    session_keys = sorted(
        [k for k in conv if k.startswith('session_') and not k.endswith('date_time')],
        key=lambda x: int(x.split('_')[1]))
    for sk in session_keys:
        for turn in conv[sk]:
            turns.append({'session': sk, 'speaker': turn.get('speaker', ''),
                          'dia_id': turn.get('dia_id', ''), 'text': turn.get('text', '')})
    return turns, session_keys


# ── Graph building (matching original locomo_eval.py) ──

def build_graph(turns, session_keys):
    embedder = get_embedder()
    graph = StarGraph()
    num_sessions = len(session_keys)
    for turn in turns:
        text = turn['text']
        if not text.strip():
            continue
        session_num = int(turn['session'].split('_')[1])
        embedding = embedder.encode(text)
        anchor = Anchor.create(
            text=text, source_session=turn['session'], embedding=embedding,
            tags=[turn['session'], turn['speaker']],
            importance=0.5, emotional_valence=0.0)
        hours_ago = (num_sessions - session_num + 1) * 4
        anchor.created_at = time.time() - hours_ago * 3600
        anchor.last_activated_at = anchor.created_at
        graph.add_anchor(anchor)
    return graph

def build_edges(graph):
    anchors = list(graph.anchors.values())
    if not anchors:
        return
    dim = len(anchors[0].embedding)
    ann = ANNIndex(dim=dim)
    for a in anchors:
        if a.embedding:
            ann.add(a.id, a.embedding)
    tag_groups = {}
    for a in anchors:
        for tag in a.tags:
            tag_groups.setdefault(tag, []).append(a.id)
    for tag, aids in tag_groups.items():
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a = graph.anchors.get(aids[i])
                b = graph.anchors.get(aids[j])
                if a and b and a.embedding and b.embedding:
                    sim = cosine_sim(a.embedding, b.embedding)
                    sim = min(1.0, sim * 1.25)
                    if sim > 0.48:
                        graph.add_edge(a.id, b.id, weight=sim)
    for a in anchors:
        if not a.embedding:
            continue
        neighbors = ann.query(a.embedding, k=15)
        for nid, score in neighbors:
            if nid != a.id and (min(a.id, nid), max(a.id, nid)) not in graph.edges:
                if score > 0.55:
                    graph.add_edge(a.id, nid, weight=score * 0.9)


# ── Method 1: Classic HybridFusion (for paper baseline) ──

def evaluate_hybridfusion(graph, qa_pairs, embedder):
    bm25 = BM25Index()
    for aid, anchor in graph.anchors.items():
        bm25.add(aid, anchor.text)
    ret = HybridFusionRetriever(graph)

    results = []
    for qa in qa_pairs:
        question = qa['question']
        answer = str(qa.get('answer') or qa.get('adversarial_answer', ''))
        category = str(qa.get('category', '4'))
        q_emb = embedder.encode(question)

        result = ret.retrieve(question, q_emb, top_k=12)
        seen = set()
        texts = []
        for c in result.constellations[:12]:
            for an in c.anchors:
                if an.id not in seen:
                    seen.add(an.id)
                    texts.append(an.text[:200])

        for aid_bm, _ in bm25.search(question, top_k=20):
            if aid_bm not in seen:
                an = graph.anchors.get(aid_bm)
                if an:
                    seen.add(aid_bm)
                    texts.append(an.text[:200])
                if len(texts) >= 40:
                    break

        combined = ' '.join(texts[:40])
        results.append({
            'hit': has_answer(answer, combined),
            'f1': f1_score(combined[:2000], answer),
            'category': category,
            'question': question,
        })
    return results


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='3 conversations only')
    parser.add_argument('--locomo-path', type=str,
                        default='C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from a partial results JSON file')
    args = parser.parse_args()

    seed_everything(42)
    dataset = load_locomo(args.locomo_path)
    num_conv = 3 if args.quick else len(dataset)

    print(f'LoCoMo-10 评估 — {num_conv} 对话, {sum(len(c["qa"]) for c in dataset[:num_conv])} QA')
    print('=' * 60)
    print('方法: HybridFusion + BM25 补充 (论文基线)')
    print('=' * 60)

    embedder = get_embedder()
    all_results = []
    total_hits = 0
    total_qa = 0
    total_f1 = 0.0

    for ci, conv in enumerate(dataset[:num_conv]):
        cid = conv.get('sample_id', f'conv{ci}')
        turns, session_keys = extract_all_turns(conv)
        graph = build_graph(turns, session_keys)
        n = len(graph.anchors)
        qa_pairs = conv['qa']

        results = evaluate_hybridfusion(graph, qa_pairs, embedder)
        conv_hits = sum(1 for r in results if r['hit'])
        conv_f1 = sum(r['f1'] for r in results) / max(1, len(results))

        total_hits += conv_hits
        total_qa += len(results)
        total_f1 += sum(r['f1'] for r in results)

        print(f'  [{ci+1}/{num_conv}] {cid}: {n} anchors')
        print(f'    QA: {conv_hits}/{len(qa_pairs)} = {conv_hits/len(qa_pairs)*100:.1f}%  F1={conv_f1:.4f}')
        all_results.extend(results)

    # Global report
    overall_rate = total_hits / total_qa * 100
    overall_f1 = total_f1 / total_qa

    print()
    print('=' * 60)
    print(f'  总计: {total_hits}/{total_qa} = {overall_rate:.1f}%  F1={overall_f1:.4f}')
    print()

    # Per category
    cat_names = {'1': 'Temporal', '2': 'Short Mem', '3': 'Long Mem',
                  '4': 'Composite', '5': 'Adversarial'}
    cats = defaultdict(lambda: {'hits': 0, 'total': 0, 'f1': 0.0})
    for r in all_results:
        c = r['category']
        cats[c]['hits'] += 1 if r['hit'] else 0
        cats[c]['total'] += 1
        cats[c]['f1'] += r['f1']

    print('  Category breakdown:')
    for c in sorted(cats.keys()):
        d = cats[c]
        rate = d['hits'] / d['total'] * 100
        avg_f1 = d['f1'] / d['total']
        name = cat_names.get(str(c), f'cat{c}')
        print(f'    Cat {c} ({name:>10s}): {d["hits"]}/{d["total"]} = {rate:.1f}%  F1={avg_f1:.4f}')

    # Save JSON
    out_json = {
        "config": {"conversations": num_conv, "total_qa": total_qa},
        "method": "HybridFusion+BM25",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "overall": {
            "has_answer": total_hits / total_qa,
            "has_answer_pct": round(overall_rate, 1),
            "f1": round(overall_f1, 4),
        },
        "by_category": {
            c: {"hits": d['hits'], "total": d['total'],
                "has_answer": round(d['hits']/d['total'], 3) if d['total']>0 else 0,
                "has_answer_pct": round(d['hits']/d['total']*100, 1) if d['total']>0 else 0,
                "f1": round(d['f1']/d['total'], 4) if d['total']>0 else 0}
            for c, d in sorted(cats.items())
        },
    }
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locomo_results.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f'\n  JSON → {json_path}')

    # Save CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locomo_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "total", "hits", "has_answer_pct", "f1"])
        for c in sorted(cats.keys()):
            d = cats[c]
            w.writerow([cat_names.get(str(c), f'cat{c}'),
                       d['total'], d['hits'],
                       f"{d['hits']/d['total']*100:.1f}" if d['total']>0 else "0",
                       f"{d['f1']/d['total']:.4f}" if d['total']>0 else "0"])
        w.writerow(["Overall", total_qa, total_hits, f"{overall_rate:.1f}", f"{overall_f1:.4f}"])
    print(f'  CSV → {csv_path}')


if __name__ == "__main__":
    main()
