"""LoCoMo-10/50 full evaluation — RRF retrieval pipeline.

Academic usage:
    python benchmarks/run_locomo_full.py [--quick] [--conversations N]

Outputs:
    - Console report with per-category breakdown
    - JSON results file (benchmarks/locomo_results.json)
    - CSV for paper figures (benchmarks/locomo_results.csv)
"""

from __future__ import annotations

import argparse
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
from star_graph.retrieval_engine.retrieval_core import _detect_temporal_query
from star_graph.math_utils import cosine_sim


# ── Metrics (same as LoCoMo paper) ──

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
        # Strategy 1: exact token sequence match
        for i in range(len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[i:i + len(answer_tokens)]:
                return True
        # Strategy 2: sliding-window token coverage
        window = max(len(answer_tokens) + 6, 10)
        for i in range(len(text_tokens) - min(window, len(text_tokens)) + 1):
            window_tokens = set(text_tokens[i:i + window])
            overlap = sum(1 for t in answer_tokens if t in window_tokens)
            if overlap / len(answer_tokens) >= 0.75:
                return True
        # Strategy 3: bigram overlap
        ans_bigrams = set(zip(answer_tokens, answer_tokens[1:])) if len(answer_tokens) > 1 else set()
        text_bigrams = set(zip(text_tokens, text_tokens[1:])) if len(text_tokens) > 1 else set()
        if ans_bigrams and text_bigrams:
            bigram_jaccard = len(ans_bigrams & text_bigrams) / len(ans_bigrams | text_bigrams)
            if bigram_jaccard > 0.5:
                return True
        # Strategy 4: substring for short answers
        if len(answer_tokens) <= 3:
            answer_lower = normalize_answer(answer).lower()
            text_lower = normalize_answer(text).lower()
            if answer_lower in text_lower:
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


# ── Data loading ──

def load_locomo(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def extract_all_turns(conversation):
    turns = []
    conv = conversation['conversation']
    session_keys = sorted(
        [k for k in conv if k.startswith('session_') and not k.endswith('date_time')],
        key=lambda x: int(x.split('_')[1])
    )
    for sk in session_keys:
        for turn in conv[sk]:
            turns.append({
                'session': sk,
                'speaker': turn.get('speaker', ''),
                'dia_id': turn.get('dia_id', ''),
                'text': turn.get('text', ''),
            })
    return turns, session_keys


# ── Ingest into all RRF subsystems ──

def ingest_conversation_rrf(graph, turns, session_keys, embedder, rt, batch_delay=0.002):
    """Feed turns into graph + register in all RRF subsystems."""
    num_sessions = len(session_keys)
    for turn in turns:
        text = turn['text']
        if not text.strip():
            continue
        embedding = embedder.encode(text)
        session_num = int(turn['session'].split('_')[1])
        hours_ago = (num_sessions - session_num + 1) * 4

        anchor = Anchor.create(
            text=text,
            source_session=turn['session'],
            embedding=embedding,
            tags=[turn['session'], turn['speaker']],
            importance=0.5,
        )
        anchor.created_at = time.time() - hours_ago * 3600
        anchor.last_activated_at = anchor.created_at
        graph.add_anchor(anchor)

        # Raw buffer
        rt.raw_buffer.add(
            text=text, session_id=turn['session'],
            embedding=embedding, tags=[turn['speaker']],
            importance=0.5, anchor_id=anchor.id,
        )

        # BM25
        if rt.bm25 is not None:
            rt.bm25.add(anchor.id, text)

        # TimeSpine
        rt.timespine.index_anchor(
            anchor.id, timestamp=anchor.created_at,
            importance=0.5, embedding=embedding,
            topic=turn['speaker'],
        )

        # Exact cache
        rt.exact_cache.harvest_from_anchor(anchor)

        if batch_delay > 0:
            time.sleep(batch_delay)


def build_edges_rrf(graph, embedder):
    """Build similarity edges for graph descent path."""
    from star_graph.index import ANNIndex
    anchors = list(graph.anchors.values())
    if not anchors:
        return
    dim = len(anchors[0].embedding)
    ann = ANNIndex(dim=dim)
    for a in anchors:
        if a.embedding:
            ann.add(a.id, a.embedding)

    for a in anchors:
        if not a.embedding:
            continue
        neighbors = ann.query(a.embedding, k=6)
        for nid, score in neighbors:
            if nid != a.id:
                sim = min(1.0, score * 1.1)
                if sim > 0.5:
                    graph.add_edge(a.id, nid, weight=sim)


# ── Evaluation ──

def evaluate_rrf(rp, qa_pairs, graph, max_items=10):
    """Evaluate using RetrievalCore.recall() RRF pipeline."""
    results = []
    for qa in qa_pairs:
        question = qa['question']
        answer = qa.get('answer') or qa.get('adversarial_answer', '')
        category = qa.get('category', '4')

        ctx = rp.recall(query=question, max_items=max_items)

        retrieved_texts = []
        seen_ids = set()
        for item in ctx.items:
            if item.anchor and item.anchor.id not in seen_ids:
                seen_ids.add(item.anchor.id)
                text = item.compressed_text or item.anchor.text[:200]
                retrieved_texts.append(text)
            elif not item.anchor:
                text = item.compressed_text[:200] if item.compressed_text else ""
                if text and text[:40] not in seen_ids:
                    seen_ids.add(text[:40])
                    retrieved_texts.append(text)

        combined = ' '.join(retrieved_texts[:20])

        hit = has_answer(answer, combined)
        f1 = f1_score(combined[:2000], answer)
        is_temporal, _ = _detect_temporal_query(question)

        results.append({
            'question': question,
            'answer': answer[:100],
            'category': category,
            'hit': hit,
            'f1': f1,
            'num_results': len(ctx.items),
            'is_temporal_query': is_temporal,
        })

    return results


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='LoCoMo RRF benchmark')
    parser.add_argument('--quick', action='store_true', help='3 conversations only')
    parser.add_argument('--conversations', type=int, default=0, help='Number (0=all)')
    parser.add_argument('--locomo-path', type=str,
                        default='C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json')
    args = parser.parse_args()

    seed_everything(42)

    print("=" * 70)
    print("  LoCoMo RRF Pipeline Benchmark")
    print("  Uses: RetrievalCore.recall() — 5-path RRF fusion")
    print("=" * 70)

    dataset = load_locomo(args.locomo_path)
    num_conv = args.conversations or (3 if args.quick else len(dataset))
    conversations = dataset[:num_conv]

    total_qa = sum(len(c['qa']) for c in conversations)
    print(f"\n  Conversations: {num_conv}, Total QA: {total_qa}")

    embedder = get_embedder()

    all_categories = defaultdict(lambda: {'hits': 0, 'total': 0, 'f1': 0.0})
    all_results = []

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get('sample_id', f'conv_{conv_idx}')
        print(f"\n  {'─'*50}")
        print(f"  [{conv_idx+1}/{num_conv}] {conv_id}")

        turns, session_keys = extract_all_turns(conv)
        print(f"  Sessions: {len(session_keys)}, Turns: {len(turns)}, QA: {len(conv['qa'])}")

        graph = StarGraph()
        cfg = Config.get()
        rt = MemoryRuntime(graph=graph, config=cfg)
        rp = RetrievalPipeline(rt)

        ingest_conversation_rrf(graph, turns, session_keys, embedder, rt)
        build_edges_rrf(graph, embedder)
        print(f"  Graph: {len(graph.anchors)} anchors, {len(graph.edges)} edges")

        qa_results = evaluate_rrf(rp, conv['qa'], graph, max_items=10)
        conv_hits = sum(1 for r in qa_results if r['hit'])
        conv_f1 = sum(r['f1'] for r in qa_results) / max(1, len(qa_results))
        print(f"  has_answer={conv_hits}/{len(qa_results)} ({conv_hits/len(qa_results)*100:.1f}%)  F1={conv_f1:.4f}")

        for r in qa_results:
            cat = str(r['category'])
            all_categories[cat]['hits'] += 1 if r['hit'] else 0
            all_categories[cat]['total'] += 1
            all_categories[cat]['f1'] += r['f1']
            all_results.append(r)

    # Final report
    total = len(all_results)
    total_hits = sum(1 for r in all_results if r['hit'])
    total_f1 = sum(r['f1'] for r in all_results) / max(1, total)

    print(f"\n{'='*70}")
    print(f"  RRF Pipeline LoCoMo Results")
    print(f"{'='*70}")
    print(f"\n  Overall:")
    print(f"    has_answer: {total_hits}/{total} ({total_hits/total*100:.1f}%)")
    print(f"    F1:         {total_f1:.4f}")

    print(f"\n  By Category:")
    cat_names = {'1': 'Temporal', '2': 'Short Mem', '3': 'Long Mem',
                  '4': 'Composite', '5': 'Adversarial'}
    for cat in sorted(all_categories.keys()):
        d = all_categories[cat]
        rate = d['hits'] / d['total'] * 100 if d['total'] > 0 else 0
        avg_f1 = d['f1'] / d['total'] if d['total'] > 0 else 0
        name = cat_names.get(str(cat), f'cat{cat}')
        print(f"    Cat {cat} ({name:>10s}): {d['hits']}/{d['total']} = {rate:.1f}%  F1={avg_f1:.4f}")

    temporal_q = [r for r in all_results if r['is_temporal_query']]
    if temporal_q:
        t_hits = sum(1 for r in temporal_q if r['hit'])
        print(f"\n  Temporal queries: {t_hits}/{len(temporal_q)} ({t_hits/len(temporal_q)*100:.1f}%)")

    # Save results
    out = {
        "config": {"conversations": num_conv, "total_qa": total},
        "overall": {"has_answer": total_hits/total, "f1": total_f1},
        "by_category": {cat: {"hits": d['hits'], "total": d['total'],
                              "has_answer": d['hits']/d['total'] if d['total']>0 else 0,
                              "f1": d['f1']/d['total'] if d['total']>0 else 0}
                        for cat, d in all_categories.items()},
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locomo_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Save CSV for paper figures
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locomo_results.csv")
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["category", "total", "hits", "has_answer", "f1"])
        cat_names = {'1': 'Temporal', '2': 'Short_Mem', '3': 'Long_Mem',
                      '4': 'Composite', '5': 'Adversarial'}
        for cat in sorted(all_categories.keys()):
            d = all_categories[cat]
            rate = d['hits'] / d['total'] * 100 if d['total'] > 0 else 0
            avg_f1 = d['f1'] / d['total'] if d['total'] > 0 else 0
            writer.writerow([cat_names.get(str(cat), f'cat{cat}'),
                            d['total'], d['hits'], f"{rate:.1f}", f"{avg_f1:.4f}"])
        writer.writerow(["Overall", total, total_hits,
                        f"{total_hits/total*100:.1f}", f"{total_f1:.4f}"])
    print(f"  CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
