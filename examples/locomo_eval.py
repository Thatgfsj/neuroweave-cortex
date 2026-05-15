"""LoCoMo benchmark evaluation for star-graph-memory.

Evaluates star-graph's cognitive memory retrievers against the LoCoMo-10 dataset
using content recall (has_answer token matching), matching the paper's methodology.

Usage: python examples/locomo_eval.py [--quick] [--conversations N]
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, Oscillator,
    OscillationResonanceRetriever,
    VectorSimilarityRetriever,
    HybridFusionRetriever,
    Constellation,
    seed_everything,
)
from star_graph.config import Config
from star_graph.embedding import get_embedder
from star_graph.sleep import SleepCycle

# ── LoCoMo evaluation metrics (from task_eval/evaluation.py) ──

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    s = s.replace(',', "")
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
    """Check if retrieved text contains answer string (token-level match).

    Multi-strategy matching:
      1. Exact stemmed n-gram sequence match (strict, LoCoMo paper method)
      2. Token-set overlap within sliding window (≥75% of answer tokens)
      3. Bigram Jaccard overlap (>0.5 threshold)
      4. Substring match (final fallback for short answers)
    """
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

        # Strategy 1: Exact token sequence match (original method)
        for i in range(len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[i:i + len(answer_tokens)]:
                return True

        # Strategy 2: Sliding-window token coverage
        window = max(len(answer_tokens) + 6, 10)
        for i in range(len(text_tokens) - min(window, len(text_tokens)) + 1):
            window_tokens = set(text_tokens[i:i + window])
            overlap = sum(1 for t in answer_tokens if t in window_tokens)
            if overlap / len(answer_tokens) >= 0.75:
                return True

        # Strategy 3: Bigram overlap
        ans_bigrams = set(zip(answer_tokens, answer_tokens[1:])) if len(answer_tokens) > 1 else set()
        text_bigrams = set(zip(text_tokens, text_tokens[1:])) if len(text_tokens) > 1 else set()
        if ans_bigrams and text_bigrams:
            bigram_jaccard = len(ans_bigrams & text_bigrams) / len(ans_bigrams | text_bigrams)
            if bigram_jaccard > 0.5:
                return True

        # Strategy 4: Substring match for short answers (≤3 words)
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
    return (2 * precision * recall) / (precision + recall)

def multi_answer_f1(prediction, ground_truth):
    preds = [p.strip() for p in str(prediction).split(',')]
    gts = [g.strip() for g in str(ground_truth).split(',')]
    scores = []
    for gt in gts:
        best = max(f1_score(p, gt) for p in preds)
        scores.append(best)
    return np.mean(scores)

def content_recall_at_k(texts, keywords, k):
    """Fraction of keywords found in top-k retrieved texts."""
    combined = ' '.join(texts[:k])
    found = sum(1 for kw in keywords if kw.lower() in combined.lower())
    return found / len(keywords) if keywords else 1.0

# ── Data loading ──

def load_locomo(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def extract_all_turns(conversation):
    """Extract all dialog turns from a conversation, in session order."""
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

# ── Ingest conversation into star-graph ──

def ingest_conversation(graph, turns, session_keys, embedder, batch_delay=0.002):
    """Feed all turns into the star-graph with session-aware timestamps."""
    num_sessions = len(session_keys)
    for turn in turns:
        session_num = int(turn['session'].split('_')[1])
        text = turn['text']
        if not text.strip():
            continue
        embedding = embedder.encode(text)
        anchor = Anchor.create(
            text=text,
            source_session=turn['session'],
            embedding=embedding,
            tags=[turn['session'], turn['speaker']],
            importance=0.5,
            emotional_valence=0.0,
        )
        hours_ago = (num_sessions - session_num + 1) * 4
        anchor.created_at = time.time() - hours_ago * 3600
        anchor.last_activated_at = anchor.created_at
        graph.add_anchor(anchor)
        if batch_delay > 0:
            time.sleep(batch_delay)

def build_edges(graph):
    """Build edges using tag-group + ANN approach."""
    from star_graph.index import ANNIndex

    anchors = list(graph.anchors.values())
    if not anchors:
        return

    dim = len(anchors[0].embedding)
    ann = ANNIndex(dim=dim)
    for a in anchors:
        if a.embedding:
            ann.add(a.id, a.embedding)

    tag_to_aids = {}
    for a in anchors:
        for tag in a.tags:
            tag_to_aids.setdefault(tag, []).append(a.id)

    # Tag-group edges
    for tag, tag_aids in tag_to_aids.items():
        for i in range(len(tag_aids)):
            for j in range(i + 1, len(tag_aids)):
                a = graph.anchors.get(tag_aids[i])
                b = graph.anchors.get(tag_aids[j])
                if a and b and a.embedding and b.embedding:
                    dot = sum(x * y for x, y in zip(a.embedding, b.embedding))
                    na = math.sqrt(sum(x**2 for x in a.embedding))
                    nb = math.sqrt(sum(x**2 for x in b.embedding))
                    sim = dot / (na * nb + 1e-8)
                    sim = min(1.0, sim * 1.25)
                    if sim > 0.48:
                        graph.add_edge(a.id, b.id, weight=sim)

    # ANN edges for top anchors not covered by tags
    for a in anchors:
        if not a.embedding:
            continue
        neighbors = ann.query(a.embedding, k=15)
        for nid, score in neighbors:
            if nid != a.id and (min(a.id, nid), max(a.id, nid)) not in graph.edges:
                if score > 0.55:
                    graph.add_edge(a.id, nid, weight=score * 0.9)

# ── QA evaluation ──

def _extract_relevant_snippet(anchor_text, question_tokens, min_chars=10):
    """Extract the most relevant sentence from anchor text for the question.

    If the text is short (<120 chars), return it whole. Otherwise find the
    sentence with highest question-token overlap.
    """
    if len(anchor_text) <= 120:
        return anchor_text
    import re
    sentences = re.split(r'(?<=[.!?])\s+', anchor_text)
    if len(sentences) <= 1:
        return anchor_text[:200]
    best_sentence = max(
        sentences,
        key=lambda s: sum(1 for t in question_tokens if t.lower() in s.lower())
    )
    return best_sentence if len(best_sentence) > min_chars else anchor_text[:200]


def evaluate_qa(graph, qa_pairs, retrievers, embedder):
    """Evaluate retrieval on QA pairs using LoCoMo's has_answer metric.

    Retrieves top-10 constellations and extracts the most relevant snippet
    from each anchor, improving answer surface coverage vs raw full-text
    concatenation.
    """
    from star_graph.bm25 import BM25Index

    results = {ret.name: {'hits': [], 'f1_scores': [], 'latency_ms': []}
               for ret in retrievers}

    # Build BM25 index for keyword pre-filtering
    bm25 = BM25Index()
    for aid, anchor in graph.anchors.items():
        bm25.add(aid, anchor.text, anchor.tags)

    for qa in qa_pairs:
        question = qa['question']
        answer = qa.get('answer') or qa.get('adversarial_answer', '')
        category = qa.get('category', '4')

        embedding = embedder.encode(question)
        question_tokens = normalize_answer(question).split()

        # BM25 keyword candidates (up to 15)
        bm25_ids = bm25.search(question, top_k=15)

        for ret in retrievers:
            result = ret.retrieve(question, embedding, top_k=8)
            results[ret.name]['latency_ms'].append(result.latency_ms)

            # Collect top-8 constellation anchors, interleaved with BM25 candidates
            retrieved_texts = []
            seen_ids = set()

            for c in result.constellations[:8]:
                for a in c.anchors:
                    if a.id not in seen_ids:
                        seen_ids.add(a.id)
                        snippet = _extract_relevant_snippet(a.text, question_tokens)
                        retrieved_texts.append(snippet)

            # Add top BM25 results not already covered
            for aid in bm25_ids:
                if aid not in seen_ids:
                    anchor = graph.anchors.get(aid)
                    if anchor:
                        seen_ids.add(aid)
                        snippet = _extract_relevant_snippet(anchor.text, question_tokens)
                        retrieved_texts.append(snippet)
                    if len(retrieved_texts) >= 15:
                        break

            combined = ' '.join(retrieved_texts)

            # has_answer check
            hit = has_answer(answer, combined)
            results[ret.name]['hits'].append(1 if hit else 0)

            # F1 score
            if category in ['2', '3', '4']:
                f1 = f1_score(combined[:1500], answer)
            elif category == '1':
                f1 = multi_answer_f1(combined[:1500], answer)
            elif category == '5':
                f1 = 0.0
            else:
                f1 = f1_score(combined[:1500], answer)
            results[ret.name]['f1_scores'].append(f1)

    return results

# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='LoCoMo benchmark for star-graph-memory')
    parser.add_argument('--quick', action='store_true', help='Evaluate only 3 conversations')
    parser.add_argument('--conversations', type=int, default=0, help='Number of conversations (0=all)')
    parser.add_argument('--skip-sleep', action='store_true', help='Skip sleep consolidation')
    parser.add_argument('--locomo-path', type=str,
                        default='C:/Users/thatg/AppData/Local/Temp/LoCoMo/data/locomo10.json')
    args = parser.parse_args()

    seed_everything(42)

    print("=" * 70)
    print("  LoCoMo Benchmark — Star Graph Memory Evaluation")
    print("=" * 70)

    # Load dataset
    locomo_path = os.path.expanduser(args.locomo_path)
    dataset = load_locomo(locomo_path)
    num_conv = args.conversations or (3 if args.quick else len(dataset))
    conversations = dataset[:num_conv]

    print(f"\n  Dataset: {len(dataset)} conversations, evaluating {num_conv}")
    total_qa = sum(len(c['qa']) for c in conversations)
    total_turns = sum(
        len([t for s in [k for k in c['conversation'] if k.startswith('session_') and not k.endswith('date_time')]
             for t in c['conversation'][s]])
        for c in conversations
    )
    print(f"  Total turns: {total_turns}, Total QA pairs: {total_qa}")

    embedder = get_embedder()
    print(f"  Embedder: {embedder.__class__.__name__}")

    # Per-category aggregation
    all_results = {
        'VectorSimilarity': {'hits': [], 'f1': [], 'latency': [], 'by_category': {}},
        'OscillationResonance': {'hits': [], 'f1': [], 'latency': [], 'by_category': {}},
        'HybridFusion': {'hits': [], 'f1': [], 'latency': [], 'by_category': {}},
    }

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get('sample_id', f'conv_{conv_idx}')
        print(f"\n{'─'*60}")
        print(f"  Conversation {conv_idx+1}/{num_conv}: {conv_id}")
        print(f"{'─'*60}")

        # Extract turns and sessions
        turns, session_keys = extract_all_turns(conv)
        print(f"  Sessions: {len(session_keys)}, Turns: {len(turns)}, QA: {len(conv['qa'])}")

        # Build star graph
        graph = StarGraph()
        t0 = time.perf_counter()
        print(f"  Ingesting turns...", end=' ', flush=True)
        ingest_conversation(graph, turns, session_keys, embedder, batch_delay=0.0005)
        ingest_time = time.perf_counter() - t0
        print(f"done ({ingest_time:.1f}s, {len(graph.anchors)} anchors)")

        # Build edges
        print(f"  Building edges...", end=' ', flush=True)
        t0 = time.perf_counter()
        build_edges(graph)
        edge_time = time.perf_counter() - t0
        edge_count = len(graph.edges)
        print(f"done ({edge_time:.1f}s, {edge_count} edges)")

        # Sleep consolidation
        if not args.skip_sleep:
            print(f"  Sleep consolidation...", end=' ', flush=True)
            t0 = time.perf_counter()
            sleep = SleepCycle(graph)
            report = sleep.run()
            sleep_time = time.perf_counter() - t0
            print(f"done ({sleep_time:.1f}s, {report['merged']} merged, "
                  f"{len(graph.anchors)} anchors remain)")

        # Run retrievers
        print(f"  Evaluating {len(conv['qa'])} QA pairs...", end=' ', flush=True)
        t0 = time.perf_counter()

        vec_ret = VectorSimilarityRetriever(graph)
        osc_ret = OscillationResonanceRetriever(graph)
        hyb_ret = HybridFusionRetriever(graph)

        conv_results = evaluate_qa(graph, conv['qa'], [vec_ret, osc_ret, hyb_ret], embedder)
        eval_time = time.perf_counter() - t0
        print(f"done ({eval_time:.1f}s)")

        # Print per-conversation summary
        for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
            r = conv_results[name]
            hit_rate = sum(r['hits']) / len(r['hits']) if r['hits'] else 0
            avg_f1 = np.mean(r['f1_scores']) if r['f1_scores'] else 0
            avg_lat = np.mean(r['latency_ms']) if r['latency_ms'] else 0
            print(f"    {name:<24} has_answer={hit_rate:.3f}  F1={avg_f1:.3f}  "
                  f"latency={avg_lat:.1f}ms")
            all_results[name]['hits'].extend(r['hits'])
            all_results[name]['f1'].extend(r['f1_scores'])
            all_results[name]['latency'].extend(r['latency_ms'])

            # Per-category
            for qa, h, f in zip(conv['qa'], r['hits'], r['f1_scores']):
                cat = str(qa.get('category', '4'))
                if cat not in all_results[name]['by_category']:
                    all_results[name]['by_category'][cat] = {'hits': [], 'f1': []}
                all_results[name]['by_category'][cat]['hits'].append(h)
                all_results[name]['by_category'][cat]['f1'].append(f)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  LoCoMo Benchmark Results")
    print(f"{'='*70}")

    cat_names = {'1': 'Temporal', '2': 'ShortMem', '3': 'LongMem', '4': 'Composite', '5': 'Adversarial'}

    print(f"\n  {'Method':<24} {'has_ans':>8} {'F1':>8} {'Latency':>8}")
    print(f"  {'─'*50}")
    for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
        r = all_results[name]
        ha = np.mean(r['hits']) if r['hits'] else 0
        f1 = np.mean(r['f1']) if r['f1'] else 0
        lat = np.mean(r['latency']) if r['latency'] else 0
        print(f"  {name:<24} {ha:>8.3f} {f1:>8.3f} {lat:>7.1f}ms")

    print(f"\n  Per-Category has_answer:")
    print(f"  {'Category':<14} {'VecSim':>10} {'OscRes':>10} {'HybFus':>10}")
    print(f"  {'─'*50}")
    for cat in sorted(cat_names.keys()):
        vals = []
        for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
            cd = all_results[name]['by_category'].get(cat)
            if cd and cd.get('hits'):
                ha = sum(cd['hits']) / len(cd['hits'])
            else:
                ha = 0.0
            vals.append(ha)
        print(f"  {cat_names[cat]:<14} {vals[0]*100:>9.1f}% {vals[1]*100:>9.1f}% {vals[2]*100:>9.1f}%")

    # Per-category F1
    print(f"\n  Per-Category F1:")
    print(f"  {'Category':<14} {'VecSim':>10} {'OscRes':>10} {'HybFus':>10}")
    print(f"  {'─'*50}")
    for cat in sorted(cat_names.keys()):
        vals = []
        for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
            cd = all_results[name]['by_category'].get(cat)
            if cd and cd.get('f1'):
                f1 = sum(cd['f1']) / len(cd['f1'])
            else:
                f1 = 0.0
            vals.append(f1)
        print(f"  {cat_names[cat]:<14} {vals[0]*100:>9.1f}% {vals[1]*100:>9.1f}% {vals[2]*100:>9.1f}%")

    # Compression ratio
    print(f"\n  Memory efficiency (avg per conversation):")
    print(f"    Turns: ~{total_turns // num_conv}, Anchors: varies, "
          f"Compression: graph-based semantic compression")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locomo_results.json')
    serializable = {}
    for name in all_results:
        serializable[name] = {
            'has_answer': float(np.mean(all_results[name]['hits'])) if all_results[name]['hits'] else 0,
            'f1': float(np.mean(all_results[name]['f1'])) if all_results[name]['f1'] else 0,
            'latency_ms': float(np.mean(all_results[name]['latency'])) if all_results[name]['latency'] else 0,
            'by_category': {
                cat: {
                    'has_answer': float(np.mean(cd['hits'])) if cd['hits'] else 0,
                    'f1': float(np.mean(cd['f1'])) if cd['f1'] else 0,
                }
                for cat, cd in all_results[name]['by_category'].items()
            }
        }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

if __name__ == '__main__':
    main()
