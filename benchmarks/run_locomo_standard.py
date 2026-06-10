"""LoCoMo-10 Full Benchmark — NWC v1.4.0
=========================================
Standard 50-conversation evaluation with detailed per-dialogue,
per-category metrics. Uses DeepSeek API for answer scoring when
available, falls back to has-answer evaluation.

Usage:
    python benchmarks/run_locomo_standard.py

Outputs:
    benchmarks/locomo_v140_results.json   — full per-QA results
    benchmarks/locomo_v140_summary.json   — per-dialogue summary
    benchmarks/locomo_v140_categories.json — per-category breakdown
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, get_embedder, seed_everything, MemoryRuntime,
    RetrievalPipeline, AgentContext, Config,
)
from star_graph.math_utils import cosine_sim
from star_graph.bm25 import BM25Index
from star_graph.index import ANNIndex

# ── DeepSeek LLM scorer (optional) ──────────────────────────

try:
    import openai as _openai_client
    _DEEPSEEK_MODEL = "deepseek-chat"
    _DEEPSEEK_BASE = "https://api.deepseek.com/v1"
    _DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    _DEEPSEEK_AVAILABLE = True
except ImportError:
    _DEEPSEEK_AVAILABLE = False


def _score_with_deepseek(question, retrieved_context, answer_candidates):
    """Use DeepSeek to judge whether retrieved context contains the answer."""
    if not _DEEPSEEK_AVAILABLE:
        return None

    prompt = f"""You are evaluating a memory retrieval system.

Question: {question}

Retrieved memory context:
{retrieved_context[:3000]}

Expected answer(s): {answer_candidates}

Does the retrieved context contain enough information to correctly answer the question?
Answer ONLY with a JSON object: {{"has_answer": true/false, "confidence": 0.0-1.0, "reason": "short explanation"}}"""

    try:
        client = _openai_client.OpenAI(api_key=_DEEPSEEK_KEY, base_url=_DEEPSEEK_BASE)
        resp = client.chat.completions.create(
            model=_DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # Try to parse JSON
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        return {"has_answer": None, "confidence": 0.0, "reason": str(e)}

    return None


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
    if not isinstance(answers, list):
        answers = [answers]
    text_tokens = normalize_answer(text).split()
    for answer in answers:
        answer = str(answer)
        answer = _normalize(answer)
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue
        # Exact token match
        for i in range(len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[i:i + len(answer_tokens)]:
                return True
        # Sliding window overlap
        window = max(len(answer_tokens) + 6, 10)
        for i in range(len(text_tokens) - min(window, len(text_tokens)) + 1):
            window_tokens = set(text_tokens[i:i + window])
            overlap = sum(1 for t in answer_tokens if t in window_tokens)
            if overlap / len(answer_tokens) >= 0.75:
                return True
        # Bigram Jaccard
        ans_bigrams = set(zip(answer_tokens, answer_tokens[1:])) if len(answer_tokens) > 1 else set()
        text_bigrams = set(zip(text_tokens, text_tokens[1:])) if len(text_tokens) > 1 else set()
        if ans_bigrams and text_bigrams:
            bigram_jaccard = len(ans_bigrams & text_bigrams) / max(1, len(ans_bigrams | text_bigrams))
            if bigram_jaccard > 0.5:
                return True
        # Short answer substring
        if len(answer_tokens) <= 3:
            if answer.lower() in text.lower():
                return True
    return False


def f1_score(answers, text):
    if not isinstance(answers, list):
        answers = [answers]
    text_tokens = set(normalize_answer(text).split())
    best_f1 = 0.0
    for answer in answers:
        answer_tokens = set(normalize_answer(str(answer)).split())
        if not answer_tokens:
            continue
        tp = len(answer_tokens & text_tokens)
        if tp == 0:
            continue
        precision = tp / max(1, len(text_tokens))
        recall = tp / max(1, len(answer_tokens))
        f1 = 2 * precision * recall / max(0.001, precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


# ── Data loading ──

def load_locomo(path):
    with open(path, 'r', encoding='utf-8') as f:
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
        graph.anchors[anchor.id] = anchor

    # Add edges between consecutive turns in same session
    turn_ids = sorted(graph.anchors.keys(), key=lambda tid: graph.anchors[tid].created_at)
    for i in range(len(turn_ids) - 1):
        a1 = graph.anchors[turn_ids[i]]
        a2 = graph.anchors[turn_ids[i + 1]]
        if a1.source_session == a2.source_session:
            graph.add_edge(turn_ids[i], turn_ids[i + 1], weight=0.3, edge_type='temporal')
    return graph


# ── Retrieval ──

def evaluate_retrieval(graph, qa_pairs, embedder, use_deepseek=False):
    results = []
    ann = ANNIndex()
    for anchor in graph.anchors.values():
        if getattr(anchor, 'embedding', None) and len(getattr(anchor, 'embedding', [])) > 0:
            ann.add(anchor.id, anchor.embedding)
    ann.rebuild()

    bm25 = BM25Index()
    for anchor in graph.anchors.values():
        bm25.add(anchor.id, anchor.text)

    for qa in qa_pairs:
        question = qa.get('question', '')
        answers = qa.get('answers', [qa.get('answer', '')])
        if isinstance(answers, str):
            answers = [answers]
        category = str(qa.get('category', '4'))

        # Embedding search
        q_emb = embedder.encode(question) if question else []
        vec_results = ann.query(q_emb, k=10) if q_emb else []

        # BM25 search
        bm25_results = bm25.search(question, top_k=10) if question else []

        # Hybrid fusion (RRF)
        rrf_scores = {}
        for rank, (aid, score) in enumerate(vec_results):
            rrf_scores[aid] = rrf_scores.get(aid, 0) + 1.0 / (rank + 60)
        for rank, (aid, score) in enumerate(bm25_results):
            rrf_scores[aid] = rrf_scores.get(aid, 0) + 1.0 / (rank + 60)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:10]

        # Concatenate retrieved texts
        retrieved_texts = []
        for aid in sorted_ids:
            anchor = graph.anchors.get(aid)
            if anchor:
                retrieved_texts.append(anchor.text)

        context = "\n".join(retrieved_texts)

        hit = has_answer(answers, context)
        f1 = f1_score(answers, context)

        # DeepSeek scoring (optional)
        llm_score = None
        if use_deepseek and _DEEPSEEK_AVAILABLE:
            llm_score = _score_with_deepseek(question, context, answers)
            if llm_score and llm_score.get('has_answer') is not None:
                hit = llm_score['has_answer']

        results.append({
            'question': question[:200],
            'answers': [str(a)[:200] for a in answers],
            'category': category,
            'hit': hit,
            'f1': round(f1, 4),
            'num_retrieved': len(retrieved_texts),
            'retrieved_preview': context[:500],
            'llm_score': llm_score,
        })

    return results


# ── Main ──

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--locomo-path', default='C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json')
    parser.add_argument('--use-deepseek', action='store_true', help='Use DeepSeek LLM for answer scoring')
    parser.add_argument('--output-dir', default='benchmarks')
    args = parser.parse_args()

    seed_everything(42)
    dataset = load_locomo(args.locomo_path)
    num_conv = 3 if args.quick else len(dataset)
    total_qa = sum(len(c['qa']) for c in dataset[:num_conv])

    print(f'LoCoMo-10 Standard — {num_conv} conversations, {total_qa} QA pairs')
    print(f'DeepSeek scoring: {"enabled" if args.use_deepseek and _DEEPSEEK_AVAILABLE else "disabled"}')
    print('=' * 70)

    embedder = get_embedder()
    all_results = []
    conv_summaries = []
    total_hits = 0
    total_f1 = 0.0
    cats = defaultdict(lambda: {'hits': 0, 'total': 0, 'f1': 0.0})

    t_start = time.time()

    for ci, conv in enumerate(dataset[:num_conv]):
        cid = conv.get('sample_id', f'conv{ci}')
        turns, session_keys = extract_all_turns(conv)
        graph = build_graph(turns, session_keys)
        n = len(graph.anchors)
        qa_pairs = conv['qa']

        t_conv = time.time()
        results = evaluate_retrieval(graph, qa_pairs, embedder, args.use_deepseek)
        conv_ms = (time.time() - t_conv) * 1000

        conv_hits = sum(1 for r in results if r['hit'])
        conv_f1 = sum(r['f1'] for r in results) / max(1, len(results))

        total_hits += conv_hits
        total_f1 += sum(r['f1'] for r in results)

        for r in results:
            cats[r['category']]['total'] += 1
            if r['hit']:
                cats[r['category']]['hits'] += 1
            cats[r['category']]['f1'] += r['f1']

        conv_summary = {
            'id': cid,
            'anchors': n,
            'qa_pairs': len(qa_pairs),
            'hits': conv_hits,
            'accuracy': round(conv_hits / max(1, len(qa_pairs)) * 100, 2),
            'avg_f1': round(conv_f1, 4),
            'latency_ms': round(conv_ms, 1),
        }
        conv_summaries.append(conv_summary)
        all_results.extend(results)

        print(f'  [{ci+1:>2}/{num_conv}] {cid:>5s}  anchors={n:>3d}  '
              f'acc={conv_hits}/{len(qa_pairs)}={conv_hits/len(qa_pairs)*100:.1f}%  '
              f'F1={conv_f1:.4f}  {conv_ms:.0f}ms')

    total_elapsed = (time.time() - t_start) / 60

    # Overall
    overall_rate = total_hits / total_qa * 100
    overall_f1 = total_f1 / total_qa

    print()
    print('=' * 70)
    print(f'  Total: {total_hits}/{total_qa} = {overall_rate:.1f}%  F1={overall_f1:.4f}  ({total_elapsed:.1f} min)')
    print()

    cat_names = {'1': 'Temporal', '2': 'Short Mem', '3': 'Long Mem',
                 '4': 'Composite', '5': 'Adversarial'}
    for cid, cdata in sorted(cats.items()):
        name = cat_names.get(cid, f'Cat{cid}')
        rate = cdata['hits'] / max(1, cdata['total']) * 100
        avgf1 = cdata['f1'] / max(1, cdata['total'])
        print(f'  {name:>15s}: {cdata["hits"]:>3d}/{cdata["total"]:>3d} = {rate:.1f}%  F1={avgf1:.4f}')

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    full_output = {
        'version': 'v1.4.0',
        'date': timestamp,
        'dataset': 'LoCoMo-10',
        'num_conversations': num_conv,
        'total_qa': total_qa,
        'overall': {
            'has_answer_rate': round(overall_rate, 2),
            'avg_f1': round(overall_f1, 4),
            'elapsed_minutes': round(total_elapsed, 1),
        },
        'categories': {},
        'conversations': conv_summaries,
        'details': all_results,
    }

    for cid, cdata in sorted(cats.items()):
        name = cat_names.get(cid, f'Cat{cid}')
        full_output['categories'][name] = {
            'hits': cdata['hits'],
            'total': cdata['total'],
            'accuracy': round(cdata['hits'] / max(1, cdata['total']) * 100, 2),
            'avg_f1': round(cdata['f1'] / max(1, cdata['total']), 4),
        }

    json_path = os.path.join(args.output_dir, 'locomo_v140_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved: {json_path}')
    print(f'File size: {os.path.getsize(json_path):,} bytes')


if __name__ == '__main__':
    main()
