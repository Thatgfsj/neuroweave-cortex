"""LoCoMo LLM-enhanced benchmark evaluation for star-graph-memory.

Extends the base locomo_eval with MiniMax LLM for:
  1. RAG answer generation — LLM reads retrieved context, answers question
  2. LLM-as-judge — LLM scores answer correctness against ground truth

This produces a more meaningful evaluation than token-matching alone,
since many correct answers use different wording than the ground truth.

Usage: python examples/locomo_llm_eval.py [--quick] [--conversations N] [--model NAME]
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

# ── MiniMax LLM client ──

class MiniMaxLLM:
    """OpenAI-compatible client targeting MiniMax API (M2.7 model)."""

    def __init__(self, model: str = "MiniMax-M2.7"):
        self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = os.environ.get("MINIMAX_API_BASE_URL",
                                        "https://api.minimaxi.com/v1")
        self.model = model

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip think blocks from MiniMax M2.7 reasoning model output.

        Handles complete <think>...</think>, unclosed <think>..., and
        trailing fragments from truncated JSON output.
        """
        import re as _re
        text = _re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
        idx = text.find('<think>')
        if idx >= 0:
            text = text[:idx]
        return text.strip()

    def _call(self, system: str, prompt: str,
              max_tokens: int = 512, temperature: float = 0.2) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            response = client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            raw = response.choices[0].message.content or ""
            return self._strip_thinking(raw)
        except ImportError:
            return "__NO_OPENAI__"
        except Exception as e:
            return f"__ERROR__:{e}"

    def answer_question(self, question: str, context: str) -> str:
        system = (
            "You are a precise QA assistant. Answer the question concisely "
            "using ONLY the provided context. If the context doesn't contain "
            "the answer, say 'I don't know based on the context.' "
            "Keep answers under 3 sentences."
        )
        prompt = f"Context:\n{context[:3000]}\n\nQuestion: {question}\n\nAnswer:"
        return self._call(system, prompt, max_tokens=400)

    def judge_correctness(self, question: str, ground_truth: str,
                          generated_answer: str) -> dict:
        system = (
            "You are an answer evaluator. Compare the generated answer to "
            "the ground truth answer. Score as:\n"
            "- 1.0: completely correct (semantically equivalent)\n"
            "- 0.5: partially correct (some overlap but incomplete or partially wrong)\n"
            "- 0.0: completely wrong or contradictory\n"
            "Reply with JSON only, no thinking, no markdown: "
            '{"score": <float>, "reason": "<brief>"}'
        )
        prompt = (
            f"Question: {question}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Generated Answer: {generated_answer}\n\n"
            f"Score:"
        )
        raw = self._call(system, prompt, max_tokens=500, temperature=0.0)
        try:
            import re as _re
            # Strip any remaining think blocks first
            raw = self._strip_thinking(raw)
            # Extract JSON — be more lenient
            match = _re.search(r'\{\s*"score"\s*:\s*([\d.]+)\s*[,}]', raw)
            if match:
                score = float(match.group(1))
                reason_match = _re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
                reason = reason_match.group(1) if reason_match else ""
                return {"score": max(0.0, min(1.0, score)), "reason": reason}
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass
        return {"score": 0.0, "reason": f"parse_error:{raw[:80]}"}


# ── Token-matching metrics (from base locomo_eval) ──

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
    return (2 * precision * recall) / (precision + recall)

def multi_answer_f1(prediction, ground_truth):
    preds = [p.strip() for p in str(prediction).split(',')]
    gts = [g.strip() for g in str(ground_truth).split(',')]
    scores = [max(f1_score(p, gt) for p in preds) for gt in gts]
    return np.mean(scores)


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


# ── Graph construction (same as base eval) ──

def ingest_conversation(graph, turns, session_keys, embedder, batch_delay=0.0005):
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
    for a in anchors:
        if not a.embedding:
            continue
        neighbors = ann.query(a.embedding, k=15)
        for nid, score in neighbors:
            if nid != a.id and (min(a.id, nid), max(a.id, nid)) not in graph.edges:
                if score > 0.55:
                    graph.add_edge(a.id, nid, weight=score * 0.9)


def _extract_relevant_snippet(anchor_text, question_tokens, min_chars=10):
    if len(anchor_text) <= 120:
        return anchor_text
    sentences = re.split(r'(?<=[.!?])\s+', anchor_text)
    if len(sentences) <= 1:
        return anchor_text[:200]
    best_sentence = max(
        sentences,
        key=lambda s: sum(1 for t in question_tokens if t.lower() in s.lower())
    )
    return best_sentence if len(best_sentence) > min_chars else anchor_text[:200]


# ── LLM-enhanced evaluation ──

def evaluate_qa_llm(graph, qa_pairs, retrievers, embedder, llm: MiniMaxLLM,
                    max_samples: int = 0):
    """RAG + LLM evaluation: retrieve context, LLM answers, judge correctness."""
    from star_graph.bm25 import BM25Index

    results = {}
    for ret in retrievers:
        results[ret.name] = {
            'hits': [], 'f1_scores': [], 'latency_ms': [],
            'llm_hits': [], 'llm_judge_scores': [],
            'llm_answers': [], 'llm_judge_reasons': [],
        }

    bm25 = BM25Index()
    for aid, anchor in graph.anchors.items():
        tag_text = " ".join(anchor.tags) if anchor.tags else ""
        bm25.add(aid, f"{anchor.text} {tag_text}")

    samples_processed = 0
    for qa in qa_pairs:
        question = qa['question']
        answer = qa.get('answer') or qa.get('adversarial_answer', '')
        category = qa.get('category', '4')

        embedding = embedder.encode(question)
        question_tokens = normalize_answer(question).split()
        bm25_ids = bm25.search(question, top_k=15)

        for ret in retrievers:
            result = ret.retrieve(question, embedding, top_k=8)
            results[ret.name]['latency_ms'].append(result.latency_ms)

            retrieved_texts = []
            seen_ids = set()
            for c in result.constellations[:8]:
                for a in c.anchors:
                    if a.id not in seen_ids:
                        seen_ids.add(a.id)
                        snippet = _extract_relevant_snippet(a.text, question_tokens)
                        retrieved_texts.append(snippet)
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

            # Token-match metrics (baseline)
            hit = has_answer(answer, combined)
            results[ret.name]['hits'].append(1 if hit else 0)
            if category in ['2', '3', '4']:
                f1 = f1_score(combined[:1500], answer)
            elif category == '1':
                f1 = multi_answer_f1(combined[:1500], answer)
            elif category == '5':
                f1 = 0.0
            else:
                f1 = f1_score(combined[:1500], answer)
            results[ret.name]['f1_scores'].append(f1)

            # LLM answer generation
            llm_answer = llm.answer_question(question, combined)
            results[ret.name]['llm_answers'].append(llm_answer)

            # LLM-as-judge
            judgement = llm.judge_correctness(question, answer, llm_answer)
            results[ret.name]['llm_judge_scores'].append(judgement.get("score", 0.0))
            results[ret.name]['llm_judge_reasons'].append(judgement.get("reason", ""))

            # Token-match on LLM answer (complement to judge)
            llm_hit = has_answer(answer, llm_answer)
            results[ret.name]['llm_hits'].append(1 if llm_hit else 0)

        samples_processed += 1
        if max_samples > 0 and samples_processed >= max_samples:
            break

    return results


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='LoCoMo LLM-enhanced benchmark')
    parser.add_argument('--quick', action='store_true', help='Evaluate only 2 conversations')
    parser.add_argument('--conversations', type=int, default=0)
    parser.add_argument('--skip-sleep', action='store_true')
    parser.add_argument('--model', type=str, default='MiniMax-M2.7',
                       help='LLM model name (MiniMax-M2.7 is the default)')
    parser.add_argument('--max-samples', type=int, default=0,
                       help='Max QA pairs per conversation (0=all)')
    parser.add_argument('--locomo-path', type=str,
                       default='C:/Users/thatg/AppData/Local/Temp/LoCoMo/data/locomo10.json')
    args = parser.parse_args()

    seed_everything(42)

    print("=" * 70)
    print("  LoCoMo LLM-Enhanced Benchmark — Star Graph Memory + MiniMax")
    print("=" * 70)

    # Init LLM
    llm = MiniMaxLLM(model=args.model)
    if not llm.api_key:
        print("\n  ERROR: MINIMAX_API_KEY not set in environment.")
        print("  Set it via: setx MINIMAX_API_KEY <your-key>")
        sys.exit(1)
    print(f"\n  LLM: MiniMax gateway ({args.model})")

    # Load dataset
    locomo_path = os.path.expanduser(args.locomo_path)
    dataset = load_locomo(locomo_path)
    num_conv = args.conversations or (2 if args.quick else len(dataset))
    conversations = dataset[:num_conv]

    print(f"  Dataset: {len(dataset)} conversations, evaluating {num_conv}")
    total_qa = sum(len(c['qa']) for c in conversations)
    total_turns = sum(
        len([t for s in [k for k in c['conversation']
             if k.startswith('session_') and not k.endswith('date_time')]
             for t in c['conversation'][s]])
        for c in conversations
    )
    print(f"  Total turns: {total_turns}, Total QA pairs: {total_qa}")
    if args.max_samples > 0:
        print(f"  Max samples/conversation: {args.max_samples}")

    embedder = get_embedder()
    print(f"  Embedder: {embedder.__class__.__name__}")

    all_results = {
        'VectorSimilarity': {'hits': [], 'f1': [], 'latency': [],
                            'llm_hits': [], 'llm_judge': [], 'by_category': {}},
        'OscillationResonance': {'hits': [], 'f1': [], 'latency': [],
                                 'llm_hits': [], 'llm_judge': [], 'by_category': {}},
        'HybridFusion': {'hits': [], 'f1': [], 'latency': [],
                         'llm_hits': [], 'llm_judge': [], 'by_category': {}},
    }

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get('sample_id', f'conv_{conv_idx}')
        print(f"\n{'─'*60}")
        print(f"  Conversation {conv_idx+1}/{num_conv}: {conv_id}")
        print(f"{'─'*60}")

        turns, session_keys = extract_all_turns(conv)
        print(f"  Sessions: {len(session_keys)}, Turns: {len(turns)}, "
              f"QA: {len(conv['qa'])}")

        graph = StarGraph()
        t0 = time.perf_counter()
        print(f"  Ingesting turns...", end=' ', flush=True)
        ingest_conversation(graph, turns, session_keys, embedder)
        ingest_time = time.perf_counter() - t0
        print(f"done ({ingest_time:.1f}s, {len(graph.anchors)} anchors)")

        print(f"  Building edges...", end=' ', flush=True)
        t0 = time.perf_counter()
        build_edges(graph)
        edge_time = time.perf_counter() - t0
        print(f"done ({edge_time:.1f}s, {len(graph.edges)} edges)")

        if not args.skip_sleep:
            print(f"  Sleep consolidation...", end=' ', flush=True)
            t0 = time.perf_counter()
            sleep = SleepCycle(graph)
            report = sleep.run()
            sleep_time = time.perf_counter() - t0
            print(f"done ({sleep_time:.1f}s, {report['merged']} merged, "
                  f"{len(graph.anchors)} anchors remain)")

        vec_ret = VectorSimilarityRetriever(graph)
        osc_ret = OscillationResonanceRetriever(graph)
        hyb_ret = HybridFusionRetriever(graph)

        print(f"  Evaluating {len(conv['qa'])} QA pairs with LLM...", flush=True)
        t0 = time.perf_counter()
        conv_results = evaluate_qa_llm(
            graph, conv['qa'], [vec_ret, osc_ret, hyb_ret], embedder, llm,
            max_samples=args.max_samples,
        )
        eval_time = time.perf_counter() - t0
        print(f"  done ({eval_time:.1f}s)")

        for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
            r = conv_results[name]
            hit_rate = sum(r['hits']) / len(r['hits']) if r['hits'] else 0
            avg_f1 = np.mean(r['f1_scores']) if r['f1_scores'] else 0
            avg_lat = np.mean(r['latency_ms']) if r['latency_ms'] else 0
            llm_hit_rate = sum(r['llm_hits']) / len(r['llm_hits']) if r['llm_hits'] else 0
            llm_judge_avg = np.mean(r['llm_judge_scores']) if r['llm_judge_scores'] else 0
            print(f"    {name:<24} token-ha={hit_rate:.3f}  F1={avg_f1:.3f}  "
                  f"llm-ha={llm_hit_rate:.3f}  llm-judge={llm_judge_avg:.3f}  "
                  f"lat={avg_lat:.1f}ms")

            all_results[name]['hits'].extend(r['hits'])
            all_results[name]['f1'].extend(r['f1_scores'])
            all_results[name]['latency'].extend(r['latency_ms'])
            all_results[name]['llm_hits'].extend(r['llm_hits'])
            all_results[name]['llm_judge'].extend(r['llm_judge_scores'])

            if args.max_samples > 0:
                n = min(args.max_samples, len(conv['qa']))
                qa_iter = conv['qa'][:n]
            else:
                qa_iter = conv['qa']
            for qa, h, f, lh, lj in zip(
                qa_iter, r['hits'], r['f1_scores'],
                r['llm_hits'], r['llm_judge_scores'],
            ):
                cat = str(qa.get('category', '4'))
                if cat not in all_results[name]['by_category']:
                    all_results[name]['by_category'][cat] = {
                        'hits': [], 'f1': [], 'llm_hits': [], 'llm_judge': [],
                    }
                all_results[name]['by_category'][cat]['hits'].append(h)
                all_results[name]['by_category'][cat]['f1'].append(f)
                all_results[name]['by_category'][cat]['llm_hits'].append(lh)
                all_results[name]['by_category'][cat]['llm_judge'].append(lj)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  LoCoMo LLM-Enhanced Benchmark Results")
    print(f"{'='*70}")

    cat_names = {'1': 'Temporal', '2': 'ShortMem', '3': 'LongMem',
                 '4': 'Composite', '5': 'Adversarial'}

    print(f"\n  {'Method':<24} {'tok-ha':>8} {'F1':>8} {'llm-ha':>8} "
          f"{'llm-judge':>10} {'Latency':>8}")
    print(f"  {'─'*70}")
    for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
        r = all_results[name]
        ha = np.mean(r['hits']) if r['hits'] else 0
        f1 = np.mean(r['f1']) if r['f1'] else 0
        lat = np.mean(r['latency']) if r['latency'] else 0
        lh = np.mean(r['llm_hits']) if r['llm_hits'] else 0
        lj = np.mean(r['llm_judge']) if r['llm_judge'] else 0
        print(f"  {name:<24} {ha:>8.3f} {f1:>8.3f} {lh:>8.3f} "
              f"{lj:>10.3f} {lat:>7.1f}ms")

    print(f"\n  Per-Category LLM-Judge Scores:")
    print(f"  {'Category':<14} {'VecSim':>10} {'OscRes':>10} {'HybFus':>10}")
    print(f"  {'─'*50}")
    for cat in sorted(cat_names.keys()):
        vals = []
        for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
            cd = all_results[name]['by_category'].get(cat)
            if cd and cd.get('llm_judge'):
                lj = sum(cd['llm_judge']) / len(cd['llm_judge'])
            else:
                lj = 0.0
            vals.append(lj)
        print(f"  {cat_names[cat]:<14} {vals[0]*100:>9.1f}% {vals[1]*100:>9.1f}% "
              f"{vals[2]*100:>9.1f}%")

    print(f"\n  Per-Category LLM has_answer (token-match on LLM output):")
    print(f"  {'Category':<14} {'VecSim':>10} {'OscRes':>10} {'HybFus':>10}")
    print(f"  {'─'*50}")
    for cat in sorted(cat_names.keys()):
        vals = []
        for name in ['VectorSimilarity', 'OscillationResonance', 'HybridFusion']:
            cd = all_results[name]['by_category'].get(cat)
            if cd and cd.get('llm_hits'):
                lh = sum(cd['llm_hits']) / len(cd['llm_hits'])
            else:
                lh = 0.0
            vals.append(lh)
        print(f"  {cat_names[cat]:<14} {vals[0]*100:>9.1f}% {vals[1]*100:>9.1f}% "
              f"{vals[2]*100:>9.1f}%")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'locomo_llm_results.json')
    serializable = {}
    for name in all_results:
        r = all_results[name]
        serializable[name] = {
            'token_has_answer': float(np.mean(r['hits'])) if r['hits'] else 0,
            'f1': float(np.mean(r['f1'])) if r['f1'] else 0,
            'llm_has_answer': float(np.mean(r['llm_hits'])) if r['llm_hits'] else 0,
            'llm_judge_score': float(np.mean(r['llm_judge'])) if r['llm_judge'] else 0,
            'latency_ms': float(np.mean(r['latency'])) if r['latency'] else 0,
            'by_category': {
                cat: {
                    'token_has_answer': float(np.mean(cd['hits'])) if cd['hits'] else 0,
                    'f1': float(np.mean(cd['f1'])) if cd['f1'] else 0,
                    'llm_has_answer': float(np.mean(cd['llm_hits'])) if cd['llm_hits'] else 0,
                    'llm_judge_score': float(np.mean(cd['llm_judge'])) if cd['llm_judge'] else 0,
                }
                for cat, cd in r['by_category'].items()
            }
        }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

if __name__ == '__main__':
    main()
