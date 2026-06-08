"""Baseline: Pure Vector Search (same embedder as NWC, no cognitive modules).

Fair comparison: same LoCoMo-10 data, same embedding model (via get_embedder()),
same has_answer metric. This measures the performance of raw embedding-only retrieval.
"""
import sys, os, json, csv, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import get_embedder, seed_everything
from star_graph.math_utils import cosine_sim as _cosine_sim
from benchmarks.run_locomo_full import has_answer, f1_score, load_locomo, extract_all_turns

seed_everything(42)
dataset = load_locomo(
    'C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json')

embedder = get_embedder()
cats_all = defaultdict(lambda: {'hits': 0, 'total': 0, 'f1': 0.0})
total_hits = total_qa = total_f1 = 0

print('Baseline: Pure Vector Search (same embedding, no cognitive modules)')
print(f'10 conversations, {sum(len(c["qa"]) for c in dataset)} QA pairs\n')

for ci, conv in enumerate(dataset):
    cid = conv.get('sample_id', f'conv{ci}')
    turns, session_keys = extract_all_turns(conv)
    qa_pairs = conv['qa']
    t0 = time.time()

    # Build vector store: just text + embedding pairs
    texts = {}
    embs = {}
    for turn in turns:
        text = turn['text']
        if not text.strip(): continue
        texts[turn['session'] + '_' + turn.get('dia_id', str(len(texts)))] = text
        embs[turn['session']] = embedder.encode(text)  # just keep last

    # Actually, collect all embeddings properly
    store = []
    for turn in turns:
        text = turn['text']
        if not text.strip(): continue
        emb = embedder.encode(text)
        store.append((text, emb))

    conv_hits = conv_f1 = 0
    for qa in qa_pairs:
        q = qa['question']
        a = str(qa.get('answer') or qa.get('adversarial_answer', ''))
        cat = str(qa.get('category', '4'))
        q_emb = embedder.encode(q)

        # Pure cosine: sort all by similarity, take top-40
        scored = [(_cosine_sim(q_emb, emb), text) for text, emb in store]
        scored.sort(key=lambda x: -x[0])
        combined = ' '.join(text[:200] for _, text in scored[:40])

        hit = has_answer(a, combined)
        f1v = f1_score(combined[:2000], a)
        if hit: conv_hits += 1
        conv_f1 += f1v
        cats_all[cat]['hits'] += 1 if hit else 0
        cats_all[cat]['total'] += 1
        cats_all[cat]['f1'] += f1v

    total_hits += conv_hits
    total_qa += len(qa_pairs)
    total_f1 += conv_f1
    dt = time.time() - t0
    print(f'[{ci+1}/10] {cid}: {len(store)} docs, {len(qa_pairs)} QA '
          f'-> {conv_hits}/{len(qa_pairs)} = {conv_hits/len(qa_pairs)*100:.1f}%  ({dt:.0f}s)')

overall_rate = total_hits / total_qa * 100
overall_f1 = total_f1 / total_qa
print(f'\n{"="*60}')
print(f'  Pure Vector Search 总计: {total_hits}/{total_qa} = {overall_rate:.1f}%  F1={overall_f1:.4f}')

cat_names = {'1':'Temporal','2':'Short Mem','3':'Long Mem','4':'Composite','5':'Adversarial'}
print('\n  Category breakdown:')
for c in sorted(cats_all.keys()):
    d = cats_all[c]
    print(f'    Cat {c} ({cat_names.get(str(c),c):>10s}): {d["hits"]}/{d["total"]} = {d["hits"]/d["total"]*100:.1f}%')

# Save
out = {
    'method': 'PureVector',
    'overall': {'has_answer_pct': round(overall_rate,1), 'f1': round(overall_f1,4)},
    'by_category': {c: {'has_answer_pct': round(d['hits']/d['total']*100,1) if d['total']>0 else 0}
                     for c,d in sorted(cats_all.items())}
}
with open('benchmarks/baseline_vector_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'\n  Saved → benchmarks/baseline_vector_results.json')
