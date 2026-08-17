"""Revision v2.0 paper figures.

Reads revision_results/*.json and renders vector PDFs to paper/figures/:
  fig1_architecture.pdf — six-layer hierarchy + signal flow (static diagram)
  fig2_main_heatmap.pdf — systems × categories on LoCoMo-10
  fig3_ablation.pdf     — ablation bars with bootstrap CI
  fig4_growth.pdf       — memory compression: input tokens → retained anchors
  fig5_longmemeval.pdf  — systems × tasks on LongMemEval (if data present)

Usage:
    python benchmarks/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'paper', 'figures')
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'revision_results')

CATS = ['Temporal', 'Short', 'Long', 'Composite', 'Adversarial']
SYS_SHORT = {
    'vanilla_rag': 'Vanilla RAG', 'bm25': 'BM25', 'hybrid_rrf': 'Hybrid',
    'nwc_full': 'NWC Hyb.', 'a1_no_lifecycle': 'A1 No Lifecycle',
    'a2_no_activation': 'A2 No Activation', 'a3_no_lexical': 'A3 No Lexical',
    'a4_no_confidence': 'A4 No Confidence', 'a5_single_layer': 'A5 Single Layer',
}


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axis('off')
    layers = [
        ('L6 Identity', 'cognitive profile ~600 tok'),
        ('L5 Belief', 'strength + evidence'),
        ('L4 Pattern', 'behaviour patterns'),
        ('L3 Semantic', 'concept graph, typed edges'),
        ('L2 Episodic', 'memory anchors'),
        ('L1 Sensory', 'importance gate'),
    ]
    y = 5.2
    for name, desc in layers:
        ax.add_patch(plt.Rectangle((0.1, y - 0.32), 2.6, 0.55,
                                   facecolor='#dbe9f7', edgecolor='#2b6cb0',
                                   linewidth=1.2, zorder=3))
        ax.text(1.4, y + 0.05, name, ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=4)
        ax.text(3.1, y, desc, ha='left', va='center', fontsize=8.5,
                color='#444', zorder=4)
        if y < 5.2:
            ax.annotate('', xy=(1.4, y - 0.3), xytext=(1.4, y + 0.26),
                        arrowprops=dict(arrowstyle='->', color='#2b6cb0',
                                        lw=1.4), zorder=2)
        y -= 0.78
    ax.text(0.1, 0.05,
            'sleep consolidation  |  three-dimensional decay  |  '
            'spreading-activation retrieval',
            fontsize=9, color='#c05621', style='italic')
    ax.text(5.0, 5.45, 'compression: tokens → structure', fontsize=9,
            color='#2b6cb0')
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_architecture.pdf'))
    plt.close(fig)


def fig2_main_heatmap(summary):
    systems = ['vanilla_rag', 'bm25', 'hybrid_rrf', 'nwc_full']
    labels = [SYS_SHORT[s] for s in systems]
    data = np.array([
        [summary[s]['by_category'].get(c, {}).get('acc', np.nan) for c in CATS]
        for s in systems
    ])
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    im = ax.imshow(data, cmap='YlGnBu', vmin=0, vmax=70, aspect='auto')
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(CATS, fontsize=9)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(systems)):
        for j in range(len(CATS)):
            v = data[i, j]
            ax.text(j, i, f'{v:.0f}' if not np.isnan(v) else '--',
                    ha='center', va='center', fontsize=8,
                    color='black' if v > 40 else 'white')
    ax.set_title('LoCoMo-10 has-answer accuracy (%) by category',
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_main_heatmap.pdf'))
    plt.close(fig)


def fig3_ablation(stats):
    configs = ['nwc_full', 'a1_no_lifecycle', 'a2_no_activation',
               'a3_no_lexical', 'a4_no_confidence', 'a5_single_layer']
    labels = [SYS_SHORT[c] for c in configs]
    accs = [stats[c]['acc'] * 100 for c in configs]
    los = [stats[c]['ci95'][0] * 100 for c in configs]
    his = [stats[c]['ci95'][1] * 100 for c in configs]
    err = [[acc - lo for acc, lo in zip(accs, los)],
           [hi - acc for acc, hi in zip(accs, his)]]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = ['#2b6cb0'] + ['#cbd5e0'] * 4 + ['#e2e8f0']
    bars = ax.bar(range(len(configs)), accs, yerr=err, capsize=4,
                  color=colors, edgecolor='#2d3748', linewidth=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('has-answer accuracy (%)')
    for i, (acc, hi) in enumerate(zip(accs, his)):
        ax.text(i, hi + 0.8, f'{acc:.1f}', ha='center', fontsize=8.5)
    ax.set_ylim(0, max(his) * 1.15)
    ax.set_title('Ablation: contribution of each fusion signal (LoCoMo-10)',
                 fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_ablation.pdf'))
    plt.close(fig)


def fig4_growth():
    """Tokens in vs anchors retained per conversation (real data from
    E:\\shiyan\\results\\locomo_v140_results.json + the LoCoMo dataset)."""
    import re
    from benchmarks.run_locomo_full import load_locomo, extract_all_turns
    dataset = load_locomo(r'E:\locomo-10\data\locomo10.json')
    with open(r'E:\shiyan\results\locomo_v140_results.json') as f:
        res = json.load(f)
    conv_map = {c['id']: c for c in res['conversations']}
    tokens_in, anchors, conv_ids = [], [], []
    for conv in dataset:
        cid = conv.get('sample_id', '')
        turns, _ = extract_all_turns(conv)
        tokens = sum(len(t['text'].split()) for t in turns)
        if cid in conv_map:
            tokens_in.append(tokens)
            anchors.append(conv_map[cid]['anchors'])
            conv_ids.append(cid)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.scatter(tokens_in, anchors, c='#2b6cb0', s=40, edgecolor='white')
    for x, y, cid in zip(tokens_in, anchors, conv_ids):
        ax.annotate(cid.replace('conv-', ''), (x, y), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    z = np.polyfit(tokens_in, anchors, 1)
    xs = np.linspace(min(tokens_in), max(tokens_in), 50)
    ax.plot(xs, np.polyval(z, xs), '--', color='#c05621', lw=1)
    ax.set_xlabel('input tokens per conversation')
    ax.set_ylabel('anchors retained')
    ax.set_title('Memory growth is sublinear: anchors vs input tokens',
                 fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_growth.pdf'))
    plt.close(fig)


def fig5_longmemeval(lme):
    if not os.path.exists(lme):
        print('[skip] fig5: no longmemeval data')
        return
    with open(lme) as f:
        d = json.load(f)
    ov = d.get('overall', {})
    labels = [SYS_SHORT.get(k, k) for k in ov]
    vals = [v['has_answer_pct'] for v in ov.values()]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(range(len(labels)), vals, color='#2b6cb0',
           edgecolor='#2d3748', linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('has-answer accuracy (%)')
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title('LongMemEval-small: has-answer accuracy', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_longmemeval.pdf'))
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_architecture()
    print('fig1_architecture.pdf done')

    summary_path = os.path.join(RES_DIR, 'revision_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        fig2_main_heatmap(summary)
        print('fig2_main_heatmap.pdf done')
    else:
        print('[skip] fig2: no revision_summary.json')

    stats_path = os.path.join(RES_DIR, 'statistics.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        fig3_ablation(stats)
        print('fig3_ablation.pdf done')
    else:
        print('[skip] fig3: no statistics.json')

    fig4_growth()
    print('fig4_growth.pdf done')
    fig5_longmemeval(os.path.join(RES_DIR, 'longmemeval_summary.json'))


if __name__ == '__main__':
    main()
