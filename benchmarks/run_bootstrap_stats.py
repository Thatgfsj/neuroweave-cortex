"""Bootstrap statistics for revision v2.0 results.

Reads per-QA hit arrays produced by run_revision_exp.py and computes:
  - accuracy per configuration
  - 95% bootstrap confidence intervals (paired over QA pairs)
  - paired bootstrap p-values (nwc_full vs every other configuration)

Usage:
    python benchmarks/run_bootstrap_stats.py

Output:
    benchmarks/revision_results/statistics.json
"""

from __future__ import annotations

import json
import os
import random
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'revision_results')
RNG = random.Random(42)


def bootstrap_ci(hits_a, hits_b=None, n_resamples=2000, alpha=0.05):
    """Paired bootstrap: CI on mean(a) and on mean(a)-mean(b)."""
    n = len(hits_a)
    if hits_b is None:
        diffs = None
    else:
        assert len(hits_b) == n
        diffs = [a - b for a, b in zip(hits_a, hits_b)]
    sample_means, sample_diffs = [], []
    for _ in range(n_resamples):
        idx = [RNG.randrange(n) for _ in range(n)]
        mean_a = sum(hits_a[i] for i in idx) / n
        sample_means.append(mean_a)
        if diffs is not None:
            sample_diffs.append(sum(diffs[i] for i in idx) / n)
    sample_means.sort()
    lo = sample_means[int(n_resamples * alpha / 2)]
    hi = sample_means[int(n_resamples * (1 - alpha / 2))]
    result = {'acc': sum(hits_a) / n, 'ci95': [round(lo, 4), round(hi, 4)]}
    if diffs is not None:
        sample_diffs.sort()
        d_lo = sample_diffs[int(n_resamples * alpha / 2)]
        d_hi = sample_diffs[int(n_resamples * (1 - alpha / 2))]
        p = min(
            (sum(1 for d in sample_diffs if d <= 0) / n_resamples) * 2,
            1.0,
        ) if abs(d_lo) > 1e-12 or abs(d_hi) > 1e-12 else 1.0
        result['diff_mean'] = sum(diffs) / n
        result['diff_ci95'] = [round(d_lo, 4), round(d_hi, 4)]
        result['p_value'] = round(p, 4)
    return result


def main():
    configs = ['vanilla_rag', 'bm25', 'hybrid_rrf', 'nwc_full',
               'a1_no_lifecycle', 'a2_no_activation', 'a3_no_lexical',
               'a4_no_confidence', 'a5_single_layer']
    stats = {}
    hits = {}
    for c in configs:
        path = os.path.join(OUT_DIR, f'{c}_qa.json')
        if not os.path.exists(path):
            print(f'[skip] {c}: no {path}')
            continue
        with open(path) as f:
            data = json.load(f)
        hits[c] = data['hits']
        stats[c] = bootstrap_ci(hits[c])
        print(f'{c:>16s}: {stats[c]["acc"]*100:5.1f}%  '
              f'CI [{stats[c]["ci95"][0]*100:.1f}, {stats[c]["ci95"][1]*100:.1f}]')

    if 'nwc_full' in hits:
        print('\npaired bootstrap vs nwc_full:')
        for c, h in hits.items():
            if c == 'nwc_full':
                continue
            stats[c]['vs_nwc_full'] = bootstrap_ci(hits['nwc_full'], h)
            r = stats[c]['vs_nwc_full']
            print(f'  {c:>16s}: diff={r["diff_mean"]*100:+.1f}pp '
                  f'CI [{r["diff_ci95"][0]*100:+.1f}, '
                  f'{r["diff_ci95"][1]*100:+.1f}] p={r["p_value"]:.3f}')

    with open(os.path.join(OUT_DIR, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nsaved → revision_results/statistics.json')


if __name__ == '__main__':
    main()
