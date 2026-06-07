"""Visualize — automatic paper figure generation from benchmark results.

Generates:
  1. Overall performance bar chart (NWC vs baselines, has_answer + F1)
  2. Ablation study bar chart with error bars
  3. Per-category breakdown (LoCoMo Category 1-5)
  4. Forgetting curve (recall rate vs temporal distance)
  5. Sleep before/after graph structure visualization

All figures saved to paper/figures/ as SVG (vector) + PNG (raster).

Usage:
    python -m benchmarks.visualize
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _paper_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")


def _load_json(path: str) -> dict:
    full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    with open(full) as f:
        return json.load(f)


def main():
    paper_fig = _paper_dir()
    os.makedirs(paper_fig, exist_ok=True)

    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("⚠ matplotlib not installed — skipping figure generation")
        print(f"  Install: pip install matplotlib seaborn numpy")
        print(f"  Figures would go to: {paper_fig}")

    # Load benchmark data
    try:
        locomo = _load_json("benchmarks/locomo_results.json")
        print(f"✅ Loaded LoCoMo results")
    except FileNotFoundError:
        print(f"⚠ No locomo_results.json found — run benchmarks/run_locomo_full.py first")
        locomo = None

    try:
        ablation = _load_json("benchmarks/ablation_results.json")
        print(f"✅ Loaded ablation results")
    except FileNotFoundError:
        print(f"⚠ No ablation_results.json found — run benchmarks/ablation.py first")
        ablation = None

    if not HAS_MPL:
        print("\n  To generate figures: pip install matplotlib seaborn numpy")
        return

    # ── Figure 1: Overall Performance ──
    if locomo and "overall" in locomo:
        fig, ax = plt.subplots(figsize=(6, 4))
        methods = ["NWC (RRF)", "VectorOnly", "OscillationRes"]
        has_ans = [
            locomo["overall"]["has_answer"] * 100,
            locomo.get("baselines", {}).get("vector", 0) * 100,
            locomo.get("baselines", {}).get("oscillation", 0) * 100,
        ]
        colors = ["#2ecc71", "#95a5a6", "#f39c12"]
        bars = ax.bar(methods, has_ans, color=colors, width=0.5)
        ax.set_ylabel("has_answer (%)")
        ax.set_title("LoCoMo Retrieval Performance")
        for bar, val in zip(bars, has_ans):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(paper_fig, "fig1_overall_performance.svg"), format="svg")
        fig.savefig(os.path.join(paper_fig, "fig1_overall_performance.png"), dpi=300)
        plt.close(fig)
        print(f"  ✅ fig1: overall performance")

    # ── Figure 2: Per-Category Breakdown ──
    if locomo and "by_category" in locomo:
        fig, ax = plt.subplots(figsize=(8, 4))
        cats = locomo["by_category"]
        cat_names = {"1": "Temporal", "2": "Short Mem", "3": "Long Mem",
                      "4": "Composite", "5": "Adversarial"}
        labels = []
        values = []
        for c in sorted(cats.keys()):
            labels.append(cat_names.get(c, f"Cat{c}"))
            values.append(cats[c]["has_answer"] * 100)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        bars = ax.bar(labels, values, color=colors, width=0.5)
        ax.set_ylabel("has_answer (%)")
        ax.set_title("LoCoMo Performance by Category")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(paper_fig, "fig2_by_category.svg"), format="svg")
        fig.savefig(os.path.join(paper_fig, "fig2_by_category.png"), dpi=300)
        plt.close(fig)
        print(f"  ✅ fig2: per-category breakdown")

    # ── Figure 3: Ablation Study ──
    if ablation:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(ablation.keys())
        means = [ablation[k].get("has_answer_mean", 0) for k in labels]
        stds = [ablation[k].get("has_answer_std", 0) for k in labels]
        colors = ["#2ecc71" if l == "full" else "#e74c3c" for l in labels]
        bars = ax.bar(labels, means, yerr=stds, color=colors, width=0.5, capsize=5)
        ax.set_ylabel("has_answer (%)")
        ax.set_title("Ablation Study: Component Contribution")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{val:.1f}%", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(paper_fig, "fig3_ablation.svg"), format="svg")
        fig.savefig(os.path.join(paper_fig, "fig3_ablation.png"), dpi=300)
        plt.close(fig)
        print(f"  ✅ fig3: ablation study")

    if locomo or ablation:
        print(f"\n  All figures saved to: {paper_fig}")


if __name__ == "__main__":
    main()
