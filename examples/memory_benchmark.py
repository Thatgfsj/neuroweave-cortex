"""Memory Benchmark Suite — 5-category quantitative evaluation.

Categories:
  1. Long Context Recall — find info from deep in history
  2. Cross-session Recall — recall across session boundaries
  3. Semantic Compression — compression ratio vs. recall tradeoff
  4. Forgetting Resistance — important memories survive pruning
  5. Memory Interference — handle conflicting/evolving information

Metrics:
  - TopK Hit Rate (Recall@1, @3, @5)
  - Temporal Recall (recency-weighted precision)
  - Conflict Resolution Accuracy
  - Schema Merge Accuracy
  - Compression Ratio

Baselines:
  - Raw history (full context)
  - TF-IDF vector search
  - Star Graph (oscillation resonance + sleep consolidation)

Run: python examples/memory_benchmark.py [--quick] [--full]
"""

from __future__ import annotations

import os
import sys

# Make the repository root (for ``star_graph``) and this directory (for the
# sibling ``_benchmark_*`` support modules) importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __package__:
    from ._benchmark_runner import run_benchmark
else:
    from _benchmark_runner import run_benchmark


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Star Graph Memory Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer sessions/turns")
    parser.add_argument("--full", action="store_true", help="Full mode: 12 sessions × 420 turns")
    args = parser.parse_args()

    quick = args.quick or not args.full
    run_benchmark(quick=quick)
