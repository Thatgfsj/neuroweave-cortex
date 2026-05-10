"""CLI entry points for star-graph-memory."""

import argparse
import sys
from pathlib import Path

from .anchor import Anchor
from .storage import Storage


def add() -> None:
    """Add a new anchor point from command line or stdin."""
    parser = argparse.ArgumentParser(description="Add an anchor to the star graph")
    parser.add_argument("text", nargs="*", help="Anchor text (≤200 chars)")
    parser.add_argument("--tags", nargs="*", default=[], help="Tags")
    parser.add_argument("--importance", type=float, default=0.5)
    parser.add_argument("--graph", default=None, help="Graph file path")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else sys.stdin.read().strip()
    if not text:
        print("Error: no text provided", file=sys.stderr)
        sys.exit(1)

    store = Storage(args.graph)
    graph = store.load()
    anchor = Anchor.create(text, tags=list(args.tags),
                           importance=args.importance)
    graph.add_anchor(anchor)
    store.save(graph)
    print(f"Added anchor: {anchor.id}")


def query() -> None:
    """Query the star graph by resonance."""
    parser = argparse.ArgumentParser(description="Query the star graph")
    parser.add_argument("context", nargs="*", help="Context text to resonate with")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--min-activation", type=float, default=0.1)
    parser.add_argument("--graph", default=None)
    args = parser.parse_args()

    context = " ".join(args.context) if args.context else sys.stdin.read().strip()
    if not context:
        print("Error: no context provided", file=sys.stderr)
        sys.exit(1)

    from .resonance import Resonator

    store = Storage(args.graph)
    graph = store.load()
    resonator = Resonator(graph)
    constellations = resonator.resonate(context, spread_steps=args.steps,
                                        min_activation=args.min_activation)

    if not constellations:
        print("No resonant constellations found.")
        return

    for i, c in enumerate(constellations, 1):
        print(f"\n── Constellation {i} (weight={c.total_weight:.2f}) ──")
        for a in c.anchors[:10]:
            print(f"  [{a.retention_score:.2f}] {a.text}")


def sleep() -> None:
    """Run a sleep consolidation cycle."""
    parser = argparse.ArgumentParser(description="Run sleep consolidation")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--retention", type=float, default=0.15)
    parser.add_argument("--edge-prune", type=float, default=0.1)
    parser.add_argument("--merge-threshold", type=float, default=0.85)
    args = parser.parse_args()

    from .sleep import SleepCycle

    store = Storage(args.graph)
    graph = store.load()
    cycle = SleepCycle(graph)
    result = cycle.run(
        similarity_threshold=args.merge_threshold,
        retention_threshold=args.retention,
        edge_prune_threshold=args.edge_prune,
    )
    store.save(graph)

    print(f"Sleep cycle complete ({result['duration_seconds']}s)")
    for entry in result["log"]:
        print(f"  {entry}")
    print(f"\nBefore: {result['stats_before']}")
    print(f"After:  {result['stats_after']}")
