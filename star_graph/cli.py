"""CLI entry points — v0.2 with schema viewing, ghost listing, stats."""

import argparse
import sys
from pathlib import Path

from .anchor import Anchor
from .storage import Storage


def add() -> None:
    parser = argparse.ArgumentParser(description="Add an anchor to the star graph")
    parser.add_argument("text", nargs="*", help="Anchor text (≤200 chars)")
    parser.add_argument("--tags", nargs="*", default=[], help="Tags")
    parser.add_argument("--importance", type=float, default=0.5)
    parser.add_argument("--emotional", type=float, default=0.0, help="Emotional valence (-1..+1)")
    parser.add_argument("--surprise", type=float, default=0.5)
    parser.add_argument("--graph", default=None, help="Graph file path")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else sys.stdin.read().strip()
    if not text:
        print("Error: no text provided", file=sys.stderr)
        sys.exit(1)

    store = Storage(args.graph)
    graph = store.load()
    anchor = Anchor.create(
        text, tags=list(args.tags),
        importance=args.importance,
        emotional_valence=args.emotional,
        surprise=args.surprise,
    )
    graph.add_anchor(anchor)
    store.save(graph)
    print(f"Added: {anchor.id}")


def query() -> None:
    parser = argparse.ArgumentParser(description="Query the star graph by resonance")
    parser.add_argument("context", nargs="*", help="Context text")
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
    constellation, action = resonator.predictive_retrieve(context)

    if constellation:
        print(f"Action: {action}  |  Anchors: {len(constellation.anchors)}  |  Weight: {constellation.total_weight:.2f}")
        for a in constellation.anchors[:10]:
            cortical = " [cortical]" if a.is_cortical else ""
            print(f"  [{a.retention_score:.2f}]{cortical} {a.text}")
    else:
        print("No resonant constellations — novel experience.")


def sleep_cmd() -> None:
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

    print(f"Sleep #{result['cycle']} complete ({result['duration_seconds']}s)")
    for entry in result["log"]:
        print(f"  {entry}")
    s = result["stats_after"]
    print(f"\nAnchors: {s['anchors']}  |  Edges: {s['edges']}  |  Ghosts: {s['ghosts']}")
    print(f"Schemas: {s['schemas']}  |  Constellations: {s['constellations']}")
    print(f"Avg retention: {s['avg_retention']:.3f}  |  Cortical: {s['avg_hippocampal_dep']:.3f}")


def stats_cmd() -> None:
    parser = argparse.ArgumentParser(description="Show graph statistics")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--schemas", action="store_true", help="List schemas")
    parser.add_argument("--ghosts", action="store_true", help="List ghosts")
    args = parser.parse_args()

    store = Storage(args.graph)
    graph = store.load()
    s = graph.stats()

    print(f"Star Graph v0.2  |  {store.path}")
    print(f"Anchors: {s['anchors']}  |  Edges: {s['edges']}  |  Ghosts: {s['ghosts']}")
    print(f"Schemas: {s['schemas']}  |  Constellations: {s['constellations']}")
    print(f"Cortical index: {s['cortical_index']}")
    print(f"Avg retention: {s['avg_retention']:.3f}  |  Avg edge weight: {s['avg_edge_weight']:.3f}")
    print(f"Avg hippocampal dep: {s['avg_hippocampal_dep']:.3f} (0=cortical, 1=hippocampal)")

    if args.schemas and graph.schemas:
        print("\n── Schemas ──")
        for schema in graph.schemas.values():
            print(f"  [{schema.confidence:.2f}] {schema.template[:80]}...")
            print(f"    instances: {len(schema.instance_ids)}")

    if args.ghosts and graph.ghosts:
        print("\n── Ghosts ──")
        for ghost in graph.ghosts.values():
            print(f"  {ghost.id[:16]}...  revivals: {ghost.revival_count}")
