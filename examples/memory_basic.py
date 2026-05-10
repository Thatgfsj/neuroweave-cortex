#!/usr/bin/env python3
"""Basic end-to-end demo: add memories, sleep, retrieve.

Run: python examples/memory_basic.py

Expected output:
- Add 10 conversational memories
- Query "Where does the user live?" → retrieves relevant memories
- Run sleep cycle → shows merge/prune stats
- Query again → shows consolidated results
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import StarGraph, Anchor
from star_graph.sleep import SleepCycle
from star_graph.online import OnlineConsolidator
from star_graph.retriever import (
    OscillationResonanceRetriever,
    VectorSimilarityRetriever,
    compare_retrievers,
)
from star_graph.storage import Storage


def main():
    print("=" * 60)
    print("Star Graph Memory — Basic Demo")
    print("=" * 60)

    # ── Step 1: Initialize ───────────────────────────
    graph = StarGraph()
    online = OnlineConsolidator(graph, interval=3)

    # ── Step 2: Add example memories ─────────────────
    memories = [
        ("User lives in Beijing, Chaoyang district, near Wangjing.",
         ["personal", "location"], 0.3),
        ("User's favorite food is Italian pasta, especially carbonara.",
         ["personal", "food"], 0.5),
        ("User works as a software engineer at a tech company.",
         ["work", "location"], 0.4),
        ("User has a cat named Snowball who is 2 years old.",
         ["personal", "pets"], 0.6),
        ("User commutes by subway, taking about 40 minutes each way.",
         ["personal", "commute"], 0.1),
        ("On weekends user likes hiking in the mountains.",
         ["personal", "hobbies"], 0.5),
        ("User is learning Japanese, practices 30 minutes daily.",
         ["personal", "learning"], 0.4),
        ("User's favorite restaurant is a ramen shop near work.",
         ["food", "location"], 0.5),
        ("User graduated with a CS degree in 2020.",
         ["education", "work"], 0.3),
        ("User is planning a trip to Tokyo for cherry blossom season.",
         ["travel", "learning"], 0.7),
    ]

    print("\n── Adding Memories ──")
    for text, tags, emotion in memories:
        anchor = Anchor.create(
            text, tags=tags,
            emotional_valence=emotion,
            importance=0.5,
        )
        graph.add_anchor(anchor)

        # Auto-connect related anchors by shared tags
        for existing_id, existing in graph.anchors.items():
            if existing_id == anchor.id:
                continue
            shared_tags = set(tags) & set(existing.tags)
            if shared_tags:
                weight = 0.4 + 0.3 * len(shared_tags)
                graph.add_edge(anchor.id, existing_id,
                               weight=min(1.0, weight),
                               edge_type="topical")

        online.record_interaction(anchor)
        print(f"  + [{emotion}] {text[:60]}...")

    print(f"\nGraph: {len(graph.anchors)} anchors, {len(graph.edges)} edges")

    # ── Step 3: Query ────────────────────────────────
    print("\n── Retrieval ──")
    queries = [
        "Where does the user live?",
        "What food does the user like?",
        "What are the user's hobbies?",
        "Tell me about the user's cat",
    ]

    osc_ret = OscillationResonanceRetriever(graph)
    vec_ret = VectorSimilarityRetriever(graph)

    for query in queries[:2]:
        print(f"\n  Query: {query}")
        print(f"  {'─' * 40}")

        result = osc_ret.retrieve(query)
        print(f"  [OscillationResonance] {result.latency_ms:.1f}ms, "
              f"score={result.top_score:.3f}")
        for c in result.constellations[:2]:
            for a in c.anchors[:3]:
                print(f"    · {a.text[:80]}")

        result2 = vec_ret.retrieve(query)
        print(f"  [VectorSimilarity]   {result2.latency_ms:.1f}ms, "
              f"score={result2.top_score:.3f}")
        for c in result2.constellations[:2]:
            for a in c.anchors[:3]:
                print(f"    · {a.text[:80]}")

    # ── Step 4: Retriever comparison ─────────────────
    print("\n── Retriever Comparison ──")
    comparisons = compare_retrievers(graph, queries)
    for comp in comparisons:
        vec = comp["vector_similarity"]
        osc = comp["oscillation_resonance"]
        print(f"  Q: {comp['query'][:40]}")
        print(f"    VectorSim:  {vec['latency_ms']:.1f}ms  score={vec['top_score']:.3f}")
        print(f"    OscReson:   {osc['latency_ms']:.1f}ms  score={osc['top_score']:.3f}")

    # ── Step 5: Sleep cycle ──────────────────────────
    print("\n── Running Sleep Cycle ──")
    cycle = SleepCycle(graph)
    before = graph.stats()
    result = cycle.run()
    after = graph.stats()

    print(f"  Before: {before['anchors']} anchors, {before['edges']} edges, "
          f"{before['ghosts']} ghosts")
    print(f"  After:  {after['anchors']} anchors, {after['edges']} edges, "
          f"{after['ghosts']} ghosts")
    print(f"  Merged: {result['merged']}, Pruned: {result['pruned_anchors']}, "
          f"Schemas: {result['schemas_formed']}")
    for entry in result['log']:
        print(f"  · {entry}")

    # ── Step 6: Re-query after sleep ─────────────────
    print("\n── After Sleep: 'Where does the user live?' ──")
    result = osc_ret.retrieve("Where does the user live?")
    print(f"  [OscillationResonance] {result.latency_ms:.1f}ms")
    for c in result.constellations[:2]:
        for a in c.anchors[:3]:
            status = "[cortical]" if a.is_cortical else "[hippocampal]"
            print(f"    {status} [{a.retention_score:.2f}] {a.text[:80]}")

    # ── Step 7: Save ─────────────────────────────────
    store = Storage()
    store.save(graph)
    print(f"\nSaved to: {store.path}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
