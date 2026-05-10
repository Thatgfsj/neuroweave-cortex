"""Emergence Demo — 5-stage cognitive lifecycle.

Shows the system progressing through:
  Stage 1: Raw inputs — user learns Python ecosystem
  Stage 2: Graph formation — anchors connect by similarity
  Stage 3: Sleep + Abstraction — "Backend Python Developer" emerges
  Stage 4: Old details ghost — episodic memories fade, gist remains
  Stage 5: Ghost reactivation — "Redis timeout" revives dormant trace

This demonstrates the full cognitive lifecycle, not just storage.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, SleepCycle, seed_everything,
    get_embedder, AbstractionEngine,
)

seed_everything(42)


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main():
    banner("Stage 1: Raw Input — User Learns Python Ecosystem")

    graph = StarGraph()
    from star_graph.ghost import GhostSubsystem
    graph._ghost_subsystem = GhostSubsystem()
    embedder = get_embedder()

    # All anchors tightly clustered around Python backend development
    memories_s1 = [
        ("User is building a Flask web application with SQLAlchemy ORM.",
         ["python", "backend", "flask"], 0.7, 0.5),
        ("User created REST API endpoints in Flask for user management.",
         ["python", "backend", "flask", "api"], 0.8, 0.4),
        ("User configured Flask app factory pattern and blueprints.",
         ["python", "backend", "flask", "architecture"], 0.6, 0.3),
    ]

    memories_s2 = [
        ("User migrated from Flask to FastAPI for async support.",
         ["python", "backend", "fastapi", "async"], 0.7, 0.5),
        ("User built async REST endpoints with FastAPI and Pydantic schemas.",
         ["python", "backend", "fastapi", "api"], 0.8, 0.4),
        ("User deployed FastAPI with uvicorn and Docker container.",
         ["python", "backend", "fastapi", "deployment"], 0.6, 0.3),
    ]

    memories_s3 = [
        ("User added Redis caching layer to the FastAPI backend.",
         ["python", "backend", "fastapi", "redis"], 0.5, 0.4),
        ("Redis connection timeout caused 500 errors in production.",
         ["python", "backend", "redis", "incident"], -0.8, 0.9),
        ("User fixed Redis timeout by increasing connection pool size.",
         ["python", "backend", "redis", "fix"], 0.6, 0.6),
        # A deliberately weak anchor — will be pruned → ghost
        ("User briefly considered using MongoDB instead of PostgreSQL.",
         ["database", "mongodb"], 0.02, 0.1),
    ]

    all_sessions = [("Week 1: Flask basics", memories_s1),
                    ("Week 2: FastAPI migration", memories_s2),
                    ("Week 3: Production issues", memories_s3)]

    weak_anchor_id = None
    for session_name, memories in all_sessions:
        print(f"\n  [{session_name}]")
        for text, tags, emotion, surprise in memories:
            anchor = Anchor.create(text, source_session=session_name,
                                   tags=tags, emotional_valence=emotion,
                                   surprise=surprise)
            graph.add_anchor(anchor)
            # Track the weak MongoDB anchor for later ghost demonstration
            if "MongoDB" in text:
                weak_anchor_id = anchor.id
                anchor.vector.recency = 0.05  # simulate age
                anchor.vector.frequency = 0.0
            print(f"    + [{emotion:+.1f}] {text[:70]}...")

    print(f"\n  Graph: {graph.stats()['anchors']} anchors, {graph.stats()['edges']} edges")

    banner("Stage 2: Graph Formation — Anchors Connect by Semantic Similarity")

    # Add edges using embedding similarity
    ids = list(graph.anchors.keys())
    for i, aid_a in enumerate(ids):
        for aid_b in ids[i + 1:]:
            a, b = graph.anchors[aid_a], graph.anchors[aid_b]
            if a.embedding and b.embedding:
                from star_graph.sleep import SleepCycle
                sim = SleepCycle._embedding_similarity(a.embedding, b.embedding)
                if sim > 0.5:
                    graph.add_edge(aid_a, aid_b, weight=sim, edge_type="topical")

    print(f"\n  Graph after edge formation: {graph.stats()['anchors']} anchors, "
          f"{graph.stats()['edges']} edges")
    print(f"  Avg edge weight: {graph.stats()['avg_edge_weight']:.2f}")

    # Show high-weight connections
    strong_edges = sorted(graph.edges.items(), key=lambda x: -x[1].weight)[:5]
    print("\n  Strongest connections:")
    for (a, b), edge in strong_edges:
        print(f"    {graph.anchors[a].text[:50]}...")
        print(f"    <-> {graph.anchors[b].text[:50]}...")
        print(f"    weight: {edge.weight:.3f} [{edge.edge_type}]")
        print()

    banner("Stage 3: Sleep Cycle — Abstraction Emerges")

    cycle = SleepCycle(graph)
    result = cycle.run(retention_threshold=0.20, similarity_threshold=0.55)

    print(f"\n  Sleep results:")
    print(f"    Merged: {result['merged']}, Pruned: {result['pruned_anchors']}")
    print(f"    Schemas: {result['schemas_formed']}")
    print(f"    Ghosts: {result['ghosts_created']}")
    print(f"    Cortical memories: {result['stats_after']['avg_hippocampal_dep']:.2f} avg hippocampal dep")

    if result['log']:
        print(f"\n  Sleep log:")
        for entry in result['log']:
            print(f"    [{entry[:80]}...]" if len(entry) > 80 else f"    [{entry}]")

    # Show abstractions
    if graph.abstracts:
        print(f"\n  Emerged abstract concepts:")
        for aid, abstract in graph.abstracts.items():
            print(f"    [{abstract.label}] confidence={abstract.confidence:.2f}")
            print(f"      {abstract.description}")
            print(f"      Sources: {len(abstract.source_anchor_ids)} anchors")
    else:
        print(f"\n  No abstractions emerged yet (need more related anchors or lower threshold)")

    banner("Stage 4: Old Details Fade — Ghosts Remain")

    # Run additional sleep cycles to consolidate and prune
    for i in range(3):
        cycle.run(retention_threshold=0.20, similarity_threshold=0.55)

    stats = graph.stats()
    ghost_stats = {}
    if hasattr(cycle, '_ghost_subsystem') and cycle.graph._ghost_subsystem:
        ghost_stats = cycle.graph._ghost_subsystem.stats if hasattr(cycle.graph._ghost_subsystem, 'stats') else {}

    print(f"\n  After multiple sleep cycles:")
    print(f"    Anchors: {stats['anchors']} (some pruned)")
    print(f"    Edges: {stats['edges']}")
    print(f"    Ghosts: {stats['ghosts']}")
    print(f"    Cortical: {stats['avg_hippocampal_dep']:.2f} avg hippocampal dep")

    # Show ghost details
    if hasattr(graph, '_ghost_subsystem') and graph._ghost_subsystem:
        gs = graph._ghost_subsystem
        if gs.ghosts:
            for gid, ghost in gs.ghosts.items():
                print(f"\n    Ghost [{gid[:12]}...]:")
                print(f"      Shadow: '{ghost.semantic_shadow}'")
                print(f"      Emotion trace: {ghost.emotion_trace:+.2f}")
                print(f"      Reactivation prob: {ghost.reactivation_probability:.2f}")
                desc, conf = ghost.partial_recall()
                print(f"      Fuzzy recall: {desc}")
        else:
            print(f"\n    No ghosts created yet (all anchors still above retention threshold)")
            if weak_anchor_id and weak_anchor_id in graph.anchors:
                a = graph.anchors[weak_anchor_id]
                print(f"    Weak anchor '{a.text[:50]}...' retention: {a.retention_score:.3f}")

    banner("Stage 5: New Input — Ghost Reactivation")

    # New input that could resonate with the MongoDB ghost
    new_text = "User is considering database options: MongoDB vs PostgreSQL for the backend."
    new_embedding = embedder.encode(new_text)
    print(f"\n  New input: '{new_text}'")

    if hasattr(graph, '_ghost_subsystem') and graph._ghost_subsystem:
        gs = graph._ghost_subsystem
        matches = gs.check_resonance(new_embedding, threshold=0.35)
        if matches:
            print(f"\n  Ghosts that resonated:")
            for ghost, score in matches:
                print(f"    [{ghost.id[:12]}...] score={score:.3f}")
                print(f"      Shadow: {ghost.semantic_shadow}")
                print(f"      Emotion trace: {ghost.emotion_trace:+.2f}")
                if score > 0.35:
                    revived = gs.try_revive(ghost.id, new_text, new_embedding,
                                           ["python", "redis", "incident"])
                    if revived:
                        graph.add_anchor(revived)
                        print(f"      REVIVED! State: {revived.state.value}")
        else:
            print(f"\n  No ghosts resonated with this input.")

        # Also try fuzzy recall
        fuzzy = gs.fuzzy_recall(new_embedding, threshold=0.5)
        if fuzzy:
            print(f"\n  Fuzzy recall (sub-threshold resonance):")
            for desc, conf in fuzzy[:3]:
                print(f"    {desc}")

    final_stats = graph.stats()
    print(f"\n  Final graph state:")
    print(f"    Anchors: {final_stats['anchors']}")
    print(f"    Ghosts: {final_stats['ghosts']}")
    print(f"    Schemas: {final_stats['schemas']}")
    print(f"    Abstracts: {len(graph.abstracts) if hasattr(graph, 'abstracts') else 0}")
    print(f"    Avg retention: {final_stats['avg_retention']:.2f}")

    banner("Demo Complete")
    print("\n  Cognitive lifecycle demonstrated:")
    print("    1. Raw episodic encoding (ACTIVE state)")
    print("    2. Semantic graph formation (embedding similarity edges)")
    print("    3. Sleep consolidation + abstraction emergence")
    print("    4. Old details → ghosts (GHOST state with fuzzy recall)")
    print("    5. Ghost reactivation (REACTIVATED state, savings effect)")
    print()


if __name__ == "__main__":
    main()
