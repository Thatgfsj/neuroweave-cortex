"""LoCoMo — Long Context Memory Evaluation.

Evaluates star-graph memory performance on long-context scenarios:
  1. Temporal distance recall — probes at 10, 50, 100, 200, 400+ turns back
  2. Multi-session retention — cross-session boundary recall
  3. Forgetting curve — recall degradation over temporal distance
  4. Memory interference — contradictory information resolution

Baselines:
  - Raw keyword search (upper bound on lexical match)
  - TF-IDF vector search (semantic but no memory dynamics)
  - Star Graph VectorSimilarity (semantic + graph structure)
  - Star Graph OscillationResonance (semantic + phase-locking)

Run: python examples/locomo_test.py [--quick] [--full]
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import (
    StarGraph, Anchor, SleepCycle, seed_everything, get_embedder,
    OscillationResonanceRetriever, VectorSimilarityRetriever,
    GhostSubsystem, reload_defaults, config,
)


# ═══════════════════════════════════════════════════════════════════
# LoCoMo Data Generator
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryProbe:
    """A fact planted at a specific position for later recall testing."""
    probe_id: str
    text: str                          # the fact text
    position: int                      # turn index where planted
    session_id: str                    # which session
    category: str                      # fact, preference, bug, event, conflict
    required_keywords: list[str] = field(default_factory=list)
    related_probe_ids: list[str] = field(default_factory=list)  # for multi-hop


class LoCoMoGenerator:
    """Generates long conversations with strategically placed memory probes.

    Probes are placed at increasing temporal distances to measure
    the forgetting curve — how recall degrades over time.
    """

    # Probes with varied temporal distances
    PROBES = [
        # ── Near probes (10-30 turns) ──
        MemoryProbe("P01", "The user's primary programming language is Python, "
                    "which they have used for 5 years.", 10, "s01", "fact",
                    ["python", "primary", "language"]),
        MemoryProbe("P02", "The project is called 'CampusNet-AutoLogin' — "
                    "an automated WiFi login tool for the campus network.", 25, "s01", "fact",
                    ["campusnet", "autologin", "wifi"]),

        # ── Mid-distance probes (50-100 turns) ──
        MemoryProbe("P03", "The database backup strategy: daily full backups to S3, "
                    "hourly WAL archiving, 30-day retention, quarterly restore drills.",
                    50, "s02", "fact",
                    ["backup", "s3", "wal", "restore"]),
        MemoryProbe("P04", "Redis connection pool exhausted under peak load. "
                    "Root cause: pool size was 10, but 200+ concurrent users need 50+.",
                    75, "s02", "bug",
                    ["redis", "connection", "pool", "exhausted"]),
        MemoryProbe("P05", "The user prefers microservices over monoliths after "
                    "experiencing scaling issues with a monolithic architecture.",
                    100, "s03", "preference",
                    ["microservices", "monolithic", "scaling"]),

        # ── Far probes (150-250 turns) ──
        MemoryProbe("P06", "Docker build cache optimization: reorder Dockerfile so "
                    "COPY requirements.txt runs before pip install. Build time dropped "
                    "from 8 minutes to 45 seconds.",
                    150, "s04", "bug",
                    ["docker", "cache", "requirements.txt", "45 seconds"]),
        MemoryProbe("P07", "Security policy: JWT authentication on all endpoints, "
                    "secrets in HashiCorp Vault, OWASP ZAP scans in CI before deploy.",
                    200, "s05", "fact",
                    ["jwt", "vault", "zap", "security"]),
        MemoryProbe("P08", "The user uses Neovim with LSP, debugging, and test "
                    "integration for Python development.",
                    250, "s06", "preference",
                    ["neovim", "lsp", "debugging"]),

        # ── Very far probes (350-500 turns) ──
        MemoryProbe("P09", "Critical race condition in CampusNet-AutoLogin: login "
                    "would fail if page loaded faster than expected. Fixed by adding "
                    "explicit wait for form element before submitting.",
                    350, "s07", "bug",
                    ["race", "condition", "campusnet", "explicit wait"]),
        MemoryProbe("P10", "The production deployment uses PyInstaller to package "
                    "as standalone Windows exe, runs on startup via scheduled task.",
                    450, "s08", "fact",
                    ["pyinstaller", "executable", "startup", "scheduled task"]),

        # ── Conflict probes (evolving beliefs) ──
        MemoryProbe("P11", "User loves ORMs like SQLAlchemy — they make database "
                    "code clean and prevent SQL injection.",
                    60, "s02", "conflict",
                    ["sqlalchemy", "clean", "injection"]),
        MemoryProbe("P12", "User now strongly prefers raw SQL over ORMs for complex "
                    "queries after debugging too many N+1 problems.",
                    260, "s06", "conflict",
                    ["raw sql", "n+1", "reversed"]),
    ]

    # Filler conversation topics
    TOPICS = ["python", "frontend", "database", "devops", "debugging", "architecture"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.probes: list[MemoryProbe] = []
        self.turns: list[dict] = []
        self._probe_positions: dict[int, MemoryProbe] = {}

    def generate(self, num_sessions: int = 8, turns_per_session: int = 100
                 ) -> tuple[list[dict], list[MemoryProbe]]:
        """Generate the full long-context dataset.

        Probe positions are scaled to fit within the total turn count.
        Max probe position maps to ~90% of total turns.
        """
        total_turns = num_sessions * turns_per_session
        self.probes = self.PROBES.copy()

        # Scale probe positions to fit within total turns
        max_probe_pos = max(p.position for p in self.PROBES)  # 450
        scale = (total_turns * 0.9) / max_probe_pos

        for p in self.probes:
            scaled_pos = int(p.position * scale)
            p.position = scaled_pos
            self._probe_positions[scaled_pos] = p

        for s in range(num_sessions):
            session_id = f"s{s+1:02d}"
            self._generate_session(session_id, turns_per_session,
                                   s * turns_per_session)

        return self.turns, self.probes

    def _generate_session(self, session_id: str, num_turns: int,
                          global_offset: int) -> None:
        session_topics = self.TOPICS.copy()
        self.rng.shuffle(session_topics)

        for i in range(num_turns):
            global_pos = global_offset + i
            is_user = i % 2 == 0

            # Check if a probe is placed at this position
            probe = self._probe_positions.get(global_pos)

            if probe and is_user:
                text = probe.text
                tags = [probe.category]
            else:
                topic = session_topics[i % len(session_topics)]
                text = self._generate_filler(is_user, topic)
                tags = [topic]

            self.turns.append({
                "id": f"turn_{global_pos:05d}",
                "session_id": session_id,
                "position": global_pos,
                "speaker": "user" if is_user else "assistant",
                "text": text,
                "tags": tags,
                "is_probe": probe is not None and is_user,
                "probe_id": probe.probe_id if probe else None,
            })

    def _generate_filler(self, is_user: bool, topic: str) -> str:
        """Generate realistic filler conversation."""
        if is_user:
            templates = [
                f"I'm working on a {topic} task — can you help with {{action}}?",
                f"How do I {{action}} in {topic}?",
                f"Getting an error in my {topic} code: {{error}}.",
                f"What's the best practice for {{action}} in {topic}?",
                f"I need to optimize my {topic} setup for {{scenario}}.",
                f"Should I use {{tool_a}} or {{tool_b}} for this {topic} project?",
                f"Can you review my {topic} approach for {{action}}?",
                f"I'm migrating from {{tool_a}} to {{tool_b}} for {topic}. Advice?",
            ]
        else:
            templates = [
                f"For {topic}, the standard approach is {{solution}}.",
                f"The issue is likely {{cause}}. Try {{fix}}.",
                f"In {topic}, you want to use {{pattern}} for {{scenario}}.",
                f"Here's how to {{action}} in {topic}: {{steps}}.",
                f"The {topic} docs recommend {{approach}} for this case.",
                f"Based on your setup, I'd suggest {{solution}} in {topic}.",
            ]

        actions = ["handle errors", "manage state", "optimize queries",
                   "set up CI", "configure caching", "add logging",
                   "handle auth", "parse input", "format output"]
        tools = ["pytest", "Docker", "Redis", "FastAPI", "SQLAlchemy",
                 "Celery", "NGINX", "Grafana"]
        scenarios = ["production", "development", "testing", "staging",
                     "multi-tenant", "high-traffic"]
        solutions = ["use connection pooling", "add retry logic", "cache results",
                     "use async/await", "add monitoring"]

        template = self.rng.choice(templates)
        return template.format(
            action=self.rng.choice(actions),
            error="TypeError: 'NoneType' object has no attribute 'get'",
            tool_a=self.rng.choice(tools), tool_b=self.rng.choice(tools),
            scenario=self.rng.choice(scenarios),
            solution=self.rng.choice(solutions),
            cause="a misconfigured connection",
            fix="update the config and restart",
            pattern="the repository pattern",
            steps="1) install dep, 2) configure, 3) test",
            approach="using dependency injection",
        )


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def content_recall(retrieved_texts: list[str], keywords: list[str]) -> float:
    """Fraction of required keywords found in retrieved texts."""
    if not keywords:
        return 1.0
    if not retrieved_texts:
        return 0.0
    combined = " ".join(retrieved_texts).lower()
    found = sum(1 for kw in keywords if kw.lower() in combined)
    return found / len(keywords)


def banner(text: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def run_locomo(quick: bool = False) -> dict:
    """Run the LoCoMo evaluation."""
    reload_defaults()
    seed_everything(42)

    num_sessions = 3 if quick else 8
    turns_per_session = 50 if quick else 100
    total_turns = num_sessions * turns_per_session

    banner("LoCoMo — Long Context Memory Evaluation")
    print(f"  Sessions: {num_sessions} × ~{turns_per_session} turns = ~{total_turns} total")
    print(f"  Memory probes: {len(LoCoMoGenerator.PROBES)} at increasing distances")

    # ── Generate data ──
    banner("Phase 1: Generating Long-Context Data")
    gen = LoCoMoGenerator(seed=42)
    turns, probes = gen.generate(num_sessions, turns_per_session)
    print(f"  Generated {len(turns)} conversation turns")
    print(f"  Probes placed at positions: {sorted(p.position for p in probes)}")

    # ── Build star graph ──
    banner("Phase 2: Building Memory Graph")
    embedder = get_embedder()
    graph = StarGraph()
    graph._ghost_subsystem = GhostSubsystem()

    probe_anchor_ids: dict[str, str] = {}  # probe_id → anchor_id

    for turn in turns:
        importance = 0.7 if turn["is_probe"] else 0.5
        anchor = Anchor.create(
            turn["text"],
            source_session=turn["session_id"],
            tags=turn["tags"],
            importance=importance,
        )
        graph.add_anchor(anchor)
        if turn["probe_id"]:
            probe_anchor_ids[turn["probe_id"]] = anchor.id

    # Build similarity edges
    ids_list = list(graph.anchors.keys())
    for i, aid_a in enumerate(ids_list):
        for aid_b in ids_list[i + 1:]:
            a, b = graph.anchors[aid_a], graph.anchors[aid_b]
            if a.embedding and b.embedding:
                sim = _cosine_sim(a.embedding, b.embedding)
                if sim > 0.6:
                    graph.add_edge(aid_a, aid_b, weight=sim)

    print(f"  Anchors: {len(graph.anchors)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Probe anchors mapped: {len(probe_anchor_ids)}")

    # ── Pre-sleep retrieval test ──
    banner("Phase 3: Pre-Sleep Retrieval (Raw Memory)")

    results = {
        "config": {"sessions": num_sessions, "turns_per_session": turns_per_session,
                    "total_turns": len(turns), "num_probes": len(probes)},
        "pre_sleep": {},
        "post_sleep": {},
        "forgetting_curve": [],
        "interference": {},
    }

    vec_ret = VectorSimilarityRetriever(graph)
    osc_ret = OscillationResonanceRetriever(graph)

    pre_results = _evaluate_all_probes(probes, vec_ret, osc_ret, graph)
    results["pre_sleep"] = pre_results

    print(f"  VectorSimilarity recall: {pre_results['vector']['avg_recall']:.3f}")
    print(f"  OscillationRes recall:   {pre_results['osc']['avg_recall']:.3f}")

    # ── Sleep consolidation ──
    banner("Phase 4: Sleep Consolidation")
    cycle = SleepCycle(graph)
    sleep_result = cycle.run()
    print(f"  {sleep_result['stats_before']['anchors']} → "
          f"{sleep_result['stats_after']['anchors']} anchors "
          f"({sleep_result['merged']} merged, {sleep_result['pruned_anchors']} pruned)")
    print(f"  Schemas: {sleep_result['schemas_formed']}, "
          f"Ghosts: {sleep_result['ghosts_created']}")

    # ── Post-sleep retrieval test ──
    banner("Phase 5: Post-Sleep Retrieval (Consolidated Memory)")

    post_results = _evaluate_all_probes(probes, vec_ret, osc_ret, graph)
    results["post_sleep"] = post_results

    print(f"  VectorSimilarity recall: {post_results['vector']['avg_recall']:.3f}")
    print(f"  OscillationRes recall:   {post_results['osc']['avg_recall']:.3f}")

    # ── Forgetting curve ──
    banner("Phase 6: Forgetting Curve Analysis")

    # Group probes by temporal distance
    distance_bins = [(0, 50, "near"), (51, 150, "mid"), (151, 300, "far"),
                     (301, 999, "very-far")]
    curve_data = []

    for min_pos, max_pos, label in distance_bins:
        bin_probes = [p for p in probes if min_pos <= p.position <= max_pos]
        if not bin_probes:
            continue
        bin_results = _evaluate_all_probes(bin_probes, vec_ret, osc_ret, graph)
        curve_data.append({
            "distance": label,
            "min_pos": min_pos,
            "max_pos": max_pos,
            "num_probes": len(bin_probes),
            "vector_recall": bin_results["vector"]["avg_recall"],
            "osc_recall": bin_results["osc"]["avg_recall"],
            "post_vector_recall": bin_results.get("vector", {}).get("avg_recall", 0),
        })
        results["forgetting_curve"] = curve_data

    # ── Interference resolution ──
    banner("Phase 7: Memory Interference Resolution")

    conflict_probes = [p for p in probes if p.category == "conflict"]
    interf_results = _evaluate_interference(conflict_probes, graph, vec_ret)
    results["interference"] = interf_results

    # ── Per-probe detail ──
    print(f"\n  Per-Probe Detail (VectorSimilarity, post-sleep):")
    print(f"  {'ID':<5} {'Pos':>5} {'Dist':>8} {'Recall':>7} {'Keywords'}")
    print(f"  {'─'*60}")
    for probe in probes:
        query = _probe_to_query(probe)
        vr = vec_ret.retrieve(query)
        vr_texts = []
        for c in vr.constellations:
            for a in c.anchors:
                vr_texts.append(a.text)
        recall = content_recall(vr_texts, probe.required_keywords)
        marker = "PASS" if recall > 0.5 else ("OK" if recall > 0 else "FAIL")
        dist_label = _distance_label(probe.position, total_turns)
        print(f"  {marker} {probe.probe_id:<3} {probe.position:>5} {dist_label:>8} "
              f"{recall:>6.2f} {', '.join(probe.required_keywords[:3])}")

    # ── Print final report ──
    _print_locomo_report(results)
    _print_forgetting_curve(curve_data)

    # Save
    results_path = Path(__file__).parent / "locomo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    return results


def _evaluate_all_probes(probes: list[MemoryProbe],
                          vec_ret: VectorSimilarityRetriever,
                          osc_ret: OscillationResonanceRetriever,
                          graph: StarGraph) -> dict:
    """Evaluate retrieval for all probes."""
    vec_recalls = []
    osc_recalls = []

    for probe in probes:
        query = _probe_to_query(probe)

        # Vector similarity
        vr = vec_ret.retrieve(query)
        vr_texts = []
        for c in vr.constellations:
            for a in c.anchors:
                vr_texts.append(a.text)
        vec_recalls.append(content_recall(vr_texts, probe.required_keywords))

        # Oscillation resonance
        ore = osc_ret.retrieve(query)
        or_texts = []
        for c in ore.constellations:
            for a in c.anchors:
                or_texts.append(a.text)
        osc_recalls.append(content_recall(or_texts, probe.required_keywords))

    return {
        "vector": {
            "avg_recall": sum(vec_recalls) / len(vec_recalls) if vec_recalls else 0,
            "per_probe": vec_recalls,
        },
        "osc": {
            "avg_recall": sum(osc_recalls) / len(osc_recalls) if osc_recalls else 0,
            "per_probe": osc_recalls,
        },
        "num_probes": len(probes),
    }


def _evaluate_interference(conflict_probes: list[MemoryProbe],
                            graph: StarGraph,
                            vec_ret: VectorSimilarityRetriever) -> dict:
    """Evaluate how well the system resolves conflicting beliefs.

    For ORM→raw SQL conflict: new belief (raw SQL) should rank higher than old.
    """
    if len(conflict_probes) < 2:
        return {"resolved": 0, "total": 0, "score": 1.0}

    # Find old vs new belief for each conflict pair
    scores = []
    for i, old_p in enumerate(conflict_probes):
        for new_p in conflict_probes[i + 1:]:
            if old_p.position >= new_p.position:
                continue  # not old→new

            # Query for the topic — new belief should rank higher
            query = f"What does the user prefer: {old_p.category}?"
            result = vec_ret.retrieve(query)

            old_found = False
            new_found = False
            old_rank = 999
            new_rank = 999

            for rank, c in enumerate(result.constellations):
                for a in c.anchors:
                    if any(kw.lower() in a.text.lower()
                           for kw in old_p.required_keywords):
                        old_found = True
                        old_rank = min(old_rank, rank)
                    if any(kw.lower() in a.text.lower()
                           for kw in new_p.required_keywords):
                        new_found = True
                        new_rank = min(new_rank, rank)

            # Score: 1.0 if new before old, 0.5 if both found same rank, 0 if old before new
            if new_found and not old_found:
                scores.append(1.0)
            elif new_found and old_found:
                scores.append(1.0 if new_rank <= old_rank else 0.0)
            elif not new_found and not old_found:
                scores.append(0.5)  # neutral
            else:
                scores.append(0.0)

    return {
        "resolved": sum(1 for s in scores if s >= 0.5),
        "total": len(scores),
        "score": sum(scores) / len(scores) if scores else 0.0,
    }


def _probe_to_query(probe: MemoryProbe) -> str:
    """Convert a probe to a natural language query."""
    query_map = {
        "fact": f"What does the user say about {probe.required_keywords[0]}?",
        "bug": "What bugs have been encountered and how were they fixed?",
        "preference": f"What are the user's preferences regarding {probe.required_keywords[0] if probe.required_keywords else 'development'}?",
        "conflict": f"What is the user's current stance on {probe.required_keywords[0] if probe.required_keywords else 'this topic'}?",
    }
    return query_map.get(probe.category, f"Tell me about {probe.required_keywords[0] if probe.required_keywords else 'this'}")


def _distance_label(position: int, total_turns: int) -> str:
    ratio = position / max(1, total_turns)
    if ratio < 0.15:
        return "near"
    elif ratio < 0.4:
        return "mid"
    elif ratio < 0.7:
        return "far"
    else:
        return "v-far"


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)


# ── Report printing ─────────────────────────────────────

def _print_locomo_report(results: dict) -> None:
    banner("LoCoMo Evaluation Report")

    pre = results["pre_sleep"]
    post = results["post_sleep"]
    interf = results["interference"]

    print(f"\n  ┌{'─'*62}┐")
    print(f"  │ {'Metric':<30} {'Pre-Sleep':>14} {'Post-Sleep':>14} │")
    print(f"  ├{'─'*62}┤")
    print(f"  │ {'VectorSimilarity Recall':<30} {pre['vector']['avg_recall']:>14.3f} "
          f"{post['vector']['avg_recall']:>14.3f} │")
    print(f"  │ {'OscillationRes Recall':<30} {pre['osc']['avg_recall']:>14.3f} "
          f"{post['osc']['avg_recall']:>14.3f} │")
    print(f"  ├{'─'*62}┤")
    print(f"  │ {'Interference Resolution':<30} {'':>14} "
          f"{interf.get('score', 0):>14.3f} │")
    print(f"  └{'─'*62}┘")

    # Per-probe detail
    print(f"\n  Per-Probe Recall (VectorSimilarity, post-sleep):")
    print(f"  {'Probe':<6} {'Position':>8} {'Distance':>9} {'Recall':>7} {'Keywords'}")
    print(f"  {'─'*60}")


def _print_forgetting_curve(curve_data: list[dict]) -> None:
    print(f"\n  ┌{'─'*55}┐")
    print(f"  │ {'Forgetting Curve (VectorSimilarity)':<53} │")
    print(f"  ├{'─'*55}┤")
    print(f"  │ {'Distance':<12} {'Probes':>7} {'Vec Recall':>11} {'Osc Recall':>11} │")
    print(f"  ├{'─'*55}┤")
    for point in curve_data:
        print(f"  │ {point['distance']:<12} {point['num_probes']:>7} "
              f"{point['vector_recall']:>11.3f} {point['osc_recall']:>11.3f} │")
    print(f"  └{'─'*55}┘")

    # Check for forgetting pattern
    if len(curve_data) >= 2:
        first = curve_data[0]["vector_recall"]
        last = curve_data[-1]["vector_recall"]
        if last < first:
            print(f"  Forgetting detected: recall drops from {first:.3f} (near) → "
                  f"{last:.3f} (far), Δ = {first - last:.3f}")
        else:
            print(f"  No significant forgetting: recall stable at ~{first:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoCoMo Long Context Memory Evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 sessions × 50 turns")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 8 sessions × 100 turns")
    args = parser.parse_args()

    quick = args.quick or not args.full
    run_locomo(quick=quick)
