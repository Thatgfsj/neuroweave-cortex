"""Memory Evolution Engine — from static storage to dynamic cognitive memory.

Core mechanisms:
  1. Time decay:     w_t = w_0 * e^(-λt)  — older memories fade
  2. Frequency boost: importance += α * (1 - importance)  — repeated recall strengthens
  3. Conflict resolution: belief state transitions, not raw overwrites
  4. Interference:   proactive (old→new) + retroactive (new→old) interference
  5. Importance scoring: I = α·F + β·N + γ·G + δ·E  (frequency, novelty, goal, emotion)
  6. Anchor evolution: episodic anchors → semantic knowledge over time

This is the engine that makes star-graph a COGNITIVE memory system, not just a graph DB.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, MemoryState
from .graph import StarGraph, Edge
from .config import Config


# ── Evolution event types ───────────────────────────────

@dataclass
class EvolutionEvent:
    """A single evolution step applied to the memory graph."""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""        # decay, boost, conflict, interference, evolve
    anchor_id: str = ""
    description: str = ""
    old_value: float = 0.0
    new_value: float = 0.0


@dataclass
class BeliefTransition:
    """Tracks how a belief changes over time — not just overwrite, but evolve."""
    anchor_id: str
    topic: str                  # what this belief is about
    old_belief: str             # previous belief text
    new_belief: str             # new belief text
    old_anchor_id: str          # anchor containing old belief
    new_anchor_id: str          # anchor containing new belief
    transition_time: float = field(default_factory=time.time)
    confidence_shift: float = 0.0  # how much confidence changed


# ── Engine ──────────────────────────────────────────────

class MemoryEvolutionEngine:
    """Applies cognitive memory dynamics: decay, boost, conflict, interference.

    This is NOT a one-shot operation. It runs periodically (or on-demand)
    to evolve the memory graph toward a more efficient, consolidated state.

    Usage:
        engine = MemoryEvolutionEngine(graph)
        report = engine.evolve()  # run one evolution cycle
        print(engine.evolution_summary())
    """

    def __init__(self, graph: StarGraph, config: Config | None = None):
        self.graph = graph
        self.cfg = config or Config.get()
        self.events: list[EvolutionEvent] = []
        self.belief_transitions: list[BeliefTransition] = []
        self._access_counts: dict[str, int] = defaultdict(int)
        self._last_evolve_time: float = time.time()
        self._cycle_count: int = 0

    # ── Main evolution cycle ─────────────────────────────

    def evolve(self, current_time: float | None = None) -> dict:
        """Run one full evolution cycle. Returns summary stats."""
        if current_time is None:
            current_time = time.time()

        self._cycle_count += 1
        elapsed = current_time - self._last_evolve_time
        self._last_evolve_time = current_time
        self.events = []

        # 1. Apply time decay
        decay_stats = self._apply_time_decay(elapsed)

        # 2. Apply frequency-based importance boost
        boost_stats = self._apply_frequency_boost()

        # 3. Detect and resolve conflicts
        conflict_stats = self._resolve_conflicts()

        # 4. Apply interference (proactive + retroactive)
        interference_stats = self._apply_interference()

        # 5. Evolve anchors (merge highly similar, long-stable anchors)
        evolution_stats = self._evolve_anchors()

        # 6. Thermal degradation (HOT→WARM→COLD→DEAD)
        thermal_stats = self._apply_thermal_degradation()

        return {
            "cycle": self._cycle_count,
            "elapsed_hours": elapsed / 3600,
            "decay": decay_stats,
            "boost": boost_stats,
            "conflicts": conflict_stats,
            "interference": interference_stats,
            "evolution": evolution_stats,
            "thermal": thermal_stats,
            "total_events": len(self.events),
        }

    # ── 1. Time Decay ────────────────────────────────────

    def _apply_time_decay(self, elapsed_seconds: float) -> dict:
        """w_t = w_0 * e^(-λt) — memories weaken over time without access."""
        if elapsed_seconds <= 0:
            return {"decayed": 0}

        c = self.cfg.anchor.decay
        half_life_seconds = c.base_half_life_hours * 3600
        lam = math.log(2) / half_life_seconds  # decay constant

        decayed = 0
        for anchor in self.graph.anchors.values():
            # Time since last activation
            hours_since_access = (time.time() - anchor.last_activated_at) / 3600

            # State-dependent decay rate
            if anchor.state == MemoryState.DORMANT:
                effective_lam = lam / c.dormant_half_life_factor
            elif anchor.state == MemoryState.GHOST:
                effective_lam = lam / c.ghost_half_life_factor
            elif anchor.state == MemoryState.REACTIVATED:
                effective_lam = lam / c.reactivated_half_life_factor
            else:
                effective_lam = lam

            old_recency = anchor.vector.recency
            anchor.decay(hours_since_access, c.base_half_life_hours)
            new_recency = anchor.vector.recency

            if abs(old_recency - new_recency) > 0.001:
                decayed += 1
                self.events.append(EvolutionEvent(
                    event_type="decay",
                    anchor_id=anchor.id,
                    description=f"recency: {old_recency:.3f}→{new_recency:.3f}",
                    old_value=old_recency,
                    new_value=new_recency,
                ))

            # Also decay edge weights for dormant edges
            for key, edge in list(self.graph.edges.items()):
                if edge.source == anchor.id or edge.target == anchor.id:
                    hours_since_edge = (time.time() - edge.last_activated_at) / 3600
                    if hours_since_edge > c.base_half_life_hours * 2:
                        old_w = edge.weight
                        if hasattr(edge, 'apply_decay'):
                            edge.apply_decay(hours_since_edge)
                        else:
                            edge.weaken(self.cfg.sleep.hebbian.decay_log_factor * 0.001)
                        if abs(old_w - edge.weight) > 0.001:
                            self.events.append(EvolutionEvent(
                                event_type="edge_decay",
                                anchor_id=f"{edge.source}<->{edge.target}",
                                description=f"edge weight: {old_w:.3f}->{edge.weight:.3f}",
                                old_value=old_w,
                                new_value=edge.weight,
                            ))

        return {"decayed": decayed}

    # ── 2. Frequency Enhancement ─────────────────────────

    def _apply_frequency_boost(self) -> dict:
        """importance += α * (1 - importance) — repeated recall strengthens memory."""
        boosted = 0
        for anchor in self.graph.anchors.values():
            if anchor.replay_count > 0:
                old_imp = anchor.vector.importance
                # Diminishing returns: each replay adds less
                boost_per_replay = 0.03
                for _ in range(min(anchor.replay_count, 10)):
                    anchor.vector.importance = min(
                        1.0,
                        anchor.vector.importance + boost_per_replay * (1.0 - anchor.vector.importance)
                    )
                new_imp = anchor.vector.importance

                if abs(old_imp - new_imp) > 0.001:
                    boosted += 1
                    self.events.append(EvolutionEvent(
                        event_type="boost",
                        anchor_id=anchor.id,
                        description=f"importance (replays={anchor.replay_count}): {old_imp:.3f}→{new_imp:.3f}",
                        old_value=old_imp,
                        new_value=new_imp,
                    ))

        return {"boosted": boosted}

    # ── 3. Conflict Resolution ───────────────────────────

    def _resolve_conflicts(self) -> dict:
        """Detect belief conflicts and create transitions (NOT overwrites).

        A conflict exists when two anchors:
        1. Share topic tags
        2. Are temporally separated (one older, one newer)
        3. Either: have opposite emotional valence, OR have different conclusions
           about the same topic (detected via embedding similarity + tag overlap)

        Old belief gets reduced confidence but is PRESERVED as history.
        New belief gets higher activation weight.
        A 'contradiction' edge links them, creating a belief timeline.
        """
        resolved = 0
        anchors_list = list(self.graph.anchors.values())

        for i, a in enumerate(anchors_list):
            for b in anchors_list[i + 1:]:
                a_tags = set(a.tags)
                b_tags = set(b.tags)
                shared_tags = a_tags & b_tags
                if not shared_tags:
                    continue

                # Must have temporal separation
                time_diff = abs(a.created_at - b.created_at)
                if time_diff < 1.0:  # created at same time, not a conflict
                    continue

                newer = a if a.created_at > b.created_at else b
                older = b if a.created_at > b.created_at else a

                # Conflict signal 1: opposite valence on shared topic
                valence_opposed = (
                    abs(a.vector.emotional_valence - b.vector.emotional_valence) > 0.8
                    and a.vector.emotional_valence * b.vector.emotional_valence < 0
                )

                # Conflict signal 2: high semantic similarity but different enough
                # to suggest a changed position (same topic, different conclusion)
                semantic_conflict = False
                if a.embedding and b.embedding:
                    sim = _cosine_sim(a.embedding, b.embedding)
                    # High similarity on same topic but not nearly identical
                    # suggests same subject, different position
                    semantic_conflict = (0.55 < sim < 0.85
                                        and len(shared_tags) >= 1
                                        and time_diff > 3600)  # at least 1 hour apart

                if valence_opposed or semantic_conflict:
                    # Don't overwrite — create a contradiction edge with confidence
                    edge_key = self.graph._key(newer.id, older.id)
                    if edge_key not in self.graph.edges:
                        conflict_confidence = 0.7 if valence_opposed else 0.55
                        self.graph.add_edge(
                            newer.id, older.id,
                            weight=0.3,
                            edge_type="contradiction",
                            confidence=conflict_confidence,
                            source_type="inferred",
                        )

                    # Reduce confidence of old belief, boost new
                    older.vector.importance = max(0.3, older.vector.importance * 0.92)
                    newer.vector.importance = min(1.0, newer.vector.importance * 1.08)

                    transition = BeliefTransition(
                        anchor_id=newer.id,
                        topic=", ".join(sorted(shared_tags)),
                        old_belief=older.text[:100],
                        new_belief=newer.text[:100],
                        old_anchor_id=older.id,
                        new_anchor_id=newer.id,
                        confidence_shift=abs(newer.vector.importance - older.vector.importance),
                    )
                    self.belief_transitions.append(transition)
                    resolved += 1

                    self.events.append(EvolutionEvent(
                        event_type="conflict",
                        anchor_id=newer.id,
                        description=f"belief transition: {transition.topic}",
                        old_value=older.vector.importance,
                        new_value=newer.vector.importance,
                    ))

        return {"resolved": resolved}

    # ── 4. Interference ──────────────────────────────────

    def _apply_interference(self) -> dict:
        """Apply proactive and retroactive interference between similar memories.

        Proactive: old memories interfere with new similar ones → reduce new plasticity
        Retroactive: new memories interfere with old similar ones → reduce old activation
        """
        stats = {"proactive": 0, "retroactive": 0}
        c = self.cfg.competition

        anchors_list = list(self.graph.anchors.values())
        for i, a in enumerate(anchors_list):
            for b in anchors_list[i + 1:]:
                if not a.embedding or not b.embedding:
                    continue
                sim = _cosine_sim(a.embedding, b.embedding)
                if sim < c.interference_threshold:
                    continue

                # These two are similar enough to interfere
                older = a if a.created_at < b.created_at else b
                newer = b if a.created_at < b.created_at else a
                time_gap_hours = (newer.created_at - older.created_at) / 3600

                # Proactive: old memory reduces stability of new (harder to consolidate)
                if time_gap_hours > 0 and time_gap_hours < 168:  # within 1 week
                    old_stability = newer.vector.stability
                    newer.vector.stability *= (1.0 - c.interference_stability_factor * sim)
                    if abs(old_stability - newer.vector.stability) > 0.001:
                        stats["proactive"] += 1

                # Retroactive: new memory reduces activation of old
                old_importance = older.vector.importance
                older.vector.importance *= (1.0 - c.interference_importance_factor * sim * 0.3)
                if abs(old_importance - older.vector.importance) > 0.001:
                    stats["retroactive"] += 1

        for k, v in stats.items():
            if v > 0:
                self.events.append(EvolutionEvent(
                    event_type="interference",
                    description=f"{k}: {v} memories affected",
                    old_value=0, new_value=float(v),
                ))

        return stats

    # ── 5. Anchor Evolution ──────────────────────────────

    def _evolve_anchors(self) -> dict:
        """Long-stable, highly similar anchors can merge into semantic knowledge.

        This is the episodic→semantic transition: multiple episodes about the same
        topic gradually form an abstracted, consolidated memory.
        """
        if len(self.graph.anchors) < 3:
            return {"merged": 0}

        merged = 0
        # Find anchors that are:
        # - Very stable (stability > 0.7)
        # - High similarity
        # - Same topic cluster
        anchors_list = list(self.graph.anchors.values())
        to_merge = []

        for i, a in enumerate(anchors_list):
            if a.vector.stability < 0.7:
                continue
            for b in anchors_list[i + 1:]:
                if b.vector.stability < 0.7:
                    continue
                if not a.embedding or not b.embedding:
                    continue

                sim = _cosine_sim(a.embedding, b.embedding)
                tag_overlap = len(set(a.tags) & set(b.tags))

                if sim > 0.85 and tag_overlap >= 1:
                    to_merge.append((a.id, b.id, sim))

        # Merge only the top few per cycle (don't collapse everything at once)
        to_merge.sort(key=lambda x: -x[2])
        merge_candidates = to_merge[:max(1, len(to_merge) // 5)]

        for aid_a, aid_b, sim in merge_candidates:
            a = self.graph.anchors.get(aid_a)
            b = self.graph.anchors.get(aid_b)
            if not a or not b:
                continue

            # Merge into the older anchor
            combined_text = _merge_texts(a.text, b.text)
            a.text = combined_text[:280]
            a.tags = list(set(a.tags + b.tags))
            a.vector.stability = max(a.vector.stability, b.vector.stability)
            a.vector.importance = max(a.vector.importance, b.vector.importance) * 0.95
            a.replay_count = max(a.replay_count, b.replay_count)

            # Remove the second anchor
            self.graph.remove_anchor(b.id)
            merged += 1

            self.events.append(EvolutionEvent(
                event_type="evolve",
                anchor_id=aid_a,
                description=f"merged {aid_b[:8]} (sim={sim:.3f}) — episodic→semantic",
                old_value=0,
                new_value=sim,
            ))

        return {"merged": merged}

    # ── Public API ───────────────────────────────────────

    def record_access(self, anchor_id: str) -> None:
        """Record that an anchor was retrieved — used for frequency tracking."""
        self._access_counts[anchor_id] += 1
        if anchor_id in self.graph.anchors:
            self.graph.anchors[anchor_id].activate()

    def compute_importance(self, anchor: Anchor,
                           goal_relevance: float = 0.5) -> float:
        """I = α·F + β·N + γ·G + δ·E

        Composite importance from frequency, novelty, goal relevance, emotion.
        """
        v = anchor.vector
        # Frequency: normalized access count
        max_access = max(self._access_counts.values()) if self._access_counts else 1
        f_score = min(1.0, self._access_counts.get(anchor.id, 0) / max(1, max_access))

        # Novelty: surprise is a proxy for novelty
        n_score = v.surprise

        # Goal relevance: passed in externally
        g_score = goal_relevance

        # Emotion: absolute valence (strong emotion = more memorable)
        e_score = abs(v.emotional_valence)

        # Weights: frequency 0.30, novelty 0.25, goal 0.25, emotion 0.20
        return 0.30 * f_score + 0.25 * n_score + 0.25 * g_score + 0.20 * e_score

    def evolution_summary(self) -> dict:
        """Human-readable summary of all evolution activity."""
        event_counts = defaultdict(int)
        for e in self.events:
            event_counts[e.event_type] += 1

        return {
            "cycle": self._cycle_count,
            "total_events": len(self.events),
            "by_type": dict(event_counts),
            "belief_transitions": len(self.belief_transitions),
            "beliefs": [
                {
                    "topic": bt.topic,
                    "old": bt.old_belief[:80],
                    "new": bt.new_belief[:80],
                    "time": bt.transition_time,
                }
                for bt in self.belief_transitions[-5:]  # last 5
            ],
        }

    def print_report(self) -> None:
        """Print a human-readable evolution report."""
        s = self.evolution_summary()
        print(f"\n  ═══ Memory Evolution Report (cycle {s['cycle']}) ═══")
        print(f"  Total events: {s['total_events']}")
        for etype, count in s['by_type'].items():
            print(f"    {etype}: {count}")
        if s['belief_transitions']:
            print(f"  Belief transitions ({s['belief_transitions']}):")
            for bt in s['beliefs']:
                print(f"    [{bt['topic']}]")
                print(f"      ← {bt['old']}")
                print(f"      → {bt['new']}")


    # ── 6. Thermal degradation ─────────────────────────────

    def _apply_thermal_degradation(self) -> dict:
        """Apply thermal lifecycle transitions based on retention scores.

        Monitors all anchors and applies storage tiering:
        - COLD anchors with retention < 0.05 → finalize to DEAD (hash only)
        - Stale COLD anchors → ensure they remain in index-only mode
        - GHOST anchors with decaying reactivation → finalize to DEAD

        This is NOT deletion. It's a graduated offload:
        HOT (memory) → WARM (disk) → COLD (index) → DEAD (audit)
        """
        from .anchor import ThermalState, MemoryState
        stats = {"hot": 0, "warm": 0, "cold": 0, "dead": 0,
                 "finalized": 0, "thawed": 0}

        for anchor in self.graph.anchors.values():
            ts = anchor.thermal_state
            stats[ts.value] = stats.get(ts.value, 0) + 1

            if ts == ThermalState.DEAD:
                # Already dead — ensure it's marked as ghost
                if anchor.state != MemoryState.GHOST:
                    anchor.state = MemoryState.GHOST
                    anchor.state_history.append((MemoryState.GHOST, time.time()))
                continue

            if ts == ThermalState.COLD:
                r = anchor.retention_score
                # Check if COLD → DEAD: retention has dropped below 0.05
                if r < 0.05 and anchor.state != MemoryState.GHOST:
                    # Offload to ghost with metadata preservation
                    anchor.state = MemoryState.GHOST
                    anchor.state_history.append((MemoryState.GHOST, time.time()))
                    anchor._ghost_reactivation_prob = max(0.01, r * 0.2)
                    stats["finalized"] += 1
                    stats["dead"] += 1
                    stats["cold"] -= 1
                elif r < 0.05 and anchor.state == MemoryState.GHOST:
                    # Ghost with near-zero retention → fully dead
                    anchor._ghost_reactivation_prob = 0.0
                    stats["dead"] += 1
                    stats["cold"] -= 1

            elif ts == ThermalState.WARM and anchor.state == MemoryState.GHOST:
                # Ghost that warmed up somehow — verify revival
                stats["thawed"] += 1

        return stats

# ── Helpers ─────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)


def _merge_texts(a: str, b: str) -> str:
    """Merge two anchor texts, keeping the more informative parts."""
    if len(a) >= len(b):
        return a
    # Prefer the longer, more detailed text
    if len(b) > len(a) * 1.5:
        return b
    # Combine key parts
    return a + " | " + b[:max(50, 280 - len(a) - 3)]
