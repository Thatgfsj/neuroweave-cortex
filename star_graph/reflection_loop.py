"""Self-Reflection Loop — automatic error detection and correction.

During sleep consolidation, the system detects contradictions between beliefs,
auto-corrects the lower-confidence one, and generates correction reports.
When a corrected belief was wrong, the original ghost can be reactivated.

Key components:
  SelfCorrectionReport — structured log of what was corrected and why
  SelfReflectionLoop   — orchestrates contradiction detection and auto-correction

Wire into sleep cycle to run after merge/compression phases.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .math_utils import cosine_sim as _cosine_sim


@dataclass
class SelfCorrectionReport:
    """A structured log entry recording a self-correction.

    Tracks what belief was corrected, the new correct version, the evidence
    that triggered the correction, and how the correction propagated through
    the memory graph.
    """

    id: str
    timestamp: float = field(default_factory=time.time)
    original_belief: str = ""
    corrected_belief: str = ""
    contradiction_type: str = "direct"  # direct, update, correction, refinement
    affected_anchor_ids: list[str] = field(default_factory=list)
    weakened_belief_ids: list[str] = field(default_factory=list)
    created_ghost_ids: list[str] = field(default_factory=list)
    revived_ghost_ids: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    resolution: str = ""  # description of how it was resolved

    @classmethod
    def create(cls, original: str, correction: str,
               contradiction_type: str = "direct",
               affected: list[str] | None = None) -> SelfCorrectionReport:
        import hashlib
        rid = hashlib.blake2b(
            (original + correction + str(time.time())).encode(), digest_size=8
        ).hexdigest()
        return cls(
            id=rid,
            original_belief=original,
            corrected_belief=correction,
            contradiction_type=contradiction_type,
            affected_anchor_ids=affected or [],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "original_belief": self.original_belief[:200],
            "corrected_belief": self.corrected_belief[:200],
            "contradiction_type": self.contradiction_type,
            "affected_anchors": len(self.affected_anchor_ids),
            "weakened_beliefs": len(self.weakened_belief_ids),
            "ghosts_created": len(self.created_ghost_ids),
            "ghosts_revived": len(self.revived_ghost_ids),
            "confidence_delta": round(self.confidence_delta, 3),
            "resolution": self.resolution[:200],
        }


class SelfReflectionLoop:
    """Auto-detects and corrects contradictions during sleep consolidation.

    Workflow:
      1. Scan anchors for contradictory pairs (high embedding similarity +
         opposing emotional valence or contradictory edge types)
      2. Determine which belief is more reliable (higher confidence + stability)
      3. Weaken the lower-confidence belief
      4. Create correction edge (invalidated_by / superseded_by)
      5. Optionally create negative ghost for the old belief
      6. If a previously-weakened belief is reinforced by new evidence,
         reactivate its ghost
      7. Generate SelfCorrectionReport

    Queryable: agent can ask "what did I get wrong about X?" via
    get_corrections_for_topic().

    Usage:
        loop = SelfReflectionLoop()
        loop.set_ghost_subsystem(ghosts)
        reports = loop.run(graph, autobiography)
    """

    def __init__(self):
        c = Config.get()
        ref_cfg = getattr(c, 'reflection_loop', None)
        self.contradiction_threshold = (
            getattr(ref_cfg, 'contradiction_threshold', 0.7) if ref_cfg else 0.7
        )
        self.min_confidence_gap = (
            getattr(ref_cfg, 'min_confidence_gap', 0.15) if ref_cfg else 0.15
        )
        self.weak_en_factor = (
            getattr(ref_cfg, 'weaken_factor', 0.30) if ref_cfg else 0.30
        )
        self.ghost_on_correction = (
            getattr(ref_cfg, 'ghost_on_correction', True) if ref_cfg else True
        )
        self.max_reports = (
            getattr(ref_cfg, 'max_reports', 100) if ref_cfg else 100
        )
        self.auto_revive_threshold = (
            getattr(ref_cfg, 'auto_revive_threshold', 0.3) if ref_cfg else 0.3
        )

        self.reports: dict[str, SelfCorrectionReport] = {}
        self._ghosts = None

    def set_ghost_subsystem(self, ghosts):
        self._ghosts = ghosts

    # ── Main execution ──────────────────────────────────────

    def run(self, graph, autobiography=None) -> list[SelfCorrectionReport]:
        """Execute the full self-reflection loop.

        Returns list of new SelfCorrectionReport entries generated this cycle.
        """
        new_reports: list[SelfCorrectionReport] = []

        # Step 1: Detect contradictions between anchors
        contradictions = self._detect_contradictions(graph)

        # Step 2: Detect contradictions between autobiographical beliefs
        if autobiography:
            belief_contradictions = self._detect_belief_contradictions(autobiography)
            contradictions.extend(belief_contradictions)

        # Step 3: Resolve each contradiction
        for contra in contradictions:
            report = self._resolve_contradiction(graph, *contra)
            if report:
                self.reports[report.id] = report
                new_reports.append(report)

        # Step 4: Check for ghost revival opportunities
        if self._ghosts:
            revived = self._check_ghost_revivals(graph)
            for report in new_reports:
                report.revived_ghost_ids.extend(revived)

        # Step 5: Enforce report limit
        if len(self.reports) > self.max_reports:
            sorted_reports = sorted(self.reports.items(),
                                   key=lambda x: x[1].timestamp)
            for rid, _ in sorted_reports[:len(self.reports) - self.max_reports]:
                del self.reports[rid]

        return new_reports

    # ── Contradiction detection ─────────────────────────────

    def _detect_contradictions(self, graph) -> list[tuple]:
        """Find anchor pairs that contradict each other.

        Returns list of (anchor_a, anchor_b, contradiction_score, reason).
        """
        contradictions = []
        anchors_list = list(graph.anchors.values())

        for i, a in enumerate(anchors_list):
            if not a.embedding or not a.is_retrievable:
                continue
            for b in anchors_list[i + 1:]:
                if not b.embedding or not b.is_retrievable:
                    continue

                # High embedding similarity + opposing emotional valence
                sim = _cosine_sim(a.embedding, b.embedding)
                if sim < self.contradiction_threshold:
                    continue

                # Check for contradiction signals
                emotion_opposing = (
                    abs(a.vector.emotional_valence - b.vector.emotional_valence) > 0.5
                    and a.vector.emotional_valence * b.vector.emotional_valence < 0
                )

                # Check for explicit contradiction edge
                has_contra_edge = False
                edge_key = graph._key(a.id, b.id)
                edge = graph.edges.get(edge_key)
                if edge and edge.edge_type in ("contradicts", "invalidates", "supersedes"):
                    has_contra_edge = True

                if emotion_opposing or has_contra_edge:
                    contradiction_score = sim
                    if has_contra_edge:
                        contradiction_score += 0.2
                    if emotion_opposing:
                        contradiction_score += 0.1

                    reason = "explicit_contradiction" if has_contra_edge else "emotional_opposition"
                    contradictions.append((a, b, contradiction_score, reason))

        contradictions.sort(key=lambda x: -x[2])
        return contradictions

    def _detect_belief_contradictions(self, autobiography) -> list[tuple]:
        """Find contradictory self-beliefs."""
        contradictions = []
        beliefs = autobiography.get_beliefs(min_stability=0.1)

        for i, b1 in enumerate(beliefs):
            b1_text = b1.get("belief", "")
            if not b1_text:
                continue
            for b2 in beliefs[i + 1:]:
                b2_text = b2.get("belief", "")
                if not b2_text:
                    continue

                # Simple keyword-based contradiction detection
                negations = {"not ", "never ", "no longer ", "don't ", "doesn't ",
                           "isn't ", "aren't ", "wasn't ", "weren't "}
                b1_neg = any(n in b1_text.lower() for n in negations)
                b2_neg = any(n in b2_text.lower() for n in negations)

                # Only one is negated = potential contradiction
                if b1_neg != b2_neg:
                    # Rough overlap check
                    b1_words = set(b1_text.lower().split())
                    b2_words = set(b2_text.lower().split())
                    overlap = len(b1_words & b2_words) / max(1, min(len(b1_words), len(b2_words)))
                    if overlap > 0.3:
                        contradictions.append((
                            b1, b2, overlap, "belief_contradiction"
                        ))

        return contradictions

    # ── Resolution ──────────────────────────────────────────

    def _resolve_contradiction(self, graph, item_a, item_b,
                                contradiction_score: float,
                                reason: str) -> SelfCorrectionReport | None:
        """Resolve a contradiction by weakening the less reliable item."""
        # Determine which item is more reliable
        # For anchors: use retention_score
        # For beliefs: use stability
        if hasattr(item_a, 'retention_score'):
            score_a = item_a.retention_score
            score_b = item_b.retention_score
            text_a = item_a.text[:200]
            text_b = item_b.text[:200]
            id_a = item_a.id
            id_b = item_b.id
        else:
            # It's a belief dict
            score_a = item_a.get("stability", 0.5)
            score_b = item_b.get("stability", 0.5)
            text_a = item_a.get("belief", "")[:200]
            text_b = item_b.get("belief", "")[:200]
            id_a = item_a.get("id", "")
            id_b = item_b.get("id", "")

        # Need significant confidence gap to auto-correct
        if abs(score_a - score_b) < self.min_confidence_gap:
            return None

        if score_a >= score_b:
            winner, loser = item_a, item_b
            winner_id, loser_id = id_a, id_b
            winner_text, loser_text = text_a, text_b
        else:
            winner, loser = item_b, item_a
            winner_id, loser_id = id_b, id_a
            winner_text, loser_text = text_b, text_a

        report = SelfCorrectionReport.create(
            original=loser_text,
            correction=winner_text,
            contradiction_type="update" if reason != "explicit_contradiction" else "correction",
            affected=[loser_id, winner_id],
        )
        report.weakened_belief_ids = [loser_id]
        report.confidence_delta = abs(score_a - score_b)

        # Weaken the loser
        if hasattr(loser, 'vector'):
            loser.vector.stability *= (1.0 - self.weak_en_factor)
            loser.vector.confidence *= (1.0 - self.weak_en_factor * 0.5)
            report.resolution = (
                f"Weakened anchor '{loser_text[:80]}' (stability ×{1.0 - self.weak_en_factor:.2f}) "
                f"in favor of '{winner_text[:80]}' "
                f"(higher confidence: {score_a:.2f} vs {score_b:.2f})"
            )
        elif isinstance(loser, dict):
            loser["stability"] = loser.get("stability", 0.5) * (1.0 - self.weak_en_factor)
            report.resolution = (
                f"Weakened belief '{loser_text[:80]}' in favor of '{winner_text[:80]}'"
            )

        # Add invalidation edge in graph
        if loser_id in graph.anchors and winner_id in graph.anchors:
            graph.add_edge(
                winner_id, loser_id,
                weight=contradiction_score * 0.7,
                edge_type="invalidated_by",
                confidence=min(1.0, contradiction_score),
            )

        # Create negative ghost for the old (weakened) belief
        if self._ghosts and self.ghost_on_correction:
            ghost = self._ghosts.create_negative(
                original_text=loser_text,
                contradiction_text=winner_text,
                target_anchor_id=loser_id,
                contradiction_type=report.contradiction_type,
            )
            report.created_ghost_ids.append(ghost.id)

        return report

    # ── Ghost revival ───────────────────────────────────────

    def _check_ghost_revivals(self, graph) -> list[str]:
        """Check if any previously-weakened beliefs should be revived.

        If a ghost's original anchor was weakened but new evidence supports it,
        reactivate the ghost back to a full anchor.
        """
        if self._ghosts is None:
            return []

        revived_ids = []
        for ghost_id, ghost in list(self._ghosts.ghosts.items()):
            if ghost.revival_count > 0:
                continue
            if ghost.reactivation_probability < self.auto_revive_threshold:
                continue

            # Check if there's new evidence that supports this ghost
            # Look for anchors with high similarity to the ghost's compressed embedding
            supporting_evidence = 0.0
            for aid, anchor in graph.anchors.items():
                if not anchor.embedding or not anchor.is_retrievable:
                    continue
                resonance = ghost.resonance(anchor.embedding)
                if resonance > 0.6:
                    supporting_evidence += resonance * anchor.retention_score

            if supporting_evidence > 0.8:
                # Revive the ghost
                new_text = f"[revived] {ghost.semantic_shadow}"
                revived_anchor = self._ghosts.try_revive(
                    ghost_id,
                    new_text=new_text,
                    new_embedding=list(ghost.compressed_embedding),
                    new_tags=list(ghost.original_tags),
                )
                if revived_anchor:
                    graph.add_anchor(revived_anchor)
                    revived_ids.append(ghost_id)

        return revived_ids

    # ── Query API ───────────────────────────────────────────

    def get_corrections_for_topic(self, topic: str) -> list[dict]:
        """Query: 'what did I get wrong about X?'

        Returns correction reports whose original or corrected belief
        mentions the given topic.
        """
        topic_lower = topic.lower()
        matches = []
        for report in self.reports.values():
            if (topic_lower in report.original_belief.lower() or
                    topic_lower in report.corrected_belief.lower()):
                matches.append(report.to_dict())
        return matches

    def get_recent_corrections(self, count: int = 10) -> list[dict]:
        """Get the most recent correction reports."""
        sorted_reports = sorted(self.reports.values(),
                               key=lambda r: -r.timestamp)
        return [r.to_dict() for r in sorted_reports[:count]]

    def get_correction_stats(self) -> dict:
        """Get statistics about the correction history."""
        if not self.reports:
            return {"total_corrections": 0}

        types = defaultdict(int)
        total_delta = 0.0
        total_weakened = 0
        total_ghosts = 0
        total_revived = 0

        for report in self.reports.values():
            types[report.contradiction_type] += 1
            total_delta += report.confidence_delta
            total_weakened += len(report.weakened_belief_ids)
            total_ghosts += len(report.created_ghost_ids)
            total_revived += len(report.revived_ghost_ids)

        return {
            "total_corrections": len(self.reports),
            "by_type": dict(types),
            "avg_confidence_delta": round(total_delta / len(self.reports), 3),
            "total_weakened": total_weakened,
            "total_ghosts_created": total_ghosts,
            "total_ghosts_revived": total_revived,
        }

    @property
    def stats(self) -> dict:
        return self.get_correction_stats()
