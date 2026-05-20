"""Conflict detection and resolution for consolidation (v1.0.4).

Identifies semantic contradictions between new and existing memories during
NREM sleep, then resolves them using one of three strategies:
  - overwrite: high-confidence new fact replaces old
  - coexist:   both views are valid (opinion divergence)
  - deprecate:  old fact marked invalid (invalid_at timestamp)

Design mirrors Zep's graph node lifecycle management and Hindsight's
Background Merging approach to contradiction handling.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..anchor import Anchor
from ..math_utils import cosine_sim as _cosine_sim


class ConflictResolution(Enum):
    OVERWRITE = "overwrite"    # new fact replaces old
    COEXIST = "coexist"        # both valid (divergent views)
    DEPRECATE = "deprecate"    # old fact scheduled for removal


@dataclass
class ConflictPair:
    """A detected semantic conflict between two anchors."""

    anchor_a: Anchor          # existing memory
    anchor_b: Anchor          # new/challenging memory
    similarity: float         # embedding cosine similarity
    sentiment_divergence: float  # abs(valence_a - valence_b) when one is negative
    resolution: ConflictResolution = ConflictResolution.COEXIST
    reason: str = ""

    @property
    def is_resolved(self) -> bool:
        return bool(self.reason)


class ConflictDetector:
    """Detect and resolve contradictions between memories during sleep.

    Algorithm:
      1. For each anchor, find its nearest semantic neighbors (cosine > threshold).
      2. Among high-similarity pairs, detect sentiment divergence:
         - One anchor has negative valence, the other positive → potential contradiction
         - Same topic but opposite emotional framing
      3. Classify resolution strategy based on confidence and stability:
         - High-confidence new (stability > 0.7, recency > old) → OVERWRITE
         - Both have moderate confidence → COEXIST (keep both, tag as conflicting views)
         - Old memory has low retention → DEPRECATE (mark invalid_at)
    """

    def __init__(self,
                 similarity_threshold: float = 0.85,
                 sentiment_threshold: float = 0.3,
                 overwrite_confidence: float = 0.7,
                 deprecate_retention: float = 0.2):
        self.similarity_threshold = similarity_threshold
        self.sentiment_threshold = sentiment_threshold
        self.overwrite_confidence = overwrite_confidence
        self.deprecate_retention = deprecate_retention

    def detect(self, anchors: list[Anchor]) -> list[ConflictPair]:
        """Find all conflict pairs among the given anchors.

        For each anchor pair with high semantic overlap, check whether
        they express contradictory sentiments (one positive, one negative
        on the same topic).
        """
        conflicts: list[ConflictPair] = []
        n = len(anchors)
        if n < 2:
            return conflicts

        for i in range(n):
            a = anchors[i]
            if not a.embedding or a.invalid_at is not None:
                continue
            for j in range(i + 1, n):
                b = anchors[j]
                if not b.embedding or b.invalid_at is not None:
                    continue

                sim = _cosine_sim(a.embedding, b.embedding)
                if sim < self.similarity_threshold:
                    continue

                # Check sentiment divergence: one positive, one negative
                va = a.vector.emotional_valence
                vb = b.vector.emotional_valence
                sentiment_div = abs(va - vb)

                # Only flag when valences point in opposite directions
                opposite_signs = (va < -self.sentiment_threshold and vb > self.sentiment_threshold) or \
                                 (vb < -self.sentiment_threshold and va > self.sentiment_threshold)
                if not opposite_signs:
                    continue

                a.conflict_candidate = True
                b.conflict_candidate = True
                conflicts.append(ConflictPair(
                    anchor_a=a,
                    anchor_b=b,
                    similarity=sim,
                    sentiment_divergence=sentiment_div,
                ))

        return conflicts

    def resolve(self, conflicts: list[ConflictPair]) -> list[ConflictPair]:
        """Apply resolution strategies to each conflict pair.

        Decision logic (weighted toward preserving knowledge):
          - New anchor (b) has high stability & higher confidence → OVERWRITE old (a)
          - Both have moderate stability → COEXIST (tag both as conflicting views)
          - Old anchor (a) has very low retention → DEPRECATE old
          - Otherwise → COEXIST (conservative default)
        """
        now = time.time()
        for cp in conflicts:
            a, b = cp.anchor_a, cp.anchor_b

            # Determine recency: newer anchor is the "challenger"
            a_recency = 1.0 - min(1.0, (now - a.created_at) / (90 * 86400))
            b_recency = 1.0 - min(1.0, (now - b.created_at) / (90 * 86400))
            b_is_newer = b.created_at > a.created_at
            newer = b if b_is_newer else a
            older = a if b_is_newer else b

            # OVERWRITE: high-confidence new fact replaces old
            if newer.vector.confidence > self.overwrite_confidence and \
               newer.vector.stability > self.overwrite_confidence and \
               older.vector.stability < self.overwrite_confidence:
                cp.resolution = ConflictResolution.OVERWRITE
                cp.reason = f"high_confidence_overwrite: {newer.id[:8]} replaces {older.id[:8]}"
                older.invalid_at = now
                older.conflict_candidate = False
                newer.conflict_candidate = False
                continue

            # DEPRECATE: old memory has poor retention
            if older.vector.importance < self.deprecate_retention or \
               older.retention_score < self.deprecate_retention:
                cp.resolution = ConflictResolution.DEPRECATE
                cp.reason = f"deprecate_low_retention: {older.id[:8]}"
                older.invalid_at = now
                older.conflict_candidate = False
                continue

            # COEXIST: both views valid, keep both
            cp.resolution = ConflictResolution.COEXIST
            cp.reason = "coexist_divergent_views"
            a.conflict_candidate = False
            b.conflict_candidate = False

        return conflicts

    def detect_and_resolve(self, anchors: list[Anchor]) -> dict:
        """Run full conflict detection + resolution cycle.

        Returns a summary dict with counts per resolution strategy.
        """
        conflicts = self.detect(anchors)
        resolved = self.resolve(conflicts)

        counts = {"overwrite": 0, "coexist": 0, "deprecate": 0, "total": len(resolved)}
        for cp in resolved:
            counts[cp.resolution.value] += 1

        return {
            "conflicts_detected": len(conflicts),
            "resolutions": dict(counts),
            "pairs": resolved,
        }
