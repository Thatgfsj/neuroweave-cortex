"""Memory Revision Engine — improve low-quality memories during sleep (v1.0.4).

Identifies and revises low-confidence/high-surprise memories during NREM
sleep consolidation. Uses template-based merging by default, with optional
LLM-assisted re-summarization for higher quality.

Prioritization: surprise x (1 - confidence) — memories that were unexpected
but poorly integrated get revised first.

Design mirrors MemForge's Memory Revision approach (#65).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from ..anchor import Anchor, AnchorVector


@dataclass
class RevisionCandidate:
    """A memory flagged for revision during sleep."""

    anchor: Anchor
    revision_priority: float  # surprise × (1 - confidence), higher = revise first
    low_confidence: bool = True
    high_surprise: bool = False
    stale: bool = False

    @property
    def needs_revision(self) -> bool:
        return self.low_confidence or self.high_surprise or self.stale


@dataclass
class RevisionResult:
    """Output from a single memory revision run."""

    candidates_scanned: int
    revised: int           # count successfully revised
    merged_into_existing: int  # merged with a better anchor
    skipped: int           # no suitable revision found
    details: list[dict] = field(default_factory=list)


class MemoryRevisionEngine:
    """Revise low-quality memories during sleep consolidation.

    Algorithm:
      1. Scan anchors with confidence < threshold or surprise > threshold
      2. Sort by priority = surprise × (1 - confidence)
      3. For each candidate, find related anchors (same tags/topic)
      4. If a high-quality similar anchor exists → merge into it
      5. Otherwise, re-summarize: combine with similar anchors, extract key info
      6. Strengthen revised anchor (boost stability, confidence)

    Usage:
        engine = MemoryRevisionEngine()
        result = engine.revise(anchors, embedder)
        # → RevisionResult with counts and details
    """

    def __init__(self,
                 confidence_threshold: float = 0.35,
                 surprise_threshold: float = 0.7,
                 max_candidates: int = 50,
                 similarity_threshold: float = 0.75,
                 strengthen_boost: float = 0.15,
                 llm_fn: Callable | None = None):
        self.confidence_threshold = confidence_threshold
        self.surprise_threshold = surprise_threshold
        self.max_candidates = max_candidates
        self.similarity_threshold = similarity_threshold
        self.strengthen_boost = strengthen_boost
        self._llm_fn = llm_fn

    # ── Candidate discovery ───────────────────────────────────

    def find_candidates(self, anchors: list[Anchor]) -> list[RevisionCandidate]:
        """Find anchors that need revision.

        Priority = surprise × (1 - confidence). Higher priority means the
        memory was surprising but poorly integrated — prime revision target.
        """
        candidates: list[RevisionCandidate] = []
        now = time.time()

        for anchor in anchors:
            if anchor.invalid_at is not None:
                continue
            if anchor.state.name in ("GHOST", "FROZEN", "DEAD"):
                continue

            v = anchor.vector
            low_conf = v.confidence < self.confidence_threshold
            high_surp = v.surprise > self.surprise_threshold
            # Stale: old but never consolidated (low stability, old age)
            age_days = (now - anchor.created_at) / 86400
            stale = v.stability < 0.3 and age_days > 7 and anchor.replay_count < 2

            if not (low_conf or high_surp or stale):
                continue

            priority = v.surprise * (1.0 - v.confidence)
            if stale:
                priority += 0.2  # stale boost

            candidates.append(RevisionCandidate(
                anchor=anchor,
                revision_priority=priority,
                low_confidence=low_conf,
                high_surprise=high_surp,
                stale=stale,
            ))

        candidates.sort(key=lambda c: -c.revision_priority)
        return candidates[:self.max_candidates]

    # ── Similar anchor search ─────────────────────────────────

    def _find_similar(self, anchor: Anchor, all_anchors: list[Anchor]) -> list[Anchor]:
        """Find anchors that are similar to the given one.

        Uses tag overlap + embedding similarity when available.
        Returns anchors sorted by similarity (best first).
        """
        candidates: list[tuple[float, Anchor]] = []
        a_tags = set(anchor.tags)
        a_emb = anchor.embedding

        for other in all_anchors:
            if other.id == anchor.id:
                continue
            if other.invalid_at is not None:
                continue

            # Tag overlap bonus
            tag_overlap = len(a_tags & set(other.tags))
            tag_score = min(1.0, tag_overlap / max(1, len(a_tags))) if a_tags else 0.0

            # Embedding similarity
            emb_score = 0.0
            if a_emb and other.embedding:
                from ..math_utils import cosine_sim
                emb_score = max(0.0, cosine_sim(a_emb, other.embedding))

            # Combined score: 70% embedding + 30% tag
            score = 0.7 * emb_score + 0.3 * tag_score
            if score > 0.3:
                candidates.append((score, other))

        candidates.sort(key=lambda x: -x[0])
        return [a for _, a in candidates[:5]]

    # ── Template-based revision (offline, no LLM) ─────────────

    def _template_revise(self, anchor: Anchor,
                         similar: list[Anchor]) -> str | None:
        """Template-based text improvement — no LLM required.

        Strategy:
        - If a similar anchor has higher confidence → merge texts, keep the
          more specific version.
        - If stale → trim redundant content, keep core facts.
        - If low confidence → consolidate with nearest similar anchor.
        """
        # Find the best similar anchor (highest confidence, most specific)
        best_similar = None
        best_score = -1.0
        for s in similar:
            score = s.vector.confidence * 0.5 + s.vector.stability * 0.3 + len(s.text) / 500 * 0.2
            if score > best_score:
                best_score = score
                best_similar = s

        if best_similar is not None:
            # Merge: take the longer, more specific text as base
            if len(best_similar.text) >= len(anchor.text):
                return best_similar.text[:280]
            else:
                # Combine key phrases
                combined = _extract_key_sentence(anchor.text)
                if combined and combined not in best_similar.text:
                    return (best_similar.text[:200] + ". " + combined)[:280]
                return best_similar.text[:280]

        # Stale/low-confidence without similar: trim to one key sentence
        key = _extract_key_sentence(anchor.text)
        if key and len(key) < len(anchor.text):
            return key[:280]
        return None

    # ── Revision execution ────────────────────────────────────

    def revise(self, anchors: list[Anchor],
               embedder=None) -> RevisionResult:
        """Run the full memory revision cycle.

        Args:
            anchors: All anchors in the graph.
            embedder: Optional embedder for similarity searches.

        Returns:
            RevisionResult with counts and per-revision details.
        """
        candidates = self.find_candidates(anchors)
        if not candidates:
            return RevisionResult(candidates_scanned=0, revised=0,
                                  merged_into_existing=0, skipped=0)

        result = RevisionResult(
            candidates_scanned=len(candidates),
            revised=0,
            merged_into_existing=0,
            skipped=0,
        )

        for cand in candidates:
            anchor = cand.anchor
            similar = self._find_similar(anchor, anchors)

            # Check if we should merge into an existing high-quality anchor
            high_quality_similar = [
                s for s in similar
                if s.vector.confidence > self.confidence_threshold + 0.2
                and s.vector.stability > 0.5
            ]
            if high_quality_similar:
                best = high_quality_similar[0]
                # Transfer tags and boost the better anchor
                best.tags = list(set(best.tags + anchor.tags))
                best.vector.stability = min(1.0, best.vector.stability + self.strengthen_boost)
                best.vector.confidence = min(1.0, best.vector.confidence + 0.05)
                # Deprecate the low-quality anchor
                anchor.invalid_at = time.time()
                result.merged_into_existing += 1
                result.details.append({
                    "anchor_id": anchor.id[:8],
                    "action": "merged",
                    "into": best.id[:8],
                    "priority": round(cand.revision_priority, 3),
                })
                continue

            # Try template-based revision
            new_text = None
            if self._llm_fn is not None:
                try:
                    new_text = self._llm_fn(anchor, similar)
                except Exception:
                    pass

            if new_text is None:
                new_text = self._template_revise(anchor, similar)

            if new_text is not None and new_text != anchor.text:
                anchor.text = new_text[:280]
                anchor.vector.stability = min(1.0, anchor.vector.stability + self.strengthen_boost)
                anchor.vector.confidence = min(1.0, anchor.vector.confidence + 0.1)
                anchor.conflict_candidate = False
                result.revised += 1
                result.details.append({
                    "anchor_id": anchor.id[:8],
                    "action": "revised",
                    "priority": round(cand.revision_priority, 3),
                })
            else:
                result.skipped += 1

        return result


def _extract_key_sentence(text: str) -> str:
    """Extract the most contentful sentence from text."""
    sentences = re.split(r'[.。!！?？\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return text[:200]
    # Return the longest sentence (usually most informative)
    return max(sentences, key=len)[:280]
