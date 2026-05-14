"""Autobiographical Memory — the agent's subjective self-model.

Distinct from ReflectionNode (which analyses the user/world objectively).
SelfNarrative captures the agent's own experience: "I discussed X with the user,"
"I believe Y about the user," "my emotional tone during session Z."

This is the agent forming a sense of self through accumulated interaction history.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SelfNarrative:
    """A single autobiographical memory — the agent's first-person experience.

    Unlike ReflectionNode ("what I learned about the world"), this is
    "what I experienced / who I am as an agent."
    """

    id: str
    episode_summary: str = ""
    self_belief: str = ""
    emotional_tone: float = 0.0
    source_session: str = ""
    source_anchor_ids: list[str] = field(default_factory=list)
    formed_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    stability: float = 0.5
    access_count: int = 0
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, episode_summary: str = "",
               self_belief: str = "",
               emotional_tone: float = 0.0,
               source_session: str = "",
               source_anchor_ids: list[str] | None = None,
               tags: list[str] | None = None,
               ) -> SelfNarrative:
        """Create a self-narrative with auto-generated ID."""
        content = f"{episode_summary}|{self_belief}|{source_session}"
        nid = hashlib.blake2b(content.encode(), digest_size=8).hexdigest()
        return cls(
            id=nid,
            episode_summary=episode_summary,
            self_belief=self_belief,
            emotional_tone=float(max(-1.0, min(1.0, emotional_tone))),
            source_session=source_session,
            source_anchor_ids=source_anchor_ids or [],
            tags=tags or [],
        )

    @property
    def relevance(self) -> float:
        """Composite relevance: stability weighted by recency."""
        hours = (time.time() - self.last_accessed_at) / 3600
        recency = math.exp(-hours / (7 * 24))  # 7-day half-life
        return self.stability * recency

    def access(self):
        """Record an access, boosting stability slightly."""
        self.access_count += 1
        self.last_accessed_at = time.time()
        self.stability = min(1.0, self.stability + 0.02)

    def reinforce(self, new_belief: str = ""):
        """Reinforce this belief with additional evidence."""
        self.stability = min(1.0, self.stability + 0.15)
        self.last_accessed_at = time.time()
        if new_belief:
            self.self_belief = new_belief

    def weaken(self):
        """Weaken this belief when contradictory evidence appears."""
        self.stability = max(0.05, self.stability - 0.20)

    def degrade(self, half_life_days: float = 30.0) -> float:
        """Apply natural decay. Returns remaining stability."""
        hours = (time.time() - self.last_accessed_at) / 3600
        decay = math.exp(-hours * math.log(2) / (half_life_days * 24))
        self.stability = max(0.01, self.stability * decay)
        return self.stability


class AutobiographicalMemory:
    """Manages the agent's self-model — accumulated first-person experience.

    Call form_from_interaction() after each meaningful exchange to build
    the agent's autobiographical record. Use recall_self() to query
    "what do I know about myself/the user" rather than "what happened."
    """

    def __init__(self, max_narratives: int = 1000):
        self._narratives: dict[str, SelfNarrative] = {}
        self.max_narratives = max_narratives
        self._total_formed = 0

    # ── CRUD ──────────────────────────────────────────────

    def form_from_interaction(self,
                              episode_summary: str,
                              self_belief: str = "",
                              emotional_tone: float = 0.0,
                              source_session: str = "",
                              source_anchor_ids: list[str] | None = None,
                              tags: list[str] | None = None,
                              ) -> SelfNarrative:
        """Record a new self-narrative from an interaction.

        Args:
            episode_summary: "I discussed Redis timeout debugging with the user"
            self_belief: "I believe the user prefers hands-on debugging"
            emotional_tone: -1..+1, how the agent felt during this interaction
            source_session: Session identifier
            source_anchor_ids: Which anchor IDs this insight draws from
            tags: Classification tags
        """
        narrative = SelfNarrative.create(
            episode_summary=episode_summary,
            self_belief=self_belief,
            emotional_tone=emotional_tone,
            source_session=source_session,
            source_anchor_ids=source_anchor_ids,
            tags=tags,
        )

        # If a similar narrative already exists, reinforce rather than duplicate
        existing = self._find_similar(narrative)
        if existing:
            existing.reinforce(self_belief)
            return existing

        self._narratives[narrative.id] = narrative
        self._total_formed += 1
        self._enforce_limit()
        return narrative

    def get(self, narrative_id: str) -> SelfNarrative | None:
        n = self._narratives.get(narrative_id)
        if n:
            n.access()
        return n

    def forget(self, narrative_id: str) -> SelfNarrative | None:
        return self._narratives.pop(narrative_id, None)

    # ── Retrieval ─────────────────────────────────────────

    def recall_self(self, query: str = "", *,
                    top_k: int = 10,
                    min_stability: float = 0.05) -> list[SelfNarrative]:
        """Retrieve self-knowledge relevant to a query.

        If no query provided, returns the most stable/recent narratives.
        Uses simple keyword overlap scoring — for embedding-based search,
        use recall_self_embedding().
        """
        if not self._narratives:
            return []

        query_lower = query.lower() if query else ""

        scored: list[tuple[SelfNarrative, float]] = []
        for narrative in self._narratives.values():
            if narrative.stability < min_stability:
                continue
            score = narrative.relevance

            if query_lower:
                # Keyword overlap boost
                content = f"{narrative.episode_summary} {narrative.self_belief}".lower()
                query_words = set(query_lower.split())
                content_words = set(content.split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score *= 1.0 + 0.2 * overlap
                # Tag match boost
                tag_match = sum(1 for t in narrative.tags if t.lower() in query_lower)
                score *= 1.0 + 0.3 * tag_match

            scored.append((narrative, score))

        scored.sort(key=lambda x: -x[1])
        return [n for n, _ in scored[:top_k]]

    def recall_session(self, session_id: str) -> list[SelfNarrative]:
        """Get all self-narratives from a specific session."""
        return sorted(
            [n for n in self._narratives.values() if n.source_session == session_id],
            key=lambda n: -n.formed_at,
        )

    def get_beliefs(self, min_stability: float = 0.2) -> list[dict]:
        """Get all stable self-beliefs (the agent's current self-model)."""
        beliefs: list[dict] = []
        for n in self._narratives.values():
            if n.self_belief and n.stability >= min_stability:
                beliefs.append({
                    "belief": n.self_belief,
                    "stability": round(n.stability, 3),
                    "episode": n.episode_summary[:120],
                    "emotional_tone": n.emotional_tone,
                    "session": n.source_session,
                })
        beliefs.sort(key=lambda b: -b["stability"])
        return beliefs

    def get_emotional_profile(self, session_id: str = "") -> dict:
        """Get the agent's emotional profile across sessions.

        Returns average emotional tone and trend direction.
        """
        if session_id:
            narratives = [n for n in self._narratives.values()
                         if n.source_session == session_id]
        else:
            narratives = list(self._narratives.values())

        if not narratives:
            return {"avg_tone": 0.0, "trend": "neutral", "count": 0}

        tones = [n.emotional_tone for n in narratives]
        avg = sum(tones) / len(tones)

        if len(tones) >= 2:
            # Split into first half / second half for trend
            mid = len(tones) // 2
            first_half = sum(tones[:mid]) / mid
            second_half = sum(tones[mid:]) / (len(tones) - mid)
            if second_half - first_half > 0.1:
                trend = "improving"
            elif second_half - first_half < -0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "avg_tone": round(avg, 3),
            "trend": trend,
            "count": len(narratives),
        }

    # ── Contradiction handling ────────────────────────────

    def update_belief(self, narrative_id: str, new_belief: str) -> bool:
        """Update a self-belief when it changes."""
        n = self._narratives.get(narrative_id)
        if n is None:
            return False
        n.self_belief = new_belief
        n.last_accessed_at = time.time()
        return True

    def contradict_belief(self, belief_substring: str,
                          correction: str = "") -> list[str]:
        """Weaken all beliefs containing a substring. Returns affected IDs."""
        affected = []
        for n in self._narratives.values():
            if belief_substring.lower() in n.self_belief.lower():
                n.weaken()
                affected.append(n.id)
                if correction:
                    # Create a corrected narrative
                    corrected = SelfNarrative.create(
                        episode_summary=f"Corrected belief: {correction}",
                        self_belief=correction,
                        source_session=n.source_session,
                        tags=n.tags + ["corrected"],
                    )
                    self._narratives[corrected.id] = corrected
        return affected

    # ── Maintenance ───────────────────────────────────────

    def degrade_all(self, half_life_days: float = 30.0) -> int:
        """Apply decay to all narratives. Returns count of those below threshold."""
        to_remove = []
        for nid, n in self._narratives.items():
            remaining = n.degrade(half_life_days)
            if remaining < 0.02:
                to_remove.append(nid)
        for nid in to_remove:
            del self._narratives[nid]
        return len(to_remove)

    def _enforce_limit(self):
        """Remove weakest narratives when over capacity."""
        if len(self._narratives) <= self.max_narratives:
            return
        sorted_n = sorted(self._narratives.items(),
                         key=lambda x: x[1].relevance)
        for nid, _ in sorted_n[:len(self._narratives) - self.max_narratives]:
            del self._narratives[nid]

    def _find_similar(self, narrative: SelfNarrative) -> SelfNarrative | None:
        """Check if a very similar narrative already exists."""
        for existing in self._narratives.values():
            # Same session + similar summary
            if existing.source_session == narrative.source_session:
                words_new = set(narrative.episode_summary.lower().split())
                words_old = set(existing.episode_summary.lower().split())
                if words_new and words_old:
                    overlap = len(words_new & words_old) / max(len(words_new), len(words_old))
                    if overlap > 0.7:
                        return existing
        return None

    @property
    def stats(self) -> dict:
        if not self._narratives:
            return {"total": 0, "avg_stability": 0.0, "avg_emotional_tone": 0.0}
        return {
            "total": len(self._narratives),
            "total_formed": self._total_formed,
            "avg_stability": round(
                sum(n.stability for n in self._narratives.values()) / len(self._narratives), 3),
            "avg_emotional_tone": round(
                sum(n.emotional_tone for n in self._narratives.values()) / len(self._narratives), 3),
            "beliefs": sum(1 for n in self._narratives.values() if n.self_belief),
        }
