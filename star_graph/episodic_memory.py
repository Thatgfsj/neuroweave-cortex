"""Episodic Memory — time + context + state event streams.

Adds temporal and contextual richness to memory. Each episode captures:
  - When it happened (timestamp, session)
  - What was happening (context snapshot)
  - The outcome (action + result)
  - Emotional arc (valence trajectory)

Episodes form time-ordered streams within sessions. Auto-summarization
compresses N episodes → 1 session summary.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpisodeNode:
    """A single episode in a memory event stream."""
    id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    summary: str = ""                          # what happened
    context_snapshot: str = ""                 # what was the context
    participants: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0            # -1.0 to 1.0
    importance: float = 0.5
    action: str = ""                           # what was done
    outcome: str = ""                          # what was the result
    anchor_ids: list[str] = field(default_factory=list)  # related graph anchors
    tags: list[str] = field(default_factory=list)
    prev_episode_id: str = ""                  # previous in stream
    next_episode_id: str = ""                  # next in stream

    @property
    def is_first_in_session(self) -> bool:
        return self.prev_episode_id == ""


@dataclass
class SessionSummary:
    """Auto-generated summary of a session's episodes."""
    session_id: str
    episode_count: int
    summary_text: str
    key_topics: list[str]
    emotional_arc: list[float]      # emotional valence over time
    time_span_seconds: float
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)


class EpisodicMemory:
    """Time-ordered episodic memory with contextual recall.

    Usage:
        em = EpisodicMemory()
        ep = em.record_episode(
            session_id="session_42",
            summary="Discussed Redis timeout bug",
            context_snapshot="Debugging auth middleware",
            emotional_valence=-0.3,
            action="Increased connection pool from 10 to 20",
            outcome="Timeout resolved",
        )
        # Later:
        episodes = em.recall_session("session_42")
        relevant = em.contextual_recall(
            query="Redis connection issues",
            time_range_hours=168,
        )
        summary = em.summarize_session("session_42")
    """

    def __init__(self, max_episodes_per_session: int = 500):
        self.max_episodes_per_session = max_episodes_per_session
        self._episodes: dict[str, EpisodeNode] = {}
        self._session_index: dict[str, list[str]] = defaultdict(list)  # session_id → [ep_ids]
        self._last_in_session: dict[str, str] = {}  # session_id → last ep_id
        self._summaries: dict[str, SessionSummary] = {}
        self._total_recorded = 0

    # ── Recording ──────────────────────────────────────────

    def record_episode(self, *,
                       session_id: str = "",
                       summary: str = "",
                       context_snapshot: str = "",
                       participants: list[str] | None = None,
                       emotional_valence: float = 0.0,
                       importance: float = 0.5,
                       action: str = "",
                       outcome: str = "",
                       anchor_ids: list[str] | None = None,
                       tags: list[str] | None = None,
                       timestamp: float | None = None) -> EpisodeNode:
        """Record a new episode in a session's event stream."""
        ep_hash = hashlib.md5(
            f"{session_id}:{summary}:{time.time()}".encode()
        ).hexdigest()[:12]
        ep_id = f"ep_{ep_hash}"

        prev_id = self._last_in_session.get(session_id, "")

        episode = EpisodeNode(
            id=ep_id,
            session_id=session_id,
            timestamp=timestamp or time.time(),
            summary=summary,
            context_snapshot=context_snapshot,
            participants=participants or [],
            emotional_valence=emotional_valence,
            importance=importance,
            action=action,
            outcome=outcome,
            anchor_ids=anchor_ids or [],
            tags=tags or [],
            prev_episode_id=prev_id,
        )

        # Link previous episode to this one
        if prev_id and prev_id in self._episodes:
            self._episodes[prev_id].next_episode_id = ep_id

        self._episodes[ep_id] = episode
        self._session_index[session_id].append(ep_id)
        self._last_in_session[session_id] = ep_id
        self._total_recorded += 1

        # Enforce max episodes per session
        session_eps = self._session_index[session_id]
        if len(session_eps) > self.max_episodes_per_session:
            oldest_id = session_eps.pop(0)
            self._episodes.pop(oldest_id, None)

        return episode

    # ── Recall ─────────────────────────────────────────────

    def recall_session(self, session_id: str) -> list[EpisodeNode]:
        """Get all episodes for a session in chronological order."""
        ep_ids = self._session_index.get(session_id, [])
        return [self._episodes[eid] for eid in ep_ids if eid in self._episodes]

    def contextual_recall(self, *,
                          query: str = "",
                          session_id: str = "",
                          time_range_hours: float = 0.0,
                          min_importance: float = 0.0,
                          tags: list[str] | None = None,
                          max_items: int = 20) -> list[EpisodeNode]:
        """Recall episodes matching contextual filters.

        Filters by: text query, session, time range, importance, tags.
        """
        candidates: list[EpisodeNode] = []

        if session_id:
            candidates = self.recall_session(session_id)
        else:
            candidates = list(self._episodes.values())

        # Time range filter
        if time_range_hours > 0:
            cutoff = time.time() - time_range_hours * 3600
            candidates = [e for e in candidates if e.timestamp >= cutoff]

        # Text query filter (simple keyword match)
        if query:
            query_lower = query.lower()
            candidates = [
                e for e in candidates
                if query_lower in e.summary.lower()
                or query_lower in e.context_snapshot.lower()
                or query_lower in e.action.lower()
                or query_lower in e.outcome.lower()
            ]

        # Importance filter
        if min_importance > 0:
            candidates = [e for e in candidates if e.importance >= min_importance]

        # Tag filter
        if tags:
            tag_set = set(t.lower() for t in tags)
            candidates = [
                e for e in candidates
                if tag_set & set(t.lower() for t in e.tags)
            ]

        candidates.sort(key=lambda e: -e.timestamp)
        return candidates[:max_items]

    def get_episode(self, episode_id: str) -> EpisodeNode | None:
        return self._episodes.get(episode_id)

    # ── Summarization ──────────────────────────────────────

    def summarize_session(self, session_id: str) -> SessionSummary | None:
        """Auto-summarize a session's episodes into a single summary."""
        episodes = self.recall_session(session_id)
        if not episodes:
            return None

        # Collect key topics from tags
        tag_counter: dict[str, int] = defaultdict(int)
        for ep in episodes:
            for tag in ep.tags:
                tag_counter[tag] += 1
        top_topics = [t for t, _ in sorted(tag_counter.items(), key=lambda x: -x[1])[:5]]

        # Emotional arc
        emotional_arc = [ep.emotional_valence for ep in episodes]

        # Time span
        time_span = episodes[-1].timestamp - episodes[0].timestamp if len(episodes) > 1 else 0

        # Build summary text
        key_actions = [ep.summary for ep in episodes if ep.importance >= 0.6]
        summary_text = "; ".join(key_actions[:10]) if key_actions else episodes[-1].summary

        avg_importance = sum(ep.importance for ep in episodes) / len(episodes)

        summary = SessionSummary(
            session_id=session_id,
            episode_count=len(episodes),
            summary_text=summary_text[:500],
            key_topics=top_topics,
            emotional_arc=emotional_arc,
            time_span_seconds=time_span,
            importance=avg_importance,
        )
        self._summaries[session_id] = summary
        return summary

    def get_session_summary(self, session_id: str) -> SessionSummary | None:
        """Get a previously generated session summary."""
        return self._summaries.get(session_id)

    def get_all_summaries(self, min_importance: float = 0.0) -> list[SessionSummary]:
        """Get all session summaries."""
        summaries = list(self._summaries.values())
        if min_importance > 0:
            summaries = [s for s in summaries if s.importance >= min_importance]
        summaries.sort(key=lambda s: -s.created_at)
        return summaries

    # ── Stats ───────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "total_episodes": len(self._episodes),
            "total_sessions": len(self._session_index),
            "total_summaries": len(self._summaries),
            "total_recorded": self._total_recorded,
            "avg_episodes_per_session": round(
                len(self._episodes) / max(1, len(self._session_index)), 1),
        }
