"""Working Memory — short-term, capacity-limited, auto-clearing buffer.

Biological analogy: prefrontal cortex working memory (~7±2 chunks).
- Fixed capacity with priority-based eviction
- Items auto-expire after TTL
- Non-persistent (not saved to disk)
- High-plasticity — items can be promoted to long-term memory
- Integrated with scheduler for retrieval priority
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkingMemoryEntry:
    """A single chunk in working memory — lightweight, ephemeral."""
    text: str
    embedding: list[float] | None = None
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    source_session: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    emotional_valence: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_accessed_at

    @property
    def priority(self) -> float:
        """Priority for eviction — lower = more evictable.

        Combines importance (stable) with recency (decaying).
        Frequently accessed items resist eviction.
        """
        recency = math.exp(-self.idle_seconds / 600)  # decay over 10 min
        frequency_bonus = math.log(1 + self.access_count) * 0.15
        return self.importance * 0.6 + recency * 0.3 + frequency_bonus

    def touch(self):
        self.last_accessed_at = time.time()
        self.access_count += 1


class WorkingMemory:
    """Short-term memory buffer with limited capacity and TTL.

    Capacity follows Miller's Law (~7±2 chunks).
    Items expire after a configurable TTL.
    On overflow, lowest-priority items are evicted.
    """

    def __init__(self, max_capacity: int = 9, ttl_seconds: float = 1800.0):
        self.max_capacity = max_capacity
        self.ttl_seconds = ttl_seconds
        self._entries: list[WorkingMemoryEntry] = []

    # ── Core operations ───────────────────────────────────

    def add(self, text: str, *,
            embedding: list[float] | None = None,
            importance: float = 0.5,
            tags: list[str] | None = None,
            source_session: str = "",
            emotional_valence: float = 0.0) -> WorkingMemoryEntry:
        """Add an item to working memory. Evicts lowest-priority if full."""
        self._clear_expired()

        entry = WorkingMemoryEntry(
            text=text,
            embedding=embedding,
            importance=importance,
            tags=tags or [],
            source_session=source_session,
            emotional_valence=emotional_valence,
        )

        if len(self._entries) >= self.max_capacity:
            # Evict lowest-priority entry
            self._entries.sort(key=lambda e: e.priority)
            self._entries.pop(0)

        self._entries.append(entry)
        return entry

    def get_all(self, max_items: int | None = None) -> list[WorkingMemoryEntry]:
        """Get all active entries, most recent first."""
        self._clear_expired()
        entries = sorted(self._entries, key=lambda e: e.last_accessed_at, reverse=True)
        if max_items:
            entries = entries[:max_items]
        for e in entries:
            e.touch()
        return entries

    def get_relevant(self, query_embedding: list[float] | None = None,
                     query_text: str = "",
                     min_score: float = 0.0,
                     max_items: int | None = None) -> list[tuple[WorkingMemoryEntry, float]]:
        """Get working memory entries relevant to a query."""
        self._clear_expired()

        if not self._entries:
            return []

        scored: list[tuple[WorkingMemoryEntry, float]] = []

        for entry in self._entries:
            score = 0.5  # base score — working memory gets retrieval priority

            if query_embedding and entry.embedding:
                sem_score = _cosine_sim(query_embedding, entry.embedding)
                score = 0.7 * sem_score + 0.3 * entry.priority
            elif query_text:
                # Simple keyword overlap
                q_words = set(query_text.lower().split())
                e_words = set(entry.text.lower().split())
                overlap = len(q_words & e_words) / max(1, len(q_words))
                score = 0.6 * overlap + 0.4 * entry.priority

            if score >= min_score:
                scored.append((entry, score))
                entry.touch()

        scored.sort(key=lambda x: -x[1])
        if max_items:
            scored = scored[:max_items]
        return scored

    def promote(self, entry: WorkingMemoryEntry, manager) -> str:
        """Promote a working memory entry to a full long-term Anchor.

        Returns the new anchor ID.
        """
        anchor = manager.remember(
            text=entry.text,
            source_session=entry.source_session,
            tags=entry.tags,
            emotional_valence=entry.emotional_valence,
            importance=max(0.5, entry.importance),
        )
        self._entries = [e for e in self._entries if e is not entry]
        return anchor.id

    def clear(self):
        """Clear all working memory entries."""
        self._entries.clear()

    def clear_session(self, session_id: str):
        """Clear entries from a specific session."""
        self._entries = [e for e in self._entries
                         if e.source_session != session_id]

    # ── Internal ──────────────────────────────────────────

    def _clear_expired(self):
        """Remove entries past their TTL."""
        now = time.time()
        self._entries = [e for e in self._entries
                         if now - e.created_at < self.ttl_seconds]

    # ── Properties ────────────────────────────────────────

    @property
    def size(self) -> int:
        self._clear_expired()
        return len(self._entries)

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_capacity

    @property
    def summary(self) -> str:
        """One-line summary of working memory contents."""
        self._clear_expired()
        if not self._entries:
            return "Working memory: empty"
        items = [e.text[:60] for e in self._entries[-5:]]
        return f"Working memory ({len(self._entries)}/{self.max_capacity}): " + " | ".join(items)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
