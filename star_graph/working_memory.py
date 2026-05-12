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

    def __init__(self, max_capacity: int = 15, ttl_seconds: float = 3600.0):
        self.max_capacity = max_capacity
        self.ttl_seconds = ttl_seconds
        self._entries: list[WorkingMemoryEntry] = []
        self._exact_index: dict[str, list[WorkingMemoryEntry]] = {}  # O(1) exact key lookup

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
            evicted = self._entries.pop(0)
            self._unindex_entry(evicted)

        self._entries.append(entry)
        self._index_entry(entry)
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

    def get_exact(self, key: str) -> list[WorkingMemoryEntry]:
        """O(1) exact match lookup by entity key (e.g. 'alice-birthday').

        Returns entries matching the exact key, sorted by priority.
        Returns empty list if no match.
        """
        entries = self._exact_index.get(key, [])
        if entries:
            for e in entries:
                e.touch()
            entries.sort(key=lambda e: e.priority, reverse=True)
        return entries

    def clear(self):
        """Clear all working memory entries."""
        self._entries.clear()
        self._exact_index.clear()

    def clear_session(self, session_id: str):
        """Clear entries from a specific session."""
        self._entries = [e for e in self._entries
                         if e.source_session != session_id]

    # ── Internal ──────────────────────────────────────────

    def _index_entry(self, entry: WorkingMemoryEntry):
        """Extract exact-match keys from entry and add to O(1) lookup index."""
        keys = _extract_keys(entry.text, entry.tags)
        for key in keys:
            if key not in self._exact_index:
                self._exact_index[key] = []
            self._exact_index[key].append(entry)

    def _unindex_entry(self, entry: WorkingMemoryEntry):
        """Remove entry from exact-match index."""
        keys = _extract_keys(entry.text, entry.tags)
        for key in keys:
            bucket = self._exact_index.get(key, [])
            if entry in bucket:
                bucket.remove(entry)
            if not bucket:
                self._exact_index.pop(key, None)

    def _clear_expired(self):
        """Remove entries past their TTL."""
        now = time.time()
        expired = [e for e in self._entries
                   if now - e.created_at >= self.ttl_seconds]
        for e in expired:
            self._unindex_entry(e)
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


def _extract_keys(text: str, tags: list[str]) -> list[str]:
    """Extract exact-match keys from text for working memory indexing."""
    import re
    keys: list[str] = []

    # Entity-attribute: "Alice's birthday"
    for m in re.finditer(r"(\w+(?:\s+\w+){0,2})'s\s+(\w+)", text, re.IGNORECASE):
        e = m.group(1).strip().lower().replace(' ', '_')
        a = m.group(2).strip().lower().replace(' ', '_')
        if len(e) > 1 and len(a) > 1:
            keys.append(f"{e}-{a}")

    # KV: "X is Y", "X = Y"
    for m in re.finditer(r'(\w+)\s(?:is|set to|equals|:|=>|->|=)\s(\w+)', text, re.IGNORECASE):
        keys.append(f"{m.group(1).lower()}-{m.group(2).lower()}")

    # Tags as keys
    for tag in tags:
        tag_norm = tag.lower().replace(' ', '_')
        if len(tag_norm) > 2:
            keys.append(f"tag:{tag_norm}")

    # First two significant words as fallback
    words = [w.lower() for w in re.findall(r'[a-zA-Z]\w{2,}', text)]
    if not keys and len(words) >= 2:
        keys.append(f"{words[0]}-{words[1]}")

    return list(dict.fromkeys(keys))[:5]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
