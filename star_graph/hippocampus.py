"""Hippocampus Buffer — transient cache between working memory and long-term memory.

Prevents long-term graph pollution: new input stays in the hippocampus until sleep
decides whether to promote, summarize, merge, or discard.

L1 (instant): ~30min, raw text, no vectorization, no graph edges.
L2 (short-term): ~24h, lightweight vectorized, local graph, sleep-processable.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HippocampusItem:
    """A single entry in the hippocampus buffer."""
    text: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    emotional_valence: float = 0.0
    source_session: str = ""
    embedding: list[float] | None = None
    created_at: float = 0.0
    access_count: int = 0
    last_accessed_at: float = 0.0
    decay_score: float = 1.0  # 1.0 = fresh, 0.0 = discard

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_accessed_at == 0.0:
            self.last_accessed_at = self.created_at


class HippocampusBuffer:
    """Two-level transient cache protecting long-term memory from noise.

    L1 — Instant buffer (30min): raw text FIFO, no graph participation.
    L2 — Short-term buffer (24h): lightweight vectorized, local graph for
         cross-item linking, sleep-processable.

    Usage:
        hb = HippocampusBuffer()
        hb.ingest("User says X", tags=["debug"])
        ...
        # Sleep decides what to do with L2 items
        decisions = hb.sleep_decide(graph, embedder)
    """

    def __init__(self,
                 l1_max_items: int = 50,
                 l1_ttl_minutes: float = 30.0,
                 l2_max_items: int = 200,
                 l2_ttl_hours: float = 24.0,
                 promote_threshold: int = 3):
        self.l1: OrderedDict[str, HippocampusItem] = OrderedDict()
        self.l2: OrderedDict[str, HippocampusItem] = OrderedDict()
        self.l1_max = l1_max_items
        self.l1_ttl = l1_ttl_minutes * 60
        self.l2_max = l2_max_items
        self.l2_ttl = l2_ttl_hours * 3600
        self.promote_threshold = promote_threshold
        self._id_counter = 0

    # ── Public API ─────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self.l1) + len(self.l2)

    def ingest(self, text: str, *,
               tags: list[str] | None = None,
               importance: float = 0.5,
               emotional_valence: float = 0.0,
               source_session: str = "",
               embedding: list[float] | None = None) -> str:
        """Accept new input into L1 buffer. Returns the item ID."""
        self._evict_expired()
        item_id = f"hc_{self._id_counter}"
        self._id_counter += 1

        item = HippocampusItem(
            text=text,
            tags=tags or [],
            importance=importance,
            emotional_valence=emotional_valence,
            source_session=source_session,
            embedding=embedding,
        )
        self.l1[item_id] = item

        # FIFO eviction from L1
        while len(self.l1) > self.l1_max:
            oldest = next(iter(self.l1))
            del self.l1[oldest]

        return item_id

    def promote(self, item_id: str) -> str | None:
        """Promote an item from L1 to L2. Returns new L2 ID or None."""
        if item_id not in self.l1:
            return None
        item = self.l1.pop(item_id)
        l2_id = f"hc2_{self._id_counter}"
        self._id_counter += 1
        self.l2[l2_id] = item

        while len(self.l2) > self.l2_max:
            oldest = next(iter(self.l2))
            del self.l2[oldest]

        return l2_id

    def access(self, item_id: str) -> HippocampusItem | None:
        """Record an access to an item. Returns the item or None."""
        for store in (self.l1, self.l2):
            if item_id in store:
                item = store[item_id]
                item.access_count += 1
                item.last_accessed_at = time.time()
                # Auto-promote L1 → L2 on threshold
                if store is self.l1 and item.access_count >= self.promote_threshold:
                    return self._get_item(self.promote(item_id) or item_id)
                return item
        return None

    def query_l1(self, text_substring: str = "", tags: list[str] | None = None) -> list[HippocampusItem]:
        """Simple L1 scan — text substring match or tag overlap."""
        results = []
        for item in reversed(self.l1.values()):
            if text_substring and text_substring.lower() not in item.text.lower():
                continue
            if tags and not set(tags).intersection(item.tags):
                continue
            results.append(item)
        return results

    def query_l2(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Cosine-similarity search in L2 (lightweight)."""
        scored = []
        for item_id, item in self.l2.items():
            if item.embedding:
                sim = _cosine_sim(query_embedding, item.embedding)
                if sim > 0.3:
                    scored.append((item_id, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def sleep_decide(self, graph, embedder) -> dict:
        """Run sleep-time decisions on L2 items. Returns decision stats.

        Decision outcomes per L2 item:
          - promote: create a full Anchor in the star graph
          - abstract: merge with similar items into an AbstractNode
          - discard: drop entirely (low importance, no access)
          - keep: retain in L2 for another cycle
        """
        now = time.time()
        stats = {"promoted": 0, "abstracted": 0, "discarded": 0, "kept": 0}
        to_remove = []

        for item_id, item in list(self.l2.items()):
            age_hours = (now - item.created_at) / 3600
            # Decay: items lose score over time, regain on access
            hours_since_access = (now - item.last_accessed_at) / 3600
            item.decay_score = max(0.0, 1.0 - hours_since_access / (self.l2_ttl / 3600))

            # Decision logic
            if item.access_count >= 3 and item.importance > 0.3:
                # Promote to long-term memory
                from .anchor import Anchor
                emb = item.embedding or (embedder.encode(item.text) if embedder else None)
                anchor = Anchor.create(
                    text=item.text,
                    source_session=item.source_session,
                    embedding=emb,
                    emotional_valence=item.emotional_valence,
                    importance=item.importance,
                    tags=item.tags,
                )
                graph.add_anchor(anchor)
                to_remove.append(item_id)
                stats["promoted"] += 1

            elif item.decay_score < 0.1 and item.importance < 0.3:
                # Discard
                to_remove.append(item_id)
                stats["discarded"] += 1

            elif item.access_count >= 5 and item.importance > 0.6:
                # High-value: promote
                from .anchor import Anchor
                emb = item.embedding or (embedder.encode(item.text) if embedder else None)
                anchor = Anchor.create(
                    text=item.text,
                    source_session=item.source_session,
                    embedding=emb,
                    emotional_valence=item.emotional_valence,
                    importance=min(1.0, item.importance + 0.1),
                    tags=item.tags,
                )
                graph.add_anchor(anchor)
                to_remove.append(item_id)
                stats["promoted"] += 1

            else:
                stats["kept"] += 1

        for item_id in to_remove:
            self.l2.pop(item_id, None)

        return stats

    def evict_expired(self) -> int:
        """Force evict all expired items. Returns count removed."""
        return self._evict_expired()

    @property
    def stats(self) -> dict:
        self._evict_expired()
        l1_decay_scores = [i.decay_score for i in self.l1.values()]
        l2_decay_scores = [i.decay_score for i in self.l2.values()]
        return {
            "l1_items": len(self.l1),
            "l2_items": len(self.l2),
            "total": len(self.l1) + len(self.l2),
            "l1_avg_decay": sum(l1_decay_scores) / max(1, len(l1_decay_scores)),
            "l2_avg_decay": sum(l2_decay_scores) / max(1, len(l2_decay_scores)),
            "promotable": sum(1 for i in self.l1.values() if i.access_count >= self.promote_threshold),
        }

    # ── Internal ───────────────────────────────────────────

    def _get_item(self, item_id: str) -> HippocampusItem | None:
        for store in (self.l1, self.l2):
            if item_id in store:
                return store[item_id]
        return None

    def _evict_expired(self) -> int:
        now = time.time()
        removed = 0
        for store, ttl in [(self.l1, self.l1_ttl), (self.l2, self.l2_ttl)]:
            expired = [iid for iid, item in store.items()
                       if now - item.created_at > ttl and item.access_count < 2]
            for iid in expired:
                del store[iid]
                removed += 1
        return removed


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
