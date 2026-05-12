"""Streaming Memory Buffer — continuous ingestion with backpressure.

v1.0-6: Enables real-time, non-blocking memory ingestion for production
workloads. Items flow through a pipeline:
  ingest → deduplicate → batch → merge → promote to graph

Supports:
- Continuous item-by-item ingestion
- Auto-batch on count and time thresholds
- Deduplication of similar items within the buffer window
- Backpressure signaling when buffer is full
- Periodic flush to the graph (non-blocking)
- Session-aware grouping

Usage:
    buffer = StreamingMemoryBuffer(manager, max_buffer=500, flush_interval_s=30)
    buffer.ingest("User mentioned Redis timeout issue", tags=["debug", "redis"])
    buffer.ingest("Fixed by increasing pool size to 20", tags=["debug", "redis"])
    # ... more items ...
    stats = buffer.flush()  # or auto-flush via background thread
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamItem:
    """A single memory item in the streaming buffer."""
    text: str
    tags: list[str] = field(default_factory=list)
    source_session: str = ""
    importance: float = 0.5
    emotional_valence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    embedding: list[float] | None = None

    def __hash__(self):
        return hash((self.text[:100], self.source_session, self.timestamp))


@dataclass
class StreamStats:
    """Statistics for the streaming buffer."""
    total_ingested: int = 0
    total_flushed: int = 0
    total_merged: int = 0       # items merged/deduped
    total_promoted: int = 0     # promoted to graph
    buffer_size: int = 0
    last_flush_at: float = 0.0
    backpressure_events: int = 0
    dropped_items: int = 0

    @property
    def merge_ratio(self) -> float:
        """Fraction of items that were merged/deduped."""
        if self.total_ingested == 0:
            return 0.0
        return self.total_merged / self.total_ingested

    def summary(self) -> str:
        return (f"ingested={self.total_ingested}, flushed={self.total_flushed}, "
                f"merged={self.total_merged}, promoted={self.total_promoted}, "
                f"buffer={self.buffer_size}, dropped={self.dropped_items}")


class StreamingMemoryBuffer:
    """Continuous memory ingestion buffer with auto-batching and backpressure.

    Design:
      ingest(item) → buffer[ ] → deduplicate → batch_ready → flush → merge → promote

    Thread-safe for concurrent ingestion. Auto-flush can be driven by:
    - Count threshold (buffer reaches N items)
    - Time threshold (N seconds since last flush)
    - Manual flush() call
    """

    def __init__(self, manager,
                 max_buffer: int = 500,
                 flush_interval_s: float = 30.0,
                 batch_size: int = 20,
                 dedup_threshold: float = 0.85,
                 max_sessions: int = 10,
                 auto_flush: bool = True):
        self._manager = manager
        self.max_buffer = max_buffer
        self.flush_interval_s = flush_interval_s
        self.batch_size = batch_size
        self.dedup_threshold = dedup_threshold
        self.max_sessions = max_sessions
        self.auto_flush = auto_flush

        # Items organized by session for better grouping
        self._buffer: dict[str, list[StreamItem]] = defaultdict(list)
        self._lock = threading.RLock()
        self._last_flush_at = time.time()
        self._embedder = None

        # Stats
        self.stats = StreamStats()

        # Background auto-flush thread
        self._flush_thread: threading.Thread | None = None
        self._running = False
        if auto_flush:
            self._start_auto_flush()

    # ── Public API ────────────────────────────────────────

    def ingest(self, text: str, *,
              tags: list[str] | None = None,
              source_session: str = "default",
              importance: float = 0.5,
              emotional_valence: float = 0.0,
              embedding: list[float] | None = None) -> bool:
        """Ingest a single memory item into the streaming buffer.

        Returns True if accepted, False if rejected (backpressure).
        """
        with self._lock:
            total = sum(len(v) for v in self._buffer.values())

            if total >= self.max_buffer:
                # Backpressure: try to flush immediately
                self.stats.backpressure_events += 1
                if total >= self.max_buffer * 1.5:
                    # Severe backpressure: drop oldest item
                    self.stats.dropped_items += 1
                    return False

                # Moderate backpressure: force a partial flush
                self._flush_internal(force=True, partial=True)

            item = StreamItem(
                text=text, tags=tags or [],
                source_session=source_session,
                importance=importance,
                emotional_valence=emotional_valence,
                embedding=embedding,
            )
            self._buffer[source_session].append(item)
            self.stats.total_ingested += 1
            self.stats.buffer_size = total + 1

        # Check if we should flush (batch size threshold)
        if total + 1 >= self.batch_size:
            self.flush()

        return True

    def ingest_batch(self, items: list[dict]) -> int:
        """Ingest multiple items at once. Returns number accepted."""
        accepted = 0
        for item in items:
            if self.ingest(**item):
                accepted += 1
        return accepted

    def flush(self, force: bool = False) -> dict:
        """Flush buffered items to the graph.

        Deduplicates within the buffer window, merges similar items,
        and promotes consolidated items as graph anchors.

        Returns a dict with flush statistics.
        """
        with self._lock:
            return self._flush_internal(force=force)

    def close(self):
        """Stop auto-flush and flush remaining items."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self.flush(force=True)

    @property
    def size(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._buffer.values())

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_buffer

    @property
    def sessions(self) -> list[str]:
        with self._lock:
            return list(self._buffer.keys())

    # ── Internal ──────────────────────────────────────────

    def _flush_internal(self, force: bool = False, partial: bool = False) -> dict:
        """Internal flush: deduplicate, merge, and promote items to graph.

        Must be called with _lock held.
        """
        now = time.time()
        if not force and (now - self._last_flush_at) < self.flush_interval_s * 0.5:
            return {"flushed": 0, "merged": 0, "promoted": 0, "reason": "too_soon"}

        total_before = sum(len(v) for v in self._buffer.values())
        if total_before == 0:
            return {"flushed": 0, "merged": 0, "promoted": 0, "reason": "empty"}

        merged_count = 0
        promoted_count = 0

        # Process each session's items
        for session_id, items in list(self._buffer.items()):
            if not items:
                continue

            # Step 1: Deduplicate by text similarity
            deduped = self._dedup_items(items)

            merged_count += len(items) - len(deduped)

            # Step 2: Group into topic clusters by tag overlap
            clusters = self._cluster_by_tags(deduped)

            # Step 3: For each cluster, create a merged anchor
            items_per_batch = min(len(deduped), self.batch_size) if partial else len(deduped)

            for cluster in clusters[:items_per_batch]:
                promoted = self._promote_cluster(cluster, session_id)
                if promoted:
                    promoted_count += 1

            # Clear this session's items
            self._buffer[session_id] = []

        # Clean up empty sessions
        empty = [k for k, v in self._buffer.items() if not v]
        for k in empty:
            del self._buffer[k]

        # Cap sessions
        while len(self._buffer) > self.max_sessions:
            oldest = min(self._buffer.keys(), key=lambda k: self._buffer[k][0].timestamp if self._buffer[k] else float('inf'))
            del self._buffer[oldest]

        self._last_flush_at = now
        self.stats.total_flushed += total_before
        self.stats.total_merged += merged_count
        self.stats.total_promoted += promoted_count
        self.stats.buffer_size = sum(len(v) for v in self._buffer.values())

        return {
            "flushed": total_before,
            "merged": merged_count,
            "promoted": promoted_count,
            "buffer_remaining": self.stats.buffer_size,
            "reason": "partial" if partial else "full",
        }

    def _dedup_items(self, items: list[StreamItem]) -> list[StreamItem]:
        """Deduplicate items by text similarity within the buffer."""
        if len(items) <= 1:
            return items

        # Get embeddings for all items
        for item in items:
            if item.embedding is None:
                item.embedding = self._get_embedder().encode(item.text)

        kept = []
        for item in items:
            is_dup = False
            for existing in kept:
                if item.embedding and existing.embedding:
                    sim = _cosine_sim(item.embedding, existing.embedding)
                    if sim > self.dedup_threshold:
                        # Merge: boost importance of existing, skip new
                        existing.importance = max(existing.importance, item.importance)
                        existing.tags = list(set(existing.tags + item.tags))
                        is_dup = True
                        break
            if not is_dup:
                kept.append(item)

        return kept

    def _cluster_by_tags(self, items: list[StreamItem]) -> list[list[StreamItem]]:
        """Group items by tag overlap into clusters for merging."""
        if not items:
            return []

        # Simple greedy clustering by tag intersection
        clusters: list[list[StreamItem]] = []
        remaining = list(items)

        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            seed_tags = set(seed.tags)

            i = 0
            while i < len(remaining):
                item_tags = set(remaining[i].tags)
                overlap = len(seed_tags & item_tags)
                # Join cluster if any tag overlap, or if no tags on either
                if overlap > 0 or (not seed_tags and not item_tags):
                    cluster.append(remaining.pop(i))
                    seed_tags |= item_tags
                else:
                    i += 1

            clusters.append(cluster)

        return clusters

    def _promote_cluster(self, items: list[StreamItem],
                        session_id: str) -> bool:
        """Promote a cluster of similar items into a single graph anchor.

        The most important item's text is used as the anchor text,
        with merged tags and averaged importance from all items.
        """
        if not items:
            return False

        # Sort by importance, best text as the anchor
        items.sort(key=lambda x: -x.importance)
        best = items[0]

        # Merge tags
        all_tags = list(set(tag for item in items for tag in item.tags))

        # Average importance weighted by item count
        avg_importance = sum(it.importance for it in items) / len(items)

        # Average emotional valence
        avg_valence = sum(it.emotional_valence for it in items) / len(items)

        # Create anchor text: primary item text + count suffix if merged
        if len(items) > 1:
            text = f"{best.text} (+{len(items) - 1} related items)"
        else:
            text = best.text

        try:
            self._manager.remember(
                text=text,
                source_session=session_id,
                tags=all_tags,
                importance=avg_importance,
                emotional_valence=avg_valence,
            )
            return True
        except Exception:
            return False

    def _get_embedder(self):
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    # ── Background auto-flush ─────────────────────────────

    def _start_auto_flush(self):
        """Start a background daemon thread for periodic auto-flush."""
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._auto_flush_loop, daemon=True)
        self._flush_thread.start()

    def _auto_flush_loop(self):
        """Background loop that flushes at regular intervals."""
        while self._running:
            time.sleep(self.flush_interval_s)
            if not self._running:
                break
            try:
                if self.size > 0:
                    self.flush()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)
