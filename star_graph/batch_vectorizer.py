"""Batch vectorizer — reduces embedding I/O by buffering and batch-encoding.

Deferred writes: buffer ≥ batch_size or > flush_interval triggers batch embedding.
SQLite backing store for optional crash recovery (avoid lost embeddings on restart).

Usage:
    bv = BatchVectorizer(graph, embedder, batch_size=32, flush_interval=30.0)
    bv.enqueue(text, anchor_id)
    # ... more enqueues ...
    bv.flush()  # or auto-triggered by size/timer
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Optional


class BatchVectorizer:
    """Buffered, batch-oriented embedding writer.

    Collects text+anchor_id pairs and batch-encodes when the buffer reaches
    batch_size or flush_interval seconds have elapsed since the first enqueue.

    Uses a background timer thread for automatic flush. Thread-safe for
    concurrent enqueues (single-writer lock).
    """

    def __init__(self, graph, embedder,
                 batch_size: int = 32,
                 flush_interval: float = 30.0,
                 db_path: str = ""):
        self._graph = graph
        self._embedder = embedder
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(1.0, flush_interval)

        self._queue: list[tuple[str, str]] = []          # [(text, anchor_id)]
        self._first_enqueue_at: float = 0.0
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._draining = False

        # SQLite journal for crash recovery (optional)
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        if db_path:
            self._init_db()

        self.total_batched: int = 0
        self.total_flushes: int = 0

    # ── SQLite journal ──────────────────────────────────────

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                anchor_id TEXT NOT NULL,
                queued_at REAL NOT NULL
            )
        """)
        conn.commit()
        self._conn = conn

    def _journal_insert(self, text: str, anchor_id: str) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT INTO pending_embeddings (text, anchor_id, queued_at) VALUES (?, ?, ?)",
                (text, anchor_id, time.time()),
            )
            self._conn.commit()
        except Exception:
            pass  # Journal is best-effort

    def _journal_delete(self) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute("DELETE FROM pending_embeddings")
            self._conn.commit()
        except Exception:
            pass

    def recover_pending(self) -> list[tuple[str, str]]:
        """Recover unprocessed entries from a previous crash. Call once on startup."""
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT text, anchor_id FROM pending_embeddings ORDER BY id"
            ).fetchall()
            if rows:
                self._journal_delete()
            return [(r[0], r[1]) for r in rows]
        except Exception:
            return []

    # ── Enqueue / flush ─────────────────────────────────────

    def enqueue(self, text: str, anchor_id: str) -> None:
        """Queue a text for batch embedding.

        If the buffer is full, flushes immediately. Otherwise starts or
        restarts the flush timer.
        """
        with self._lock:
            self._queue.append((text, anchor_id))
            self._journal_insert(text, anchor_id)

            if not self._first_enqueue_at:
                self._first_enqueue_at = time.time()

            if len(self._queue) >= self._batch_size:
                self._drain_locked()
                return

            self._schedule_timer_locked()

    def flush(self) -> int:
        """Force-flush all pending texts. Returns count of embeddings written."""
        with self._lock:
            return self._drain_locked()

    def _schedule_timer_locked(self) -> None:
        if self._timer:
            self._timer.cancel()
        elapsed = time.time() - self._first_enqueue_at
        delay = max(0.5, self._flush_interval - elapsed)
        self._timer = threading.Timer(delay, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        with self._lock:
            if self._queue:
                self._drain_locked()

    def _drain_locked(self) -> int:
        """Drain queue while holding lock. Returns count processed."""
        if self._draining or not self._queue:
            return 0

        self._draining = True
        if self._timer:
            self._timer.cancel()
            self._timer = None

        batch = list(self._queue)
        self._queue.clear()
        self._first_enqueue_at = 0.0
        self._journal_delete()
        self._draining = False

        # Batch-encode outside of lock contention window
        texts = [t for t, _ in batch]
        try:
            embeddings = self._embedder.encode(texts)
        except Exception:
            self._draining = False
            return 0

        # Write embeddings back to anchors
        written = 0
        for (_, aid), emb in zip(batch, embeddings):
            anchor = self._graph.anchors.get(aid)
            if anchor is not None:
                anchor.embedding = emb
                # Update cortical index with new embedding
                self._graph.cortical_index.append((emb, aid))
                written += 1

        self.total_batched += written
        self.total_flushes += 1
        return written

    def __len__(self) -> int:
        return len(self._queue)

    def shutdown(self) -> None:
        """Flush remaining items and stop the timer."""
        self.flush()
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
