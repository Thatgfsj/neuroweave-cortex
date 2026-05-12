"""Raw Chunk Buffer — uncompressed short-term memory tier.

Stores original dialogue segments without compression. Queried FIRST during
retrieval (before graph search). Provides BM25 + vector hybrid search.

Multi-level architecture:
  L0 Raw Buffer (last 1-2 sessions, no compression, TTL eviction)
  L1 Mid-level Anchors (compressed, rehearsed, graph-structured)
  L2 Long-term Graph (persistent, decayed, abstraction-only)

Inspired by MemGPT's OS-style multi-level memory and Zep's chunk+graph approach.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawChunk:
    """An uncompressed dialogue segment — exact original text preserved."""

    id: str
    text: str
    session_id: str
    created_at: float = field(default_factory=time.time)
    embedding: list[float] | None = None
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    anchor_id: str = ""  # linked graph anchor (dual-write)

    # BM25 pre-computed stats
    _term_freq: dict[str, int] | None = field(default=None, repr=False)
    _token_count: int = 0

    def tokenize(self) -> list[str]:
        """Lowercase word tokenization."""
        import re
        return re.findall(r'[a-zA-Z_]\w{1,}', self.text.lower())

    @property
    def term_freq(self) -> dict[str, int]:
        if self._term_freq is None:
            self._term_freq = {}
            for token in self.tokenize():
                self._term_freq[token] = self._term_freq.get(token, 0) + 1
            self._token_count = sum(self._term_freq.values())
        return self._term_freq

    @property
    def token_count(self) -> int:
        if self._token_count == 0:
            _ = self.term_freq  # triggers computation
        return self._token_count


class RawBuffer:
    """Session-based raw chunk storage with BM25 + vector hybrid retrieval.

    Eviction policy: FIFO by session — only last max_sessions are retained.
    """

    def __init__(self, max_sessions: int = 2, max_chunks_per_session: int = 500,
                 bm25_k1: float = 1.2, bm25_b: float = 0.75):
        self.max_sessions = max_sessions
        self.max_chunks_per_session = max_chunks_per_session
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        # session_id -> [chunk_ids]
        self._sessions: dict[str, list[str]] = defaultdict(list)
        # chunk_id -> RawChunk
        self._chunks: dict[str, RawChunk] = {}

        # Global IDF cache
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._total_docs: int = 0
        self._avg_doc_len: float = 0.0
        self._idf_stale: bool = True

    # ── Write ──────────────────────────────────────────────

    def add(self, text: str, session_id: str = "",
            embedding: list[float] | None = None,
            tags: list[str] | None = None,
            importance: float = 0.5,
            anchor_id: str = "") -> RawChunk:
        """Store a raw chunk. Evicts oldest session if over capacity."""
        import hashlib

        chunk_id = hashlib.blake2b(
            (text[:200] + session_id + str(time.time())).encode(),
            digest_size=8,
        ).hexdigest()

        chunk = RawChunk(
            id=chunk_id, text=text, session_id=session_id,
            embedding=embedding, tags=tags or [],
            importance=importance, anchor_id=anchor_id,
        )

        self._chunks[chunk_id] = chunk
        self._sessions[session_id].append(chunk_id)

        # Evict oldest session if over max
        while len(self._sessions) > self.max_sessions:
            oldest_session = min(self._sessions.keys(),
                                 key=lambda s: self._chunks[self._sessions[s][0]].created_at
                                 if self._sessions[s] else float('inf'))
            self._evict_session(oldest_session)

        # Trim oversized session
        if len(self._sessions[session_id]) > self.max_chunks_per_session:
            removed = self._sessions[session_id].pop(0)
            self._chunks.pop(removed, None)

        # Update stats lazily
        for token in chunk.term_freq:
            self._doc_freq[token] += 1
        self._total_docs += 1
        self._idf_stale = True

        return chunk

    def _evict_session(self, session_id: str) -> None:
        """Remove all chunks belonging to a session."""
        for chunk_id in self._sessions.pop(session_id, []):
            chunk = self._chunks.pop(chunk_id, None)
            if chunk:
                for token in chunk.term_freq:
                    self._doc_freq[token] = max(0, self._doc_freq[token] - 1)
                self._total_docs -= 1
        self._idf_stale = True

    # ── BM25 ───────────────────────────────────────────────

    def _compute_avg_doc_len(self) -> float:
        if self._chunks:
            return sum(c.token_count for c in self._chunks.values()) / len(self._chunks)
        return 1.0

    def _idf(self, term: str) -> float:
        """BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
        df = self._doc_freq.get(term, 0)
        N = self._total_docs
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_score(self, query_tokens: list[str], chunk: RawChunk) -> float:
        """BM25 scoring for a single chunk."""
        if self._avg_doc_len <= 0 or self._idf_stale:
            self._avg_doc_len = self._compute_avg_doc_len()
            self._idf_stale = False

        score = 0.0
        tf = chunk.term_freq
        doc_len = chunk.token_count

        for token in query_tokens:
            if token not in tf:
                continue
            idf = self._idf(token)
            freq = tf[token]
            numerator = freq * (self.bm25_k1 + 1.0)
            denominator = freq + self.bm25_k1 * (
                1.0 - self.bm25_b + self.bm25_b * doc_len / max(1.0, self._avg_doc_len))
            score += idf * numerator / denominator

        return score

    def tokenize_query(self, query: str) -> list[str]:
        """Tokenize and deduplicate query terms."""
        import re
        return list(dict.fromkeys(
            t.lower() for t in re.findall(r'[a-zA-Z_]\w{1,}', query) if len(t) > 1
        ))

    # ── Read / Search ──────────────────────────────────────

    def search(self, query: str, query_embedding: list[float] | None = None,
               top_k: int = 5, bm25_weight: float = 0.5,
               session_id: str = "") -> list[tuple[RawChunk, float]]:
        """Hybrid BM25 + vector search over raw chunks.

        Args:
            query: Search query text
            query_embedding: Optional embedding for vector similarity
            top_k: Max results
            bm25_weight: Weight of BM25 vs vector (1.0 = BM25 only)
            session_id: If set, prioritize chunks from this session

        Returns:
            List of (RawChunk, combined_score) sorted descending
        """
        if not self._chunks:
            return []

        query_tokens = self.tokenize_query(query)

        results: list[tuple[RawChunk, float]] = []

        for chunk in self._chunks.values():
            # BM25 score
            bm25 = self._bm25_score(query_tokens, chunk) if query_tokens else 0.0

            # Vector similarity
            vec_sim = 0.0
            if query_embedding and chunk.embedding:
                dot = sum(a * b for a, b in zip(query_embedding, chunk.embedding))
                na = math.sqrt(sum(x**2 for x in query_embedding))
                nb = math.sqrt(sum(x**2 for x in chunk.embedding))
                vec_sim = max(0.0, dot / (na * nb + 1e-8))

            # Combine: BM25 + vector, weighted
            combined = bm25_weight * self._normalize_bm25(bm25) + \
                       (1.0 - bm25_weight) * vec_sim

            # Session boost: same-session chunks get +0.15
            if session_id and chunk.session_id == session_id:
                combined += 0.15

            # Importance bonus
            combined += chunk.importance * 0.05

            if combined > 0.01:
                results.append((chunk, combined))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _normalize_bm25(self, score: float) -> float:
        """Sigmoid-normalize BM25 to [0, 1]."""
        if score <= 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-score / 3.0))

    def search_session(self, session_id: str, query: str,
                       query_embedding: list[float] | None = None,
                       top_k: int = 10) -> list[tuple[RawChunk, float]]:
        """Search within a specific session."""
        return self.search(query, query_embedding, top_k, session_id=session_id)

    def get_recent(self, n: int = 20) -> list[RawChunk]:
        """Get the N most recent chunks across all sessions."""
        sorted_chunks = sorted(
            self._chunks.values(), key=lambda c: c.created_at, reverse=True)
        return sorted_chunks[:n]

    def get_session_chunks(self, session_id: str) -> list[RawChunk]:
        """Get all chunks from a session, newest first."""
        chunk_ids = self._sessions.get(session_id, [])
        return sorted(
            (self._chunks[cid] for cid in chunk_ids if cid in self._chunks),
            key=lambda c: c.created_at, reverse=True,
        )

    @property
    def stats(self) -> dict:
        return {
            "total_sessions": len(self._sessions),
            "total_chunks": len(self._chunks),
            "session_ids": list(self._sessions.keys()),
            "avg_doc_len": round(self._avg_doc_len, 1),
        }

    def clear(self) -> None:
        self._sessions.clear()
        self._chunks.clear()
        self._doc_freq.clear()
        self._total_docs = 0
        self._avg_doc_len = 0.0
        self._idf_stale = True
