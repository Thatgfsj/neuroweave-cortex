"""Multi-Level Cognitive Cache — query, session, topic, activation, and exact-match caches.

Reduces redundant computation in the recall hot path:

  QueryCache       — LRU cache of recent query→result pairs, TTL 5 min
  SessionCache     — per-session working set of frequently accessed anchors
  TopicCache       — pre-computed tag→top_anchors mapping, rebuilt on sleep
  ActivationCache  — cached spreading activation results for hot seeds
  ExactMatchCache  — deterministic O(1) entity-pair KV lookup bypass

All caches are wired into the retrieval pipeline before full graph search.
"""

from __future__ import annotations

import math
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QueryCacheEntry:
    """Cached query result."""
    query: str
    result_ids: list[str]
    result_scores: list[float]
    created_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def is_expired(self, ttl_seconds: float = 300) -> bool:
        return self.age_seconds >= ttl_seconds


class QueryCache:
    """LRU cache of query→result pairs with configurable TTL.

    Reduces repeated retrievals for the same query within a short window.
    """

    def __init__(self, max_entries: int = 100, ttl_seconds: float = 300):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, QueryCacheEntry] = OrderedDict()

    def get(self, query: str) -> QueryCacheEntry | None:
        """Get cached result if not expired."""
        normalized = self._normalize(query)
        entry = self._cache.get(normalized)
        if entry is None:
            return None
        if entry.is_expired(self.ttl_seconds):
            del self._cache[normalized]
            return None
        # Move to end (LRU)
        self._cache.move_to_end(normalized)
        return entry

    def set(self, query: str, result_ids: list[str],
            result_scores: list[float]) -> QueryCacheEntry:
        """Cache a query result."""
        normalized = self._normalize(query)
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)
        entry = QueryCacheEntry(
            query=query,
            result_ids=result_ids,
            result_scores=result_scores,
        )
        self._cache[normalized] = entry
        return entry

    def invalidate(self, query: str | None = None) -> int:
        """Invalidate cache entries. If query is None, clear all."""
        if query is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        normalized = self._normalize(query)
        if normalized in self._cache:
            del self._cache[normalized]
            return 1
        return 0

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        expired = [
            k for k, v in self._cache.items()
            if v.is_expired(self.ttl_seconds)
        ]
        for k in expired:
            del self._cache[k]
        return len(expired)

    @staticmethod
    def _normalize(query: str) -> str:
        return query.strip().lower()

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }


class SessionCache:
    """Per-session working set of frequently accessed anchors.

    Tracks which anchors are "hot" in the current session — if the same
    anchor is retrieved 3+ times, it's promoted to the session cache.
    """

    def __init__(self, max_entries: int = 50, promote_threshold: int = 3):
        self.max_entries = max_entries
        self.promote_threshold = promote_threshold
        self._hot_anchors: dict[str, float] = {}       # anchor_id → access_count
        self._access_counts: dict[str, int] = {}       # anchor_id → raw count
        self._session_id: str = ""

    def start_session(self, session_id: str = ""):
        """Start tracking a new session."""
        self._session_id = session_id or str(time.time())
        self._hot_anchors.clear()
        self._access_counts.clear()

    def record_access(self, anchor_id: str):
        """Record an access to an anchor. Auto-promotes if threshold reached."""
        self._access_counts[anchor_id] = self._access_counts.get(anchor_id, 0) + 1
        count = self._access_counts[anchor_id]
        if count >= self.promote_threshold:
            self._hot_anchors[anchor_id] = time.time()
            self._trim()

    def get_hot(self, top_k: int = 10) -> list[str]:
        """Get the most recently promoted hot anchors."""
        sorted_hot = sorted(
            self._hot_anchors.items(),
            key=lambda x: -x[1],
        )
        return [aid for aid, _ in sorted_hot[:top_k]]

    def is_hot(self, anchor_id: str) -> bool:
        return anchor_id in self._hot_anchors

    def _trim(self):
        """Keep only the most recent hot anchors."""
        if len(self._hot_anchors) > self.max_entries:
            sorted_hot = sorted(
                self._hot_anchors.items(),
                key=lambda x: -x[1],
            )
            self._hot_anchors = dict(sorted_hot[:self.max_entries])

    def __len__(self) -> int:
        return len(self._hot_anchors)

    @property
    def stats(self) -> dict:
        return {
            "session_id": self._session_id,
            "hot_anchors": len(self._hot_anchors),
            "total_accesses": sum(self._access_counts.values()),
            "unique_accessed": len(self._access_counts),
        }


class TopicCache:
    """Pre-computed tag→top_anchors mapping, rebuilt during sleep.

    Maps each tag to its highest-retention anchors. During retrieval,
    if the query contains a known tag keyword, the pre-computed topic
    list provides fast candidate anchors without ANN search.
    """

    def __init__(self, max_per_tag: int = 20):
        self.max_per_tag = max_per_tag
        self._tag_index: dict[str, list[str]] = {}    # tag → [anchor_id...]
        self._last_rebuild: float = 0.0

    def rebuild(self, graph) -> int:
        """Rebuild the tag→anchor index from the graph. Returns tags indexed."""
        from .anchor import ThermalState
        tag_map: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for aid, anchor in graph.anchors.items():
            if not anchor.is_retrievable:
                continue
            if anchor.thermal_state in (ThermalState.FROZEN, ThermalState.DEAD):
                continue
            score = anchor.retention_score
            for tag in anchor.tags:
                if tag not in ('dormant', 'consolidating', 'ghost',
                              'mtm', 'ltm', 'summary', 'ltm_summary',
                              'topic_cluster', 'promoted_stm', 'promoted_mtm',
                              'sleep_rebuilt', 'pattern', 'abstractive_memory'):
                    tag_map[tag].append((aid, score))

        self._tag_index.clear()
        for tag, entries in tag_map.items():
            entries.sort(key=lambda x: -x[1])
            self._tag_index[tag] = [aid for aid, _ in entries[:self.max_per_tag]]

        self._last_rebuild = time.time()
        return len(self._tag_index)

    def lookup(self, tags: list[str], top_k: int = 10) -> list[str]:
        """Get candidate anchors for given tags."""
        candidates: dict[str, float] = {}
        for tag in tags:
            entries = self._tag_index.get(tag, [])
            position_bonus = 1.0
            for aid in entries:
                # Higher ranked entries get higher weight
                candidates[aid] = candidates.get(aid, 0) + position_bonus
                position_bonus *= 0.85  # diminishing returns
                if len(candidates) >= top_k * 3:
                    break

        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        return [aid for aid, _ in ranked[:top_k]]

    def lookup_text(self, query: str, top_k: int = 10) -> list[str]:
        """Match query text against known tags and return candidate anchors."""
        query_lower = query.lower()
        matched_tags = [tag for tag in self._tag_index
                       if tag.lower() in query_lower]
        return self.lookup(matched_tags, top_k=top_k)

    @property
    def age_hours(self) -> float:
        if self._last_rebuild == 0:
            return float('inf')
        return (time.time() - self._last_rebuild) / 3600

    @property
    def stats(self) -> dict:
        return {
            "tags_indexed": len(self._tag_index),
            "max_per_tag": self.max_per_tag,
            "last_rebuild_hours_ago": round(self.age_hours, 2),
        }

    def __len__(self) -> int:
        return len(self._tag_index)


class ActivationCache:
    """Cached spreading activation results for hot seed nodes.

    When the same seed anchors are used for spreading activation repeatedly
    (common in multi-turn conversations), the activations are cached to
    avoid re-running BFS traversal.
    """

    def __init__(self, max_entries: int = 50, ttl_seconds: float = 60):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, list[str], list[float]]] = OrderedDict()

    def get(self, seed_id: str) -> tuple[list[str], list[float]] | None:
        """Get cached activation result. Returns (anchor_ids, scores) or None."""
        entry = self._cache.get(seed_id)
        if entry is None:
            return None
        cached_at, anchor_ids, scores = entry
        if time.time() - cached_at >= self.ttl_seconds:
            del self._cache[seed_id]
            return None
        self._cache.move_to_end(seed_id)
        return anchor_ids, scores

    def set(self, seed_id: str, anchor_ids: list[str],
            scores: list[float]):
        """Cache an activation result."""
        while len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)
        self._cache[seed_id] = (time.time(), anchor_ids, scores)

    def invalidate(self, anchor_id: str | None = None):
        """Invalidate cache for a specific anchor or all."""
        if anchor_id is None:
            self._cache.clear()
        else:
            self._cache.pop(anchor_id, None)

    def evict_expired(self) -> int:
        now = time.time()
        expired = [k for k, (ts, _, _) in self._cache.items()
                   if now - ts >= self.ttl_seconds]
        for k in expired:
            del self._cache[k]
        return len(expired)

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }


class CognitiveCacheManager:
    """Orchestrates all four cognitive caches.

    Wired into the retrieval pipeline before full graph search:
    1. QueryCache: exact query match (fastest)
    2. SessionCache: hot anchors from current session
    3. TopicCache: tag-keyword match candidates
    4. ActivationCache: cached spreading activation results

    Usage:
        cache_mgr = CognitiveCacheManager()
        cache_mgr.topic_cache.rebuild(graph)  # during sleep
        cached = cache_mgr.lookup(query, tags=["python", "async"])
    """

    def __init__(self):
        self.query_cache = QueryCache(max_entries=100, ttl_seconds=300)
        self.session_cache = SessionCache(max_entries=50)
        self.topic_cache = TopicCache(max_per_tag=20)
        self.activation_cache = ActivationCache(max_entries=50, ttl_seconds=60)

    def lookup(self, query: str = "",
               tags: list[str] | None = None,
               seed_ids: list[str] | None = None) -> dict:
        """Multi-level cache lookup. Returns {source: [anchor_ids]}."""
        result: dict[str, list[str]] = {}

        # Level 1: Query cache
        if query:
            q_entry = self.query_cache.get(query)
            if q_entry:
                result["query_cache"] = q_entry.result_ids

        # Level 2: Session cache
        hot = self.session_cache.get_hot(top_k=10)
        if hot:
            result["session_cache"] = hot

        # Level 3: Topic cache
        if tags:
            topic_ids = self.topic_cache.lookup(tags, top_k=10)
            if topic_ids:
                result["topic_cache"] = topic_ids

        # Level 4: Activation cache (per-seed)
        activation_ids: list[str] = []
        if seed_ids:
            for sid in seed_ids[:3]:
                cached = self.activation_cache.get(sid)
                if cached:
                    activation_ids.extend(cached[0])
            if activation_ids:
                result["activation_cache"] = list(dict.fromkeys(activation_ids))[:10]

        return result

    def record_query(self, query: str, result_ids: list[str],
                     result_scores: list[float]):
        self.query_cache.set(query, result_ids, result_scores)

    def record_access(self, anchor_id: str):
        self.session_cache.record_access(anchor_id)

    def rebuild_on_sleep(self, graph) -> dict:
        """Called during sleep to rebuild topic index and evict expired entries."""
        tags_indexed = self.topic_cache.rebuild(graph)
        query_evicted = self.query_cache.evict_expired()
        activation_evicted = self.activation_cache.evict_expired()
        return {
            "tags_indexed": tags_indexed,
            "query_evicted": query_evicted,
            "activation_evicted": activation_evicted,
        }

    def start_session(self, session_id: str = ""):
        self.session_cache.start_session(session_id)

    @property
    def stats(self) -> dict:
        return {
            "query_cache": self.query_cache.stats,
            "session_cache": self.session_cache.stats,
            "topic_cache": self.topic_cache.stats,
            "activation_cache": self.activation_cache.stats,
        }


# ═══════════════════════════════════════════════════════════════
# ExactMatchCache — deterministic O(1) entity-pair KV lookup
# (merged from exact_cache.py, P2 architecture slimdown)
# ═══════════════════════════════════════════════════════════════

_ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
_KV_PATTERN = re.compile(
    r'(\w+(?:\s+\w+){0,2})\s(?:is|was|set to|equals|:|=>|->|=)\s(\w+(?:\s+\w+){0,3})',
    re.IGNORECASE,
)
_PORT_PATTERN = re.compile(r'\b(port|listen)\s+(\d+)\b', re.IGNORECASE)
_VERSION_PATTERN = re.compile(r'(\w+)\s+version\s+([\d.]+)', re.IGNORECASE)
_PATH_PATTERN = re.compile(r'(?:at|in|from|to)\s+([/\w.]+(?:/[.\w]+)+)', re.IGNORECASE)
_URL_PATTERN = re.compile(r'(https?://[^\s,]+)', re.IGNORECASE)
_PREF_PATTERN = re.compile(
    r'(prefers?|likes?|uses?|configured?|set)\s+(?:to\s+)?(\w+(?:\s+\w+){0,3})',
    re.IGNORECASE,
)
_ATTR_PATTERN = re.compile(
    r'(\w+(?:\s+\w+){0,2})\'s\s+(\w+)',
    re.IGNORECASE,
)


def extract_entity_keys(text: str, tags: list[str] | None = None) -> list[str]:
    """Extract exact-match keys from anchor text.

    Produces stable keys like "alice-birthday", "redis-port", "dark_mode-preference"
    that can be used for deterministic O(1) lookup.

    Returns up to 5 keys, sorted by confidence (longer keys = higher confidence).
    """
    keys: list[tuple[str, float]] = []
    text_lower = text.lower()
    text_clean = re.sub(r'\s+', ' ', text)

    for m in _ATTR_PATTERN.finditer(text_clean):
        entity = m.group(1).strip().lower().replace(' ', '_')
        attr = m.group(2).strip().lower().replace(' ', '_')
        if len(entity) > 2 and len(attr) > 2:
            keys.append((f"{entity}-{attr}", 0.9))

    for m in _KV_PATTERN.finditer(text_clean):
        key_part = m.group(1).strip().lower().replace(' ', '_')
        val_part = m.group(2).strip().lower().replace(' ', '_')
        if len(key_part) > 1 and len(val_part) > 0:
            keys.append((f"{key_part}-{val_part}", 0.8))

    for m in _PREF_PATTERN.finditer(text_clean):
        pref_value = m.group(2).strip().lower().replace(' ', '_')
        if len(pref_value) > 1:
            subject = ""
            before = text_clean[:m.start()].strip()
            subj_match = _ENTITY_PATTERN.search(before)
            if subj_match:
                subject = subj_match.group(1).lower().replace(' ', '_') + "-"
            keys.append((f"{subject}preference-{pref_value}", 0.85))

    for m in _PORT_PATTERN.finditer(text_clean):
        port = m.group(2)
        keys.append((f"port-{port}", 0.9))

    for m in _VERSION_PATTERN.finditer(text_clean):
        name = m.group(1).strip().lower()
        ver = m.group(2)
        keys.append((f"{name}-version-{ver}", 0.85))

    entities = _ENTITY_PATTERN.findall(text_clean)
    entities_unique = list(dict.fromkeys(e.strip().lower().replace(' ', '_') for e in entities))
    if entities_unique and tags:
        for tag in tags:
            if len(tag) > 2:
                keys.append((f"{entities_unique[0]}-{tag.lower()}", 0.7))

    for tag in (tags or []):
        tag_norm = tag.lower().replace(' ', '_')
        if len(tag_norm) > 2:
            keys.append((f"tag:{tag_norm}", 0.6))

    words = [w for w in re.findall(r'[a-z_]\w+', text_lower, re.IGNORECASE) if len(w) > 2]
    if len(words) >= 2 and not keys:
        keys.append((f"{words[0]}-{words[1]}", 0.4))

    seen: dict[str, float] = {}
    for k, conf in keys:
        k = re.sub(r'[^a-z0-9_:#-]', '', k)[:80]
        if k in seen:
            seen[k] = max(seen[k], conf)
        else:
            seen[k] = conf

    sorted_keys = sorted(seen.items(), key=lambda x: -x[1])
    return [k for k, _ in sorted_keys[:5]]


@dataclass
class ExactMatchEntry:
    """A single entry in the exact match cache."""
    key: str
    anchor_id: str
    text: str
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_hit: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self):
        self.last_hit = time.time()
        self.hit_count += 1


class ExactMatchCache:
    """Deterministic O(1) KV cache for strongly associated entity pairs.

    Bypasses fuzzy retrieval entirely. When a query contains a known entity key,
    the exact match result is returned directly without graph traversal.
    """

    def __init__(self, max_entries_per_key: int = 5):
        self.max_entries_per_key = max_entries_per_key
        self._store: dict[str, list[ExactMatchEntry]] = {}
        self._key_set: set[str] = set()
        self.hits: int = 0
        self.misses: int = 0

    def put(self, key: str, anchor_id: str, text: str,
            confidence: float = 0.5) -> ExactMatchEntry:
        entry = ExactMatchEntry(
            key=key, anchor_id=anchor_id, text=text,
            confidence=confidence,
        )
        if key not in self._store:
            self._store[key] = []
            self._key_set.add(key)

        bucket = self._store[key]
        for existing in bucket:
            if existing.anchor_id == anchor_id:
                existing.text = text
                existing.confidence = max(existing.confidence, confidence)
                existing.touch()
                return existing

        bucket.append(entry)
        if len(bucket) > self.max_entries_per_key:
            bucket.sort(key=lambda e: e.confidence * math.log(2 + e.hit_count))
            removed = bucket.pop(0)
            if not bucket:
                del self._store[key]
                self._key_set.discard(key)
        return entry

    def get(self, key: str, top_k: int = 3) -> list[ExactMatchEntry]:
        if key not in self._key_set:
            self.misses += 1
            return []
        bucket = self._store.get(key, [])
        if not bucket:
            self.misses += 1
            return []

        now = time.time()
        scored: list[tuple[ExactMatchEntry, float]] = []
        for entry in bucket:
            self.hits += 1
            hours_since = (now - entry.last_hit) / 3600
            recency = math.exp(-hours_since / 168)
            score = entry.confidence * 0.7 + recency * 0.3
            scored.append((entry, score))
            entry.touch()

        scored.sort(key=lambda x: -x[1])
        return [e for e, _ in scored[:top_k]]

    def has(self, key: str) -> bool:
        return key in self._key_set

    def harvest_from_anchor(self, anchor: Any) -> list[str]:
        text = getattr(anchor, 'text', '')
        tags = getattr(anchor, 'tags', [])
        anchor_id = getattr(anchor, 'id', '')
        salience = getattr(anchor, 'salience', 0.5)
        pre_keys = getattr(anchor, 'exact_match_keys', [])
        keys = pre_keys if pre_keys else extract_entity_keys(text, tags)
        for key in keys:
            self.put(key, anchor_id, text[:200], confidence=0.5 + salience * 0.4)
        return keys

    def query_keys(self, text: str, tags: list[str] | None = None) -> list[str]:
        return extract_entity_keys(text, tags)

    def remove_anchor(self, anchor_id: str):
        for key in list(self._store.keys()):
            self._store[key] = [e for e in self._store[key] if e.anchor_id != anchor_id]
            if not self._store[key]:
                del self._store[key]
                self._key_set.discard(key)

    def clear(self):
        self._store.clear()
        self._key_set.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return sum(len(bucket) for bucket in self._store.values())

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "entries": self.size,
            "unique_keys": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
        }
