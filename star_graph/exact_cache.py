"""Exact Match Cache — deterministic O(1) entity-pair lookup bypass.

For strongly associated entity pairs (name→birthday, place→coordinates,
config_key→value), this provides a KV cache that skips fuzzy retrieval entirely.

Retrieval flow:
  query → extract entity key → KV cache exact lookup
    ├── hit  → return directly (deterministic, O(1))
    └── miss → System-1/2 fuzzy retrieval → degraded results

Entity keys are auto-harvested from anchor text using lightweight patterns
(NER-like regex for capitalized entities, key-value pairs, and domain-specific
patterns). No LLM dependency — pure regex + frequency.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any
from typing import Optional


# ── Entity extraction patterns ────────────────────────────────

# Capitalized multi-word phrases (potential named entities)
_ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

# Key-value patterns: "X is Y", "X = Y", "X: Y"
_KV_PATTERN = re.compile(
    r'(\w+(?:\s+\w+){0,2})\s(?:is|was|set to|equals|:|=>|->|=)\s(\w+(?:\s+\w+){0,3})',
    re.IGNORECASE,
)

# Domain-specific patterns
_PORT_PATTERN = re.compile(r'\b(port|listen)\s+(\d+)\b', re.IGNORECASE)
_VERSION_PATTERN = re.compile(r'(\w+)\s+version\s+([\d.]+)', re.IGNORECASE)
_PATH_PATTERN = re.compile(r'(?:at|in|from|to)\s+([/\w.]+(?:/[.\w]+)+)', re.IGNORECASE)
_URL_PATTERN = re.compile(r'(https?://[^\s,]+)', re.IGNORECASE)

# Preference/configuration detection
_PREF_PATTERN = re.compile(
    r'(prefers?|likes?|uses?|configured?|set)\s+(?:to\s+)?(\w+(?:\s+\w+){0,3})',
    re.IGNORECASE,
)

# Entity-attribute pairs
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
    keys: list[tuple[str, float]] = []  # (key, confidence)
    text_lower = text.lower()
    text_clean = re.sub(r'\s+', ' ', text)

    # 1. Entity-attribute pairs: "Alice's birthday" → "alice-birthday"
    for m in _ATTR_PATTERN.finditer(text_clean):
        entity = m.group(1).strip().lower().replace(' ', '_')
        attr = m.group(2).strip().lower().replace(' ', '_')
        if len(entity) > 2 and len(attr) > 2:
            keys.append((f"{entity}-{attr}", 0.9))

    # 2. Key-value patterns: "port is 6379" → "port-6379"
    for m in _KV_PATTERN.finditer(text_clean):
        key_part = m.group(1).strip().lower().replace(' ', '_')
        val_part = m.group(2).strip().lower().replace(' ', '_')
        if len(key_part) > 1 and len(val_part) > 0:
            keys.append((f"{key_part}-{val_part}", 0.8))

    # 3. Preference detection: "prefers dark mode" → "preference-dark_mode"
    for m in _PREF_PATTERN.finditer(text_clean):
        pref_value = m.group(2).strip().lower().replace(' ', '_')
        if len(pref_value) > 1:
            # Try to find the subject
            subject = ""
            before = text_clean[:m.start()].strip()
            subj_match = _ENTITY_PATTERN.search(before)
            if subj_match:
                subject = subj_match.group(1).lower().replace(' ', '_') + "-"
            keys.append((f"{subject}preference-{pref_value}", 0.85))

    # 4. Port numbers
    for m in _PORT_PATTERN.finditer(text_clean):
        port = m.group(2)
        keys.append((f"port-{port}", 0.9))

    # 5. Version strings
    for m in _VERSION_PATTERN.finditer(text_clean):
        name = m.group(1).strip().lower()
        ver = m.group(2)
        keys.append((f"{name}-version-{ver}", 0.85))

    # 6. Named entities — pair adjacent capitalized entities
    entities = _ENTITY_PATTERN.findall(text_clean)
    entities_unique = list(dict.fromkeys(e.strip().lower().replace(' ', '_') for e in entities))
    # Pair first entity with each tag as key
    if entities_unique and tags:
        for tag in tags:
            if len(tag) > 2:
                keys.append((f"{entities_unique[0]}-{tag.lower()}", 0.7))

    # 7. Tags as keys themselves (for tag-based exact lookup)
    for tag in (tags or []):
        tag_norm = tag.lower().replace(' ', '_')
        if len(tag_norm) > 2:
            keys.append((f"tag:{tag_norm}", 0.6))

    # 8. Fallback: first two significant words
    words = [w for w in re.findall(r'[a-z_]\w+', text_lower, re.IGNORECASE) if len(w) > 2]
    if len(words) >= 2 and not keys:
        keys.append((f"{words[0]}-{words[1]}", 0.4))

    # Deduplicate by key, keep highest confidence
    seen: dict[str, float] = {}
    for k, conf in keys:
        k = re.sub(r'[^a-z0-9_:#-]', '', k)[:80]
        if k in seen:
            seen[k] = max(seen[k], conf)
        else:
            seen[k] = conf

    # Sort by confidence desc, return top 5
    sorted_keys = sorted(seen.items(), key=lambda x: -x[1])
    return [k for k, _ in sorted_keys[:5]]


# ── Core data structures ──────────────────────────────────────

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

    Usage:
        cache = ExactMatchCache()
        cache.put("alice-birthday", anchor_id, "Alice's birthday is May 10th")
        result = cache.get("alice-birthday")  # O(1) deterministic lookup
    """

    def __init__(self, max_entries_per_key: int = 5):
        self.max_entries_per_key = max_entries_per_key
        self._store: dict[str, list[ExactMatchEntry]] = {}
        self._key_set: set[str] = set()  # fast membership test
        self.hits: int = 0
        self.misses: int = 0

    def put(self, key: str, anchor_id: str, text: str,
            confidence: float = 0.5) -> ExactMatchEntry:
        """Insert into cache. Evicts lowest-confidence entry if key bucket is full."""
        entry = ExactMatchEntry(
            key=key, anchor_id=anchor_id, text=text,
            confidence=confidence,
        )

        if key not in self._store:
            self._store[key] = []
            self._key_set.add(key)

        bucket = self._store[key]

        # Don't duplicate anchor_id for same key
        for existing in bucket:
            if existing.anchor_id == anchor_id:
                existing.text = text
                existing.confidence = max(existing.confidence, confidence)
                existing.touch()
                return existing

        bucket.append(entry)

        # Evict lowest confidence if full
        if len(bucket) > self.max_entries_per_key:
            bucket.sort(key=lambda e: e.confidence * math.log(2 + e.hit_count))
            removed = bucket.pop(0)
            if not bucket:
                del self._store[key]
                self._key_set.discard(key)

        return entry

    def get(self, key: str, top_k: int = 3) -> list[ExactMatchEntry]:
        """Exact match lookup. Returns entries sorted by confidence × recency.

        Returns empty list if key not found.
        """
        if key not in self._key_set:
            self.misses += 1
            return []

        bucket = self._store.get(key, [])
        if not bucket:
            self.misses += 1
            return []

        # Score: confidence * recency boost
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
        """Check if a key exists in the cache (O(1))."""
        return key in self._key_set

    def harvest_from_anchor(self, anchor: Any) -> list[str]:
        """Extract and store keys from an Anchor.

        Uses anchor.text, anchor.tags, and anchor.exact_match_keys
        if available. Returns list of keys added.
        """
        text = getattr(anchor, 'text', '')
        tags = getattr(anchor, 'tags', [])
        anchor_id = getattr(anchor, 'id', '')
        salience = getattr(anchor, 'salience', 0.5)

        # Use pre-extracted keys if available
        pre_keys = getattr(anchor, 'exact_match_keys', [])
        keys = pre_keys if pre_keys else extract_entity_keys(text, tags)

        for key in keys:
            self.put(key, anchor_id, text[:200], confidence=0.5 + salience * 0.4)

        return keys

    def query_keys(self, text: str, tags: list[str] | None = None) -> list[str]:
        """Extract potential keys from a query for lookup."""
        return extract_entity_keys(text, tags)

    def remove_anchor(self, anchor_id: str):
        """Remove all entries for a given anchor ID."""
        for key in list(self._store.keys()):
            self._store[key] = [e for e in self._store[key] if e.anchor_id != anchor_id]
            if not self._store[key]:
                del self._store[key]
                self._key_set.discard(key)

    def clear(self):
        """Clear all cache entries."""
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
