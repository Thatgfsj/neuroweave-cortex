"""Symbolic pre-filter — topic/keyword gate before embedding search.

Prevents embedding collision: "python bug", "selenium issue", and
"network login" would all score similarly in vector space (all "technical"),
but a symbolic filter ensures only same-topic items pass through to the
expensive embedding comparison.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class FilterResult:
    """Result of symbolic filtering."""
    passed_ids: list[str]
    rejected_ids: list[str]
    filter_stage: str = ""       # which stage rejected them
    duration_ms: float = 0.0
    candidate_count: int = 0
    passed_count: int = 0


class SymbolicFilter:
    """Two-stage symbolic gate: topic filter → keyword filter → embedding.

    Stage 1 (Topic gate): Must share at least one tag with query topics.
    Stage 2 (Keyword gate): Must have min keyword overlap (Jaccard > threshold).

    Only items passing BOTH stages proceed to expensive embedding comparison.
    """

    # Common stop words to exclude from keyword extraction
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'because', 'but', 'and', 'or', 'if', 'while', 'about',
        'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
        'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
        'what', 'which', 'who', 'whom', 'also', 'any', 'up', 'out',
    }

    def __init__(self, min_tag_overlap: int = 1,
                 min_keyword_jaccard: float = 0.08,
                 max_keywords_per_text: int = 20):
        self.min_tag_overlap = min_tag_overlap
        self.min_keyword_jaccard = min_keyword_jaccard
        self.max_keywords_per_text = max_keywords_per_text

        # Cached keyword sets
        self._keyword_cache: dict[str, set[str]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def extract_keywords(self, text: str) -> set[str]:
        """Extract lowercased meaningful words from text."""
        if not text:
            return set()
        words = re.findall(r'[a-zA-Z_]\w{1,}', text.lower())
        return {w for w in words if w not in self.STOP_WORDS}

    def extract_entities(self, text: str) -> set[str]:
        """Extract capitalized words and technical terms as entities."""
        entities: set[str] = set()
        # CamelCase / snake_case technical terms
        for m in re.finditer(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text):
            entities.add(m.group().lower())
        for m in re.finditer(r'\b[a-z]+_[a-z]+\b', text):
            entities.add(m.group().lower())
        # Capitalized proper nouns
        for m in re.finditer(r'\b[A-Z][a-z]{2,}\b', text):
            entities.add(m.group().lower())
        return entities

    def get_keywords(self, text: str) -> set[str]:
        """Get cached keywords for a text."""
        if text in self._keyword_cache:
            self._cache_hits += 1
            return self._keyword_cache[text]
        self._cache_misses += 1
        kw = self.extract_keywords(text)
        if len(kw) > self.max_keywords_per_text:
            # Keep the longest words (more informative)
            kw = set(sorted(kw, key=len, reverse=True)[:self.max_keywords_per_text])
        self._keyword_cache[text] = kw
        return kw

    def has_topic_overlap(self, query_tags: list[str],
                          anchor_tags: list[str]) -> bool:
        """Check if query and anchor share at least min_tag_overlap tags."""
        if not query_tags or not anchor_tags:
            return self.min_tag_overlap == 0
        return len(set(query_tags) & set(anchor_tags)) >= self.min_tag_overlap

    def keyword_jaccard(self, query_kw: set[str],
                        anchor_kw: set[str]) -> float:
        """Jaccard similarity between keyword sets."""
        if not query_kw or not anchor_kw:
            return 0.0
        return len(query_kw & anchor_kw) / len(query_kw | anchor_kw)

    def filter(self, query_text: str,
               query_tags: list[str],
               candidates: list[tuple[str, str, list[str]]],
               # each: (anchor_id, anchor_text, anchor_tags)
               ) -> FilterResult:
        """Two-stage symbolic filter.

        Args:
            query_text: The query string
            query_tags: Tags associated with the query
            candidates: List of (anchor_id, anchor_text, anchor_tags) tuples

        Returns:
            FilterResult with passed_ids and rejected_ids
        """
        t0 = time.perf_counter()
        result = FilterResult(candidate_count=len(candidates))

        if not candidates:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        query_kw = self.get_keywords(query_text)

        passed: list[str] = []
        rejected: list[str] = []

        for anchor_id, anchor_text, anchor_tags in candidates:
            # Stage 1: Topic gate
            if query_tags and not self.has_topic_overlap(query_tags, anchor_tags):
                rejected.append(anchor_id)
                result.filter_stage = "topic"
                continue

            # Stage 2: Keyword gate
            if query_kw:
                anchor_kw = self.get_keywords(anchor_text)
                jac = self.keyword_jaccard(query_kw, anchor_kw)
                if jac < self.min_keyword_jaccard:
                    rejected.append(anchor_id)
                    result.filter_stage = "keyword"
                    continue

            passed.append(anchor_id)

        result.passed_ids = passed
        result.rejected_ids = rejected
        result.passed_count = len(passed)
        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    def filter_anchors(self, query_text: str,
                       query_tags: list[str],
                       anchors: dict) -> FilterResult:
        """Convenience: filter a dict of {id: Anchor} objects."""
        candidates = [
            (aid, a.text, a.tags)
            for aid, a in anchors.items()
            if a.is_retrievable
        ]
        return self.filter(query_text, query_tags, candidates)

    @property
    def stats(self) -> dict:
        return {
            "cached_keywords": len(self._keyword_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        }
