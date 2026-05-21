"""Domain-based graph partitioning for the NeuroWeave Cortex cognitive memory runtime.

Partitions the star graph into domain subspaces to reduce retrieval noise,
enable domain-scoped search, and manage cross-domain edge weights.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor


class Domain(str, Enum):
    """Cognitive domain classification for memory anchors."""
    DEVELOPMENT = "development"
    LIFESTYLE = "lifestyle"
    EMOTIONAL = "emotional"
    PROJECT = "project"
    WORLD_KNOWLEDGE = "world_knowledge"
    UNCLASSIFIED = "unclassified"


@dataclass
class DomainConfig:
    """Configuration for domain-based graph partitioning."""
    cross_domain_edge_weight: float = 0.3
    soft_isolation: bool = True
    auto_classify: bool = True
    max_domains_per_anchor: int = 2
    domain_embedding_dim: int = 384


# ── Keyword/pattern tables for classify() ──────────────────────────

_DEVELOPMENT_TAGS: frozenset[str] = frozenset({
    'code', 'debug', 'programming', 'dev', 'software',
    'bug', 'fix', 'feature', 'refactor', 'architecture',
})

_DEVELOPMENT_PATTERNS: list[re.Pattern] = [
    re.compile(r'\bfunction\b', re.IGNORECASE),
    re.compile(r'\bclass\b', re.IGNORECASE),
    re.compile(r'\bimport\b', re.IGNORECASE),
    re.compile(r'\berror\b', re.IGNORECASE),
    re.compile(r'\bbug\b', re.IGNORECASE),
    re.compile(r'\bfix\b', re.IGNORECASE),
    re.compile(r'\bdeploy\b', re.IGNORECASE),
    re.compile(r'\bcommit\b', re.IGNORECASE),
    re.compile(r'\bPR\b', re.IGNORECASE),
    re.compile(r'\bAPI\b', re.IGNORECASE),
]

_EMOTIONAL_TAGS: frozenset[str] = frozenset({
    'emotion', 'feeling', 'mood', 'sentiment', 'relationship',
})

_EMOTION_WORDS: frozenset[str] = frozenset({
    'happy', 'sad', 'frustrated', 'excited', 'worried', 'grateful',
    'angry', 'anxious', 'joyful', 'depressed', 'content', 'upset',
    'thrilled', 'nervous', 'calm', 'stressed', 'proud', 'ashamed',
    'hopeful', 'disappointed', 'loved', 'lonely',
})

_EMOTION_PATTERN: re.Pattern = re.compile(
    r'\b(' + '|'.join(_EMOTION_WORDS) + r')\b', re.IGNORECASE,
)

_LIFESTYLE_TAGS: frozenset[str] = frozenset({
    'preference', 'habit', 'daily', 'life', 'style', 'routine', 'personal',
})

_LIFESTYLE_KEYWORDS: frozenset[str] = frozenset({
    'routine', 'habit', 'prefer', 'daily',
})

_PROJECT_TAGS: frozenset[str] = frozenset({
    'project', 'milestone', 'deadline', 'stakeholder', 'deliverable',
})

_PROJECT_KEYWORDS: frozenset[str] = frozenset({
    'milestone', 'deadline', 'stakeholder',
})

_WORLD_KNOWLEDGE_TAGS: frozenset[str] = frozenset({
    'knowledge', 'fact', 'learning', 'concept', 'info', 'research', 'study',
})

_PERSONAL_INDICATORS: tuple[str, ...] = (
    'i ', "i'm", 'my ', 'me ', 'we ', 'our ',
)


class DomainManager:
    """Manages domain assignment, cross-domain edge weighting, and domain-scoped search."""

    def __init__(self, graph: StarGraph, config: DomainConfig | None = None):
        self._graph = graph
        self._config = config or DomainConfig()
        self._domain_map: dict[str, Domain] = {}
        self._cross_edge_counts: dict[tuple[Domain, Domain], int] = defaultdict(int)

    # ── Classification ─────────────────────────────────────────────

    def classify(self, text: str, tags: list[str] | None = None,
                 emotional_valence: float = 0.0,
                 memory_type: str = "") -> Domain:
        """Infer domain from text content and tags."""
        tags_set = {t.lower() for t in (tags or [])}
        text_lower = text.lower()

        if tags_set & _DEVELOPMENT_TAGS or any(
            p.search(text) for p in _DEVELOPMENT_PATTERNS
        ):
            return Domain.DEVELOPMENT

        if (tags_set & _EMOTIONAL_TAGS
                or abs(emotional_valence) > 0.5
                or _EMOTION_PATTERN.search(text)):
            return Domain.EMOTIONAL

        if tags_set & _PROJECT_TAGS or any(
            kw in text_lower for kw in _PROJECT_KEYWORDS
        ):
            return Domain.PROJECT

        if tags_set & _WORLD_KNOWLEDGE_TAGS or memory_type == 'knowledge':
            if not any(pi in text_lower for pi in _PERSONAL_INDICATORS):
                return Domain.WORLD_KNOWLEDGE

        if tags_set & _LIFESTYLE_TAGS or any(
            kw in text_lower for kw in _LIFESTYLE_KEYWORDS
        ):
            return Domain.LIFESTYLE

        return Domain.UNCLASSIFIED

    # ── Assignment ─────────────────────────────────────────────────

    def assign_domain(self, anchor_id: str,
                      domain: Domain | None = None) -> Domain:
        """Assign domain to anchor (auto-classify if None)."""
        anchor = self._graph.anchors.get(anchor_id)
        if domain is None and self._config.auto_classify and anchor is not None:
            text = anchor.text if hasattr(anchor, 'text') else ''
            tags = getattr(anchor, 'tags', [])
            valence = anchor.vector.emotional_valence if hasattr(anchor, 'vector') else 0.0
            domain = self.classify(text, tags, emotional_valence=valence)
        if domain is None:
            domain = Domain.UNCLASSIFIED

        self._domain_map[anchor_id] = domain

        if anchor is not None:
            self._sync_anchor_metadata(anchor, domain)

        return domain

    def _sync_anchor_metadata(self, anchor: Anchor, domain: Domain) -> None:
        """Store domain in anchor metadata."""
        if not hasattr(anchor, 'metadata') or anchor.metadata is None:
            anchor.metadata = {}  # type: ignore[attr-defined]
        anchor.metadata['domain'] = domain.value  # type: ignore[index]

    def get_domain(self, anchor_id: str) -> Domain:
        """Get the domain of an anchor."""
        if anchor_id in self._domain_map:
            return self._domain_map[anchor_id]

        anchor = self._graph.anchors.get(anchor_id)
        if anchor is not None and hasattr(anchor, 'metadata') and anchor.metadata:
            stored = anchor.metadata.get('domain')
            if stored:
                try:
                    return Domain(stored)
                except ValueError:
                    pass

        return Domain.UNCLASSIFIED

    def get_domain_anchors(self, domain: Domain) -> list[str]:
        """All anchor IDs in the given domain."""
        return [aid for aid, d in self._domain_map.items()
                if d == domain and aid in self._graph.anchors]

    def domain_size(self, domain: Domain) -> int:
        """Count anchors in the domain."""
        return sum(1 for aid, d in self._domain_map.items()
                   if d == domain and aid in self._graph.anchors)

    # ── Cross-domain edges ─────────────────────────────────────────

    def cross_domain_edge_allowed(self, anchor_id_a: str,
                                   anchor_id_b: str) -> bool:
        """Check if edge between two anchors should be allowed (always True)."""
        da = self._domain_map.get(anchor_id_a, Domain.UNCLASSIFIED)
        db = self._domain_map.get(anchor_id_b, Domain.UNCLASSIFIED)
        if da != db and da != Domain.UNCLASSIFIED and db != Domain.UNCLASSIFIED:
            key = (da, db) if da.value < db.value else (db, da)
            self._cross_edge_counts[key] += 1
        return True

    def cross_domain_weight(self, weight: float) -> float:
        """Reduce weight for cross-domain edges: weight x cross_domain_edge_weight."""
        return weight * self._config.cross_domain_edge_weight

    def _is_cross_domain(self, anchor_id_a: str, anchor_id_b: str) -> bool:
        """Check if two anchors belong to different classified domains."""
        da = self._domain_map.get(anchor_id_a, Domain.UNCLASSIFIED)
        db = self._domain_map.get(anchor_id_b, Domain.UNCLASSIFIED)
        if da == Domain.UNCLASSIFIED or db == Domain.UNCLASSIFIED:
            return False
        return da != db

    def edge_weight(self, weight: float, anchor_id_a: str,
                     anchor_id_b: str) -> float:
        """Return effective edge weight, reduced if cross-domain."""
        if self._is_cross_domain(anchor_id_a, anchor_id_b):
            return self.cross_domain_weight(weight)
        return weight

    def track_edge(self, anchor_id_a: str, anchor_id_b: str) -> None:
        """Register a cross-domain edge for statistics."""
        da = self._domain_map.get(anchor_id_a, Domain.UNCLASSIFIED)
        db = self._domain_map.get(anchor_id_b, Domain.UNCLASSIFIED)
        if da != db and da != Domain.UNCLASSIFIED and db != Domain.UNCLASSIFIED:
            key = (da, db) if da.value < db.value else (db, da)
            self._cross_edge_counts[key] += 1

    # ── Domain-scoped search ───────────────────────────────────────

    def search_domain(self, domain: Domain,
                      query_embedding: list[float],
                      top_k: int = 10) -> list[str]:
        """Search only within a domain using cosine similarity."""
        from .math_utils import cosine_sim

        domain_anchors = [aid for aid, d in self._domain_map.items()
                          if d == domain and aid in self._graph.anchors]
        if not domain_anchors:
            return []

        scored: list[tuple[str, float]] = []
        for aid in domain_anchors:
            anchor = self._graph.anchors.get(aid)
            if anchor is not None and anchor.embedding:
                sim = cosine_sim(query_embedding, anchor.embedding)
                scored.append((aid, sim))

        scored.sort(key=lambda x: -x[1])
        return [aid for aid, _ in scored[:top_k]]

    # ── Related domains ────────────────────────────────────────────

    def get_related_domains(self, domain: Domain,
                            min_edges: int = 3) -> list[Domain]:
        """Domains with significant cross-edges to the given domain."""
        related: dict[Domain, int] = defaultdict(int)
        for (da, db), count in self._cross_edge_counts.items():
            if da == domain:
                related[db] += count
            elif db == domain:
                related[da] += count
        return [d for d, c in sorted(related.items(), key=lambda x: -x[1])
                if c >= min_edges]

    # ── Balance ────────────────────────────────────────────────────

    def balance_domains(self) -> dict[Domain, int]:
        """Return domain distribution; flag imbalanced domains (>60% of total)."""
        dist: dict[Domain, int] = {}
        total = sum(
            self.domain_size(d) for d in Domain if d != Domain.UNCLASSIFIED
        )
        if total == 0:
            return {d: 0 for d in Domain}

        for d in Domain:
            size = self.domain_size(d)
            if d != Domain.UNCLASSIFIED and size > 0 and size / total > 0.6:
                dist[d] = -size  # Negative signals imbalance
            else:
                dist[d] = size
        return dist

    # ── Snapshot ───────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Domain distribution and cross-domain edge counts."""
        dist = {d.value: self.domain_size(d) for d in Domain}
        cross_edges = {
            f"{da.value}<->{db.value}": count
            for (da, db), count in sorted(self._cross_edge_counts.items(),
                                          key=lambda x: -x[1])
        }
        return {
            "domains": dist,
            "total_anchors": sum(dist.values()),
            "total_classified": sum(
                1 for aid, d in self._domain_map.items()
                if d != Domain.UNCLASSIFIED and aid in self._graph.anchors
            ),
            "cross_domain_edges": cross_edges,
            "config": {
                "cross_domain_edge_weight": self._config.cross_domain_edge_weight,
                "soft_isolation": self._config.soft_isolation,
                "auto_classify": self._config.auto_classify,
                "max_domains_per_anchor": self._config.max_domains_per_anchor,
            },
        }
