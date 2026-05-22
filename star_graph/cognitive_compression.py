"""Cognitive Compression — events→concepts→identity→world_model.

Upgrades existing compression.py with concept formation and world model
construction. Four levels:
  RAW (0) → CONCEPT (1) → IDENTITY (2) → WORLD_MODEL (3)
"""

from __future__ import annotations

import enum
import hashlib
import math
import time
import uuid
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any


class CompressionStage(enum.Enum):
    EVENT_CLUSTER = "event_cluster"
    SUMMARIZE = "summarize"
    CONCEPT_EXTRACT = "concept_extract"
    PATTERN_DETECT = "pattern_detect"
    TRAIT_UPDATE = "trait_update"
    WORLD_MODEL = "world_model"
    PRUNE_REDUNDANT = "prune_redundant"


@dataclass
class CognitiveCompressionResult:
    """Result of a cognitive compression cycle."""
    level: int = 0                     # 0..3
    source_anchor_ids: list[str] = field(default_factory=list)
    compressed_text: str = ""
    embedding: list[float] | None = None
    concepts_formed: list[str] = field(default_factory=list)
    patterns_detected: list[str] = field(default_factory=list)
    personality_trait_updates: dict[str, float] = field(default_factory=dict)
    world_model_updates: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.5
    evidence_count: int = 0
    contradictory_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorldModelBelief:
    """A belief about how the world works — highest compression level."""
    id: str
    belief: str
    category: str = "user_behavior"  # user_behavior | tooling | communication | decision_making
    confidence: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    stability: float = 0.5
    formed_from: list[str] = field(default_factory=list)  # trait IDs that led to this
    last_updated_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "belief": self.belief,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "stability": round(self.stability, 3),
            "evidence_count": len(self.supporting_evidence),
        }


class CognitiveCompressor:
    """Multi-level cognitive compression pipeline."""

    def __init__(self, graph=None, *,
                 config: dict[str, Any] | None = None,
                 concept_cortex=None,
                 personality_model=None,
                 embedder=None):
        self.graph = graph
        self._config = config or {}
        self._concept_cortex = concept_cortex
        self._personality_model = personality_model
        self._embedder = embedder

        self._min_cluster = self._config.get("min_cluster_size", 3)
        self._sim_threshold = self._config.get("similarity_threshold", 0.55)
        self._summary_confidence = self._config.get("summary_confidence_threshold", 0.4)
        self._concept_min_summaries = self._config.get("concept_min_summaries", 3)
        self._pattern_min = self._config.get("pattern_min_concepts", 3)
        self._trait_min = self._config.get("trait_min_patterns", 2)
        self._world_min_evidence = self._config.get("world_model_min_evidence", 5)

        self._summaries: list[CognitiveCompressionResult] = []
        self._world_model: dict[str, WorldModelBelief] = {}
        self._cycle_count: int = 0

        # Stage enable flags
        stages_enabled = self._config.get("stages_enabled", {})
        self._stages = {
            "event_cluster": stages_enabled.get("event_cluster", True),
            "summarize": stages_enabled.get("summarize", True),
            "concept_extract": stages_enabled.get("concept_extract", True),
            "pattern_detect": stages_enabled.get("pattern_detect", True),
            "trait_update": stages_enabled.get("trait_update", True),
            "world_model": stages_enabled.get("world_model", True),
            "prune_redundant": stages_enabled.get("prune_redundant", True),
        }

    # ── Full Pipeline ──────────────────────────────────────

    def compress(self, anchor_ids: list[str] | None = None,
                 max_level: int = 3) -> list[CognitiveCompressionResult]:
        """Run the full compression pipeline."""
        results: list[CognitiveCompressionResult] = []
        self._cycle_count += 1

        # Stage 1: Cluster events
        clusters: list[list[str]] = []
        if self._stages["event_cluster"]:
            clusters = self._cluster_events(anchor_ids or [])

        # Stage 2: Summarize clusters
        if self._stages["summarize"] and clusters:
            for cluster in clusters:
                if len(cluster) >= self._min_cluster:
                    summary = self._summarize_cluster(cluster)
                    if summary.confidence >= self._summary_confidence:
                        results.append(summary)
                        self._summaries.append(summary)

        # Stage 3: Extract concepts from summaries
        if self._stages["concept_extract"] and len(self._summaries) >= self._concept_min_summaries:
            concepts = self._extract_concepts()
            for r in results:
                r.concepts_formed = concepts

        # Stage 4: Detect patterns
        if self._stages["pattern_detect"] and len(results) >= self._pattern_min:
            patterns = self._detect_patterns(results)
            for r in results:
                r.patterns_detected = patterns

        # Stage 5: Update traits
        if self._stages["trait_update"] and self._personality_model:
            trait_updates = self._update_personality(results)
            for r in results:
                r.personality_trait_updates = trait_updates

        # Stage 6: Update world model
        if self._stages["world_model"] and len(self._summaries) >= self._world_min_evidence:
            wm_updates = self._update_world_model()
            for r in results:
                r.world_model_updates = wm_updates

        # Stage 7: Prune redundant
        if self._stages["prune_redundant"]:
            self._prune_redundant()

        return results

    # ── Stage implementations ──────────────────────────────

    def _cluster_events(self, anchor_ids: list[str]) -> list[list[str]]:
        """Group similar events into clusters by tag overlap."""
        if not anchor_ids or not self.graph:
            return []

        clusters: list[list[str]] = []
        assigned: set[str] = set()

        for aid in anchor_ids:
            if aid in assigned:
                continue
            cluster = [aid]
            assigned.add(aid)

            # Find similar anchors via graph neighbors
            if hasattr(self.graph, 'get_neighbors'):
                neighbors = self.graph.get_neighbors(aid)
                for n in neighbors:
                    nid = n[0] if isinstance(n, tuple) else getattr(n, 'target_id', '')
                    if nid in anchor_ids and nid not in assigned:
                        cluster.append(nid)
                        assigned.add(nid)

            if len(cluster) >= self._min_cluster:
                clusters.append(cluster)

        return clusters

    def _summarize_cluster(self, cluster_ids: list[str]) -> CognitiveCompressionResult:
        """Create a summary from a cluster of events."""
        # Extract common words / tags from anchors
        texts: list[str] = []
        all_tags: list[str] = []

        if self.graph:
            for aid in cluster_ids:
                try:
                    anchor = self.graph.get_anchor(aid)
                    if anchor:
                        texts.append(getattr(anchor, 'text', '')[:200])
                        all_tags.extend(getattr(anchor, 'tags', []))
                except Exception:
                    pass

        if not texts:
            return CognitiveCompressionResult(
                level=1,
                source_anchor_ids=cluster_ids,
                compressed_text=f"Summary of {len(cluster_ids)} related events",
                confidence=0.3,
                evidence_count=len(cluster_ids),
            )

        # Simple keyword-based summary
        from collections import Counter
        tag_counts = Counter(all_tags)
        top_tags = [t for t, _ in tag_counts.most_common(5)]

        summary_text = f"Cluster of {len(cluster_ids)} events about {', '.join(top_tags[:3])}: " + \
                       "; ".join(t[:100] for t in texts[:3])

        return CognitiveCompressionResult(
            level=1,
            source_anchor_ids=cluster_ids,
            compressed_text=summary_text,
            confidence=min(0.8, 0.4 + len(cluster_ids) * 0.05),
            evidence_count=len(cluster_ids),
        )

    def _extract_concepts(self) -> list[str]:
        """Extract concepts from accumulated summaries."""
        concepts: list[str] = []

        # Count recurring keywords across summaries
        from collections import Counter
        all_words: list[str] = []
        for s in self._summaries[-20:]:
            all_words.extend(s.compressed_text.lower().split())

        word_counts = Counter(w for w in all_words if len(w) > 3)
        for word, count in word_counts.most_common(20):
            if count >= self._concept_min_summaries:
                concepts.append(word)

        # Register with concept cortex if available
        if self._concept_cortex and concepts:
            for concept_label in concepts[:5]:
                self._concept_cortex.get_or_create_concept(
                    concept_label, domain="inferred"
                )

        return concepts[:10]

    def _detect_patterns(self, results: list[CognitiveCompressionResult]) -> list[str]:
        """Detect recurring patterns across compression results."""
        patterns: list[str] = []
        if len(results) < self._pattern_min:
            return patterns

        # Look for recurring keywords
        from collections import Counter
        all_tags: list[str] = []
        for r in results:
            for concept in r.concepts_formed:
                all_tags.append(concept)

        tag_counts = Counter(all_tags)
        for tag, count in tag_counts.most_common(10):
            if count >= self._pattern_min:
                patterns.append(f"recurring:{tag}")

        return patterns

    def _update_personality(self, results: list[CognitiveCompressionResult]) -> dict[str, float]:
        """Update personality model based on detected patterns."""
        # Update personality model if available
        if self._personality_model:
            try:
                for r in results:
                    for pattern in r.patterns_detected:
                        if hasattr(self._personality_model, 'update_trait'):
                            self._personality_model.update_trait(pattern, 0.01)
            except Exception:
                pass
        return {}

    def _update_world_model(self) -> dict[str, str]:
        """Form/update world model beliefs from accumulated summaries."""
        updates: dict[str, str] = {}

        # Derive beliefs from summary patterns
        if len(self._summaries) >= self._world_min_evidence:
            categories = self._categorize_summaries()
            for category, count in categories.items():
                if count >= self._world_min_evidence // 2:
                    belief_id = f"wm_{category}_{self._cycle_count}"
                    belief = WorldModelBelief(
                        id=belief_id,
                        belief=f"User frequently engages with {category} topics",
                        category=category,
                        confidence=min(0.7, count / 20),
                        supporting_evidence=[s.compressed_text[:50] for s in self._summaries[-10:]],
                    )
                    self._world_model[belief_id] = belief
                    updates[category] = belief.belief

        return updates

    def _categorize_summaries(self) -> dict[str, int]:
        """Categorize summaries by topic."""
        categories: dict[str, int] = defaultdict(int)
        for s in self._summaries[-50:]:
            text = s.compressed_text.lower()
            if any(kw in text for kw in ['code', 'debug', 'programming', 'develop']):
                categories['development'] += 1
            elif any(kw in text for kw in ['communicate', 'explain', 'document', 'write']):
                categories['communication'] += 1
            elif any(kw in text for kw in ['tool', 'setup', 'configure', 'install']):
                categories['tooling'] += 1
            elif any(kw in text for kw in ['decide', 'choose', 'prefer', 'option']):
                categories['decision_making'] += 1
            else:
                categories['general'] += 1
        return categories

    def _prune_redundant(self):
        """Remove redundant summaries below retention threshold."""
        if len(self._summaries) > 100:
            # Keep top 80% by confidence
            self._summaries.sort(key=lambda s: s.confidence, reverse=True)
            self._summaries = self._summaries[:80]

        # Prune weak world model beliefs
        for bid in list(self._world_model.keys()):
            if self._world_model[bid].confidence < 0.2:
                del self._world_model[bid]

    # ── Queries ────────────────────────────────────────────

    def get_world_model(self) -> list[WorldModelBelief]:
        return list(self._world_model.values())

    def get_compression_summary(self) -> dict:
        return {
            "total_summaries": len(self._summaries),
            "total_world_model_beliefs": len(self._world_model),
            "cycle_count": self._cycle_count,
            "avg_confidence": round(
                sum(s.confidence for s in self._summaries) / max(len(self._summaries), 1), 3
            ),
        }
