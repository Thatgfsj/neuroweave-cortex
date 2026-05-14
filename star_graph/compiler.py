"""Cognitive Compiler — full-chain worldview emergence from raw messages.

Pipeline:
  1000 raw messages → Sleep compression → 200 episodic summaries
  200 episodic → AbstractiveMemoryEngine → 20 concept nodes
  20 concepts → Cross-session pattern merge → 5 worldview nodes
  5 worldviews → Core profile extraction → 1 user profile

Each level has its own compression ratio and stability threshold, producing
increasingly stable and abstract representations of the user and their work.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import Config


@dataclass
class WorldviewNode:
    """A long-term stable belief about the user, formed from cross-session patterns.

    Worldview nodes represent what the system "understands" about the user —
    preferences, habits, expertise, values, and working style. They are the
    most stable and highest-level abstractions in the system.
    """

    id: str
    label: str
    description: str
    source_concept_ids: list[str]
    confidence: float = 0.5
    stability: float = 0.5
    evidence_count: int = 0
    formed_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    domain: str = ""
    worldview_type: str = "belief"  # belief, preference, habit, expertise, value

    @property
    def is_stable(self) -> bool:
        return self.stability > 0.7 and self.evidence_count >= 2

    def reinforce(self):
        self.evidence_count += 1
        self.last_reinforced_at = time.time()
        self.stability = min(1.0, self.stability + 0.10)
        self.confidence = min(1.0, self.confidence + 0.05)

    def weaken(self):
        self.stability = max(0.05, self.stability - 0.15)
        self.confidence = max(0.05, self.confidence - 0.10)

    def degrade(self, half_life_days: float = 90.0) -> float:
        hours = (time.time() - self.last_reinforced_at) / 3600
        decay = math.exp(-hours * math.log(2) / (half_life_days * 24))
        self.stability = max(0.01, self.stability * decay)
        return self.stability


@dataclass
class UserProfile:
    """The system's synthesized understanding of the user.

    Extracted from worldview consensus — the single most compressed
    representation of who the user is and how they work.
    """

    id: str
    summary: str
    preferences: list[str]
    expertise_areas: list[str]
    working_style: str
    values: list[str]
    habits: list[str]
    source_worldview_ids: list[str]
    confidence: float = 0.5
    formed_at: float = field(default_factory=time.time)
    version: int = 1

    @classmethod
    def from_worldviews(cls, worldviews: list[WorldviewNode],
                        worldview_id: str = "") -> UserProfile:
        """Synthesize a user profile from a set of worldview nodes."""
        preferences: list[str] = []
        expertise: list[str] = []
        working_styles: list[str] = []
        values: list[str] = []
        habits: list[str] = []

        for wv in worldviews:
            if wv.worldview_type == "preference":
                preferences.append(wv.description)
            elif wv.worldview_type == "expertise":
                expertise.append(wv.description)
            elif wv.worldview_type == "value":
                values.append(wv.description)
            elif wv.worldview_type == "habit":
                habits.append(wv.description)
            else:
                working_styles.append(wv.description)

        confidences = [wv.confidence for wv in worldviews]
        avg_confidence = sum(confidences) / max(1, len(confidences))

        summary_parts = []
        if expertise:
            summary_parts.append(f"Expert in: {', '.join(expertise[:3])}")
        if preferences:
            summary_parts.append(f"Prefers: {', '.join(preferences[:3])}")
        if working_styles:
            summary_parts.append(f"Works: {', '.join(working_styles[:2])}")
        if values:
            summary_parts.append(f"Values: {', '.join(values[:2])}")

        return cls(
            id=worldview_id or f"profile_{int(time.time())}",
            summary=". ".join(summary_parts) if summary_parts else "Profile forming...",
            preferences=preferences,
            expertise_areas=expertise,
            working_style="; ".join(working_styles) if working_styles else "undetermined",
            values=values,
            habits=habits,
            source_worldview_ids=[wv.id for wv in worldviews],
            confidence=avg_confidence,
        )


class CognitiveCompiler:
    """Orchestrates the full cognitive compression pipeline.

    Runs the chain:
      Raw anchors → Episodic summaries → Concept nodes → Worldviews → User profile

    Usage:
        compiler = CognitiveCompiler()
        result = compiler.compile(graph, session_groups)
        # result["worldviews"] → list of WorldviewNode
        # result["profile"] → UserProfile
    """

    def __init__(self):
        c = Config.get()
        comp_cfg = getattr(c, 'compiler', None) or {}

        self.episodic_ratio = getattr(comp_cfg, 'episodic_ratio', 0.20)
        self.concept_ratio = getattr(comp_cfg, 'concept_ratio', 0.10)
        self.worldview_ratio = getattr(comp_cfg, 'worldview_ratio', 0.25)
        self.min_worldview_confidence = getattr(comp_cfg, 'min_worldview_confidence', 0.4)
        self.min_worldview_evidence = getattr(comp_cfg, 'min_worldview_evidence', 2)
        self.worldview_half_life_days = getattr(comp_cfg, 'worldview_half_life_days', 90.0)
        self.similarity_threshold = getattr(comp_cfg, 'similarity_threshold', 0.55)
        self.max_worldviews = getattr(comp_cfg, 'max_worldviews', 20)

        self.worldviews: dict[str, WorldviewNode] = {}
        self._profile: UserProfile | None = None
        self._counter = 0

    # ── Full compilation pipeline ───────────────────────────

    def compile(self, graph, session_groups: dict[str, list] | None = None
                ) -> dict:
        """Run the full cognitive compilation pipeline.

        Args:
            graph: StarGraph instance
            session_groups: Optional dict of session_id → list of Anchor.
                            If None, anchors are grouped by source_session from graph.

        Returns:
            Dict with keys: episodic, concepts, worldviews, profile, stats
        """
        # Step 0: Prepare session groups
        if session_groups is None:
            session_groups = defaultdict(list)
            for aid, anchor in graph.anchors.items():
                if anchor.embedding and anchor.source_session:
                    session_groups[anchor.source_session].append(anchor)

        total_raw = sum(len(v) for v in session_groups.values())

        # Step 1: RAW → EPISODIC (SessionCompressor)
        episodic = self._compile_episodic(graph, session_groups)

        # Step 2: EPISODIC → CONCEPTS (AbstractiveMemoryEngine)
        concepts = self._compile_concepts(graph, episodic)

        # Step 3: CONCEPTS → WORLDVIEWS
        worldviews = self._compile_worldviews(concepts)

        # Step 4: WORLDVIEWS → USER PROFILE
        profile = self._compile_profile()

        return {
            "episodic": episodic,
            "concepts": concepts,
            "worldviews": worldviews,
            "profile": profile,
            "stats": {
                "raw_count": total_raw,
                "episodic_count": len(episodic),
                "concept_count": len(concepts),
                "worldview_count": len(worldviews),
                "profile_version": profile.version if profile else 0,
                "compression_chain": f"{total_raw}→{len(episodic)}→{len(concepts)}→{len(worldviews)}→1",
            },
        }

    def _compile_episodic(self, graph,
                          session_groups: dict[str, list]) -> list:
        """Step 1: Compress raw anchors within sessions into episodic summaries."""
        from .compression import SessionCompressor

        compressor = SessionCompressor()
        all_episodic = []

        for session_id, anchors in session_groups.items():
            if len(anchors) < 3:
                continue
            summaries = compressor.compress(anchors, session_id)
            all_episodic.extend(summaries)

            # Add compressed summaries to graph
            for summary in summaries:
                proxy = summary.to_anchor_proxy()
                graph.add_anchor(proxy)
                for src_id in summary.source_anchor_ids:
                    if src_id in graph.anchors:
                        graph.add_edge(
                            proxy.id, src_id,
                            weight=summary.confidence * 0.6,
                            edge_type="compresses",
                        )

        return all_episodic

    def _compile_concepts(self, graph, episodic_summaries: list) -> list:
        """Step 2: Cluster episodic summaries into concept nodes."""
        from .abstraction import AbstractiveMemoryEngine

        if len(episodic_summaries) < 3:
            return []

        engine = AbstractiveMemoryEngine(
            min_occurrences=2,
            similarity_threshold=self.similarity_threshold,
        )

        # Extract cross-session patterns from the graph
        patterns = engine.extract_patterns(graph)

        # Promote stable patterns to concept nodes
        concepts = engine.promote_stable_patterns(graph)

        # Also try running the MultiLevelCompressor strategic level
        from .compression import MultiLevelCompressor
        mcomp = MultiLevelCompressor()
        strategic = mcomp.compress_strategic(episodic_summaries)
        if strategic:
            mcomp.add_to_graph(graph, strategic, edge_type="concept_of")

        return concepts + strategic

    def _compile_worldviews(self, concepts: list) -> list[WorldviewNode]:
        """Step 3: Merge concept nodes into worldview nodes.

        Groups concepts by domain and extracts stable beliefs about the user.
        Concepts that share tags/domains and are semantically similar get merged
        into worldview nodes.
        """
        if not concepts:
            return []

        # Cluster concepts by domain and similarity
        clusters = self._cluster_by_domain(concepts)

        new_worldviews = []
        for domain, cluster_concepts in clusters.items():
            if len(cluster_concepts) < 2:
                continue

            worldview = self._synthesize_worldview(domain, cluster_concepts)
            if worldview.confidence >= self.min_worldview_confidence:
                self.worldviews[worldview.id] = worldview
                new_worldviews.append(worldview)

        # Enforce max worldviews limit
        if len(self.worldviews) > self.max_worldviews:
            sorted_wvs = sorted(
                self.worldviews.items(),
                key=lambda x: (x[1].stability * x[1].confidence * x[1].evidence_count),
            )
            for wv_id, _ in sorted_wvs[:len(self.worldviews) - self.max_worldviews]:
                del self.worldviews[wv_id]

        return new_worldviews

    def _compile_profile(self) -> UserProfile | None:
        """Step 4: Synthesize user profile from worldview consensus."""
        stable_worldviews = [wv for wv in self.worldviews.values()
                            if wv.is_stable]
        if not stable_worldviews:
            return self._profile

        profile = UserProfile.from_worldviews(
            stable_worldviews,
            worldview_id=self._profile.id if self._profile else "",
        )

        if self._profile:
            profile.version = self._profile.version + 1

        self._profile = profile
        return profile

    # ── Clustering and synthesis ────────────────────────────

    def _cluster_by_domain(self, concepts) -> dict[str, list]:
        """Cluster concepts by their inferred domain."""
        clusters: dict[str, list] = defaultdict(list)

        for concept in concepts:
            domain = self._infer_domain_from_concept(concept)
            clusters[domain].append(concept)

        return dict(clusters)

    def _infer_domain_from_concept(self, concept) -> str:
        """Infer domain from a concept (SummaryAnchor, AbstractNode, or PatternMemory)."""
        tags = getattr(concept, 'tags', [])
        text = getattr(concept, 'text', '') or getattr(concept, 'description', '')
        label = getattr(concept, 'label', '')

        combined = f"{' '.join(tags)} {text} {label}".lower()

        domain_keywords = {
            "python_development": {"python", "flask", "django", "fastapi", "pip", "pytest"},
            "javascript_development": {"javascript", "node", "react", "vue", "npm", "typescript"},
            "devops": {"docker", "kubernetes", "deploy", "pipeline", "terraform", "ci/cd"},
            "database": {"sql", "mysql", "postgresql", "redis", "mongo", "query"},
            "debugging": {"bug", "error", "fix", "debug", "trace", "exception"},
            "architecture": {"architecture", "design", "pattern", "component", "module"},
            "security": {"auth", "token", "permission", "encrypt", "vulnerability"},
            "testing": {"test", "unit", "integration", "mock", "assert"},
            "performance": {"optimize", "performance", "cache", "latency", "benchmark"},
            "user_preferences": {"prefer", "like", "style", "habit", "workflow"},
        }

        scores: dict[str, int] = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _synthesize_worldview(self, domain: str,
                               concepts: list) -> WorldviewNode:
        """Synthesize a worldview node from a cluster of concepts."""
        self._counter += 1

        # Determine worldview type
        worldview_type = self._infer_worldview_type(concepts, domain)

        # Build description from concept texts
        texts = []
        for c in concepts:
            t = getattr(c, 'pattern_text', '') or getattr(c, 'text', '') or getattr(c, 'description', '')
            if t:
                texts.append(t)

        # Extract key terms and generate label
        key_terms = self._extract_key_terms(texts)
        label = key_terms[0] if key_terms else f"{domain} understanding"

        # Generate description
        if worldview_type == "preference":
            description = f"User prefers {self._summarize_texts(texts, 'working with')}"
        elif worldview_type == "expertise":
            description = f"User has expertise in {self._summarize_texts(texts, 'knowledge of')}"
        elif worldview_type == "habit":
            description = f"User habitually {self._summarize_texts(texts, 'tends to')}"
        else:
            description = f"Belief about {domain}: {self._summarize_texts(texts, 'patterns in')}"

        # Confidence from concept consistency
        confidences = [getattr(c, 'confidence', 0.5) for c in concepts]
        avg_confidence = sum(confidences) / max(1, len(confidences))
        # More concepts = higher confidence
        confidence = min(1.0, avg_confidence * (1.0 + 0.1 * min(len(concepts) - 1, 5)))

        # Collect all tags
        all_tags: list[str] = []
        for c in concepts:
            t = getattr(c, 'tags', [])
            if t:
                all_tags.extend(t)
        tag_counter = Counter(all_tags)
        common_tags = [t for t, c in tag_counter.most_common(5) if c >= 2
                      and not t.startswith("domain:") and not t.startswith("level:")]

        return WorldviewNode(
            id=f"worldview_{self._counter}",
            label=label[:120],
            description=description[:500],
            source_concept_ids=[getattr(c, 'id', '') for c in concepts],
            confidence=confidence,
            stability=0.3 + avg_confidence * 0.3,
            evidence_count=len(concepts),
            tags=common_tags,
            domain=domain,
            worldview_type=worldview_type,
        )

    def _infer_worldview_type(self, concepts: list, domain: str) -> str:
        """Infer the worldview type from concept content."""
        combined = " ".join([
            getattr(c, 'pattern_text', '') or getattr(c, 'text', '') or getattr(c, 'description', '')
            for c in concepts
        ]).lower()

        if any(w in combined for w in ("prefer", "like", "dislike", "favorite")):
            return "preference"
        elif any(w in combined for w in ("expert", "experienced", "skilled", "proficient", "knowledge")):
            return "expertise"
        elif any(w in combined for w in ("always", "usually", "habit", "tend", "routine")):
            return "habit"
        elif any(w in combined for w in ("value", "important", "priority", "principle", "believe")):
            return "value"
        return "belief"

    def _summarize_texts(self, texts: list[str], context_prefix: str) -> str:
        """Generate a concise summary from concept texts."""
        if not texts:
            return f"various {context_prefix} patterns"

        # Extract the most common noun phrases
        key_terms = self._extract_key_terms(texts)
        if not key_terms:
            return f"general {context_prefix} across sessions"

        if len(key_terms) <= 2:
            return f"{context_prefix} {key_terms[0]}"
        return f"{context_prefix} {', '.join(key_terms[:3])}"

    @staticmethod
    def _extract_key_terms(texts: list[str], top_k: int = 5) -> list[str]:
        """TF-IDF style key term extraction."""
        STOP_WORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "and", "or", "but", "it", "its",
            "this", "that", "these", "those", "not", "no", "so", "as", "if",
            "then", "than", "too", "very", "just", "about", "also",
        }

        import re
        N = len(texts)
        doc_freq: Counter = Counter()
        term_freq: Counter = Counter()

        for text in texts:
            tokens = re.findall(r'[a-z0-9]+', text.lower())
            tokens = [t for t in tokens if len(t) > 3 and t not in STOP_WORDS]
            for t in set(tokens):
                doc_freq[t] += 1
            for t in tokens:
                term_freq[t] += 1

        scored = []
        for term, tf in term_freq.items():
            df = doc_freq.get(term, 1)
            idf = math.log((N + 1) / (df + 1)) + 1.0
            scored.append((term, tf * idf))

        scored.sort(key=lambda x: -x[1])
        return [term for term, _ in scored[:top_k]]

    # ── Maintenance ─────────────────────────────────────────

    def degrade_worldviews(self) -> int:
        """Apply decay to all worldviews. Returns count below threshold."""
        to_remove = []
        for wv_id, wv in self.worldviews.items():
            remaining = wv.degrade(self.worldview_half_life_days)
            if remaining < 0.02:
                to_remove.append(wv_id)
        for wv_id in to_remove:
            del self.worldviews[wv_id]
        return len(to_remove)

    def get_stable_worldviews(self, min_stability: float = 0.5) -> list[WorldviewNode]:
        return sorted(
            [wv for wv in self.worldviews.values() if wv.stability >= min_stability],
            key=lambda wv: -(wv.stability * wv.confidence),
        )

    @property
    def profile(self) -> UserProfile | None:
        return self._profile

    @property
    def stats(self) -> dict:
        wv_types = Counter(wv.worldview_type for wv in self.worldviews.values())
        return {
            "total_worldviews": len(self.worldviews),
            "stable_worldviews": sum(1 for wv in self.worldviews.values() if wv.is_stable),
            "worldview_types": dict(wv_types),
            "avg_confidence": round(
                sum(wv.confidence for wv in self.worldviews.values()) / max(1, len(self.worldviews)), 3),
            "avg_stability": round(
                sum(wv.stability for wv in self.worldviews.values()) / max(1, len(self.worldviews)), 3),
            "profile_version": self._profile.version if self._profile else 0,
        }
