"""Event→Pattern→Identity abstraction chain for cognitive memory runtime."""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor

from .math_utils import cosine_sim

TOPIC_KEYWORDS = [
    'debugging', 'architecture', 'security', 'testing', 'deployment',
    'preference', 'coding_style', 'tooling', 'performance', 'design',
    'communication', 'learning',
]

TOPIC_ALIASES: dict[str, str] = {
    'debug': 'debugging', 'bug': 'debugging', 'error': 'debugging',
    'troubleshoot': 'debugging', 'fix': 'debugging',
    'architect': 'architecture', 'structure': 'architecture',
    'pattern': 'architecture', 'system': 'architecture',
    'secure': 'security', 'vulnerability': 'security', 'auth': 'security',
    'encrypt': 'security', 'permission': 'security',
    'test': 'testing', 'qa': 'testing', 'verify': 'testing',
    'validate': 'testing', 'assert': 'testing',
    'deploy': 'deployment', 'release': 'deployment', 'pipeline': 'deployment',
    'ci': 'deployment', 'cd': 'deployment', 'prod': 'deployment',
    'prefer': 'preference', 'like': 'preference', 'favorite': 'preference',
    'choice': 'preference', 'option': 'preference',
    'style': 'coding_style', 'convention': 'coding_style',
    'format': 'coding_style', 'lint': 'coding_style',
    'tool': 'tooling', 'ide': 'tooling', 'editor': 'tooling',
    'plugin': 'tooling', 'automation': 'tooling',
    'perf': 'performance', 'speed': 'performance', 'slow': 'performance',
    'fast': 'performance', 'optimize': 'performance', 'memory': 'performance',
    'design': 'design', 'ui': 'design', 'ux': 'design',
    'interface': 'design', 'layout': 'design',
    'communicat': 'communication', 'document': 'communication',
    'write': 'communication', 'explain': 'communication', 'message': 'communication',
    'learn': 'learning', 'study': 'learning', 'research': 'learning',
    'tutorial': 'learning', 'course': 'learning',
}

TOPIC_TRAITS: dict[str, str] = {
    'debugging': "methodical about debugging and troubleshooting",
    'architecture': "prioritizes system architecture and design",
    'security': "conscious of security implications and best practices",
    'testing': "diligent about testing and validation",
    'deployment': "experienced with deployment and operations",
    'preference': "has clear and consistent preferences",
    'coding_style': "has a distinctive coding style and conventions",
    'tooling': "proficient with development tools and automation",
    'performance': "attentive to performance and optimization",
    'design': "thoughtful about design and user experience",
    'communication': "values clear communication and documentation",
    'learning': "continuously learning and adapting to new knowledge",
}

_STOP_WORDS: set[str] = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
    'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
    'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'only', 'own', 'same', 'than', 'too', 'very', 'just', 'about', 'also',
    'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'itself', 'what', 'which', 'who', 'whom', 'when', 'where',
    'how', 'if', 'then', 'else', 'because', 'while', 'although', 'here',
    'there', 'now', 'then', 'once', 'already', 'always', 'never',
    'sometimes', 'often', 'usually', 'really', 'actually', 'still',
    'using', 'used', 'use', 'make', 'made', 'get', 'got', 'need', 'needs',
    'needed', 'want', 'wants', 'wanted', 'know', 'known', 'see', 'seen',
    'take', 'took', 'come', 'came', 'go', 'went', 'say', 'said',
    'one', 'two', 'three', 'first', 'second', 'third',
}


class AbstractionLevel(str, Enum):
    """Level in the abstraction chain — event through identity."""
    EVENT = "event"
    SUMMARY = "summary"
    PATTERN = "pattern"
    IDENTITY = "identity"


@dataclass
class AbstractionNode:
    """A node at any level in the abstraction chain."""

    id: str
    level: AbstractionLevel
    text: str
    source_ids: list[str]
    confidence: float
    created_at: float
    evidence_count: int
    last_updated: float

    @property
    def is_stable(self) -> bool:
        return self.confidence >= 0.7


@dataclass
class AbstractionChainConfig:
    """Triggers and thresholds for each abstraction level."""

    summary_trigger: int = 100
    pattern_trigger: int = 10
    identity_trigger: int = 5
    min_confidence: float = 0.6
    llm_enabled: bool = False
    llm_model: str = ""
    max_chain_depth: int = 4
    topic_similarity_threshold: float = 0.7


class AbstractionChain:
    """Event→Pattern→Identity abstraction chain for cognitive memory."""

    def __init__(self, graph: StarGraph,
                 config: AbstractionChainConfig | None = None,
                 embedder=None, llm_fn=None):
        self.graph = graph
        self.config = config or AbstractionChainConfig()
        self.embedder = embedder
        self.llm_fn = llm_fn

        self._nodes: dict[str, AbstractionNode] = {}
        self._topic_events: dict[str, list[str]] = defaultdict(list)
        self._topic_summaries: dict[str, list[str]] = defaultdict(list)
        self._pattern_ids: list[str] = []
        self._identity_ids: list[str] = []
        self._source_to_derived: dict[str, list[str]] = defaultdict(list)

    # ── Public API ───────────────────────────────────────────

    def ingest_event(self, anchor_id: str) -> AbstractionNode | None:
        """Process a new event, trigger summarization if threshold met."""
        anchor = self.graph.anchors.get(anchor_id)
        if anchor is None:
            return None
        topic = self._infer_topic(anchor.text, anchor.tags)
        self._topic_events[topic].append(anchor_id)
        if len(self._topic_events[topic]) >= self.config.summary_trigger:
            return self._summarize_topic(topic)
        return None

    def check_triggers(self) -> dict[AbstractionLevel, list[AbstractionNode]]:
        """Check all levels for trigger thresholds, return new nodes created."""
        result: dict[AbstractionLevel, list[AbstractionNode]] = {
            AbstractionLevel.SUMMARY: [],
            AbstractionLevel.PATTERN: [],
            AbstractionLevel.IDENTITY: [],
        }

        for topic, event_ids in list(self._topic_events.items()):
            if len(event_ids) >= self.config.summary_trigger:
                node = self._summarize_topic(topic)
                if node:
                    result[AbstractionLevel.SUMMARY].append(node)

        for topic, summary_ids in list(self._topic_summaries.items()):
            if len(summary_ids) >= self.config.pattern_trigger:
                node = self._derive_pattern(topic, summary_ids)
                if node:
                    self._topic_summaries[topic] = self._topic_summaries[topic][self.config.pattern_trigger:]
                    result[AbstractionLevel.PATTERN].append(node)

        if len(self._pattern_ids) >= self.config.identity_trigger:
            node = self._derive_identity(self._pattern_ids)
            if node:
                self._pattern_ids = self._pattern_ids[self.config.identity_trigger:]
                result[AbstractionLevel.IDENTITY].append(node)

        return result

    def summarize_topic(self, topic: str, source_ids: list[str]) -> AbstractionNode | None:
        """Create a summary from related events on the same topic."""
        texts = []
        now = time.time()
        for aid in source_ids:
            anchor = self.graph.anchors.get(aid)
            if anchor:
                texts.append(anchor.text)
        if not texts:
            return None
        abstract_text = self._generate_summary(texts, topic)
        confidence = min(0.5 + 0.05 * len(texts), 0.95)
        node = self._create_node(
            level=AbstractionLevel.SUMMARY,
            text=abstract_text,
            source_ids=source_ids,
            confidence=confidence,
            topic=topic,
        )
        if node:
            self._topic_summaries[topic].append(node.id)
        return node

    def derive_pattern(self, summary_ids: list[str]) -> AbstractionNode | None:
        """Derive a pattern from related summaries."""
        texts = []
        tags_list: list[list[str]] = []
        for sid in summary_ids:
            node = self._nodes.get(sid)
            if node:
                texts.append(node.text)
            anchor = self.graph.anchors.get(sid)
            if anchor:
                tags_list.append(anchor.tags)
        if not texts:
            return None
        topic = self._infer_best_topic(texts, tags_list)
        abstract_text = self._generate_pattern(texts, topic)
        confidence = min(0.4 + 0.06 * len(texts), 0.9)
        node = self._create_node(
            level=AbstractionLevel.PATTERN,
            text=abstract_text,
            source_ids=summary_ids,
            confidence=confidence,
            topic=topic,
        )
        if node:
            self._pattern_ids.append(node.id)
        return node

    def derive_identity(self, pattern_ids: list[str]) -> AbstractionNode | None:
        """Derive an identity trait from related patterns."""
        texts = []
        topics: list[str] = []
        for pid in pattern_ids:
            node = self._nodes.get(pid)
            if node:
                texts.append(node.text)
            anchor = self.graph.anchors.get(pid)
            if anchor:
                topics.extend(anchor.tags)
        if not texts:
            return None
        abstract_text = self._generate_identity(texts, topics)
        confidence = min(0.35 + 0.07 * len(texts), 0.85)
        node = self._create_node(
            level=AbstractionLevel.IDENTITY,
            text=abstract_text,
            source_ids=pattern_ids,
            confidence=confidence,
            topic="identity",
        )
        if node:
            self._identity_ids.append(node.id)
        return node

    def get_chain(self, anchor_id: str) -> list[AbstractionNode]:
        """Follow the chain from an event up through summaries/patterns to identity."""
        chain: list[AbstractionNode] = []
        current_id = anchor_id
        while current_id:
            derived = self._source_to_derived.get(current_id, [])
            if not derived:
                break
            next_id = derived[0]
            node = self._nodes.get(next_id)
            if node is None:
                break
            chain.append(node)
            current_id = next_id
        return chain

    def get_identity_traits(self) -> list[AbstractionNode]:
        """All current identity-level abstractions."""
        return [self._nodes[iid] for iid in self._identity_ids if iid in self._nodes]

    def get_patterns(self) -> list[AbstractionNode]:
        """All current pattern-level abstractions."""
        return [self._nodes[pid] for pid in self._pattern_ids if pid in self._nodes]

    def get_summaries(self) -> list[AbstractionNode]:
        """All current summary-level abstractions."""
        result: list[AbstractionNode] = []
        for nid, node in self._nodes.items():
            if node.level == AbstractionLevel.SUMMARY:
                result.append(node)
        return result

    def compress_topic(self, topic: str, max_events: int = 100) -> int:
        """Compress events in a topic into summaries, return count compressed."""
        event_ids = self._topic_events.get(topic, [])
        if not event_ids:
            return 0
        to_compress = event_ids[:max_events]
        compressed = 0
        chunk_size = max(5, self.config.summary_trigger // 2)
        for i in range(0, len(to_compress), chunk_size):
            chunk = to_compress[i:i + chunk_size]
            node = self.summarize_topic(topic, chunk)
            if node:
                compressed += len(chunk)
        self._topic_events[topic] = event_ids[len(to_compress):]
        return compressed

    def snapshot(self) -> dict:
        """Chain statistics per level."""
        events_total = sum(len(v) for v in self._topic_events.values())
        summaries_total = sum(len(v) for v in self._topic_summaries.values())
        return {
            "events_pending": events_total,
            "summaries_total": summaries_total,
            "patterns_total": len(self._pattern_ids),
            "identity_traits_total": len(self._identity_ids),
            "nodes_total": len(self._nodes),
            "topics_active": len(self._topic_events),
        }

    # ── Internal methods ─────────────────────────────────────

    def _infer_topic(self, text: str, tags: list[str]) -> str:
        """Infer topic from text content and tags."""
        text_lower = text.lower()
        tag_lower = [t.lower() for t in tags]
        combined = text_lower + " " + " ".join(tag_lower)

        scores: dict[str, int] = defaultdict(int)
        for keyword in TOPIC_KEYWORDS:
            if keyword in combined:
                scores[keyword] += 2
            for tag in tag_lower:
                if keyword in tag or tag in keyword:
                    scores[keyword] += 1

        for word in combined.split():
            w = word.strip(".,!?;:\"'()[]{}")
            if w in TOPIC_ALIASES:
                scores[TOPIC_ALIASES[w]] += 1

        if scores:
            return max(scores, key=lambda k: scores[k])
        return "general"

    def _infer_best_topic(self, texts: list[str],
                          tags_list: list[list[str]]) -> str:
        """Infer the most common topic across multiple texts."""
        topics = []
        for i, text in enumerate(texts):
            tags = tags_list[i] if i < len(tags_list) else []
            topics.append(self._infer_topic(text, tags))
        if not topics:
            return "general"
        counter = Counter(topics)
        return counter.most_common(1)[0][0]

    def _summarize_topic(self, topic: str) -> AbstractionNode | None:
        """Internal: create a summary from pending events on this topic."""
        event_ids = self._topic_events.get(topic, [])
        if len(event_ids) < self.config.summary_trigger:
            return None
        chunk = event_ids[:self.config.summary_trigger]
        node = self.summarize_topic(topic, chunk)
        if node:
            self._topic_events[topic] = event_ids[self.config.summary_trigger:]
        return node

    def _derive_pattern(self, topic: str,
                        summary_ids: list[str]) -> AbstractionNode | None:
        """Internal: derive a pattern from summaries on a topic."""
        if len(summary_ids) < self.config.pattern_trigger:
            return None
        chunk = summary_ids[:self.config.pattern_trigger]
        return self.derive_pattern(chunk)

    def _derive_identity(self, pattern_ids: list[str]) -> AbstractionNode | None:
        """Internal: derive an identity trait from patterns."""
        if len(pattern_ids) < self.config.identity_trigger:
            return None
        chunk = pattern_ids[:self.config.identity_trigger]
        return self.derive_identity(chunk)

    def _create_node(self, level: AbstractionLevel, text: str,
                     source_ids: list[str], confidence: float,
                     topic: str = "") -> AbstractionNode:
        """Create an AbstractionNode and store it in the graph."""
        now = time.time()
        node_id = hashlib.blake2b(
            (text + str(now)).encode(), digest_size=8
        ).hexdigest()

        evidence_count = len(source_ids)
        if confidence < self.config.min_confidence:
            confidence = self.config.min_confidence

        node = AbstractionNode(
            id=node_id,
            level=level,
            text=text,
            source_ids=list(source_ids),
            confidence=confidence,
            created_at=now,
            evidence_count=evidence_count,
            last_updated=now,
        )
        self._nodes[node_id] = node

        tags = ["__abstraction__", f"__{level.value}__"]
        if topic:
            tags.append(topic)

        from .memory_core.anchor import Anchor
        anchor = Anchor.create(
            text=f"[{level.value.upper()}] {text}",
            source_session="abstraction_chain",
            tags=tags,
            importance=confidence,
        )
        self.graph.add_anchor(anchor)

        for sid in source_ids:
            if sid in self.graph.anchors:
                self.graph.add_edge(
                    src=sid, tgt=anchor.id,
                    weight=confidence,
                    edge_type="derived_from",
                    source_type="inferred",
                )
            self._source_to_derived[sid].append(node_id)

        return node

    # ── Non-LLM abstraction generators ──────────────────────

    def _generate_summary(self, texts: list[str], topic: str) -> str:
        """Generate a summary text from event texts (no LLM)."""
        if self.config.llm_enabled and self.llm_fn:
            return self.llm_fn("summary", texts, topic)
        phrases = []
        for t in texts:
            phrases.extend(self._extract_key_phrases(t))
        unique = list(dict.fromkeys(phrases))[:8]
        points = "; ".join(unique) if unique else "various related events"
        return f"[{len(texts)}] related events about {topic}: {points}"

    def _generate_pattern(self, texts: list[str], topic: str) -> str:
        """Generate a pattern text from summary texts (no LLM)."""
        if self.config.llm_enabled and self.llm_fn:
            return self.llm_fn("pattern", texts, topic)
        all_words: list[str] = []
        for t in texts:
            words = [w.lower().strip(".,!?;:\"'()[]{}") for w in t.split()
                     if len(w) > 4 and w.lower() not in _STOP_WORDS]
            all_words.extend(words)
        counter = Counter(all_words)
        top = counter.most_common(5)
        if not top:
            return f"Recurring pattern in {topic}: consistent themes across {len(texts)} summaries"
        parts = [f'"{kw}" appears in {min(cnt, len(texts))} summaries'
                 for kw, cnt in top]
        return f"Recurring pattern in {topic}: {'; '.join(parts)}"

    def _generate_identity(self, texts: list[str], topics: list[str]) -> str:
        """Generate an identity trait text from pattern texts (no LLM)."""
        if self.config.llm_enabled and self.llm_fn:
            return self.llm_fn("identity", texts, topics)
        trait_parts: list[str] = []
        seen_traits: set[str] = set()
        for t in topics:
            if t in TOPIC_TRAITS and t not in seen_traits:
                trait_parts.append(TOPIC_TRAITS[t])
                seen_traits.add(t)
        for t in texts:
            for kw, trait in TOPIC_TRAITS.items():
                if kw in t.lower() and trait not in seen_traits:
                    trait_parts.append(trait)
                    seen_traits.add(trait)
        if not trait_parts:
            topic_set = set(topics)
            for kw, alias in TOPIC_ALIASES.items():
                if alias in topic_set and TOPIC_TRAITS.get(alias, "") not in seen_traits:
                    trait_parts.append(TOPIC_TRAITS.get(alias, ""))
                    seen_traits.add(TOPIC_TRAITS[alias])
        if not trait_parts:
            return f"User demonstrates consistent behavior across {len(texts)} recurring patterns"
        combined = ", suggesting they are ".join(trait_parts[:3])
        return f"User consistently demonstrates traits: {combined}"

    @staticmethod
    def _extract_key_phrases(text: str, max_phrases: int = 5) -> list[str]:
        """Extract key phrases from text for summarization."""
        words = text.split()
        phrases: list[str] = []
        i = 0
        while i < len(words) and len(phrases) < max_phrases:
            w = words[i].lower().strip(".,!?;:\"'()[]{}")
            if len(w) > 4 and w not in _STOP_WORDS:
                phrase = words[i].strip(".,!?;:\"'()[]{}")
                j = i + 1
                while j < len(words) and j < i + 3:
                    nxt = words[j].lower().strip(".,!?;:\"'()[]{}")
                    if nxt in _STOP_WORDS:
                        j += 1
                        continue
                    phrase += " " + words[j].strip(".,!?;:\"'()[]{}")
                    j += 1
                    break
                phrases.append(phrase)
            i += 1
        return phrases
