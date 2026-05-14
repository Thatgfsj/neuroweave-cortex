"""Versioned Memory — cognitive trajectory tracking for belief evolution.

Tracks how beliefs evolve over time:
  用户喜欢Python → 用户偏向AI开发 → 用户研究认知架构

Instead of isolated duplicate cognitive nodes, beliefs form a linked chain
of versions. Each new observation supersedes or refines the previous belief.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BeliefVersion:
    """One version of a belief about a topic."""
    id: str
    topic: str                          # normalized topic key
    text: str                           # the belief statement
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    stability: float = 0.3
    evidence_anchor_ids: list[str] = field(default_factory=list)
    superseded_by: str = ""             # id of newer version that replaced this
    supersedes: str = ""                # id of older version this replaced
    version_number: int = 1
    source_session: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def is_current(self) -> bool:
        return self.superseded_by == ""

    @property
    def is_initial(self) -> bool:
        return self.supersedes == ""


class CognitiveTrajectory:
    """Tracks the evolution of beliefs across time.

    Usage:
        ct = CognitiveTrajectory()
        ct.record_belief("user_language", "用户喜欢Python", confidence=0.6)
        ct.record_belief("user_language", "用户偏向AI开发用Python", confidence=0.7)
        chain = ct.get_trajectory("user_language")
        current = ct.get_current_belief("user_language")
    """

    # Patterns that indicate a belief is being updated/refined
    _REFINEMENT_SIGNALS = [
        r'(?:now|currently|actually|really|actually now)\s+.+',
        r'(?:现在|其实|实际上|最近|目前)\s+.+',
        r'(?:no longer|not anymore|stopped|switched to)\s+.+',
        r'(?:不再|已经不|换了|改成了)\s+.+',
        r'(?:prefers?|likes?|uses?|works? with)\s+.+',
        r'(?:偏好|喜欢|使用|从事)\s+.+',
    ]

    def __init__(self, similarity_threshold: float = 0.45):
        self._beliefs: dict[str, BeliefVersion] = {}
        self._topic_index: dict[str, list[str]] = defaultdict(list)  # topic → [version_ids]
        self.similarity_threshold = similarity_threshold
        self._total_recorded = 0
        self._total_superseded = 0

    # ── Topic normalization ─────────────────────────────────

    @staticmethod
    def normalize_topic(text: str) -> str:
        """Normalize a belief text into a topic key."""
        text_lower = text.lower().strip()
        # Remove common filler
        text_lower = re.sub(r'\b(the|a|an|is|are|was|were|has|have|had|的|了|是|在|有)\b', '', text_lower)
        text_lower = re.sub(r'\s+', '_', text_lower)[:60]
        return text_lower

    @staticmethod
    def _topic_hash(topic: str) -> str:
        return hashlib.md5(topic.lower().encode()).hexdigest()[:12]

    # ── Core API ────────────────────────────────────────────

    def record_belief(self, topic: str, text: str, *,
                      confidence: float = 0.5,
                      evidence_anchor_ids: list[str] | None = None,
                      source_session: str = "",
                      tags: list[str] | None = None) -> BeliefVersion:
        """Record a new belief. Auto-versions if topic already exists."""
        topic_key = self.normalize_topic(topic)
        existing = self.get_current_belief(topic)

        belief_id = f"bv_{self._topic_hash(topic_key)}_{int(time.time())}"

        version = BeliefVersion(
            id=belief_id,
            topic=topic_key,
            text=text,
            confidence=confidence,
            evidence_anchor_ids=evidence_anchor_ids or [],
            source_session=source_session,
            tags=tags or [],
        )

        if existing:
            # Check if this supersedes the existing belief
            if self._is_superseding(existing.text, text):
                existing.superseded_by = belief_id
                version.supersedes = existing.id
                version.version_number = existing.version_number + 1
                version.stability = max(0.3, existing.stability + 0.1)
                self._total_superseded += 1

        self._beliefs[belief_id] = version
        self._topic_index[topic_key].append(belief_id)
        self._total_recorded += 1
        return version

    def get_current_belief(self, topic: str) -> BeliefVersion | None:
        """Get the latest (non-superseded) belief for a topic."""
        topic_key = self.normalize_topic(topic)
        version_ids = self._topic_index.get(topic_key, [])
        for vid in reversed(version_ids):
            belief = self._beliefs.get(vid)
            if belief and belief.is_current:
                return belief
        return None

    def get_trajectory(self, topic: str) -> list[BeliefVersion]:
        """Get the full evolution chain for a topic (oldest first)."""
        topic_key = self.normalize_topic(topic)
        version_ids = self._topic_index.get(topic_key, [])
        versions = [self._beliefs[vid] for vid in version_ids if vid in self._beliefs]
        versions.sort(key=lambda v: v.version_number)
        return versions

    def get_all_current_beliefs(self) -> list[BeliefVersion]:
        """Get all current (non-superseded) beliefs."""
        return [b for b in self._beliefs.values() if b.is_current]

    def get_recent_changes(self, hours: float = 168.0) -> list[BeliefVersion]:
        """Beliefs that were superseded or created recently."""
        cutoff = time.time() - hours * 3600
        recent = []
        for b in self._beliefs.values():
            if b.timestamp >= cutoff:
                recent.append(b)
        recent.sort(key=lambda b: -b.timestamp)
        return recent

    # ── Detection ───────────────────────────────────────────

    def _is_superseding(self, old_text: str, new_text: str) -> bool:
        """Heuristic: does new_text represent an evolution of old_text?"""
        old_lower = old_text.lower()
        new_lower = new_text.lower()

        # High word overlap → same topic, likely evolution
        old_words = set(re.findall(r'[a-z一-鿿]{2,}', old_lower))
        new_words = set(re.findall(r'[a-z一-鿿]{2,}', new_lower))
        if not old_words or not new_words:
            return False
        overlap = len(old_words & new_words) / min(len(old_words), len(new_words))
        if overlap >= self.similarity_threshold:
            return True

        # Check refinement signals
        for pattern in self._REFINEMENT_SIGNALS:
            if re.search(pattern, new_lower):
                return True

        return False

    def detect_from_graph(self, graph) -> list[BeliefVersion]:
        """Scan graph anchors for belief statements and record them."""
        new_beliefs = []
        # Patterns that indicate a belief about the user/state
        belief_patterns = [
            r'(?:user|用户)\s*(?:prefers?|likes?|uses?|works? with|knows?|is|has|wants?)\s+(.+?)(?:[\.\n]|$)',
            r'(?:用户|user)\s*(?:偏好|喜欢|使用|了解|熟悉|知道|想要)\s*(.+?)(?:[。\n]|$)',
        ]
        for anchor in graph.anchors.values():
            if not anchor.is_retrievable:
                continue
            for pattern in belief_patterns:
                for match in re.finditer(pattern, anchor.text, re.IGNORECASE):
                    belief_text = match.group(0).strip()[:200]
                    topic = match.group(1).strip() if match.lastindex else belief_text[:60]
                    bv = self.record_belief(
                        topic=topic,
                        text=belief_text,
                        evidence_anchor_ids=[anchor.id],
                        source_session=getattr(anchor, 'source_session', ''),
                        tags=getattr(anchor, 'tags', []),
                    )
                    new_beliefs.append(bv)
        return new_beliefs

    # ── Stats ───────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        current_count = sum(1 for b in self._beliefs.values() if b.is_current)
        return {
            "total_beliefs": len(self._beliefs),
            "current_beliefs": current_count,
            "superseded": self._total_superseded,
            "total_recorded": self._total_recorded,
            "topics_tracked": len(self._topic_index),
            "recent_changes": len(self.get_recent_changes(168.0)),
        }
