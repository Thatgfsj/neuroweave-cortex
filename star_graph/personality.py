"""Personality Model — deep user trait extraction beyond worldview compilation.

Extracts:
  - Traits: openness, conscientiousness, extraversion, agreeableness, neuroticism
  - Working style: planning, execution, debugging, learning preferences
  - Communication: verbosity, formality, code-vs-natural, question frequency
  - Expertise: domains, skill levels, learning velocity
  - Values: what the user cares about (inferred from emotional valence patterns)
  - Behavioral patterns: time-of-day activity, session length, topic switching

Updated incrementally as new anchors arrive and during sleep cycles.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonalityProfile:
    """Aggregated user personality model."""
    # Big Five traits (0-1)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Working style
    planner_vs_doer: float = 0.5        # 1.0 = heavy planner, 0.0 = jumps in
    debugger_persistence: float = 0.5   # 1.0 = persistent debugger
    learning_style: str = "balanced"    # "reading", "doing", "asking", "balanced"
    iteration_speed: float = 0.5        # 1.0 = fast iterations

    # Communication
    average_message_length: float = 0.0
    formality: float = 0.5              # 1.0 = very formal
    code_ratio: float = 0.0             # fraction of messages containing code
    question_frequency: float = 0.5     # how often user asks vs tells
    uses_emoji: bool = False

    # Expertise
    expertise_areas: dict[str, float] = field(default_factory=dict)  # domain→level
    skill_count: int = 0

    # Values
    values: dict[str, float] = field(default_factory=dict)  # value→strength

    # Meta
    confidence: float = 0.3             # overall model confidence
    evidence_count: int = 0             # how many anchors contributed
    last_updated: float = field(default_factory=time.time)
    version: int = 0


class PersonalityModel:
    """Deep personality extraction from memory graph content.

    Goes beyond the compiler's worldview nodes by extracting nuanced
    behavioral patterns, trait estimates, and implicit preferences.

    Usage:
        pm = PersonalityModel()
        pm.ingest_anchor(anchor)           # incremental update
        pm.extract_from_graph(graph)       # full scan (during sleep)
        profile = pm.profile               # current personality profile
    """

    # --- Trait keyword indicators ---
    _TRAIT_SIGNALS = {
        "openness": {
            "positive": {"curious", "explore", "learn", "new", "experiment",
                        "creative", "idea", "interesting", "discover", "try",
                        "好奇", "探索", "学习", "创新", "尝试", "新"},
            "negative": {"stick to", "familiar", "comfort zone", "same",
                        "routine", "保守", "习惯"},
        },
        "conscientiousness": {
            "positive": {"plan", "organize", "schedule", "todo", "checklist",
                        "test", "review", "refactor", "clean", "document",
                        "计划", "整理", "测试", "重构", "文档"},
            "negative": {"hack", "quick fix", "shortcut", "skip", "messy",
                        "临时", "快速修复", "跳过"},
        },
        "extraversion": {
            "positive": {"discuss", "share", "talk", "meeting", "collaborate",
                        "team", "together", "pair", "discuss",
                        "讨论", "分享", "合作", "团队"},
            "negative": {"alone", "quiet", "solo", "单独", "安静"},
        },
        "agreeableness": {
            "positive": {"agree", "nice", "good point", "makes sense", "helpful",
                        "appreciate", "thanks", "同意", "好的", "谢谢"},
            "negative": {"disagree", "wrong", "no", "bad", "terrible",
                        "不同意", "不对", "不好"},
        },
        "neuroticism": {
            "positive": {"anxious", "worry", "stress", "nervous", "frustrated",
                        "overwhelm", "panic", "焦虑", "担心", "压力"},
            "negative": {"calm", "relax", "fine", "confident", "sure",
                        "平静", "放松", "自信"},
        },
    }

    _LEARNING_STYLE_SIGNALS = {
        "reading": {"read", "doc", "documentation", "article", "blog",
                   "阅读", "文档", "文章"},
        "doing": {"try", "build", "implement", "code", "write", "run",
                 "试", "构建", "实现", "写"},
        "asking": {"ask", "question", "help", "how to", "请教", "帮助", "怎么"},
    }

    _VALUE_SIGNALS = {
        "efficiency": {"fast", "quick", "efficient", "optimize", "performance",
                      "快", "高效", "优化", "性能"},
        "simplicity": {"simple", "clean", "minimal", "straightforward",
                      "简单", "简洁", "干净"},
        "reliability": {"stable", "reliable", "robust", "test", "production",
                       "稳定", "可靠", "健壮", "测试"},
        "learning": {"learn", "understand", "knowledge", "grow",
                    "学习", "理解", "知识"},
        "autonomy": {"own", "control", "custom", "self", "自己", "控制", "自主"},
    }

    def __init__(self):
        self.profile = PersonalityProfile()
        self._message_lengths: list[int] = []
        self._code_count: int = 0
        self._question_count: int = 0
        self._total_messages: int = 0
        self._expertise_signals: dict[str, list[float]] = defaultdict(list)

    # ── Incremental update ────────────────────────────────────

    def ingest_anchor(self, anchor) -> None:
        """Update personality model from a single anchor."""
        text = anchor.text if hasattr(anchor, 'text') else str(anchor)
        text_lower = text.lower()
        tags = getattr(anchor, 'tags', []) or []
        importance = getattr(anchor, 'vector', None)
        importance = importance.importance if importance else 0.5
        emotion = getattr(anchor, 'vector', None)
        emotion = emotion.emotional_valence if emotion else 0.0

        self.profile.evidence_count += 1
        self._total_messages += 1

        # Message length
        self._message_lengths.append(len(text))

        # Code detection
        if self._has_code(text):
            self._code_count += 1

        # Question detection
        if '?' in text or any(q in text_lower for q in
                             {'how to', 'what', 'why', 'which', 'can you',
                              '怎么', '如何', '为什么', '能不能'}):
            self._question_count += 1

        # Trait signals
        for trait, signals_dict in self._TRAIT_SIGNALS.items():
            pos_score = sum(1 for s in signals_dict["positive"]
                          if s in text_lower)
            neg_score = sum(1 for s in signals_dict["negative"]
                          if s in text_lower)
            if pos_score > 0 or neg_score > 0:
                current = getattr(self.profile, trait)
                delta = 0.02 * importance * (pos_score - neg_score)
                new_val = max(0.0, min(1.0, current + delta))
                setattr(self.profile, trait, new_val)

        # Learning style signals
        for style, signals in self._LEARNING_STYLE_SIGNALS.items():
            if any(s in text_lower for s in signals):
                self.profile.learning_style = style

        # Value signals
        for value, signals in self._VALUE_SIGNALS.items():
            if any(s in text_lower for s in signals):
                current = self.profile.values.get(value, 0.0)
                self.profile.values[value] = min(1.0, current + 0.05 * importance)

        # Expertise signals from tags
        domain_tags = [t for t in tags if t.lower() not in
                      {'test', 'debug', 'chat', 'conversation'}]
        for tag in domain_tags:
            current = self.profile.expertise_areas.get(tag, 0.0)
            self.profile.expertise_areas[tag] = min(1.0, current + 0.03 * importance)

        self.profile.version += 1
        self.profile.last_updated = time.time()

    # ── Bulk extraction ──────────────────────────────────────

    def extract_from_graph(self, graph) -> PersonalityProfile:
        """Full personality extraction from graph (call during sleep)."""
        self.profile = PersonalityProfile()  # reset
        self._message_lengths = []
        self._code_count = 0
        self._question_count = 0
        self._total_messages = 0
        self._expertise_signals.clear()

        for anchor in graph.anchors.values():
            if anchor.is_retrievable:
                self.ingest_anchor(anchor)

        # Compute aggregate metrics
        if self._message_lengths:
            self.profile.average_message_length = (
                sum(self._message_lengths) / len(self._message_lengths))

        if self._total_messages > 0:
            self.profile.code_ratio = self._code_count / self._total_messages
            self.profile.question_frequency = self._question_count / self._total_messages

        # Emoji detection
        emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF☀-➿⭐]')
        self.profile.uses_emoji = self._code_count > 0 or any(
            emoji_pattern.search(a.text)
            for a in graph.anchors.values()
            if a.text and emoji_pattern.search(a.text)
        )

        # Formality: ratio of formal to informal markers
        formal = sum(1 for a in graph.anchors.values()
                    if a.text and any(f in a.text.lower()
                    for f in {'please', 'would', 'could', 'shall', 'thank',
                             'appreciate', '请', '谢谢', '麻烦'}))
        informal = sum(1 for a in graph.anchors.values()
                      if a.text and any(inf in a.text.lower()
                      for inf in {'yeah', 'cool', 'ok', 'nice', 'np',
                                 '好', '行', '嗯'}))
        total_formal = formal + informal
        if total_formal > 0:
            self.profile.formality = formal / total_formal

        # Confidence based on evidence count
        self.profile.confidence = min(0.95, 0.2 + self.profile.evidence_count * 0.005)
        self.profile.version += 1
        self.profile.last_updated = time.time()

        return self.profile

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _has_code(text: str) -> bool:
        """Detect if text contains code snippets."""
        code_indicators = [
            r'```', r'def ', r'class ', r'import ', r'from \w+ import',
            r'fn ', r'let ', r'const ', r'var ', r'function ',
            r'public class', r'private ', r'@Override',
            r'print\(', r'console\.log',
        ]
        for pattern in code_indicators:
            if re.search(pattern, text):
                return True
        return False

    @property
    def trait_scores(self) -> dict[str, float]:
        """Return Big Five trait scores."""
        return {
            "openness": self.profile.openness,
            "conscientiousness": self.profile.conscientiousness,
            "extraversion": self.profile.extraversion,
            "agreeableness": self.profile.agreeableness,
            "neuroticism": self.profile.neuroticism,
        }

    def top_expertise(self, n: int = 5) -> list[tuple[str, float]]:
        """Top N expertise areas by inferred level."""
        sorted_areas = sorted(
            self.profile.expertise_areas.items(),
            key=lambda x: -x[1],
        )
        return sorted_areas[:n]

    @property
    def stats(self) -> dict:
        return {
            "evidence_count": self.profile.evidence_count,
            "confidence": round(self.profile.confidence, 3),
            "traits": {k: round(v, 3) for k, v in self.trait_scores.items()},
            "expertise_areas": len(self.profile.expertise_areas),
            "top_expertise": self.top_expertise(3),
            "values": self.profile.values,
            "code_ratio": round(self.profile.code_ratio, 3),
            "question_frequency": round(self.profile.question_frequency, 3),
            "learning_style": self.profile.learning_style,
            "version": self.profile.version,
        }
