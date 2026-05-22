"""Perception Layer — raw user text → structured cognitive input.

Transforms raw user text into a PerceptionFrame with:
- Intent classification (question, command, statement, reflection, emotion)
- Emotion analysis (valence + arousal, lexicon-based, no LLM)
- Goal extraction (explicit + implicit)
- Concept extraction (key concepts for ConceptCortex activation)
- Entity extraction (people, tools, projects)
- Implicit need inference (what the user really wants, based on emotion + intent)
- Urgency and complexity estimation

No LLM dependency — rule-based + statistical methods.
Replaces the input side of gate.py / write_gate.py in Phase 6.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ── Data Structures ─────────────────────────────────────────────


@dataclass
class IntentSignal:
    """Decomposed intent classification result."""
    primary: str = "statement"         # question | command | statement | reflection | emotion
    secondary: list[str] = field(default_factory=list)
    confidence: float = 0.5
    keywords_matched: list[str] = field(default_factory=list)


@dataclass
class PerceptionFrame:
    """Structured cognitive input from raw user text.

    This is what phase 6 subsystems consume instead of raw text.
    """
    raw_text: str = ""
    intent: str = "statement"
    intent_confidence: float = 0.5
    explicit_goals: list[str] = field(default_factory=list)
    implicit_goals: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0       # -1..+1
    emotional_arousal: float = 0.0       # 0..1 intensity
    extracted_concepts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topic_domain: str = ""              # mapped to Domain enum
    implicit_needs: list[str] = field(default_factory=list)
    urgency: float = 0.0                # 0..1
    complexity: float = 0.0             # 0..1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text[:300],
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 3),
            "explicit_goals": self.explicit_goals,
            "implicit_goals": self.implicit_goals,
            "emotional_valence": round(self.emotional_valence, 3),
            "emotional_arousal": round(self.emotional_arousal, 3),
            "extracted_concepts": self.extracted_concepts[:10],
            "entities": self.entities,
            "topic_domain": self.topic_domain,
            "implicit_needs": self.implicit_needs[:5],
            "urgency": round(self.urgency, 3),
            "complexity": round(self.complexity, 3),
        }


# ── Built-in lexicons ───────────────────────────────────────────

# NRC Emotion Lexicon subset: word → (valence, arousal)
_NRC_LEXICON: dict[str, tuple[float, float]] = {
    # Positive words
    "good": (0.7, 0.4), "great": (0.8, 0.6), "excellent": (0.9, 0.7),
    "amazing": (0.9, 0.8), "wonderful": (0.85, 0.65), "fantastic": (0.9, 0.75),
    "happy": (0.8, 0.6), "love": (0.9, 0.7), "beautiful": (0.8, 0.5),
    "perfect": (0.85, 0.6), "nice": (0.6, 0.3), "thanks": (0.7, 0.4),
    "thank": (0.7, 0.4), "appreciate": (0.7, 0.35), "helpful": (0.7, 0.4),
    "useful": (0.65, 0.3), "works": (0.6, 0.3), "working": (0.6, 0.3),
    "fixed": (0.7, 0.5), "solved": (0.8, 0.5), "success": (0.85, 0.6),
    "better": (0.6, 0.4), "best": (0.8, 0.5), "awesome": (0.9, 0.8),
    "cool": (0.6, 0.5), "glad": (0.7, 0.5), "pleased": (0.7, 0.4),
    "excited": (0.8, 0.8), "looking forward": (0.7, 0.5),
    "like": (0.3, 0.2), "prefer": (0.2, 0.15),
    # Negative words
    "bad": (-0.6, 0.4), "terrible": (-0.85, 0.7), "horrible": (-0.9, 0.8),
    "awful": (-0.8, 0.65), "hate": (-0.85, 0.7), "angry": (-0.8, 0.8),
    "frustrated": (-0.7, 0.7), "frustrating": (-0.7, 0.65),
    "annoying": (-0.6, 0.5), "annoyed": (-0.6, 0.55),
    "sad": (-0.7, 0.3), "disappointed": (-0.65, 0.4),
    "worried": (-0.6, 0.5), "anxious": (-0.65, 0.6),
    "confused": (-0.4, 0.5), "lost": (-0.5, 0.4),
    "broken": (-0.6, 0.5), "broken": (-0.6, 0.5), "fails": (-0.7, 0.5),
    "error": (-0.5, 0.4), "bug": (-0.45, 0.35), "crash": (-0.6, 0.6),
    "issue": (-0.4, 0.3), "problem": (-0.5, 0.35), "wrong": (-0.6, 0.4),
    "doesn't work": (-0.65, 0.5), "not working": (-0.65, 0.5),
    "difficult": (-0.4, 0.4), "hard": (-0.3, 0.35),
    "slow": (-0.4, 0.3), "painful": (-0.7, 0.6), "ugly": (-0.5, 0.4),
    "stupid": (-0.7, 0.6), "ridiculous": (-0.6, 0.55),
    "never": (-0.3, 0.4), "waste": (-0.6, 0.45),
    # Intensifiers
    "very": (0, 0.3), "really": (0, 0.35), "extremely": (0, 0.5),
    "absolutely": (0, 0.5), "completely": (0, 0.4), "totally": (0, 0.4),
    # Negations (flip valence)
    "not": (0, 0), "no": (0, 0),
}

# Intent detection patterns
_INTENT_PATTERNS: dict[str, list[str]] = {
    "question": [
        r"\?", r"^(what|how|why|when|where|who|which|can|could|would|will|is|are|do|does|did|should|shall|may|might|has|have|am)\b",
        r"\b(tell me|explain|describe|show me|help me understand|clarify|elaborate)\b",
        r"\b(what's|what is|how's|how do|how to|how can|why is|why do)\b",
        r"\b(any idea|do you know|can you|would you|could you help)\b",
    ],
    "command": [
        r"^(do|make|create|add|remove|delete|update|change|fix|run|start|stop|build|install|set|get|find|show|open|close|write|read|check|test|deploy|commit|push|pull|merge|review|refactor|implement|migrate|convert|generate|export|import)\b",
        r"^(please|kindly|just|simply|go ahead|now)\s+\w",
        r"\b(need to|must|have to|got to|gotta)\b",
        r"^[A-Za-z\s]+!$",
    ],
    "reflection": [
        r"\b(i think|i believe|i feel|in my opinion|from my perspective|i've been thinking|i realize|i notice|it seems|it appears)\b",
        r"\b(reflecting|pondering|wondering|considering|contemplating)\b",
        r"\b(maybe|perhaps|possibly|probably|might be|could be)\b",
    ],
    "emotion": [
        r"^(wow|oh|ah|ugh|yay|damn|jeez|geez|omg|lol|lmao|haha|hehe)\b",
        r"!{2,}", r"\b(so|very|really|extremely|absolutely|totally)\s+(happy|sad|angry|excited|frustrated|annoyed|disappointed|glad|thrilled)\b",
        r"\b(i('m| am) (so|really|very|extremely) (happy|sad|angry|excited|frustrated|annoyed|disappointed|glad|thrilled))\b",
    ],
    "statement": [
        # Default — matches anything that doesn't match above
    ],
}

# Goal extraction patterns
_GOAL_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern, type: explicit | implicit)
    (r"(?:i (?:want|need|have to|must|should|would like|plan|intend|aim) to) (.+?)(?:\.|$|,|;)", "explicit"),
    (r"(?:my goal is|the goal is|objective is|target is) (.+?)(?:\.|$|,|;)", "explicit"),
    (r"(?:i'm trying to|i am trying to) (.+?)(?:\.|$|,|;)", "explicit"),
    (r"(?:i (?:hope|wish) to) (.+?)(?:\.|$|,|;)", "implicit"),
    (r"(?:ideally|eventually) (.+?)(?:\.|$|,|;)", "implicit"),
    (r"(?:we need to|we should|we have to) (.+?)(?:\.|$|,|;)", "explicit"),
    (r"(?:task is to|mission is to) (.+?)(?:\.|$|,|;)", "explicit"),
    (r"(?:priority is|focus is) (.+?)(?:\.|$|,|;)", "explicit"),
]

# Entity extraction patterns
_ENTITY_PATTERNS: list[str] = [
    r"\b(?:project|repo|repository|library|package|module|framework|tool|language|database)\s+['\"]?(\w[\w\s./-]+?)(?:['\"]?\b)",
    r"\b(?:using|with|in|on)\s+['\"]?([A-Z][a-zA-Z0-9]+)\b",
    r"\b([A-Z][a-zA-Z0-9]+(?:\.(?:js|py|rs|go|ts|java|rb|cpp|cs))?)\b",
]

# Domain classification keywords
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "development": ["code", "programming", "debug", "compile", "deploy", "test", "bug",
                     "feature", "refactor", "commit", "push", "pull", "merge", "api",
                     "function", "class", "module", "library", "framework", "package"],
    "lifestyle": ["food", "sleep", "exercise", "health", "travel", "hobby", "music",
                   "movie", "book", "game", "sport", "fitness"],
    "emotional": ["feel", "emotion", "angry", "happy", "sad", "frustrated", "excited",
                   "worried", "stress", "anxiety", "love", "hate"],
    "project": ["deadline", "milestone", "sprint", "release", "planning", "roadmap",
                 "meeting", "report", "presentation", "client", "stakeholder"],
    "world_knowledge": ["news", "research", "paper", "study", "science", "history",
                         "politics", "economy", "technology", "trend"],
}


# ── Perception Layer ────────────────────────────────────────────


class PerceptionLayer:
    """Transforms raw user input into structured PerceptionFrame.

    Pipeline: intent → emotion → goals → concepts → entities → needs → urgency → complexity

    No LLM dependency. Uses keyword patterns, lexicon lookup, and statistical methods.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._intent_confidence_threshold = self._config.get("intent_confidence_threshold", 0.5)
        self._concept_max_count = self._config.get("concept_max_count", 10)
        self._need_max_count = self._config.get("implicit_need_max_count", 5)
        self._max_text_length = self._config.get("max_text_length", 8000)

        # Compile intent patterns
        self._compiled_intents: dict[str, list[re.Pattern]] = {}
        for intent, patterns in _INTENT_PATTERNS.items():
            self._compiled_intents[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]

        # Compile goal patterns
        self._compiled_goals: list[tuple[re.Pattern, str]] = [
            (re.compile(p, re.IGNORECASE), gtype) for p, gtype in _GOAL_PATTERNS
        ]

        # Compile entity patterns
        self._compiled_entities = [re.compile(p, re.IGNORECASE) for p in _ENTITY_PATTERNS]

        # Stop words for concept extraction
        self._stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
            'both', 'either', 'neither', 'this', 'that', 'these', 'those',
            'it', 'its', 'they', 'them', 'their', 'i', 'me', 'my', 'we', 'us',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'what', 'which', 'who', 'whom', 'when', 'where', 'how',
            'if', 'then', 'else', 'because', 'while', 'although',
            'here', 'there', 'now', 'then', 'just', 'also', 'very',
            'really', 'quite', 'rather', 'some', 'any', 'more', 'most',
            'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too',
            'about', 'up', 'out', 'off', 'over', 'under', 'again',
            'further', 'once', 'all', 'each', 'every',
        }

    # ── Main API ──────────────────────────────────────────

    def perceive(self, text: str, *,
                 session_context: dict | None = None,
                 recent_history: list[str] | None = None) -> PerceptionFrame:
        """Full perception pipeline: intent → emotion → goals → concepts → needs."""

        text = text[:self._max_text_length].strip()
        if not text:
            return PerceptionFrame(raw_text=text)

        # Stage 1: Intent
        intent_signal = self.detect_intent(text)

        # Stage 2: Emotion
        valence, arousal = self.analyze_emotion(text)

        # Stage 3: Goals
        explicit, implicit = self.extract_goals(text, intent_signal.primary)

        # Stage 4: Concepts
        concepts = self.extract_concepts(text)

        # Stage 5: Entities
        entities = self.extract_entities(text)

        # Stage 6: Domain
        domain = self.classify_domain(text, concepts)

        # Stage 7: Implicit Needs
        needs = self.infer_needs(text, (valence, arousal), intent_signal)

        # Stage 8: Urgency + Complexity
        urgency = self.estimate_urgency(text, arousal)
        complexity = self.estimate_complexity(text, concepts)

        return PerceptionFrame(
            raw_text=text,
            intent=intent_signal.primary,
            intent_confidence=intent_signal.confidence,
            explicit_goals=explicit,
            implicit_goals=implicit,
            emotional_valence=valence,
            emotional_arousal=arousal,
            extracted_concepts=concepts,
            entities=entities,
            topic_domain=domain,
            implicit_needs=needs,
            urgency=urgency,
            complexity=complexity,
        )

    def perceive_batch(self, texts: list[str]) -> list[PerceptionFrame]:
        """Batch perception."""
        return [self.perceive(t) for t in texts]

    # ── Stage 1: Intent Detection ──────────────────────────

    def detect_intent(self, text: str) -> IntentSignal:
        """Classify user intent using keyword patterns."""
        scores: dict[str, float] = {}
        matched_keywords: dict[str, list[str]] = {}

        for intent, patterns in self._compiled_intents.items():
            if intent == "statement":
                continue  # default, computed last
            score = 0.0
            matched = []
            for pattern in patterns:
                if pattern.search(text):
                    score += 1.0
                    matched.append(pattern.pattern[:60])
            if score > 0:
                scores[intent] = min(1.0, score / len(patterns))
                matched_keywords[intent] = matched

        # Fallback: statement
        if not scores:
            return IntentSignal(primary="statement", confidence=0.6,
                                keywords_matched=[])

        primary = max(scores, key=scores.get)
        confidence = scores[primary]

        # Secondary intents
        secondary = [k for k in scores if k != primary and scores[k] > 0.3]
        secondary.sort(key=lambda k: scores[k], reverse=True)

        return IntentSignal(
            primary=primary,
            secondary=secondary[:3],
            confidence=confidence,
            keywords_matched=matched_keywords.get(primary, []),
        )

    # ── Stage 2: Emotion Analysis ──────────────────────────

    def analyze_emotion(self, text: str) -> tuple[float, float]:
        """Lexicon-based emotion analysis. Returns (valence, arousal).

        Valence: -1 (negative) to +1 (positive)
        Arousal: 0 (calm) to 1 (excited)
        """
        words = text.lower().split()
        if not words:
            return (0.0, 0.0)

        valences: list[float] = []
        arousals: list[float] = []
        negate_next = False

        for i, word in enumerate(words):
            word_clean = word.strip(".,!?;:()[]{}\"'")

            # Handle negation: "not good" → flip valence
            if word_clean in ("not", "no", "never", "neither", "nor"):
                negate_next = True
                continue

            if word_clean in _NRC_LEXICON:
                v, a = _NRC_LEXICON[word_clean]
                if negate_next and v != 0:
                    v = -v
                valences.append(v)
                arousals.append(a)
                negate_next = False
            else:
                # Check 2-gram and 3-gram
                if i >= 1:
                    bigram = f"{words[i-1]} {word_clean}"
                    if bigram in _NRC_LEXICON:
                        v, a = _NRC_LEXICON[bigram]
                        valences.append(v)
                        arousals.append(a)
                if i >= 2:
                    trigram = f"{words[i-2]} {words[i-1]} {word_clean}"
                    if trigram in _NRC_LEXICON:
                        v, a = _NRC_LEXICON[trigram]
                        valences.append(v)
                        arousals.append(a)

                negate_next = False

        if not valences:
            return (0.0, 0.0)

        # Weighted average: later words slightly more important (recency)
        weights = [1.0 + i * 0.05 for i in range(len(valences))]
        total_w = sum(weights)

        avg_valence = sum(v * w for v, w in zip(valences, weights)) / total_w
        avg_arousal = sum(a * w for a, w in zip(arousals, weights)) / total_w

        return (max(-1.0, min(1.0, avg_valence)),
                max(0.0, min(1.0, avg_arousal)))

    # ── Stage 3: Goal Extraction ───────────────────────────

    def extract_goals(self, text: str, intent: str) -> tuple[list[str], list[str]]:
        """Extract explicit and implicit goals from text."""
        explicit: list[str] = []
        implicit: list[str] = []

        for pattern, gtype in self._compiled_goals:
            for match in pattern.finditer(text):
                goal_text = match.group(1).strip().rstrip(".,;")
                if len(goal_text) > 10:  # filter noise
                    if gtype == "explicit":
                        explicit.append(goal_text)
                    else:
                        implicit.append(goal_text)

        # Intent-based implicit goals
        if intent == "question" and not explicit:
            # User asking how to do something implies they want to do it
            implicit.append(f"understand: {text[:100]}")

        if intent == "command":
            # Command implies a goal to achieve something
            implicit.append(f"complete: {text[:100]}")

        return (explicit[:5], implicit[:3])

    # ── Stage 4: Concept Extraction ────────────────────────

    def extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts using TF-IDF-like keyword extraction."""
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_+#-]{2,}\b', text.lower())
        if not words:
            return []

        # Filter stop words
        meaningful = [w for w in words if w not in self._stop_words and len(w) > 2]

        # Frequency with position weighting (first occurrence more important)
        word_scores: dict[str, float] = {}
        seen: set[str] = set()
        for i, word in enumerate(meaningful):
            freq = meaningful.count(word) / len(meaningful)
            position_weight = 1.0 / (1.0 + i * 0.1)  # earlier words weight more
            if word not in seen:
                word_scores[word] = freq * position_weight * 2.0
                seen.add(word)
            else:
                word_scores[word] += freq * position_weight * 0.5

        # Sort by score descending
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [w for w, s in sorted_words[:self._concept_max_count]]

    # ── Stage 5: Entity Extraction ─────────────────────────

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities (tools, projects, languages)."""
        entities: list[str] = []

        for pattern in self._compiled_entities:
            for match in pattern.finditer(text):
                entity = match.group(1).strip()
                if len(entity) > 1 and entity.lower() not in self._stop_words:
                    entities.append(entity)

        # Dedup preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique[:10]

    # ── Stage 6: Domain Classification ─────────────────────

    def classify_domain(self, text: str, concepts: list[str]) -> str:
        """Classify text into a domain based on keyword overlap."""
        domain_scores: dict[str, float] = {}
        text_lower = text.lower()

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if kw in text_lower:
                    score += 1.0
            # Also check concepts
            for concept in concepts:
                if concept in keywords:
                    score += 0.5
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return "unclassified"

        return max(domain_scores, key=domain_scores.get)

    # ── Stage 7: Implicit Need Inference ───────────────────

    def infer_needs(self, text: str, emotion: tuple[float, float],
                    intent: IntentSignal) -> list[str]:
        """Infer what the user really needs, based on emotion + intent + text signals."""
        valence, arousal = emotion
        needs: list[str] = []

        # Emotion + intent → need rules
        if valence < -0.3 and intent.primary == "question":
            needs.append("needs help solving a frustrating problem")
        elif valence < -0.3 and intent.primary == "emotion":
            needs.append("needs emotional validation or venting")
        elif valence < -0.3 and intent.primary == "statement":
            needs.append("needs acknowledgment of difficulty")

        if valence > 0.3 and intent.primary == "statement":
            needs.append("wants to share positive experience")
        elif valence > 0.3 and intent.primary == "question":
            needs.append("wants to learn or explore with enthusiasm")

        if arousal > 0.6:
            needs.append("high emotional intensity — may need calming or focus")
            if valence > 0:
                needs.append("excited — channel energy into action")
            else:
                needs.append("agitated — may need step-by-step guidance")

        # Explicit signals in text
        if re.search(r'\b(help|assist|support|guide|advise)\b', text, re.IGNORECASE):
            needs.append("explicitly asking for help")
        if re.search(r'\b(confirm|verify|validate|check|double-check)\b', text, re.IGNORECASE):
            needs.append("seeking confirmation or validation")
        if re.search(r'\b(opinion|think|thoughts|what do you|how would you)\b', text, re.IGNORECASE):
            needs.append("wants opinion or perspective")
        if re.search(r'\b(explain|understand|why|how come|what causes)\b', text, re.IGNORECASE):
            needs.append("seeking understanding or explanation")
        if re.search(r'\b(deadline|urgent|asap|quickly|fast|now|immediately)\b', text, re.IGNORECASE):
            needs.append("has time pressure or urgency")

        # If nothing detected, note general engagement
        if not needs:
            if intent.primary == "question":
                needs.append("seeking information")
            elif intent.primary == "command":
                needs.append("wants task completion")
            else:
                needs.append("general engagement")

        return needs[:self._need_max_count]

    # ── Stage 8: Urgency + Complexity ──────────────────────

    def estimate_urgency(self, text: str, arousal: float) -> float:
        """Estimate urgency from keywords + emotional arousal."""
        urgency_keywords = r'\b(urgent|asap|emergency|critical|immediately|now|quick|fast|hurry|deadline|soon|today|tonight)\b'
        kw_hits = len(re.findall(urgency_keywords, text, re.IGNORECASE))
        kw_score = min(1.0, kw_hits * 0.25)
        return kw_score * 0.4 + arousal * 0.6

    def estimate_complexity(self, text: str, concepts: list[str]) -> float:
        """Estimate cognitive complexity from text length + concept diversity."""
        word_count = len(text.split())
        length_score = min(1.0, word_count / 200)  # 200+ words = max complexity
        concept_diversity = min(1.0, len(concepts) / 10)
        return length_score * 0.5 + concept_diversity * 0.3 + 0.2  # base complexity
