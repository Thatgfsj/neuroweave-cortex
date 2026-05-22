"""Importance Engine v2 — cognitive gatekeeper with real perception analysis.

Integrates with PerceptionLayer for intent/emotion analysis, GoalSystem for
goal-relationship scoring, and maintains a proper topic tracker with time decay.

Formula:
    importance = emotion_signal × 0.20
               + repetition_signal × 0.25
               + goal_relation × 0.25
               + content_richness × 0.15
               + future_recall_likelihood × 0.15

Content richness factors:
    - Entity density (named entities, technical terms)
    - Specificity (concrete vs vague language)
    - Actionability (is this a task/command/decision?)
    - Information density (meaningful words / total words)

Noise gate:
    0.00–0.15 → DISCARD  (greetings, filler, pure noise)
    0.15–0.35 → LOW      (casual chat, disposable)
    0.35–0.60 → NORMAL   (standard, participates in consolidation)
    0.60–1.00 → CORE     (forced to Semantic/Pattern, low decay)
"""

from dataclasses import dataclass, field
from collections import defaultdict
import time
import re
import math
from typing import Optional


# ── Levels ─────────────────────────────────────────────────

class ImportanceLevel:
    DISCARD = "discard"   # < 0.15
    LOW = "low"           # 0.15–0.35
    NORMAL = "normal"     # 0.35–0.60
    CORE = "core"         # > 0.60

    @staticmethod
    def classify(score: float) -> str:
        if score < 0.15:
            return ImportanceLevel.DISCARD
        elif score < 0.35:
            return ImportanceLevel.LOW
        elif score < 0.60:
            return ImportanceLevel.NORMAL
        else:
            return ImportanceLevel.CORE


# ── Data types ─────────────────────────────────────────────

@dataclass
class ContentRichness:
    """Structural analysis of the input itself, independent of context."""
    word_count: int = 0
    entity_count: int = 0
    technical_term_count: int = 0
    specificity_score: float = 0.0      # concrete vs vague
    actionability_score: float = 0.0    # is it a task/command?
    information_density: float = 0.0    # meaningful / total words
    composite: float = 0.0


@dataclass
class ImportanceSignal:
    emotion_signal: float = 0.0
    repetition_signal: float = 0.0
    goal_relation: float = 0.0
    content_richness: float = 0.0
    future_recall_likelihood: float = 0.0

    def compute(self, weights: "ImportanceWeights") -> float:
        return (
            self.emotion_signal * weights.emotion
            + self.repetition_signal * weights.repetition
            + self.goal_relation * weights.goal
            + self.content_richness * weights.richness
            + self.future_recall_likelihood * weights.future
        )


@dataclass
class ImportanceWeights:
    emotion: float = 0.20
    repetition: float = 0.25
    goal: float = 0.25
    richness: float = 0.15
    future: float = 0.15


@dataclass
class ImportanceResult:
    score: float
    level: str
    signal: ImportanceSignal
    richness: ContentRichness
    should_store: bool
    decay_rate: float
    consolidation_priority: int


# ── Topic tracker with decay ───────────────────────────────

@dataclass
class TopicEntry:
    topic: str
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    recent_burst: int = 0  # count in current burst window

    def decay_factor(self, now: float, half_life_days: float = 30.0) -> float:
        """Exponential decay: relevance halves every half_life_days without reinforcement."""
        days = (now - self.last_seen) / 86400
        return math.exp(-days * math.log(2) / half_life_days)

    def effective_weight(self, now: float) -> float:
        return self.count * self.decay_factor(now) + self.recent_burst * 0.5


# ── Technical term patterns ────────────────────────────────

TECH_PATTERNS = [
    # English
    r'\b(?:AI|ML|LLM|API|SDK|CLI|GPU|CPU|SQL|NoSQL|HTTP|REST|gRPC|MCP)\b',
    r'\b(?:memory|cognit\w+|neural|graph|vector|embed\w+|model|train\w+|infer\w+|architect\w+|system|design|pattern|framework|backend|frontend|deploy|pipeline|workflow)\b',
    r'\b(?:debug\w+|optimiz\w+|refactor\w+|benchmark\w+|test\w+|compil\w+|runtim\w+|schedul\w+|consolidat\w+|retriev\w+|abstract\w+|compress\w+)\b',
    # Chinese tech terms
    r'(?:架构|系统|框架|模块|引擎|算法|模型|训练|推理|检索|存储|压缩|抽象|模式|设计|实现|部署|优化|调试|测试|重构|管道|工作流|平台|接口|协议|数据库|向量|图|网络|层|索引|缓存|路由|调度|演化|巩固|遗忘|激活|抑制|注意力|概念|信念|人格|认知|神经|海马)',
    r'(?:代码|开发|编程|开源|工具|配置|安装|集成|兼容|性能|安全|加密|签名|证书|验证|持久化|序列化|反序列化|多线程|异步|并行|分布式|微服务|容器|编排|监控|日志|追踪|指标|仪表盘)',
]

# ── Vague language signals (lower specificity) ─────────────

VAGUE_PATTERNS = [
    r'\b(?:thing|stuff|something|whatever|maybe|kind of|sort of|probably|possibly|somehow)\b',
    r'\b(?:nice|good|bad|ok|fine|great|cool|awesome|interesting)\b',
]


# ── Engine ─────────────────────────────────────────────────

class ImportanceEngine:
    """Cognitive gatekeeper v2 — deep perception integration.

    Usage:
        engine = ImportanceEngine(goal_system=gs)
        result = engine.evaluate(text, perception_frame=frame, active_goals=[...])
        if result.should_store:
            memory_manager.remember(text, importance=result.score)
    """

    def __init__(self, weights: Optional[ImportanceWeights] = None,
                 goal_system=None, perception_layer=None):
        self.weights = weights or ImportanceWeights()
        self._goal_system = goal_system
        self._perception_layer = perception_layer
        self._topic_tracker: dict[str, TopicEntry] = defaultdict(
            lambda: TopicEntry(topic="", count=0, first_seen=0.0))
        self._total_evaluated = 0
        self._total_discarded = 0
        self._start_time = time.time()

    # ── Main API ──────────────────────────────────────────

    def evaluate(self, text: str, *,
                 perception_frame=None,           # PerceptionFrame from perception.py
                 emotion_valence: float = 0.0,
                 active_goals: list[str] | None = None,
                 extracted_concepts: list[str] | None = None,
                 extracted_entities: list[str] | None = None,
                 topic_tags: list[str] | None = None,
                 intent: str = "") -> ImportanceResult:
        """Full importance evaluation with perception integration."""
        self._total_evaluated += 1
        now = time.time()

        # If we have a perception frame, use its richer analysis
        if perception_frame is not None:
            emotion_valence = getattr(perception_frame, 'valence', emotion_valence)
            intent = getattr(perception_frame, 'intent', intent)
            extracted_concepts = getattr(perception_frame, 'extracted_concepts', extracted_concepts)
            extracted_entities = getattr(perception_frame, 'extracted_entities', extracted_entities)

        signal = ImportanceSignal()
        richness = self._analyze_richness(text, extracted_entities or [])

        signal.emotion_signal = self._score_emotion(emotion_valence, intent)
        signal.repetition_signal = self._score_repetition(topic_tags or [], extracted_concepts or [], now)
        signal.goal_relation = self._score_goal_relation(text, active_goals or [], extracted_concepts or [])
        signal.content_richness = richness.composite
        signal.future_recall_likelihood = self._score_future(
            signal.repetition_signal, signal.goal_relation, signal.emotion_signal,
            richness.actionability_score)

        # Force-discard pure casual/small-talk
        if richness.composite < 0.08 and richness.entity_count == 0 and richness.technical_term_count == 0:
            score = 0.05
            level = ImportanceLevel.DISCARD
            self._total_discarded += 1
            return ImportanceResult(
                score=0.05, level=ImportanceLevel.DISCARD, signal=signal,
                richness=richness, should_store=False, decay_rate=1.0,
                consolidation_priority=0)

        score = signal.compute(self.weights)
        score = min(1.0, max(0.0, score))
        level = ImportanceLevel.classify(score)

        if level == ImportanceLevel.DISCARD:
            self._total_discarded += 1

        # Update topic tracker
        for tag in (topic_tags or []) + (extracted_concepts or []):
            entry = self._topic_tracker[tag]
            if entry.count == 0:
                entry.topic = tag
                entry.first_seen = now
            entry.count += 1
            entry.last_seen = now
            # Burst detection: if seen within last hour, increment burst
            if now - entry.last_seen < 3600:
                entry.recent_burst += 1
            else:
                entry.recent_burst = 1

        return ImportanceResult(
            score=score, level=level, signal=signal, richness=richness,
            should_store=(level != ImportanceLevel.DISCARD),
            decay_rate=self._decay_rate(level),
            consolidation_priority=(2 if level == ImportanceLevel.CORE
                                    else 1 if level == ImportanceLevel.NORMAL else 0),
        )

    def evaluate_batch(self, items: list[dict]) -> list[ImportanceResult]:
        return [self.evaluate(
            item.get("text", ""),
            perception_frame=item.get("perception_frame"),
            emotion_valence=item.get("emotion_valence", 0.0),
            active_goals=item.get("active_goals", []),
            extracted_concepts=item.get("concepts", []),
            extracted_entities=item.get("entities", []),
            topic_tags=item.get("tags", []),
            intent=item.get("intent", ""),
        ) for item in items]

    def filter_noise(self, items: list[dict]) -> list[dict]:
        results = self.evaluate_batch(items)
        return [item for item, r in zip(items, results) if r.should_store]

    # ── Richness analysis ─────────────────────────────────

    def _analyze_richness(self, text: str, entities: list[str]) -> ContentRichness:
        if not text:
            return ContentRichness()

        # CJK text: count characters, not space-delimited words
        cjk_chars = len(re.findall(r'[一-鿿]', text))
        if cjk_chars > len(text) * 0.3:
            word_count = max(cjk_chars, 1)
        else:
            word_count = len(text.split())

        # Very short with no entities = likely noise
        if word_count < 5 and not entities:
            return ContentRichness(word_count=word_count, composite=0.05)

        # Entity density
        entity_count = len(entities)
        tech_count = 0
        for pattern in TECH_PATTERNS:
            tech_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Specificity: penalize vague language
        vague_count = 0
        for pattern in VAGUE_PATTERNS:
            vague_count += len(re.findall(pattern, text, re.IGNORECASE))
        specificity = 1.0 - min(1.0, vague_count / max(1, word_count) * 5)

        # Casual/small-talk detection
        casual_patterns = [
            r'^(?:你好|hi|hello|hey|早|晚安|再见|bye|谢谢|thanks|ok|好的|嗯|哦|哈哈|呵呵){1}$',
            r'^(?:.*(?:天气|吃饭|睡觉|累了|困了|饿了|渴了|热|冷))(?:.*(?:不错|还行|一般|不好))?$',
        ]
        for cp in casual_patterns:
            if re.match(cp, text.strip(), re.IGNORECASE):
                return ContentRichness(word_count=word_count, composite=0.05, specificity_score=0.1)

        # Actionability
        actionable = 0.0
        if re.search(r'(?:需要|要|必须|应该|立即|现在|修复|解决|实现|构建|部署|测试|调试|优化)', text):
            actionable += 0.4
        if re.search(r'(?:怎么|如何|为什么|什么原因|哪里出错)', text):
            actionable += 0.3
        if re.search(r'(?:决定|选择|方案|架构|设计|规划|策略)', text):
            actionable += 0.3
        actionable = min(1.0, actionable)

        # Information density
        stopwords = {'的', '了', '是', '我', '你', '他', '她', '它', '们', '这', '那', '很', '也', '都', '就',
                     '在', '不', '和', '与', 'a', 'an', 'the', 'is', 'was', 'are', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                     'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once'}
        if cjk_chars > len(text) * 0.3:
            meaningful = sum(1 for ch in text if ch not in stopwords and not ch.isspace()
                            and ch not in '，。！？、；：""''（）…—')
            density = meaningful / max(1, cjk_chars)
        else:
            words = text.split()
            meaningful = sum(1 for w in words if w.lower() not in stopwords)
            density = meaningful / max(1, word_count)

        # Composite
        composite = (
            min(1.0, (entity_count + tech_count) / max(1, word_count) * 10) * 0.35
            + specificity * 0.25
            + actionable * 0.25
            + density * 0.15
        )

        return ContentRichness(
            word_count=word_count, entity_count=entity_count, technical_term_count=tech_count,
            specificity_score=specificity, actionability_score=actionable,
            information_density=density, composite=composite,
        )

    # ── Scoring components ─────────────────────────────────

    @staticmethod
    def _score_emotion(valence: float, intent: str) -> float:
        """Emotion signal: intensity matters, not polarity. Intent type weights differently."""
        intensity = abs(valence)
        intent_weight = {
            "command": 0.7, "question": 0.5, "reflection": 0.3,
            "emotion": 0.8, "statement": 0.3, "goal_statement": 0.9,
        }.get(intent, 0.3)
        return min(1.0, intensity * 0.5 + intent_weight * 0.5)

    def _score_repetition(self, tags: list[str], concepts: list[str], now: float) -> float:
        """Topic repetition with time decay.

        Cold start with no history: if input has tags/concepts, assume it's intentional.
        Once tracker has data, use real repetition scoring.
        """
        candidates = tags + concepts
        if not candidates:
            return 0.08  # no topics at all → likely casual
        if not self._topic_tracker:
            return 0.28  # cold start with tagged/concepted input → assume intentional
        weights = []
        for c in candidates:
            entry = self._topic_tracker.get(c)
            if entry:
                weights.append(entry.effective_weight(now))
        if not weights:
            return 0.18  # new topics in an established session
        max_weight = max(weights)
        if max_weight >= 50:
            return 0.95
        elif max_weight >= 20:
            return 0.80
        elif max_weight >= 10:
            return 0.65
        elif max_weight >= 5:
            return 0.50
        elif max_weight >= 3:
            return 0.35
        return 0.22

    def _score_goal_relation(self, text: str, active_goals: list[str],
                             concepts: list[str]) -> float:
        """How related is this to active goals? Uses concept overlap + keyword matching."""
        if not active_goals:
            return 0.15
        text_lower = text.lower()
        text_words = set(text_lower.split())
        concept_set = set(c.lower() for c in concepts)

        best_score = 0.0
        for goal in active_goals:
            goal_lower = goal.lower()
            goal_words = set(goal_lower.split())

            # Direct word overlap
            word_overlap = text_words & goal_words
            word_score = len(word_overlap) / max(1, len(goal_words))

            # Concept overlap (higher weight — concepts are pre-extracted semantics)
            concept_overlap = concept_set & goal_words
            concept_score = len(concept_overlap) / max(1, len(goal_words))

            # Combined
            goal_score = word_score * 0.4 + concept_score * 0.6
            best_score = max(best_score, goal_score)

        return min(1.0, best_score * 1.5)  # boost — goal alignment is critical

    @staticmethod
    def _score_future(repetition: float, goal_relation: float,
                      emotion: float, actionability: float) -> float:
        """Predict likelihood of future recall."""
        return (repetition * 0.35 + goal_relation * 0.30
                + emotion * 0.15 + actionability * 0.20)

    @staticmethod
    def _decay_rate(level: str) -> float:
        return {"discard": 1.0, "low": 0.25, "normal": 0.08, "core": 0.015}.get(level, 0.1)

    # ── Stats ──────────────────────────────────────────────

    def stats(self) -> dict:
        now = time.time()
        hot = sorted(
            [(t.topic, t.count, t.recent_burst, t.effective_weight(now))
             for t in self._topic_tracker.values() if t.count >= 2],
            key=lambda x: -x[3],
        )[:10]
        return {
            "total_evaluated": self._total_evaluated,
            "total_discarded": self._total_discarded,
            "discard_rate": self._total_discarded / max(1, self._total_evaluated),
            "tracked_topics": len(self._topic_tracker),
            "hot_topics": [{"topic": t, "count": c, "burst": b, "weight": round(w, 1)}
                          for t, c, b, w in hot],
            "uptime_hours": (now - self._start_time) / 3600,
        }
