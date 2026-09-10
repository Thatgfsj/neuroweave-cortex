"""Importance Engine v2.1 — cognitive gatekeeper with real perception analysis.

Integrates with PerceptionLayer for intent/emotion analysis, GoalSystem for
goal-relationship scoring, and maintains a proper topic tracker with time decay.

Weights adopted after A/B benchmarking (benchmarks/AB_README.md — W-code+nov):
the v2 signal set plus a novelty signal (1 − topic overlap with known
knowledge); novelty is required by the project's human-like memory goals.

Formula (see ImportanceWeights for the authoritative values):
    importance = emotion_signal       × w_emotion   (default 0.162)
               + repetition_signal    × w_repetition (default 0.198)
               + goal_relation        × w_goal       (default 0.252)
               + retrieval_feedback   × w_retrieval_feedback (default 0.108)
               + content_richness     × w_richness   (default 0.108)
               + future_recall_likelihood × w_future (default 0.072)
               + novelty_signal       × w_novelty    (default 0.100)

Content richness factors:
    - Entity density (named entities, technical terms)
    - Specificity (concrete vs vague language)
    - Actionability (is this a task/command/decision?)
    - Information density (meaningful words / total words)

Noise gate (thresholds configurable via `importance_engine` in
star_graph/defaults.yaml):
    0.00–0.15 → DISCARD  (greetings, filler, pure noise)
    0.15–0.35 → LOW      (casual chat, disposable)
    0.35–0.60 → NORMAL   (standard, participates in consolidation)
    0.60–1.00 → CORE     (forced to Semantic/Pattern, low decay)

A separate richness fast gate force-discards inputs with richness composite
< 0.08 and zero entities/technical terms, regardless of the weighted score.
"""

from dataclasses import dataclass, field
from collections import defaultdict
import time
import re
import math
from typing import Optional


# ── Config-driven thresholds ───────────────────────────────
# All gate thresholds live in defaults.yaml under `importance_engine`.
# Values below are fallbacks used when the section/key is absent.

DEFAULT_DISCARD_THRESHOLD = 0.15
DEFAULT_LOW_THRESHOLD = 0.35
DEFAULT_NORMAL_THRESHOLD = 0.60
DEFAULT_RICHNESS_GATE_COMPOSITE = 0.08
DEFAULT_RICHNESS_GATE_FORCE_SCORE = 0.05


def _cfg_float(dotted_path: str, fallback: float) -> float:
    """Read a float from the global Config by dotted path, falling back safely."""
    try:
        from .config import Config
        value = Config.get().get_path(dotted_path, fallback)
        return float(value) if value is not None else fallback
    except Exception:
        return fallback


# ── Levels ─────────────────────────────────────────────────

class ImportanceLevel:
    DISCARD = "discard"   # < importance_engine.discard_threshold (default 0.15)
    LOW = "low"           # discard–low (default 0.15–0.35)
    NORMAL = "normal"     # low–normal (default 0.35–0.60)
    CORE = "core"         # >= normal (default >= 0.60)

    @staticmethod
    def classify(score: float) -> str:
        discard = _cfg_float("importance_engine.discard_threshold",
                             DEFAULT_DISCARD_THRESHOLD)
        low = _cfg_float("importance_engine.low_threshold",
                         DEFAULT_LOW_THRESHOLD)
        normal = _cfg_float("importance_engine.normal_threshold",
                            DEFAULT_NORMAL_THRESHOLD)
        if score < discard:
            return ImportanceLevel.DISCARD
        elif score < low:
            return ImportanceLevel.LOW
        elif score < normal:
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
    retrieval_feedback: float = 0.0    # new: retrieval_count × success_rate
    content_richness: float = 0.0
    future_recall_likelihood: float = 0.0
    novelty_signal: float = 0.0        # 1 − overlap with known knowledge

    def compute(self, weights: "ImportanceWeights") -> float:
        return (
            self.emotion_signal * weights.emotion
            + self.repetition_signal * weights.repetition
            + self.goal_relation * weights.goal
            + self.retrieval_feedback * weights.retrieval_feedback
            + self.content_richness * weights.richness
            + self.future_recall_likelihood * weights.future
            + self.novelty_signal * weights.novelty
        )


@dataclass
class ImportanceWeights:
    # W-code+nov (A/B benchmarked, benchmarks/AB_README.md): v2 signals ×0.90
    # + novelty 0.100 — sums to exactly 1.000.
    emotion: float = 0.162
    repetition: float = 0.198
    goal: float = 0.252
    retrieval_feedback: float = 0.108
    richness: float = 0.108
    future: float = 0.072
    novelty: float = 0.100


@dataclass
class ImportanceResult:
    score: float
    level: str
    signal: ImportanceSignal
    richness: ContentRichness
    should_store: bool
    decay_rate: float
    consolidation_priority: int
    retrieval_feedback: float = 0.0


# ── Topic tracker with decay ───────────────────────────────

BURST_WINDOW_SECONDS = 3600  # mentions within 1h of each other count as a burst
BURST_MAX = 20               # cap so a sustained burst saturates instead of growing unbounded


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
        # Track retrieval count per semantic signature for feedback loop
        self._retrieval_counts: dict[str, float] = {}  # concept_key → retrieval_count
        self._retrieval_success_rate: dict[str, float] = {}
        # Known-knowledge vocabulary for novelty scoring (term → seen count).
        # Updated AFTER scoring so a text never overlaps itself.
        self._known_terms: dict[str, int] = {}

    # ── Main API ──────────────────────────────────────────

    def evaluate(self, text: str, *,
                 perception_frame=None,           # PerceptionFrame from perception.py
                 emotion_valence: float = 0.0,
                 active_goals: list[str] | None = None,
                 extracted_concepts: list[str] | None = None,
                 extracted_entities: list[str] | None = None,
                 topic_tags: list[str] | None = None,
                 intent: str = "",
                 retrieval_count: int = 0) -> ImportanceResult:
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
        signal.retrieval_feedback = self._score_retrieval_feedback(
            extracted_concepts or [], topic_tags or [], retrieval_count)
        signal.future_recall_likelihood = self._score_future(
            signal.repetition_signal, signal.goal_relation, signal.emotion_signal,
            richness.actionability_score)
        terms = self._extract_terms(text, extracted_concepts or [],
                                    extracted_entities or [], topic_tags or [])
        signal.novelty_signal = self._score_novelty(terms)

        # Force-discard pure casual/small-talk. NOTE: this richness fast gate is
        # an independent structural pre-filter, distinct from θ_discard below —
        # it bypasses the weighted score entirely when the input has no
        # entities/technical terms and near-zero content richness.
        richness_gate = _cfg_float("importance_engine.richness_fast_gate.composite_below",
                                   DEFAULT_RICHNESS_GATE_COMPOSITE)
        if (richness.composite < richness_gate
                and richness.entity_count == 0
                and richness.technical_term_count == 0):
            force_score = _cfg_float("importance_engine.richness_fast_gate.force_score",
                                     DEFAULT_RICHNESS_GATE_FORCE_SCORE)
            level = ImportanceLevel.DISCARD
            self._total_discarded += 1
            return ImportanceResult(
                score=force_score, level=ImportanceLevel.DISCARD, signal=signal,
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
            # Burst detection: compare against the *previous* last_seen BEFORE
            # updating it. The old order wrote last_seen = now first, which made
            # `now - entry.last_seen < 3600` always true and burst counts grow
            # forever. Outside the window the burst restarts at 1; inside it the
            # count increments but is capped at BURST_MAX.
            # elapsed == 0 (same tag in both tags and concepts of one call) or
            # slightly negative (clock skew) counts as within-window.
            elapsed = now - entry.last_seen
            if elapsed < BURST_WINDOW_SECONDS:
                entry.recent_burst = min(entry.recent_burst + 1, BURST_MAX)
            else:
                entry.recent_burst = 1  # first sighting (last_seen == 0) or window expired
            entry.last_seen = now

        # Fold this text's terms into known knowledge AFTER scoring so the
        # novelty of the current text is never diluted by itself.
        for term in terms:
            self._known_terms[term] = self._known_terms.get(term, 0) + 1

        return ImportanceResult(
            score=score, level=level, signal=signal, richness=richness,
            should_store=(level != ImportanceLevel.DISCARD),
            decay_rate=self._decay_rate(level),
            consolidation_priority=(2 if level == ImportanceLevel.CORE
                                    else 1 if level == ImportanceLevel.NORMAL else 0),
            retrieval_feedback=signal.retrieval_feedback,
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
    def _extract_terms(text: str, concepts: list[str], entities: list[str],
                       tags: list[str]) -> set[str]:
        """Keyword set used for novelty overlap (latin words + CJK bigrams).

        Concepts/entities/tags are added as whole terms so structured signals
        participate in the overlap even when tokenization would split them.
        """
        terms: set[str] = set()
        for word in re.findall(r"[a-z0-9]{2,}", text.lower()):
            terms.add(word)
        # CJK bigrams (single hanzi are too granular for meaningful overlap)
        for seq in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for i in range(len(seq) - 1):
                terms.add(seq[i:i + 2])
        for item in concepts + entities + tags:
            if item:
                terms.add(str(item).lower())
        return terms

    def _score_novelty(self, terms: set[str]) -> float:
        """1 − max overlap with known knowledge (plan.md: 新话题 vs 已存储知识).

        Empty known-knowledge or empty terms → fully novel (1.0). The known
        vocabulary is updated after scoring, so first-seen texts are always
        novelty 1.0 and verbatim repeats converge to 0.
        """
        if not self._known_terms or not terms:
            return 1.0
        overlap = sum(1 for term in terms if term in self._known_terms)
        return max(0.0, 1.0 - overlap / len(terms))

    def _score_retrieval_feedback(self, concepts: list[str], tags: list[str],
                                   retrieval_count: int) -> float:
        """Score based on retrieval history: how often was this topic successfully recalled?

        Uses tracked retrieval_count per concept/tag, combined with the
        retrieval success rate. Memories that are frequently retrieved but
        highly relevant get a boost; never-retrieved memories get a small penalty.
        """
        candidates = [c.lower() for c in concepts] + [t.lower() for t in tags]
        if not candidates:
            # Use global retrieval_count if available
            if retrieval_count > 0:
                return min(0.35, 0.05 + retrieval_count * 0.02)
            return 0.05

        # Average retrieval feedback across matched candidates
        scores = []
        for key in candidates:
            count = self._retrieval_counts.get(key, 0)
            rate = self._retrieval_success_rate.get(key, 0.5)
            if count >= 3:
                scores.append(min(0.95, 0.1 + count * 0.05 + rate * 0.3))
            elif count >= 1:
                scores.append(0.2 + count * 0.08)
            else:
                scores.append(0.05)  # never retrieved

        return min(1.0, sum(scores) / len(scores))

    def record_retrieval(self, key: str, success: bool = True) -> None:
        """Called after each retrieval to update the feedback loop."""
        self._retrieval_counts[key] = self._retrieval_counts.get(key, 0) + 1
        current_rate = self._retrieval_success_rate.get(key, 0.5)
        # Exponential moving average
        self._retrieval_success_rate[key] = current_rate * 0.7 + (1.0 if success else 0.0) * 0.3

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
