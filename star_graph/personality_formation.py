"""Personality Formation v2 — real cognitive profile inference.

NOW WITH ACTUAL INFERENCE:
1. Cognitive style detection from interaction patterns (question types, abstraction level, decision basis)
2. Value system inference from Belief System projections
3. Behavioral pattern extraction with reinforcement and decay
4. Evolution trajectory from topic trends over time
5. Identity marker synthesis from stable behavior clusters

This is NOT a counter. This analyzes actual memory/belief data to form a profile.
"""

from dataclasses import dataclass, field
from collections import defaultdict, Counter
import time
import math
import re
from typing import Optional


# ── Profile data types ─────────────────────────────────────

@dataclass
class CognitiveStyle:
    """Inferred cognitive style from interaction analysis."""
    abstraction_level: str = "mixed"       # concrete / mixed / abstract / highly_abstract
    abstraction_score: float = 0.5
    decision_basis: str = "balanced"       # intuitive / balanced / analytical / systematic
    analytical_score: float = 0.5
    communication_style: str = "mixed"     # direct / detailed / concise / exploratory
    question_frequency: float = 0.3        # how often user asks vs states
    confidence: float = 0.2


@dataclass
class ValueSystem:
    """Inferred value priorities."""
    efficiency_over_convention: float = 0.5
    autonomy_over_guidance: float = 0.5
    depth_over_breadth: float = 0.5
    novelty_over_stability: float = 0.5
    pragmatism_over_purity: float = 0.5
    confidence: float = 0.15
    evidence_count: int = 0


@dataclass
class BehavioralPattern:
    pattern: str
    frequency: int = 0
    strength: float = 0.3
    first_seen: float = 0.0
    last_seen: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    category: str = "general"

    def reinforce(self, evidence_id: str = ""):
        self.frequency += 1
        self.strength = min(1.0, 0.3 + math.log(self.frequency + 1) * 0.25)
        self.last_seen = time.time()
        if evidence_id:
            self.evidence_ids.append(evidence_id)

    def decay(self, days_since_last: float):
        self.strength = max(0.05, self.strength * math.exp(-days_since_last / 120))


@dataclass
class EvolutionTrajectory:
    trending_toward: list[str] = field(default_factory=list)
    emerging_interests: list[str] = field(default_factory=list)
    fading_interests: list[str] = field(default_factory=list)
    stable_core: list[str] = field(default_factory=list)
    long_term_growth_areas: list[str] = field(default_factory=list)
    last_updated: float = 0.0


@dataclass
class CognitiveProfile:
    cognitive_style: CognitiveStyle = field(default_factory=CognitiveStyle)
    value_system: ValueSystem = field(default_factory=ValueSystem)
    behavioral_patterns: dict[str, BehavioralPattern] = field(default_factory=dict)
    identity_markers: list[str] = field(default_factory=list)
    evolution: EvolutionTrajectory = field(default_factory=EvolutionTrajectory)
    formed_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    data_points: int = 0
    memory_contributions: int = 0

    def summary(self) -> str:
        parts = []
        if self.identity_markers:
            parts.append(self.identity_markers[0])
        cs = self.cognitive_style
        if cs.abstraction_score > 0.6:
            parts.append("high-level abstract thinker")
        elif cs.abstraction_score > 0.4:
            parts.append("balanced concrete/abstract thinking")
        vs = self.value_system
        if vs.autonomy_over_guidance > 0.65:
            parts.append("strongly autonomous")
        if vs.efficiency_over_convention > 0.65:
            parts.append("efficiency-driven")
        if vs.depth_over_breadth > 0.65:
            parts.append("depth-oriented")
        if vs.pragmatism_over_purity > 0.65:
            parts.append("pragmatic builder")
        return " / ".join(parts) if parts else "profile forming..."

    def to_prompt_context(self) -> str:
        lines = ["# User Cognitive Profile"]
        lines.append(f"Summary: {self.summary()}")

        cs = self.cognitive_style
        if cs.confidence > 0.2:
            lines.append(f"\n## Cognitive Style (confidence: {cs.confidence:.2f})")
            lines.append(f"- Abstraction: {cs.abstraction_level} ({cs.abstraction_score:.2f})")
            lines.append(f"- Decision-making: {cs.decision_basis} ({cs.analytical_score:.2f})")
            lines.append(f"- Communication: {cs.communication_style}")

        vs = self.value_system
        if vs.confidence > 0.2:
            lines.append(f"\n## Values (confidence: {vs.confidence:.2f})")
            for name, key, threshold in [
                ("Efficiency > Convention", "efficiency_over_convention", 0.6),
                ("Autonomy > Guidance", "autonomy_over_guidance", 0.6),
                ("Depth > Breadth", "depth_over_breadth", 0.6),
                ("Novelty > Stability", "novelty_over_stability", 0.6),
                ("Pragmatism > Purity", "pragmatism_over_purity", 0.6),
            ]:
                v = getattr(vs, key)
                if v > threshold:
                    lines.append(f"- {name}: {v:.2f}")
                elif v < 0.4:
                    lines.append(f"- {name.replace('>', '<')}: {1-v:.2f}")

        if self.identity_markers:
            lines.append("\n## Identity Markers")
            for m in self.identity_markers[:5]:
                lines.append(f"- {m}")

        strong = sorted(
            [(p.pattern, p.strength, p.frequency) for p in self.behavioral_patterns.values()
             if p.strength > 0.5],
            key=lambda x: -x[1],
        )[:5]
        if strong:
            lines.append("\n## Behavior Patterns")
            for pat, strength, freq in strong:
                lines.append(f"- {pat} (×{freq}, {strength:.2f})")

        et = self.evolution
        if et.trending_toward or et.emerging_interests:
            lines.append("\n## Evolution")
            if et.trending_toward:
                lines.append(f"- Trending: {', '.join(et.trending_toward[:3])}")
            if et.emerging_interests:
                lines.append(f"- Emerging: {', '.join(et.emerging_interests[:3])}")
            if et.long_term_growth_areas:
                lines.append(f"- Long-term growth: {', '.join(et.long_term_growth_areas[:3])}")

        return "\n".join(lines)


# ── Abstraction level detection patterns ───────────────────

ABSTRACT_SIGNALS = [
    r'\b(?:架构|系统|框架|模式|范式|方法论|哲学|本质|根本|底层|抽象|概念|理论|原理|原则)\b',
    r'\b(?:architect\w+|system|framework|paradigm|methodolog\w+|philosoph\w+|principle|pattern|abstract\w+|conceptual|theoretical|fundamental)\b',
]

CONCRETE_SIGNALS = [
    r'\b(?:代码|bug|报错|调试|运行|安装|配置|部署|具体|实现|写|改|修|跑)\b',
    r'\b(?:code|bug|error|debug|run|install|config|deploy|fix|implement|write|build|compile)\b',
]

ANALYTICAL_SIGNALS = [
    r'\b(?:分析|比较|权衡|评估|基准|指标|数据|论证|推理|为什么|原因|所以|因此|结论)\b',
    r'\b(?:analyz\w+|compar\w+|benchmark|metric|data|reason\w+|why|because|therefore|conclusion|eval\w+|assess\w+)\b',
]


# ── Value inference patterns ───────────────────────────────

VALUE_SIGNALS = {
    "efficiency_over_convention": {
        "positive": [r'\b(?:效率|简洁|快速|简单|轻量|最少|避免重复|自动化|不要重复)\b',
                     r'\b(?:efficien\w+|simple|fast|lightweight|minimal|automate|DRY)\b'],
        "negative": [r'\b(?:规范|标准|惯例|传统|正规|流程)\b',
                     r'\b(?:convention|standard|traditional|formal|process)\b'],
    },
    "autonomy_over_guidance": {
        "positive": [r'\b(?:自由|灵活|自主|自己决定|不依赖|独立|控制权|自由度)\b',
                     r'\b(?:freedom|flexib\w+|autonom\w+|independent|control|self-direct)\b'],
        "negative": [r'\b(?:指导|引导|教程|模板|框架限制|规范约束)\b',
                     r'\b(?:guid\w+|tutorial|template|restrict\w+|constrain\w+)\b'],
    },
    "depth_over_breadth": {
        "positive": [r'\b(?:深入|深度|底层|彻底|完全理解|细节|原理|源码)\b',
                     r'\b(?:deep|thorough|detail|fundamental|internals|source)\b'],
        "negative": [r'\b(?:广泛|多个|很多|各种|全栈|面广|涉猎)\b',
                     r'\b(?:broad|wide|many|various|full.?stack|surface)\b'],
    },
    "novelty_over_stability": {
        "positive": [r'\b(?:新的|创新|实验|尝试|探索|前沿|最新|突破)\b',
                     r'\b(?:new|innovati\w+|experiment\w+|explor\w+|cutting.?edge|breakthrough)\b'],
        "negative": [r'\b(?:稳定|成熟|可靠|验证过|久经考验|生产级)\b',
                     r'\b(?:stabl\w+|mature|reliable|proven|production|battle.?tested)\b'],
    },
    "pragmatism_over_purity": {
        "positive": [r'\b(?:能用|实用|够用|先跑通|快速迭代|MVP|交付|上线|发布)\b',
                     r'\b(?:pragmat\w+|practical|shipping|MVP|deliver|release|iterat\w+)\b'],
        "negative": [r'\b(?:完美|纯粹|优雅|理想|最佳|完备|完全正确)\b',
                     r'\b(?:perfect|pure|elegant|ideal|optimal|complete|correct)\b'],
    },
}


# ── Personality Formation Engine v2 ────────────────────────

class PersonalityFormationEngine:
    """Forms cognitive profile from beliefs, interaction data, and memory patterns.

    Usage:
        pfe = PersonalityFormationEngine(belief_system=bs)
        pfe.ingest_memory({"content": "...", "tags": [...], "concepts": [...]})
        pfe.analyze_interaction(user_message="...", intent="question", concepts=[...])
        pfe.sync_from_beliefs()
        profile = pfe.get_profile()
        injection = profile.to_prompt_context()
    """

    def __init__(self, belief_system=None):
        self._belief_system = belief_system
        self._profile = CognitiveProfile()
        self._topic_tracker: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "first_seen": 0.0, "last_seen": 0.0, "recent": 0})
        self._question_history: list[dict] = []
        self._interaction_history: list[dict] = []
        self._total_interactions = 0

    # ── Data ingestion ─────────────────────────────────────

    def ingest_memory(self, memory: dict):
        """Ingest a single memory for pattern extraction."""
        self._profile.memory_contributions += 1
        self._profile.last_updated = time.time()
        now = time.time()

        text = memory.get("content", memory.get("text", ""))
        tags = memory.get("tags", [])
        concepts = memory.get("concepts", memory.get("extracted_concepts", []))
        importance = memory.get("importance", 0.5)

        # Track topics
        for topic in tags + concepts:
            tracker = self._topic_tracker[topic]
            if tracker["count"] == 0:
                tracker["first_seen"] = now
            tracker["count"] += 1
            tracker["last_seen"] = now
            if now - tracker["last_seen"] < 86400 * 7:  # 7 days
                tracker["recent"] += 1

        # Extract behavioral patterns from text
        patterns = self._extract_patterns(text, tags, concepts, importance)
        for pat in patterns:
            if pat not in self._profile.behavioral_patterns:
                self._profile.behavioral_patterns[pat] = BehavioralPattern(
                    pattern=pat, first_seen=now)
            self._profile.behavioral_patterns[pat].reinforce(
                memory.get("id", memory.get("anchor_id", "")))

    def analyze_interaction(self, user_message: str, *,
                            intent: str = "statement",
                            concepts: list[str] | None = None,
                            emotion_valence: float = 0.0):
        """Analyze a user interaction for cognitive style inference."""
        self._total_interactions += 1
        self._profile.data_points += 1
        self._interaction_history.append({
            "message": user_message[:500], "intent": intent,
            "concepts": concepts or [], "emotion": emotion_valence,
            "timestamp": time.time(),
        })
        if len(self._interaction_history) > 1000:
            self._interaction_history = self._interaction_history[-500:]

        # Question tracking
        if intent == "question":
            self._question_history.append({
                "message": user_message[:300],
                "concepts": concepts or [],
                "timestamp": time.time(),
            })
            if len(self._question_history) > 200:
                self._question_history = self._question_history[-100:]

        # Update cognitive style
        self._infer_cognitive_style(user_message, intent)

    # ── Cognitive style inference ──────────────────────────

    def _infer_cognitive_style(self, text: str, intent: str):
        cs = self._profile.cognitive_style

        # Abstraction level
        abstract_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in ABSTRACT_SIGNALS)
        concrete_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in CONCRETE_SIGNALS)
        total = abstract_count + concrete_count + 1

        current_abstract = abstract_count / total
        cs.abstraction_score = cs.abstraction_score * 0.85 + current_abstract * 0.15

        if cs.abstraction_score > 0.7:
            cs.abstraction_level = "highly_abstract"
        elif cs.abstraction_score > 0.5:
            cs.abstraction_level = "abstract"
        elif cs.abstraction_score > 0.3:
            cs.abstraction_level = "mixed"
        else:
            cs.abstraction_level = "concrete"

        # Analytical
        analytical_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in ANALYTICAL_SIGNALS)
        cs.analytical_score = cs.analytical_score * 0.85 + min(1.0, analytical_count / 5) * 0.15

        if cs.analytical_score > 0.7:
            cs.decision_basis = "systematic"
        elif cs.analytical_score > 0.5:
            cs.decision_basis = "analytical"
        elif cs.analytical_score > 0.3:
            cs.decision_basis = "balanced"
        else:
            cs.decision_basis = "intuitive"

        # Question frequency
        is_question = 1.0 if intent == "question" else 0.0
        cs.question_frequency = cs.question_frequency * 0.9 + is_question * 0.1

        # Communication style
        if len(text) > 500:
            cs.communication_style = "detailed"
        elif len(text) < 30:
            cs.communication_style = "direct"
        elif cs.question_frequency > 0.4:
            cs.communication_style = "exploratory"
        else:
            cs.communication_style = "concise"

        cs.confidence = min(1.0, cs.confidence + 0.01)

    # ── Pattern extraction ─────────────────────────────────

    def _extract_patterns(self, text: str, tags: list[str], concepts: list[str],
                          importance: float) -> list[str]:
        """Extract behavioral patterns from memory content."""
        patterns = []

        # Tag-based patterns
        for tag in tags:
            if len(tag) > 2:
                patterns.append(f"engages_with_{tag}")

        # Content-based patterns
        if re.search(r'(?:需要|想要|希望|计划|打算|目标)', text):
            patterns.append("goal_oriented_communication")

        if re.search(r'(?:问题|bug|错误|失败|不行|不能|不对|出问题)', text):
            patterns.append("problem_focused_interaction")

        if re.search(r'(?:设计|架构|方案|选择|决定)', text):
            patterns.append("design_level_thinking")

        if re.search(r'(?:实现|编码|写|改|修|跑|测试)', text):
            patterns.append("implementation_focused")

        if re.search(r'(?:文档|记录|说明|注释|readme)', text):
            patterns.append("documentation_aware")

        if importance > 0.6:
            patterns.append("high_value_interaction")

        return patterns

    # ── Value system inference ─────────────────────────────

    def infer_values_from_text(self, text: str):
        """Infer value priorities from text content."""
        vs = self._profile.value_system
        now = time.time()

        for value_key, signals in VALUE_SIGNALS.items():
            pos_score = sum(len(re.findall(p, text, re.IGNORECASE)) for p in signals["positive"])
            neg_score = sum(len(re.findall(p, text, re.IGNORECASE)) for p in signals["negative"])

            if pos_score + neg_score > 0:
                net = (pos_score - neg_score) / (pos_score + neg_score + 1)
                current = getattr(vs, value_key)
                # Move toward net signal
                new_value = current * 0.85 + (0.5 + net * 0.5) * 0.15
                setattr(vs, value_key, min(1.0, max(0.0, new_value)))
                vs.evidence_count += 1

        vs.confidence = min(1.0, vs.confidence + 0.02)
        vs.last_updated = now

    # ── Evolution analysis ─────────────────────────────────

    def analyze_evolution(self):
        """Detect trends, emerging/fading interests from topic history."""
        now = time.time()
        et = self._profile.evolution
        et.last_updated = now

        topics = []
        for topic, tracker in self._topic_tracker.items():
            days_active = (now - tracker["first_seen"]) / 86400 if tracker["first_seen"] else 1
            recency = tracker["recent"] / max(1, tracker["count"])
            topics.append((topic, tracker["count"], recency, days_active))

        # Stable core: frequent, long-running topics
        et.stable_core = [t for t, c, r, d in topics if c >= 5 and d > 30][:10]

        # Emerging: high recent activity, low total history
        emerging = [(t, r) for t, c, r, d in topics if c >= 3 and r > 0.5 and d < 60]
        et.emerging_interests = [t for t, r in sorted(emerging, key=lambda x: -x[1])[:5]]

        # Fading: was frequent, now low recency
        fading = [(t, r) for t, c, r, d in topics if c >= 5 and r < 0.2]
        et.fading_interests = [t for t, r in sorted(fading, key=lambda x: x[1])[:5]]

        # Trending: emerging topics people are actively engaging with
        et.trending_toward = et.emerging_interests[:2]

        # Long-term growth: topics with sustained increase
        et.long_term_growth_areas = [t for t, c, r, d in topics if c >= 8 and d > 60 and r > 0.4][:5]

    # ── Identity markers ───────────────────────────────────

    def synthesize_identity_markers(self):
        """Synthesize identity markers from stable patterns and beliefs."""
        markers = []

        # From behavioral patterns
        strong_patterns = sorted(
            [(p.pattern, p.strength) for p in self._profile.behavioral_patterns.values()
             if p.strength > 0.6 and p.frequency >= 3],
            key=lambda x: -x[1],
        )
        pattern_map = {
            "design_level_thinking": "系统架构思维",
            "implementation_focused": "实践驱动型开发者",
            "problem_focused_interaction": "问题解决导向",
            "goal_oriented_communication": "目标驱动型沟通",
            "high_value_interaction": "深度技术交流者",
            "documentation_aware": "文档意识强",
        }
        for pat, strength in strong_patterns:
            label = pattern_map.get(pat, pat.replace("_", " ").title())
            if label not in markers:
                markers.append(label)

        # From cognitive style
        cs = self._profile.cognitive_style
        if cs.abstraction_score > 0.7:
            markers.append("高度抽象思维者")
        if cs.analytical_score > 0.7:
            markers.append("系统性分析者")

        # From value system
        vs = self._profile.value_system
        if vs.efficiency_over_convention > 0.7:
            markers.append("效率优先主义者")
        if vs.autonomy_over_guidance > 0.7:
            markers.append("高度自主型")
        if vs.pragmatism_over_purity > 0.7:
            markers.append("实用主义建设者")
        if vs.depth_over_breadth > 0.7:
            markers.append("深度钻研者")

        self._profile.identity_markers = markers[:10]

    # ── Belief sync ────────────────────────────────────────

    def sync_from_beliefs(self):
        """Synchronize value system and identity from Belief System."""
        if not self._belief_system:
            return

        profile = self._belief_system.to_cognitive_profile()
        vs = self._profile.value_system
        now = time.time()

        # Value beliefs → value system
        for b in profile.get("values", []):
            self.infer_values_from_text(b["statement"])

        # Identity beliefs → identity markers
        for b in profile.get("identity_markers", []):
            statement = b["statement"]
            if statement not in self._profile.identity_markers:
                self._profile.identity_markers.append(statement)

        # Preference beliefs → behavioral patterns
        for b in profile.get("preferences", []):
            if b["statement"] not in self._profile.behavioral_patterns:
                self._profile.behavioral_patterns[b["statement"]] = BehavioralPattern(
                    pattern=b["statement"],
                    frequency=1,
                    strength=b.get("strength", 0.5),
                    category="preference",
                )

    # ── Main evolution cycle ───────────────────────────────

    def evolve(self):
        """Run full evolution: analyze trajectory, decay old patterns, synthesize markers."""
        self.analyze_evolution()
        self.synthesize_identity_markers()

        # Decay old behavioral patterns
        now = time.time()
        for pat in list(self._profile.behavioral_patterns.values()):
            days = (now - pat.last_seen) / 86400 if pat.last_seen else 0
            if days > 30:
                pat.decay(days)
            if pat.strength < 0.05 and pat.frequency < 2:
                del self._profile.behavioral_patterns[pat.pattern]

        self._profile.last_updated = now

    # ── Access ─────────────────────────────────────────────

    def get_profile(self) -> CognitiveProfile:
        return self._profile

    def get_injection(self) -> str:
        return self._profile.to_prompt_context()

    def get_summary(self) -> str:
        return self._profile.summary()

    def to_dict(self) -> dict:
        p = self._profile
        return {
            "cognitive_style": {
                "abstraction_level": p.cognitive_style.abstraction_level,
                "abstraction_score": p.cognitive_style.abstraction_score,
                "decision_basis": p.cognitive_style.decision_basis,
                "analytical_score": p.cognitive_style.analytical_score,
                "communication_style": p.cognitive_style.communication_style,
                "question_frequency": p.cognitive_style.question_frequency,
                "confidence": p.cognitive_style.confidence,
            },
            "value_system": {
                "efficiency_over_convention": p.value_system.efficiency_over_convention,
                "autonomy_over_guidance": p.value_system.autonomy_over_guidance,
                "depth_over_breadth": p.value_system.depth_over_breadth,
                "novelty_over_stability": p.value_system.novelty_over_stability,
                "pragmatism_over_purity": p.value_system.pragmatism_over_purity,
                "confidence": p.value_system.confidence,
                "evidence_count": p.value_system.evidence_count,
            },
            "identity_markers": p.identity_markers,
            "evolution_trajectory": {
                "trending_toward": p.evolution.trending_toward,
                "emerging_interests": p.evolution.emerging_interests,
                "fading_interests": p.evolution.fading_interests,
                "stable_core": p.evolution.stable_core,
                "long_term_growth_areas": p.evolution.long_term_growth_areas,
            },
            "behavioral_patterns": {
                k: {"pattern": v.pattern, "frequency": v.frequency, "strength": v.strength}
                for k, v in sorted(p.behavioral_patterns.items(),
                                   key=lambda x: -x[1].strength)[:20]
            },
            "data_points": p.data_points,
            "memory_contributions": p.memory_contributions,
            "summary": p.summary(),
        }
