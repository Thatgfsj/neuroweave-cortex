"""Belief System v2 — auto-extracts beliefs from memory clusters.

Beliefs are NOT manually added. They are EXTRACTED from:
1. Repeated patterns in memory clusters (cognitive_compression output)
2. Behavior patterns detected by personality formation
3. Explicit user statements about preferences/values
4. Contradiction between old and new evidence

Lifecycle:
    memory clusters → extract candidates → form belief (strength 0.3)
    → accumulating evidence → reinforce (strength↑)
    → contradictory evidence → challenge (strength↓)
    → strength < 0.1 AND stability < 0.2 → retire
    → two similar beliefs → merge (stronger than either)

Integration points:
    - cognitive_compression.py: clusters → extract_belief_candidates()
    - concept_cortex.py: related concepts → belief activation
    - memory graph: evidence links → belief strength
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, Counter
import time
import re


# ── Data types ─────────────────────────────────────────────

@dataclass
class Belief:
    id: str
    statement: str
    category: str = "general"
    strength: float = 0.3        # starts low, builds with evidence
    stability: float = 0.2       # resistance to change
    evidence_ids: list[str] = field(default_factory=list)
    evidence_quality: float = 0.0  # average quality of supporting evidence
    contradiction_ids: list[str] = field(default_factory=list)
    contradiction_severity: float = 0.0
    formed_at: float = 0.0
    last_reinforced: float = 0.0
    last_challenged: float = 0.0
    source: str = "extracted"     # extracted, manual, merged, inferred
    keywords: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.strength * self.stability

    def reinforce(self, evidence_id: str = "", evidence_quality: float = 0.5):
        boost = 0.03 + evidence_quality * 0.05
        self.strength = min(1.0, self.strength + boost)
        self.stability = min(1.0, self.stability + boost * 0.3)
        if evidence_id:
            self.evidence_ids.append(evidence_id)
        n = len(self.evidence_ids)
        self.evidence_quality = (self.evidence_quality * (n - 1) + evidence_quality) / max(1, n)
        self.last_reinforced = time.time()

    def challenge(self, contradiction_id: str = "", severity: float = 0.3):
        penalty = severity * 0.15 / max(0.2, self.stability)
        self.strength = max(0.0, self.strength - penalty)
        self.stability = max(0.05, self.stability - penalty * 0.2)
        if contradiction_id:
            self.contradiction_ids.append(contradiction_id)
        n = len(self.contradiction_ids)
        self.contradiction_severity = (self.contradiction_severity * (n - 1) + severity) / max(1, n)
        self.last_challenged = time.time()

    def is_stable(self) -> bool:
        return self.stability >= 0.65 and self.strength >= 0.6

    def is_weak(self) -> bool:
        return self.strength < 0.2

    def should_retire(self) -> bool:
        return self.strength < 0.08 and self.stability < 0.15


CATEGORIES = {
    "preference":   "what the user likes/dislikes/prefers",
    "value":        "what the user considers important/right/wrong",
    "worldview":    "how the user understands the world/systems",
    "identity":     "who the user believes they are",
    "capability":   "what the user can do, knows, or is learning",
    "methodology":  "how the user approaches problems and work",
}

CATEGORY_KEYWORDS = {
    "preference":   ["喜欢", "偏好", "倾向", "prefer", "like", "favorite", "常用", "习惯用"],
    "value":        ["重视", "重要", "优先", "价值", "关键", "核心", "坚持", "原则", "认为"],
    "worldview":    ["架构", "系统", "设计哲学", "本质", "根本上", "底层", "世界观"],
    "identity":     ["我是", "我在做", "定位", "角色", "方向", "领域", "专注", "identity"],
    "capability":   ["掌握", "熟悉", "精通", "经验", "用过", "开发过", "研究过", "skill"],
    "methodology":  ["方式", "方法", "流程", "步骤", "approach", "method", "策略", "先...再"],
}

EXPLICIT_BELIEF_SIGNALS = [
    # "I believe / I think / In my opinion" patterns
    (r'(?:我认为|我觉得|我相信|我的观点是|我的看法是|在我看来|我相信|我一直认为|我坚持)', 0.7),
    # "I prefer / I like" patterns
    (r'(?:我喜欢|我偏好|我倾向于|我更愿意|我习惯|我常用的|我的首选)', 0.6),
    # "In my experience / I've found" patterns
    (r'(?:根据我的经验|我发现|我注意到|我观察到|实践表明|实际使用中)', 0.5),
    # "The key is / What matters is" patterns
    (r'(?:关键是|重要的是|核心是|本质是|说到底|根本问题)', 0.6),
    # English equivalents
    (r'\b(?:I believe|I think|in my opinion|I prefer|I\'ve found|the key is|what matters)\b', 0.55),
]


# ── Belief System ──────────────────────────────────────────

class BeliefSystem:
    """Autonomous belief extraction and evolution from memory patterns.

    Usage:
        bs = BeliefSystem()
        # Auto-extract from memory clusters
        candidates = bs.extract_candidates_from_clusters(compressed_summaries)
        for c in candidates:
            bs.form_belief(c["statement"], category=c["category"],
                          evidence_ids=c["evidence_ids"])
        # Later, reinforce or challenge
        bs.reinforce("belief_0001", "m_089", quality=0.8)
    """

    def __init__(self):
        self._beliefs: dict[str, Belief] = {}
        self._by_category: dict[str, list[str]] = defaultdict(list)
        self._counter = 0
        self._extraction_history: list[dict] = []  # track what we've already extracted

    # ── CRUD ───────────────────────────────────────────────

    def form_belief(self, statement: str, *,
                    category: str = "",
                    initial_strength: float = 0.3,
                    evidence_ids: list[str] | None = None,
                    source: str = "extracted",
                    keywords: list[str] | None = None) -> Belief:
        """Form a belief. Category auto-detected if not provided."""
        if not category:
            category = self._classify_category(statement)
        if not keywords:
            keywords = self._extract_keywords(statement)
        now = time.time()
        self._counter += 1
        belief = Belief(
            id=f"belief_{self._counter:04d}",
            statement=statement,
            category=category,
            strength=initial_strength,
            evidence_ids=evidence_ids or [],
            formed_at=now,
            last_reinforced=now,
            source=source,
            keywords=keywords,
        )
        # Don't create duplicates — check similarity
        existing = self._find_similar(statement)
        if existing:
            existing.reinforce(evidence_id=(evidence_ids or [""])[0] if evidence_ids else "")
            existing.keywords = list(set(existing.keywords + keywords))
            return existing

        self._beliefs[belief.id] = belief
        self._by_category[category].append(belief.id)
        return belief

    def get(self, belief_id: str) -> Optional[Belief]:
        return self._beliefs.get(belief_id)

    def list_all(self, category: str | None = None, min_strength: float = 0.0) -> list[Belief]:
        ids = self._by_category.get(category, []) if category else list(self._beliefs.keys())
        return [self._beliefs[i] for i in ids if self._beliefs[i].strength >= min_strength]

    def list_stable(self) -> list[Belief]:
        return [b for b in self._beliefs.values() if b.is_stable()]

    # ── Auto-extraction from memory clusters ────────────────

    def extract_candidates_from_clusters(self, summaries: list[str],
                                         memory_ids: list[list[str]] | None = None) -> list[dict]:
        """Given compressed memory cluster summaries, extract belief candidates.

        This is the MAIN entry point for automated belief formation.
        Called after cognitive_compression produces cluster summaries.
        """
        candidates = []
        for i, summary in enumerate(summaries):
            # Check for explicit belief signals
            for pattern, confidence in EXPLICIT_BELIEF_SIGNALS:
                matches = re.findall(pattern, summary, re.IGNORECASE)
                if matches:
                    # Extract the surrounding context as the belief statement
                    for match in matches:
                        statement = self._clean_belief_statement(summary, match, confidence)
                        category = self._classify_category(statement)
                        evidence = memory_ids[i] if memory_ids and i < len(memory_ids) else []
                        # Skip if too similar to existing
                        if self._find_similar(statement) is None:
                            candidates.append({
                                "statement": statement,
                                "category": category,
                                "confidence": confidence,
                                "evidence_ids": evidence,
                                "source_summary": summary[:200],
                            })
        return candidates

    def extract_patterns_from_memories(self, memories: list[dict],
                                       min_occurrences: int = 3) -> list[dict]:
        """Scan a list of memories for repeated themes → belief candidates.

        Each memory dict should have: content (str), tags (list[str]), concepts (list[str]).
        """
        # Count keyword co-occurrence
        keyword_counter = Counter()
        concept_counter = Counter()
        memory_groups: dict[str, list[int]] = defaultdict(list)

        for i, mem in enumerate(memories):
            text = mem.get("content", mem.get("text", ""))
            tags = mem.get("tags", [])
            concepts = mem.get("concepts", [])
            for tag in tags:
                keyword_counter[tag] += 1
                memory_groups[tag].append(i)
            for concept in concepts:
                concept_counter[concept] += 1
                memory_groups[concept].append(i)

        candidates = []
        # Frequent themes → belief candidates
        for keyword, count in keyword_counter.most_common(30):
            if count >= min_occurrences:
                # Build statement from the keyword and its memory context
                related_memories = memory_groups.get(keyword, [])
                context = " ".join(
                    memories[idx].get("content", "")[:100]
                    for idx in related_memories[:3]
                )
                statement = self._synthesize_belief(keyword, count, context)
                if statement and self._find_similar(statement) is None:
                    candidates.append({
                        "statement": statement,
                        "category": self._classify_category(statement),
                        "evidence_ids": [memories[idx].get("id", "") for idx in related_memories],
                        "occurrence_count": count,
                        "confidence": min(0.8, count / 15),
                    })

        return candidates

    # ── Lifecycle ──────────────────────────────────────────

    def reinforce(self, belief_id: str, evidence_id: str = "", quality: float = 0.5) -> bool:
        b = self._beliefs.get(belief_id)
        if not b:
            return False
        b.reinforce(evidence_id, quality)
        return True

    def challenge(self, belief_id: str, contradiction_id: str = "", severity: float = 0.3) -> bool:
        b = self._beliefs.get(belief_id)
        if not b:
            return False
        b.challenge(contradiction_id, severity)
        return True

    def evolve(self, belief_id: str, new_statement: str) -> Optional[Belief]:
        b = self._beliefs.get(belief_id)
        if not b:
            return None
        b.statement = new_statement
        b.stability = max(0.2, b.stability - 0.1)
        b.keywords = self._extract_keywords(new_statement)
        return b

    def merge(self, id1: str, id2: str, new_statement: str = "") -> Optional[Belief]:
        b1 = self._beliefs.get(id1)
        b2 = self._beliefs.get(id2)
        if not b1 or not b2:
            return None
        merged_statement = new_statement or self._synthesize_merge_statement(b1, b2)
        return self.form_belief(
            statement=merged_statement,
            category=b1.category,
            initial_strength=max(b1.strength, b2.strength) + 0.1,
            evidence_ids=list(set(b1.evidence_ids + b2.evidence_ids)),
            source="merged",
            keywords=list(set(b1.keywords + b2.keywords)),
        )

    def retire(self, belief_id: str) -> bool:
        b = self._beliefs.get(belief_id)
        if not b:
            return False
        b.strength = 0.0
        b.stability = 0.0
        return True

    def prune_retired(self):
        self._beliefs = {k: v for k, v in self._beliefs.items() if not v.should_retire()}

    # ── Contradiction detection ────────────────────────────

    def detect_contradiction(self, belief_id: str, new_statement: str) -> Optional[dict]:
        """Check if a new statement contradicts an existing belief."""
        b = self._beliefs.get(belief_id)
        if not b:
            return None
        # Simple negation + keyword overlap check
        b_words = set(b.statement.lower().split())
        n_words = set(new_statement.lower().split())
        overlap = b_words & n_words
        if not overlap:
            return None
        overlap_ratio = len(overlap) / max(1, len(b_words))
        # Check for negation patterns
        negation_patterns = [r'\b(?:不|没有|不是|并非|不再|错了|不对)\b', r'\b(?:not?|never|no|don\'t|doesn\'t)\b']
        has_negation = any(re.search(p, new_statement, re.IGNORECASE) for p in negation_patterns)
        if has_negation and overlap_ratio > 0.3:
            severity = overlap_ratio * 0.8
            return {"contradiction": True, "severity": severity, "overlap_ratio": overlap_ratio}
        return None

    # ── Projection ─────────────────────────────────────────

    def to_cognitive_profile(self) -> dict:
        """Project all beliefs into a structured cognitive profile."""
        profile = {"preferences": [], "values": [], "worldview": [],
                   "identity_markers": [], "capabilities": [], "methodologies": []}
        for b in self.list_all(min_strength=0.3):
            entry = {"statement": b.statement, "strength": b.strength,
                     "stability": b.stability, "confidence": b.confidence,
                     "evidence_count": len(b.evidence_ids)}
            cat = b.category
            if cat in profile:
                profile[cat].append(entry)
            elif cat == "general":
                # Try to put it in the best-fitting category
                profile["preferences"].append(entry)
        return profile

    # ── Internal helpers ───────────────────────────────────

    def _find_similar(self, statement: str, threshold: float = 0.65) -> Optional[Belief]:
        """Find an existing belief that's too similar (avoid duplicates)."""
        s_words = set(statement.lower().split())
        if not s_words:
            return None
        for b in self._beliefs.values():
            b_words = set(b.statement.lower().split())
            if not b_words:
                continue
            jaccard = len(s_words & b_words) / len(s_words | b_words)
            if jaccard > threshold:
                return b
        return None

    @staticmethod
    def _classify_category(text: str) -> str:
        """Auto-detect belief category from text content."""
        scores = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text.lower())
            scores[cat] = score
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    @staticmethod
    def _extract_keywords(text: str, max_kw: int = 8) -> list[str]:
        """Extract key terms from belief statement."""
        # Simple: pick meaningful words (2+ chars Chinese, 3+ chars English)
        words = re.findall(r'[一-鿿]{2,}|[a-zA-Z]{3,}', text)
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'what',
                     'when', 'where', 'which', 'about', 'their', 'they', 'been', 'would',
                     'could', 'should', 'there', 'think', 'because', 'about', 'which',
                     '的', '了', '是', '在', '和', '也', '就', '都', '而', '及', '与',
                     '着', '或', '一个', '没有', '我们', '你们', '他们', '它们', '这个', '那个'}
        return [w for w in words if w.lower() not in stopwords][:max_kw]

    @staticmethod
    def _clean_belief_statement(text: str, signal: str, confidence: float) -> str:
        """Extract a clean belief statement from text around a signal phrase."""
        # Find signal position and take surrounding context
        idx = text.lower().find(signal.lower()) if signal.lower() in text.lower() else text.find(signal)
        if idx >= 0:
            start = max(0, idx - 10)
            end = min(len(text), idx + len(signal) + 120)
            statement = text[start:end].strip()
        else:
            statement = text[:200]
        # Clean up
        statement = re.sub(r'\s+', ' ', statement).strip()
        # Remove leading/trailing partial words
        if len(statement) > 200:
            statement = statement[:200] + "..."
        return statement

    def _synthesize_belief(self, keyword: str, count: int, context: str) -> Optional[str]:
        """Synthesize a belief statement from a frequent keyword and its context."""
        # Look for explicit opinion patterns in the context
        for pattern, _ in EXPLICIT_BELIEF_SIGNALS:
            m = re.search(pattern, context, re.IGNORECASE)
            if m:
                return self._clean_belief_statement(context, m.group(), 0.5)

        # Fallback: construct from keyword
        templates = {
            "preference": f"用户关注 {keyword}",
            "value": f"用户重视 {keyword}",
            "methodology": f"用户倾向于使用 {keyword}",
            "capability": f"用户熟悉 {keyword}",
        }
        cat = self._classify_category(keyword)
        template = templates.get(cat, f"用户经常涉及 {keyword}")
        return template

    @staticmethod
    def _synthesize_merge_statement(b1: Belief, b2: Belief) -> str:
        """Synthesize a merged belief from two similar beliefs."""
        # Use the longer, more specific statement as base
        base = b1.statement if len(b1.statement) >= len(b2.statement) else b2.statement
        # Append key distinction from the other
        other_keywords = [kw for kw in b2.keywords if kw not in b1.keywords]
        if other_keywords:
            return f"{base}（含 {', '.join(other_keywords[:3])}）"
        return base

    # ── Stats ──────────────────────────────────────────────

    def stats(self) -> dict:
        total = len(self._beliefs)
        return {
            "total_beliefs": total,
            "stable_beliefs": len(self.list_stable()),
            "weak_beliefs": len([b for b in self._beliefs.values() if b.is_weak()]),
            "by_category": {c: len(ids) for c, ids in self._by_category.items()},
            "avg_strength": sum(b.strength for b in self._beliefs.values()) / max(1, total),
            "avg_stability": sum(b.stability for b in self._beliefs.values()) / max(1, total),
            "avg_evidence": sum(len(b.evidence_ids) for b in self._beliefs.values()) / max(1, total),
        }
