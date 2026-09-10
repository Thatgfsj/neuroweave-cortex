"""A/B experiment core for Neuroweave-Cortex — config injection + evaluation harness.

This module provides the shared machinery for the two A/B experiments:

  Experiment 1 (importance weights, write-side gate):
      W-code        — code defaults (emotion=.18 repetition=.22 goal=.28
                      retrieval_feedback=.12 richness=.12 future=.08)
      W-plan        — plan.md legacy design (emotion=.25 repetition=.30 goal=.20
                      novelty=.15 future=.10)
      W-code+nov    — code defaults + novelty=0.10, all other weights scaled
                      by 0.90 so the total is exactly 1.00

  Experiment 2 (retrieval formula, recall ranking):
      R-hybrid      — HybridFusionRetriever (α=.50 β=.12 γ=.18 δ=.08 ε=.12)
      R-plan        — plan.md five-factor linear fusion implemented here as
                      PlanFiveFactorRetriever (post-hoc reranker over the graph)

HARD CONSTRAINT HONORED: this module only ADDS new files. It never monkey-patches
or edits existing star_graph/ / nwc/ sources. All variants are injected through
public subclassing / constructor arguments:

  * Importance weights   → ImportanceWeights subclass + ImportanceEngine subclass
                           (novelty lives only in the subclass; see
                           NoveltyImportanceEngine below).
  * Retrieval formula    → Retriever subclass (PlanFiveFactorRetriever) with the
                           same ``retrieve()`` interface as HybridFusionRetriever.

Evaluation protocol (both modes):
  clean env (fresh tempdir storage) → batch write (importance gate applies here)
  → per-query recall (retrieval formula applies here) → hit@k / MRR / has_answer
  → markdown table. All randomness flows through seeded ``random.Random``.

Synthetic mode: N fact memories + M noise chit-chat lines, queries paraphrase
facts, ground truth = anchor id of the source fact (anchor ids are deterministic:
blake2b(text + source_session)).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Offline-first embeddings: sentence-transformers' online model resolution can
# hang for minutes when the HF cache is cold. With HF_HUB_OFFLINE=1 a cached
# model still loads, and a cold cache raises fast → EmbeddingProvider falls
# back to TF-IDF/hash (deterministic, offline). Export AB_ONLINE_EMBEDDINGS=1
# to opt into online model download.
if os.environ.get("AB_ONLINE_EMBEDDINGS", "") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Make repo-root imports work when invoked as a script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from star_graph import MemoryManager
from star_graph.anchor import Anchor, EmbedderRegistry
from star_graph.embedding import reset_embedder
from star_graph.graph import Constellation
from star_graph.importance_engine import (
    ImportanceEngine,
    ImportanceLevel,
    ImportanceResult,
    ImportanceSignal,
    ImportanceWeights,
)
from star_graph.retrieval_engine.retriever import (
    HybridFusionRetriever,
    RetrievalResult,
    Retriever,
)

# ═══════════════════════════════════════════════════════════════
# 1. Config presets
# ═══════════════════════════════════════════════════════════════

#: Default HybridFusionRetriever weights (match retriever.py defaults exactly).
HYBRID_FUSION_WEIGHTS = {"alpha": 0.50, "beta": 0.12, "gamma": 0.18,
                         "delta": 0.08, "epsilon": 0.12}

#: plan.md five-factor weights (plan.md "检索公式" section).
PLAN_FIVE_FACTOR_WEIGHTS = {"semantic": 0.35, "temporal": 0.20,
                            "importance": 0.25, "relationship": 0.10,
                            "frequency": 0.10}

#: W-code — current code defaults (ImportanceWeights defaults in
#: star_graph/importance_engine.py). novelty weight is 0 (signal unused).
W_CODE_WEIGHTS: dict[str, float] = {
    "emotion": 0.18, "repetition": 0.22, "goal": 0.28,
    "retrieval_feedback": 0.12, "richness": 0.12, "future": 0.08,
    "novelty": 0.0,
}

#: W-plan — plan.md legacy design. NOTE: plan.md's design has no
#: retrieval_feedback / richness terms; those weights are 0 here.
W_PLAN_WEIGHTS: dict[str, float] = {
    "emotion": 0.25, "repetition": 0.30, "goal": 0.20,
    "retrieval_feedback": 0.0, "richness": 0.0, "future": 0.10,
    "novelty": 0.15,
}


def _w_code_plus_nov(novelty: float = 0.10) -> dict[str, float]:
    """W-code + novelty: code defaults with novelty added; the six base weights
    are scaled by (1 - novelty) so the total stays exactly 1.00.

    With novelty=0.10 (scale 0.90):
        emotion=0.162 repetition=0.198 goal=0.252
        retrieval_feedback=0.108 richness=0.108 future=0.072 novelty=0.10
    """
    scale = 1.0 - novelty
    w = {k: v * scale for k, v in W_CODE_WEIGHTS.items() if k != "novelty"}
    w["novelty"] = novelty
    return w


@dataclass
class ABArm:
    """One arm of an A/B experiment."""
    name: str
    description: str
    # importance weights (exp 1); keys match ImportanceWeights fields + "novelty"
    importance_weights: Optional[dict[str, float]] = None
    # retrieval config (exp 2): "hybrid" or "plan"
    retrieval: str = "hybrid"
    # fixed retrieval config used during evaluation of exp-1 arms
    eval_retrieval: str = "hybrid"
    extra: dict[str, Any] = field(default_factory=dict)

    def weights_summary(self) -> str:
        if not self.importance_weights:
            return "-"
        return ", ".join(f"{k}={v:.3f}" for k, v in self.importance_weights.items())


def importance_arms() -> list[ABArm]:
    """The three arms of experiment 1 (importance weights)."""
    nov = 0.10
    return [
        ABArm(
            name="W-code",
            description="代码默认权重 (importance_engine.py v2 defaults)",
            importance_weights=dict(W_CODE_WEIGHTS),
        ),
        ABArm(
            name="W-plan",
            description="plan.md 旧设计权重 (novelty=0.15, 无 retrieval_feedback/richness)",
            importance_weights=dict(W_PLAN_WEIGHTS),
        ),
        ABArm(
            name="W-code+nov",
            description=f"代码默认 + novelty={nov}（其余六项 ×{1 - nov:.2f} 归一化，合计=1.00）",
            importance_weights=_w_code_plus_nov(nov),
        ),
    ]


def retrieval_arms() -> list[ABArm]:
    """The two arms of experiment 2 (retrieval formula)."""
    return [
        ABArm(
            name="R-hybrid",
            description="HybridFusionRetriever 现行默认 α=.50 β=.12 γ=.18 δ=.08 ε=.12",
            retrieval="hybrid",
        ),
        ABArm(
            name="R-plan",
            description="plan.md 五因子: semantic.35 + temporal.20 + importance.25 "
                        "+ relationship.10 + frequency.10",
            retrieval="plan",
        ),
    ]


# ═══════════════════════════════════════════════════════════════
# 2. Novelty injection (no edits to importance_engine.py)
# ═══════════════════════════════════════════════════════════════

# Bilingual stopword set for keyword extraction (kept local — do not modify
# star_graph modules).
_NOVELTY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "about",
    "as", "by", "from", "that", "this", "these", "those", "it", "its",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "his", "her",
    "they", "them", "their", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "just", "very", "really",
    "so", "if", "then", "than", "when", "what", "which", "who", "how",
    "where", "why", "not", "no", "yes",
    "的", "了", "是", "我", "你", "他", "她", "它", "们", "这", "那",
    "很", "也", "都", "就", "在", "不", "和", "与", "有", "个",
}
_CJK_RE = re.compile(r"[一-鿿]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def extract_keywords(text: str) -> set[str]:
    """Language-agnostic keyword set: latin/digit tokens (len>2, not stopwords)
    plus CJK character bigrams. Used symmetrically for incoming text and for
    stored anchors, so overlap is comparable on both sides."""
    keys: set[str] = set()
    lowered = text.lower()
    for tok in _TOKEN_RE.findall(lowered):
        if len(tok) > 2 and tok not in _NOVELTY_STOPWORDS:
            keys.add(tok)
    cjk = _CJK_RE.findall(text)
    for i in range(len(cjk) - 1):
        keys.add(cjk[i] + cjk[i + 1])
    return keys


def anchor_keyword_set(anchor: Anchor) -> set[str]:
    """Keyword set for a stored anchor, from its tags + exact_match_keys + text.

    plan.md defines novelty as overlap against *already stored knowledge*; the
    anchor-side surface is whatever the anchor carries: ``tags``, the entity
    keys harvested into the exact-match cache, and the raw text tokens."""
    keys: set[str] = set()
    for tag in anchor.tags or []:
        keys |= extract_keywords(str(tag))
        low = str(tag).lower()
        if len(low) > 1:
            keys.add(low)
    for k in anchor.exact_match_keys or []:
        keys |= extract_keywords(str(k))
    keys |= extract_keywords(anchor.text or "")
    return keys


class NoveltyImportanceWeights(ImportanceWeights):
    """ImportanceWeights + a novelty field (dataclass subclass — no source edit)."""

    def __init__(self, base: dict[str, float]):
        super().__init__(
            emotion=base.get("emotion", 0.0),
            repetition=base.get("repetition", 0.0),
            goal=base.get("goal", 0.0),
            retrieval_feedback=base.get("retrieval_feedback", 0.0),
            richness=base.get("richness", 0.0),
            future=base.get("future", 0.0),
        )
        self.novelty = base.get("novelty", 0.0)


class NoveltyImportanceSignal(ImportanceSignal):
    """ImportanceSignal + a novelty_signal field, with compute() extended."""

    def __init__(self, base: ImportanceSignal, novelty_signal: float = 0.0):
        super().__init__(
            emotion_signal=base.emotion_signal,
            repetition_signal=base.repetition_signal,
            goal_relation=base.goal_relation,
            retrieval_feedback=base.retrieval_feedback,
            content_richness=base.content_richness,
            future_recall_likelihood=base.future_recall_likelihood,
        )
        self.novelty_signal = novelty_signal

    def compute(self, weights: ImportanceWeights) -> float:
        base = super().compute(weights)
        return base + self.novelty_signal * getattr(weights, "novelty", 0.0)


class NoveltyImportanceEngine(ImportanceEngine):
    """ImportanceEngine with a plan.md novelty signal.

    novelty = 1 − overlap, where overlap is the max fraction of the incoming
    text's keyword set already covered by any stored anchor's keyword set
    (tags / exact_match_keys / text). "New topic vs stored knowledge."

    Minimal-invasion strategy: subclass (NOT edit). ``evaluate()`` calls
    ``super().evaluate()`` so all base signals (emotion, repetition, goal,
    richness, topic tracker updates, noise force-discard) behave identically,
    then re-scores the result with the configured weights including novelty:

        score = Σ base_signalᵢ × wᵢ  +  novelty_signal × w_novelty

    The noise force-discard short-circuit (richness<0.08, no entities/tech
    terms → score 0.05/DISCARD) is intentionally preserved: the noise gate is
    orthogonal to weighting and keeping it identical across arms removes a
    confound.
    """

    def __init__(self, weights: NoveltyImportanceWeights, graph=None, **kw):
        super().__init__(weights=weights, **kw)
        # ``graph`` is a live StarGraph; anchors stored so far define "known".
        self._graph = graph
        self._novelty_last = 0.0

    def _anchor_sets(self) -> list[set[str]]:
        if self._graph is None:
            return []
        return [anchor_keyword_set(a) for a in self._graph.anchors.values()]

    @staticmethod
    def _novelty(text_keys: set[str], anchor_sets: list[set[str]]) -> float:
        if not text_keys or not anchor_sets:
            return 1.0  # nothing known yet / nothing comparable → fully novel
        best = 0.0
        for akeys in anchor_sets:
            if not akeys:
                continue
            overlap = len(text_keys & akeys) / len(text_keys)
            if overlap > best:
                best = overlap
            if best >= 1.0:
                break
        return 1.0 - best

    def evaluate(self, text: str, **kwargs) -> ImportanceResult:
        text_keys = extract_keywords(text)
        anchor_sets = self._anchor_sets()
        novelty = self._novelty(text_keys, anchor_sets)
        self._novelty_last = novelty

        result = super().evaluate(text, **kwargs)

        # Preserve the base noise force-discard short-circuit untouched.
        if result.score <= 0.051 and result.level == ImportanceLevel.DISCARD \
                and not result.should_store and result.richness.composite < 0.08 \
                and result.richness.entity_count == 0:
            return result

        signal = NoveltyImportanceSignal(result.signal, novelty_signal=novelty)
        score = signal.compute(self.weights)
        score = min(1.0, max(0.0, score))
        level = ImportanceLevel.classify(score)

        was_discard = result.level == ImportanceLevel.DISCARD
        result.score = score
        result.level = level
        result.should_store = (level != ImportanceLevel.DISCARD)
        result.decay_rate = self._decay_rate(level)
        result.consolidation_priority = (2 if level == ImportanceLevel.CORE
                                         else 1 if level == ImportanceLevel.NORMAL else 0)
        if level == ImportanceLevel.DISCARD and not was_discard:
            self._total_discarded += 1
        elif was_discard and level != ImportanceLevel.DISCARD:
            self._total_discarded = max(0, self._total_discarded - 1)
        return result


def build_importance_engine(arm: ABArm, graph) -> tuple[ImportanceEngine, bool]:
    """Build the importance engine for an arm. Returns (engine, uses_novelty).

    For arms with novelty weight > 0 this returns a NoveltyImportanceEngine
    wired to the live graph; otherwise a plain ImportanceEngine with the arm's
    weights (W-code path exercises the untouched default class).
    """
    weights = dict(arm.importance_weights or W_CODE_WEIGHTS)
    if weights.get("novelty", 0.0) > 0.0:
        w = NoveltyImportanceWeights(weights)
        return NoveltyImportanceEngine(weights=w, graph=graph), True
    w = ImportanceWeights(
        emotion=weights.get("emotion", 0.18),
        repetition=weights.get("repetition", 0.22),
        goal=weights.get("goal", 0.28),
        retrieval_feedback=weights.get("retrieval_feedback", 0.12),
        richness=weights.get("richness", 0.12),
        future=weights.get("future", 0.08),
    )
    return ImportanceEngine(weights=w), False


# ═══════════════════════════════════════════════════════════════
# 3. Retrieval arm: plan.md five-factor retriever (subclass, no source edit)
# ═══════════════════════════════════════════════════════════════

class PlanFiveFactorRetriever(Retriever):
    """plan.md five-factor linear fusion:

        final = semantic×0.35 + temporal×0.20 + importance×0.25
              + relationship_strength×0.10 + recall_frequency×0.10

    Factor sources (all existing fields, no schema changes):
        semantic              cosine(query_emb, anchor.embedding)
                              (text-bigram overlap fallback when no embedding)
        temporal              recency factor exp(-hours_since_last_access / 168)
                              — same 1-week decay constant HybridFusion uses
        importance            anchor.vector.importance
        relationship_strength mean traversal weight of graph.neighbors(aid)
        recall_frequency      anchor.vector.frequency
    """

    def __init__(self, graph, weights: Optional[dict[str, float]] = None):
        super().__init__(graph)
        w = dict(PLAN_FIVE_FACTOR_WEIGHTS)
        if weights:
            w.update(weights)
        self.w_semantic = w["semantic"]
        self.w_temporal = w["temporal"]
        self.w_importance = w["importance"]
        self.w_relationship = w["relationship"]
        self.w_frequency = w["frequency"]

    @property
    def name(self) -> str:
        return "PlanFiveFactor"

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return max(0.0, dot / (na * nb + 1e-8))

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        def bigrams(s):
            return {s[i:i + 2] for i in range(len(s) - 1)}
        ba, bb = bigrams(a.lower()), bigrams(b.lower())
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)

    def _relationship_strength(self, anchor_id: str) -> float:
        neighbors = self.graph.neighbors(anchor_id)  # [(nid, traversal_weight)]
        if not neighbors:
            return 0.0
        return sum(w for _, w in neighbors) / len(neighbors)

    def score_anchor(self, query: str, embedding: Optional[list[float]],
                     anchor: Anchor, now: float) -> tuple[float, dict[str, float]]:
        if anchor.embedding and embedding:
            semantic = self._cosine(embedding, anchor.embedding)
        else:
            semantic = self._text_overlap(query, anchor.text)
        hours_since = max(0.0, (now - anchor.last_activated_at) / 3600.0)
        temporal = math.exp(-hours_since / 168.0)
        importance = float(anchor.vector.importance)
        relationship = self._relationship_strength(anchor.id)
        frequency = float(anchor.vector.frequency)
        final = (self.w_semantic * semantic
                 + self.w_temporal * temporal
                 + self.w_importance * importance
                 + self.w_relationship * relationship
                 + self.w_frequency * frequency)
        return max(0.0, final), {
            "semantic": semantic, "temporal": temporal, "importance": importance,
            "relationship_strength": relationship, "recall_frequency": frequency,
        }

    def retrieve(self, query: str, embedding: list[float] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()
        if not embedding:
            from star_graph.embedding import get_embedder
            embedding = get_embedder().encode(query)
        now = time.time()
        scored: list[tuple[float, Anchor, dict[str, float]]] = []
        for anchor in self.graph.anchors.values():
            final, components = self.score_anchor(query, embedding, anchor, now)
            scored.append((final, anchor, components))
        scored.sort(key=lambda x: -x[0])
        result = RetrievalResult(
            constellations=[Constellation(anchors=[a], edges=[])
                            for _, a, _ in scored[:top_k]],
            latency_ms=(time.perf_counter() - t0) * 1000,
            method=self.name,
            top_score=scored[0][0] if scored else 0.0,
        )
        return result


def build_retriever(kind: str, graph, weights: Optional[dict[str, float]] = None) -> Retriever:
    """Build a retriever by arm key. "hybrid" → HybridFusionRetriever (code
    defaults, no source edit); "plan" → PlanFiveFactorRetriever (this module)."""
    if kind == "hybrid":
        return HybridFusionRetriever(graph, **HYBRID_FUSION_WEIGHTS)
    if kind == "plan":
        return PlanFiveFactorRetriever(graph, weights=weights)
    raise ValueError(f"unknown retrieval arm: {kind}")


# ═══════════════════════════════════════════════════════════════
# 4. Datasets
# ═══════════════════════════════════════════════════════════════

@dataclass
class MemoryRecord:
    text: str
    tags: list[str]
    is_fact: bool  # False → noise/chit-chat, gate may discard


@dataclass
class QueryCase:
    query: str
    fact_index: int  # index into the fact list; resolved to anchor id post-write


_PEOPLE = ["Mara", "Julian", "Priya", "Dmitri", "Sofia", "Kenji", "Amara",
           "Lars", "Ines", "Victor", "Noor", "Tomas"]
_ITEMS = ["lens cleaning kit", "camping stove", "vinyl records", "tea tin",
          "spare keyboard", "hiking boots", "watercolor paints", "seed archive",
          "film camera", "chess clock", "espresso grinder", "saxophone"]
_PLACES = ["studio cabinet", "garage shelf", "balcony locker", "attic trunk",
           "kitchen drawer", "basement rack", "office locker", "garden shed"]
_TOPICS = ["photography", "cooking", "marathon training", "beekeeping",
           "robotics club", "jazz piano", "pottery", "astro photography",
           "board games", "rock climbing", "baking", "birdwatching"]
_NOISE_LINES = [
    "haha that's funny", "ok cool", "nice", "yeah I guess so", "lol",
    "sure whatever you say", "hmm maybe", "sounds good", "thanks a lot",
    "good morning", "how are you today", "the weather is fine",
    "I don't know really", "just thinking out loud", "anyway ok",
    "that was a long day", "see you later", "brb", "hello there",
    "you know what I mean", "it happens", "fine fine", "alright then",
    "well ok then", "yep", "nope maybe", "cool cool", "oh I see",
]


def build_synthetic_corpus(n_facts: int = 200, n_noise: int = 100,
                           seed: int = 42) -> tuple[list[MemoryRecord], list[QueryCase]]:
    """Deterministic synthetic corpus: fact memories + noise chit-chat.

    Facts interleave a person / item / place / topic from fixed vocabularies so
    queries can paraphrase them while sharing distinguishing tokens. Noise lines
    are generic filler the importance gate should (mostly) discard.

    Returns (records, queries); queries reference facts by index.
    """
    rng = random.Random(seed)
    facts: list[MemoryRecord] = []
    seen: set[str] = set()
    i = 0
    while len(facts) < n_facts:
        person = rng.choice(_PEOPLE)
        item = rng.choice(_ITEMS)
        place = rng.choice(_PLACES)
        topic = rng.choice(_TOPICS)
        template = rng.choice([
            f"{person} keeps the {item} in the {place} next to the {topic} notes.",
            f"{person} stores a spare {item} at the {place} for the {topic} project.",
            f"The {item} that {person} uses for {topic} lives in the {place}.",
            f"{person} moved the {item} to the {place} after the {topic} workshop.",
            f"For {topic}, {person} always keeps the {item} inside the {place}.",
        ])
        if template in seen:
            continue
        seen.add(template)
        facts.append(MemoryRecord(
            text=template,
            tags=[person.lower(), topic.lower().replace(" ", "_")],
            is_fact=True,
        ))
        i += 1

    noise = [MemoryRecord(text=rng.choice(_NOISE_LINES), tags=[], is_fact=False)
             for _ in range(n_noise)]

    records: list[MemoryRecord] = []
    fact_pos: list[int] = []
    # Interleave facts and noise in write order (seeded shuffle).
    merged = [(r, True) for r in facts] + [(r, False) for r in noise]
    rng.shuffle(merged)
    for rec, is_fact in merged:
        if is_fact:
            fact_pos.append(len(records))
        records.append(rec)

    queries: list[QueryCase] = []
    query_rng = random.Random(seed + 1)
    step = max(1, n_facts // 50)
    for idx in range(0, n_facts, step):
        fact = facts[idx]
        person = fact.tags[0].capitalize()
        item = next(t for t in _ITEMS if t in fact.text)
        topic = fact.tags[1].replace("_", " ")
        q = query_rng.choice([
            f"Where is the {item}?",
            f"Who has the {item} and where is it kept?",
            f"What did {person} say about {topic}?",
            f"Which place stores the {item} for {topic}?",
            f"{person} — {item}: where?",
        ])
        queries.append(QueryCase(query=q, fact_index=idx))
    return records, queries


LOCOMO_CANDIDATE_PATHS = [
    r"C:/Users/thatg/AppData/Local/Temp/locomo-10/data/locomo10.json",
    "benchmarks/data/locomo10.json",
    "benchmarks/data/locomo.json",
]


def load_locomo_corpus(max_sessions: int = 3) -> Optional[tuple[list[MemoryRecord], list[QueryCase]]]:
    """Load LoCoMo if locally available. Each dialogue turn becomes a memory;
    each QA becomes a query whose relevant anchors are resolved post-write by
    answer-text containment (pragmatic approximation — documented in README).

    Returns None when the dataset file is not found (auto-degrade to synthetic).
    """
    path = next((p for p in LOCOMO_CANDIDATE_PATHS if os.path.exists(p)), None)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception:
        return None

    records: list[MemoryRecord] = []
    queries: list[QueryCase] = []
    import unicodedata

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKD", str(s))).strip().lower()

    turns: list[str] = []
    qas: list[dict] = []
    for conv in dataset[:max_sessions]:
        for turn in conv.get("conversation", {}).values() \
                if isinstance(conv.get("conversation"), dict) else conv.get("conversation", []):
            speaker = turn.get("speaker", "")
            text = turn.get("text", "") or turn.get("content", "")
            if not text:
                continue
            turns.append(f"{speaker}: {text}")
            records.append(MemoryRecord(
                text=f"{speaker}: {text}",
                tags=[f"speaker:{speaker}".lower()],
                is_fact=True,
            ))
        for qa in conv.get("qa", []) or []:
            qas.append({"question": qa.get("question", ""),
                        "answer": qa.get("answer", "")})

    # Map QA → fact index by answer containment (resolved to anchor post-write).
    turn_norm = [norm(t.text) for t in records]
    for qa in qas:
        ans = norm(qa["answer"])
        if len(ans) < 4:
            continue
        for idx, tn in enumerate(turn_norm):
            if ans in tn:
                queries.append(QueryCase(query=qa["question"], fact_index=idx))
                break
    if not records or not queries:
        return None
    return records, queries


# ═══════════════════════════════════════════════════════════════
# 5. Experiment runners
# ═══════════════════════════════════════════════════════════════

@dataclass
class ArmMetrics:
    arm: str
    description: str
    n_written: int = 0
    n_gate_stored: int = 0
    n_gate_discarded: int = 0
    n_gate_errors: int = 0
    avg_stored_importance: float = 0.0
    avg_novelty: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    has_answer: float = 0.0
    n_queries: int = 0
    avg_recall_ms: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_markdown_row(self) -> str:
        return (f"| {self.arm} | {self.n_gate_stored}/{self.n_gate_stored + self.n_gate_discarded} "
                f"| {self.avg_stored_importance:.3f} | {self.avg_novelty:.3f} "
                f"| {self.hit_at_1:.3f} | {self.hit_at_3:.3f} | {self.hit_at_5:.3f} "
                f"| {self.mrr:.3f} | {self.has_answer:.3f} "
                f"| {self.avg_recall_ms:.1f} |")


def _reset_embedder_state() -> None:
    """Embedder reset between arms so no state leaks from one arm's clean
    environment into the next.

    TF-IDF/hash backends carry corpus state (vocabulary growth), so they are
    fully reset per arm. The sentence-transformers backend is stateless and
    deterministic — keeping the loaded instance avoids a costly per-arm model
    reload while remaining fair (identical inputs → identical vectors)."""
    from star_graph import embedding as _emb
    provider = getattr(_emb, "_provider", None)
    if provider is not None and getattr(provider, "backend", "") == "sentence-transformers":
        EmbedderRegistry._embedder = provider
        return
    reset_embedder()
    EmbedderRegistry._embedder = None


@dataclass
class _WrittenEnv:
    manager: MemoryManager
    tmpdir: str
    fact_anchor_ids: dict[int, str]  # fact_index → anchor id
    stored_importances: list[float]
    novelties: list[float]
    gate_errors: int


def _make_clean_manager(tmpdir: str) -> MemoryManager:
    """Fresh MemoryManager on a private tempdir — no persisted state leaks."""
    storage = os.path.join(tmpdir, "memory.db")
    return MemoryManager(storage_path=storage)


def _write_phase(arm: ABArm, records: list[MemoryRecord],
                 tmpdir: str, verbose: bool = False) -> _WrittenEnv:
    """Write every record through the arm's importance gate into a clean env.

    The arm's ImportanceEngine IS the gate (score → anchor.vector.importance,
    should_store → store or drop). manager.remember's internal write gate is
    disabled by default (defaults.yaml write_gate.enabled: false), so the arm's
    engine is the single gate under test — no confound.
    """
    manager = _make_clean_manager(tmpdir)
    engine, _uses_nov = build_importance_engine(arm, manager.graph)
    fact_anchor_ids: dict[int, str] = {}
    stored_importances: list[float] = []
    novelties: list[float] = []
    gate_errors = 0
    fact_counter = -1
    for rec in records:
        if rec.is_fact:
            fact_counter += 1
        try:
            result = engine.evaluate(rec.text, topic_tags=rec.tags,
                                     extracted_entities=list(rec.tags))
        except Exception:
            gate_errors += 1
            continue
        if hasattr(engine, "_novelty_last"):
            novelties.append(engine._novelty_last)
        if not result.should_store:
            continue
        try:
            anchor = manager.remember(
                rec.text, tags=rec.tags, importance=result.score,
                source_session=f"ab_{arm.name}",
            )
        except Exception:
            gate_errors += 1
            continue
        if anchor is None:
            continue  # rejected elsewhere (eviction etc.)
        stored_importances.append(result.score)
        if rec.is_fact:
            fact_anchor_ids[fact_counter] = anchor.id
    if verbose:
        print(f"  [{arm.name}] wrote {len(fact_anchor_ids)} fact anchors, "
              f"graph size {len(manager.graph.anchors)}", flush=True)
    return _WrittenEnv(manager=manager, tmpdir=tmpdir,
                       fact_anchor_ids=fact_anchor_ids,
                       stored_importances=stored_importances,
                       novelties=novelties, gate_errors=gate_errors)


def _teardown_env(env: _WrittenEnv) -> None:
    try:
        env.manager.shutdown_batch_vectorizer()
    except Exception:
        pass
    shutil.rmtree(env.tmpdir, ignore_errors=True)


def _evaluate_retrieval(retriever: Retriever, queries: list[QueryCase],
                        fact_anchor_ids: dict[int, str],
                        top_k: int = 10) -> tuple[dict[str, float], int]:
    """Run queries against a retriever, return metrics dict + recall ms total.

    hit@k   — fraction of queries whose answer anchor is within top-k
    MRR     — mean 1/rank of the first answer hit (0 if never hit, cap top_k)
    has_answer — fraction of queries where any retrieved item is relevant
              (over the full top_k window; identical to hit@top_k when each
              query has exactly one ground-truth anchor)
    """
    hits = {1: 0, 3: 0, 5: 0}
    any_hit = 0
    rr_sum = 0.0
    ms_sum = 0.0
    n = 0
    for qc in queries:
        gt = fact_anchor_ids.get(qc.fact_index)
        if not gt:
            continue
        t0 = time.perf_counter()
        result = retriever.retrieve(qc.query, top_k=top_k)
        ms_sum += (time.perf_counter() - t0) * 1000
        ranked_ids: list[str] = []
        for c in result.constellations:
            for a in c.anchors:
                if a.id not in ranked_ids:
                    ranked_ids.append(a.id)
        n += 1
        if gt in ranked_ids:
            any_hit += 1
            rank = ranked_ids.index(gt) + 1
            rr_sum += 1.0 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1
    if n == 0:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0,
                "has_answer": 0.0, "avg_ms": 0.0}, 0
    metrics = {
        "hit@1": hits[1] / n,
        "hit@3": hits[3] / n,
        "hit@5": hits[5] / n,
        "mrr": rr_sum / n,
        "has_answer": any_hit / n,
        "avg_ms": ms_sum / n,
    }
    return metrics, n


def run_importance_arm(arm: ABArm, records: list[MemoryRecord],
                       queries: list[QueryCase], top_k: int = 10,
                       verbose: bool = False) -> ArmMetrics:
    """Experiment 1: the arm's importance config gates the WRITE phase;
    evaluation retrieval uses the FIXED default hybrid retriever so any metric
    difference is attributable to the weight config alone."""
    tmpdir = tempfile.mkdtemp(prefix=f"nwc_ab_imp_{arm.name}_")
    _reset_embedder_state()
    env = _write_phase(arm, records, tmpdir, verbose=verbose)
    try:
        retriever = build_retriever(arm.eval_retrieval, env.manager.graph)
        metrics, n_q = _evaluate_retrieval(retriever, queries, env.fact_anchor_ids,
                                           top_k=top_k)
    finally:
        _teardown_env(env)
    n_stored = len(env.stored_importances)
    m = ArmMetrics(
        arm=arm.name, description=arm.description,
        n_written=len(records),
        n_gate_stored=n_stored,
        n_gate_discarded=env.gate_errors,  # refined below
        n_gate_errors=env.gate_errors,
        avg_stored_importance=(sum(env.stored_importances) / n_stored) if n_stored else 0.0,
        avg_novelty=(sum(env.novelties) / len(env.novelties)) if env.novelties else 0.0,
        hit_at_1=metrics["hit@1"], hit_at_3=metrics["hit@3"],
        hit_at_5=metrics["hit@5"], mrr=metrics["mrr"],
        has_answer=metrics["has_answer"], n_queries=n_q,
        avg_recall_ms=metrics["avg_ms"],
        weights=dict(arm.importance_weights or {}),
    )
    m.n_gate_discarded = m.n_written - n_stored
    return m


def run_retrieval_arm(arm: ABArm, records: list[MemoryRecord],
                      queries: list[QueryCase], top_k: int = 10,
                      shared_env: Optional[_WrittenEnv] = None,
                      verbose: bool = False) -> ArmMetrics:
    """Experiment 2: the WRITE phase is identical for all arms (code-default
    importance engine), then the arm's retrieval formula ranks recall.

    Pass ``shared_env`` to evaluate multiple retrievers on one identical graph
    (preferred: removes write nondeterminism entirely). When shared_env is not
    given a fresh clean env is built for this arm.
    """
    owned = False
    if shared_env is None:
        tmpdir = tempfile.mkdtemp(prefix=f"nwc_ab_ret_{arm.name}_")
        _reset_embedder_state()
        write_arm = ABArm(name="fixed-write",
                          description="code-default importance for all retrieval arms",
                          importance_weights=dict(W_CODE_WEIGHTS))
        shared_env = _write_phase(write_arm, records, tmpdir, verbose=verbose)
        owned = True
    try:
        retriever = build_retriever(arm.retrieval, shared_env.manager.graph)
        metrics, n_q = _evaluate_retrieval(retriever, queries,
                                           shared_env.fact_anchor_ids, top_k=top_k)
        n_stored = len(shared_env.stored_importances)
        m = ArmMetrics(
            arm=arm.name, description=arm.description,
            n_written=len(records), n_gate_stored=n_stored,
            n_gate_discarded=len(records) - n_stored,
            avg_stored_importance=(sum(shared_env.stored_importances) / n_stored)
            if n_stored else 0.0,
            hit_at_1=metrics["hit@1"], hit_at_3=metrics["hit@3"],
            hit_at_5=metrics["hit@5"], mrr=metrics["mrr"],
            has_answer=metrics["has_answer"], n_queries=n_q,
            avg_recall_ms=metrics["avg_ms"],
        )
        return m
    finally:
        if owned:
            _teardown_env(shared_env)


# ═══════════════════════════════════════════════════════════════
# 6. Reporting
# ═══════════════════════════════════════════════════════════════

def metrics_table(results: list[ArmMetrics], title: str) -> str:
    """Render arm metrics as a markdown table (ready to paste into reports)."""
    lines = [
        f"## {title}",
        "",
        "| 配置 | 存储量/总量 | 平均重要度 | 平均novelty | hit@1 | hit@3 | hit@5 | MRR | has_answer | 平均召回耗时(ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(r.to_markdown_row() for r in results)
    return "\n".join(lines)


def weights_table(arms: list[ABArm]) -> str:
    lines = [
        "### 权重配置",
        "",
        "| 配置 | 说明 |",
        "|---|---|",
    ]
    for a in arms:
        lines.append(f"| {a.name} | {a.description} |")
    return "\n".join(lines)


def save_report(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
