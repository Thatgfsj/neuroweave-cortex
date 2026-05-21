"""Context-aware routing engine for NeuroWeave Cortex cognitive memory runtime.

Routes queries through a weighted multi-factor scoring pipeline that combines
embedding similarity, task relevance, domain matching, recency, user intent,
and agent state into a single composite score per memory candidate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .math_utils import cosine_sim

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor


@dataclass
class RoutingContext:
    """Contextual signals for multi-factor routing decisions."""

    task_type: str = "conversation"  # coding | debugging | planning | reflection | conversation
    active_goals: list[str] = field(default_factory=list)
    user_intent: str = ""            # question | command | statement | exploration
    agent_state: str = ""            # exploring | validating | executing | idle
    domain_hint: str = ""            # preferred domain to search
    time_preference: str = "recent"  # recent | relevant | all
    max_items: int = 10


@dataclass
class RoutingWeights:
    """Dynamic weight vector for multi-factor composite scoring."""

    w_similarity: float = 0.30
    w_task_relevance: float = 0.25
    w_domain_match: float = 0.15
    w_recency: float = 0.15
    w_user_intent: float = 0.10
    w_agent_state: float = 0.05

    def validate(self) -> None:
        """Raise ValueError if weights do not sum to 1.0."""
        total = (self.w_similarity + self.w_task_relevance + self.w_domain_match
                 + self.w_recency + self.w_user_intent + self.w_agent_state)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"RoutingWeights must sum to 1.0, got {total}")


class ContextRouter:
    """Multi-factor context routing engine for memory retrieval.

    Scores each candidate anchor against a query using a weighted combination of
    six signal dimensions: similarity, task relevance, domain match, recency,
    user intent, and agent state. Weights are dynamically adjusted per task type.
    """

    _TASK_WEIGHT_MAP: dict[str, RoutingWeights] = {
        "debugging": RoutingWeights(
            w_similarity=0.20, w_task_relevance=0.30, w_domain_match=0.20,
            w_recency=0.20, w_user_intent=0.05, w_agent_state=0.05,
        ),
        "coding": RoutingWeights(
            w_similarity=0.25, w_task_relevance=0.30, w_domain_match=0.20,
            w_recency=0.10, w_user_intent=0.10, w_agent_state=0.05,
        ),
        "planning": RoutingWeights(
            w_similarity=0.20, w_task_relevance=0.35, w_domain_match=0.15,
            w_recency=0.10, w_user_intent=0.15, w_agent_state=0.05,
        ),
        "reflection": RoutingWeights(
            w_similarity=0.15, w_task_relevance=0.20, w_domain_match=0.10,
            w_recency=0.15, w_user_intent=0.25, w_agent_state=0.15,
        ),
        "conversation": RoutingWeights(
            w_similarity=0.35, w_task_relevance=0.15, w_domain_match=0.10,
            w_recency=0.20, w_user_intent=0.15, w_agent_state=0.05,
        ),
    }

    _TASK_KEYWORDS: dict[str, list[str]] = {
        "debugging": ["debug", "error", "bug", "fix", "traceback", "exception",
                      "crash", "log", "stack", "breakpoint", "knowledge", "fact"],
        "coding": ["code", "function", "api", "implementation", "class", "module",
                   "library", "syntax", "refactor", "test", "import", "config"],
        "planning": ["goal", "plan", "architecture", "design", "milestone",
                     "roadmap", "strategy", "priority", "task", "todo"],
        "reflection": ["lesson", "insight", "reflection", "pattern", "takeaway",
                       "summary", "retrospective", "analysis", "observation"],
    }

    _INTENT_QUESTION_STARTS = frozenset({
        "what", "why", "how", "which", "when", "where", "who",
        "can", "do", "is", "are", "should", "could", "would", "will",
        "does", "did", "has", "have", "am", "was", "were",
    })

    _INTENT_COMMAND_VERBS = frozenset({
        "do", "get", "find", "show", "create", "fix", "make", "set",
        "run", "build", "add", "remove", "delete", "update", "change",
        "open", "close", "start", "stop", "read", "write", "copy",
        "move", "list", "check", "test", "deploy", "install", "configure",
        "generate", "compile", "debug", "search", "fetch", "compute",
    })

    _INTENT_EXPLORATION_PATTERNS = (
        "explore", "browse", "what if", "maybe", "consider", "think about",
        "look into", "investigate", "try", "experiment",
    )

    _STATE_KEYWORDS: dict[str, list[str]] = {
        "exploring": ["explore", "discover", "browse", "search", "investigate",
                      "research", "look", "scan", "survey"],
        "validating": ["verify", "validate", "test", "check", "confirm",
                       "ensure", "proof", "correct", "lint"],
        "executing": ["do", "run", "execute", "perform", "action", "apply",
                      "implement", "build", "deploy", "code"],
        "idle": [],
    }

    def __init__(self, graph: StarGraph | None = None,
                 embedder=None,
                 domain_manager=None):
        self.graph = graph
        self._embedder = embedder
        self.domain_manager = domain_manager

    def compute_weights(self, context: RoutingContext) -> RoutingWeights:
        """Return dynamically adjusted routing weights for the given context."""
        weights = self._TASK_WEIGHT_MAP.get(
            context.task_type,
            self._TASK_WEIGHT_MAP["conversation"],
        )
        weights.validate()
        return weights

    def score_item(self, item: Anchor,
                   query_embedding: list[float],
                   context: RoutingContext,
                   weights: RoutingWeights) -> float:
        """Compute composite score (0-1) for a single memory candidate."""
        sim = 0.0
        if query_embedding and getattr(item, "embedding", None):
            sim = max(0.0, cosine_sim(query_embedding, item.embedding))

        task_rel = self.task_relevance(item, context.task_type)
        domain_bonus = self.domain_match_bonus(item, context.domain_hint)
        rec = self.recency_score(item)
        intent_match = self._intent_match(item, context.user_intent)
        state_match = self._agent_state_match(item, context.agent_state)

        score = (
            weights.w_similarity * sim
            + weights.w_task_relevance * task_rel
            + weights.w_domain_match * domain_bonus
            + weights.w_recency * rec
            + weights.w_user_intent * intent_match
            + weights.w_agent_state * state_match
        )
        return min(1.0, max(0.0, score))

    def rank_items(self, items: list[Anchor],
                   query_embedding: list[float],
                   context: RoutingContext) -> list[Anchor]:
        """Return items ranked by composite score, highest first."""
        weights = self.compute_weights(context)
        scored: list[tuple[float, Anchor]] = []
        for item in items:
            if not getattr(item, "is_retrievable", True):
                continue
            s = self.score_item(item, query_embedding, context, weights)
            scored.append((s, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]

    def route(self, query: str,
              context: RoutingContext | None = None,
              top_k: int = 10) -> list[Anchor]:
        """Full routing pipeline: embed, score, rank, return top_k results."""
        if context is None:
            context = RoutingContext()
        if context.user_intent == "":
            context.user_intent = self.get_intent(query)

        embedder = self._get_embedder()
        query_embedding = embedder.encode(query) if callable(getattr(embedder, "encode", None)) else []

        candidates: list[Anchor] = []
        if self.graph is not None:
            candidates = list(self.graph.anchors.values())

        if not candidates:
            return []

        ranked = self.rank_items(candidates, query_embedding, context)
        return ranked[:max(1, top_k)]

    def get_intent(self, query: str) -> str:
        """Detect user intent from raw query text."""
        q = query.strip().lower()
        if not q:
            return "statement"

        if "?" in q:
            return "question"

        for pattern in self._INTENT_EXPLORATION_PATTERNS:
            if pattern in q:
                return "exploration"

        first_word = q.split()[0] if q.split() else ""
        if first_word in self._INTENT_QUESTION_STARTS:
            return "question"

        if first_word in self._INTENT_COMMAND_VERBS:
            return "command"

        return "statement"

    def task_relevance(self, anchor: Anchor, task_type: str) -> float:
        """Return 0-1 how relevant the anchor is to the given task type."""
        keywords = self._TASK_KEYWORDS.get(task_type, [])
        if not keywords or not anchor.tags:
            return 0.0

        tag_lower = {t.lower() for t in anchor.tags}

        matches = sum(1 for kw in keywords if kw in tag_lower or kw in anchor.text.lower())
        if matches == 0:
            return 0.0
        return min(1.0, matches / max(1, len(keywords)) * 2.0)

    def domain_match_bonus(self, anchor: Anchor, domain_hint: str) -> float:
        """Return 0-1 bonus for matching the domain hint."""
        if not domain_hint:
            return 0.0

        hint_lower = domain_hint.strip().lower()
        if not hint_lower:
            return 0.0

        text_lower = anchor.text.lower()
        if hint_lower in text_lower:
            return 1.0

        tag_lower = {t.lower() for t in anchor.tags}
        hint_parts = hint_lower.replace("-", " ").replace("_", " ").replace(".", " ").split()
        matches = sum(1 for part in hint_parts if any(part in t for t in tag_lower))
        matches += sum(1 for part in hint_parts if part in text_lower)

        if matches == 0:
            matched = any(part in text_lower for part in hint_parts)
            return 0.3 if matched else 0.0

        return min(1.0, matches / max(1, len(hint_parts)))

    def recency_score(self, anchor: Anchor) -> float:
        """Return 0-1 based on creation time (1 = just created)."""
        now = time.time()
        age_seconds = now - anchor.created_at
        age_hours = age_seconds / 3600.0

        if age_hours <= 0:
            return 1.0

        half_life_hours = 24.0
        return 0.5 ** (age_hours / half_life_hours)

    def _intent_match(self, anchor: Anchor, intent: str) -> float:
        """Return 0-1 how well the anchor matches the user's apparent intent."""
        if not intent:
            return 0.0

        text_lower = anchor.text.lower()
        tags_lower = {t.lower() for t in anchor.tags}

        if intent == "question":
            q_signals = ["explain", "how", "why", "what", "definition", "tutorial",
                         "documentation", "guide", "faq", "reference", "knowledge"]
            matches = sum(1 for s in q_signals if s in text_lower or s in tags_lower)
            return min(1.0, matches / 3.0)

        if intent == "command":
            c_signals = ["howto", "step", "instruction", "guide", "example",
                         "snippet", "template", "recipe", "command", "script"]
            matches = sum(1 for s in c_signals if s in text_lower or s in tags_lower)
            return min(1.0, matches / 3.0)

        if intent == "exploration":
            e_signals = ["explore", "brainstorm", "idea", "concept", "overview",
                         "topic", "domain", "survey", "catalog"]
            matches = sum(1 for s in e_signals if s in text_lower or s in tags_lower)
            return min(1.0, matches / 3.0)

        return 0.5  # statement: neutral

    def _agent_state_match(self, anchor: Anchor, agent_state: str) -> float:
        """Return 0-1 how well the anchor matches the agent's current state."""
        if not agent_state:
            return 0.0

        keywords = self._STATE_KEYWORDS.get(agent_state, [])
        if not keywords:
            return 0.0

        text_lower = anchor.text.lower()
        tags_lower = {t.lower() for t in anchor.tags}
        matches = sum(1 for kw in keywords if kw in text_lower or kw in tags_lower)
        return min(1.0, matches / max(1, len(keywords)))

    def _get_embedder(self):
        """Lazy-load the shared embedder if none was provided at init."""
        if self._embedder is None:
            from .embedding import get_embedder
            self._embedder = get_embedder()
        return self._embedder
