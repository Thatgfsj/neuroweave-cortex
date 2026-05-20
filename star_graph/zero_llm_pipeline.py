"""Zero-LLM ingestion pipeline — pure algorithmic memory intake (Mnemosyne-aligned).

Pipeline: security filter → embedding → dedup → entity extraction → classification → scoring → linking.
Only invokes LLM for optional "ambiguous judgment" gate. Target: $0/条 ingestion cost.

Usage:
    pipe = ZeroLLMPipeline(graph, embedder)
    results = pipe.ingest(["text 1", "text 2", ...])
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Callable

from .anchor import Anchor


# ── Stage 1: Security Filter ───────────────────────────────

_PII_PATTERNS = [
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'SSN'),
    (re.compile(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b'), 'CCN'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), 'EMAIL'),
    (re.compile(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'), 'PHONE'),
    (re.compile(r'\b(?:sk|api)[_-]?key[=:]\s*\S+'), 'API_KEY'),
]

_TOXICITY_PATTERNS = [
    re.compile(r'\b(hate|kill|murder|bomb|attack|terror)\b', re.I),
    re.compile(r'\b(exploit|ransomware|malware|phishing)\b', re.I),
]

_DEFAULT_SEVERITY = {"SSN": 9, "CCN": 10, "EMAIL": 5, "PHONE": 6, "API_KEY": 10}


def _filter_security(text: str) -> dict:
    """Scan for PII and toxicity. Returns {passed: bool, flags: [...], blocked: bool}."""
    flags = []
    for pattern, label in _PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            flags.append({"type": label, "count": len(matches),
                          "severity": _DEFAULT_SEVERITY.get(label, 5)})

    toxic = []
    for pattern in _TOXICITY_PATTERNS:
        for m in pattern.finditer(text):
            toxic.append(m.group(0))

    blocked = any(f["severity"] >= 9 for f in flags) or len(toxic) >= 3
    return {"passed": not blocked, "flags": flags, "toxic_matches": toxic, "blocked": blocked}


# ── Stage 2: Embedding ──────────────────────────────────────

def _embed(texts: list[str], embedder) -> list[list[float]]:
    """Batch-encode texts via embedder."""
    return embedder.encode(texts)


# ── Stage 3: Dedup ─────────────────────────────────────────

def _dedup(embeddings: list[list[float]], texts: list[str],
           graph, threshold: float = 0.92) -> list[int]:
    """Check each text against recent graph anchors. Returns indices of UNIQUE items."""
    from .math_utils import cosine_sim
    keep = []
    for i, emb in enumerate(embeddings):
        is_dup = False
        for aid, anchor in list(graph.anchors.items())[-100:]:
            if not anchor.embedding:
                continue
            if cosine_sim(emb, anchor.embedding) >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return keep


# ── Stage 4: Entity Extraction ──────────────────────────────

_ENTITY_PATTERNS = {
    "person": re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
    "organization": re.compile(r'\b(?:Inc|LLC|Corp|Ltd|GmbH|Co\.?)\b'),
    "technology": re.compile(r'\b(?:Python|Java|Rust|Go|React|Docker|Kubernetes|AWS|'
                             r'GCP|Azure|Redis|PostgreSQL|MySQL|MongoDB|Kafka|'
                             r'GraphQL|REST|gRPC|HTTP|TLS|SSL)\b'),
    "version": re.compile(r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b'),
    "date": re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    "url": re.compile(r'\bhttps?://[^\s]{4,}\b'),
}


def _extract_entities(text: str) -> dict[str, list[str]]:
    """Rule-based entity extraction (no LLM)."""
    entities: dict[str, list[str]] = {}
    for ent_type, pattern in _ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            entities[ent_type] = list(set(matches))[:5]
    return entities


# ── Stage 5: Classification ─────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "debugging": ["error", "bug", "fix", "crash", "traceback", "exception", "debug",
                  "timeout", "null", "undefined", "segfault", "stack trace"],
    "deployment": ["deploy", "release", "rollback", "staging", "production", "pipeline",
                   "CI/CD", "jenkins", "docker", "kubernetes", "helm"],
    "architecture": ["architecture", "design pattern", "microservice", "monolith",
                     "scalability", "throughput", "latency", "SLA"],
    "security": ["auth", "token", "certificate", "encrypt", "decrypt", "CVE",
                 "vulnerability", "penetration", "firewall", "OAuth"],
    "data": ["database", "migration", "schema", "query", "index", "normalization",
             "ETL", "pipeline", "warehouse", "lake"],
    "testing": ["test", "coverage", "mock", "stub", "assert", "regression",
                "unit test", "integration test", "e2e"],
    "preference": ["prefer", "like", "favorite", "choice", "recommend", "suggestion"],
    "fact": ["born", "live", "work", "graduate", "study", "know", "experience"],
    "meeting": ["meeting", "standup", "sprint", "retrospective", "planning", "demo"],
    "question": ["how to", "what is", "why does", "how do", "explain", "clarify"],
}


def _classify(text: str) -> list[str]:
    """Keyword-based topic classification."""
    lower = text.lower()
    scores = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in lower)
        if score:
            scores[topic] = score
    return sorted(scores, key=scores.get, reverse=True)[:3]


# ── Stage 6: Scoring ────────────────────────────────────────

def _score(text: str, tags: list[str]) -> dict:
    """Estimate importance, surprise, emotional_valence from text features."""
    lower = text.lower()

    # Importance heuristics
    importance_signals = sum(1 for w in ["critical", "urgent", "important", "must",
                                          "key", "essential", "vital"] if w in lower)
    importance = min(1.0, 0.3 + 0.15 * importance_signals + 0.1 * len(tags)
                     + 0.05 * min(len(text) / 50, 3))

    # Surprise heuristics
    surprise_keywords = ["surprisingly", "unexpected", "shocking", "first time",
                         "never", "discovered", "revealed", "found that"]
    surprise = min(1.0, 0.3 + 0.2 * sum(1 for w in surprise_keywords if w in lower))

    # Emotional valence
    positive = sum(1 for w in ["great", "excellent", "love", "happy", "success",
                                "solved", "fixed", "improved"] if w in lower)
    negative = sum(1 for w in ["error", "fail", "bug", "crash", "problem", "angry",
                                "frustrated", "broken", "bad"] if w in lower)
    valence = (positive - negative) / max(1, positive + negative)

    # Confidence
    confidence_signals = sum(1 for w in ["clearly", "obviously", "confirmed",
                                          "verified", "exactly"] if w in lower)
    confidence = min(0.9, 0.5 + 0.1 * confidence_signals)

    return {
        "importance": round(importance, 3),
        "surprise": round(surprise, 3),
        "emotional_valence": round(valence, 3),
        "confidence": round(confidence, 3),
    }


# ── Stage 7: Linking ────────────────────────────────────────

def _link_to_existing(anchor: Anchor, graph,
                      similarity_threshold: float = 0.65) -> list[tuple[str, float]]:
    """Link new anchor to existing related anchors via embedding similarity."""
    from .math_utils import cosine_sim
    if not anchor.embedding:
        return []
    links = []
    for aid, existing in list(graph.anchors.items())[-200:]:
        if aid == anchor.id or not existing.embedding:
            continue
        sim = cosine_sim(anchor.embedding, existing.embedding)
        if sim >= similarity_threshold:
            links.append((aid, round(sim, 3)))
    return sorted(links, key=lambda x: -x[1])[:5]


# ── Pipeline ────────────────────────────────────────────────

@dataclass
class IngestionResult:
    """Result of ingesting a single text."""
    text: str
    anchor: Anchor | None = None
    security: dict = field(default_factory=dict)
    entities: dict[str, list[str]] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    links: list[tuple[str, float]] = field(default_factory=list)
    is_duplicate: bool = False
    blocked: bool = False
    ingested: bool = False


class ZeroLLMPipeline:
    """Pure algorithmic ingestion with optional LLM ambiguity gate.

    Parameters:
        graph: the StarGraph to ingest into
        embedder: embedding provider
        llm_fn: optional callable for ambiguous judgment (takes text/context, returns dict)
        llm_threshold: confidence below which LLM is consulted
    """

    def __init__(self, graph, embedder,
                 llm_fn: Callable | None = None,
                 llm_threshold: float = 0.3):
        self._graph = graph
        self._embedder = embedder
        self._llm_fn = llm_fn
        self._llm_threshold = llm_threshold

    def ingest(self, texts: list[str], *, tags: list[str] | None = None,
               source_session: str = "") -> list[IngestionResult]:
        """Run the full zero-LLM pipeline on a batch of texts.

        Returns one IngestionResult per input text.
        """
        results: list[IngestionResult] = []
        texts_to_embed: list[int] = []  # indices of texts after filtering

        # Pass 1: security filter
        for i, text in enumerate(texts):
            r = IngestionResult(text=text)
            r.security = _filter_security(text)
            if r.security["blocked"]:
                r.blocked = True
                results.append(r)
            else:
                texts_to_embed.append(i)
                results.append(r)

        # Pass 2: batch embed
        if texts_to_embed:
            emb_texts = [texts[i] for i in texts_to_embed]
            embeddings = _embed(emb_texts, self._embedder)
            idx_to_emb = {idx: emb for idx, emb in zip(texts_to_embed, embeddings)}
        else:
            idx_to_emb = {}

        # Pass 3: dedup
        if idx_to_emb:
            all_emb = [idx_to_emb[idx] for idx in texts_to_embed]
            all_texts = [texts[idx] for idx in texts_to_embed]
            keep_indices = _dedup(all_emb, all_texts, self._graph)
            keep_set = {texts_to_embed[k] for k in keep_indices}
        else:
            keep_set = set()

        # Pass 4-7: entity, classify, score, link, create
        for i in range(len(texts)):
            r = results[i]
            if r.blocked:
                continue
            r.is_duplicate = i not in keep_set
            if r.is_duplicate:
                continue

            text = texts[i]
            emb = idx_to_emb.get(i, [])
            r.entities = _extract_entities(text)
            r.topics = _classify(text)
            r.scores = _score(text, r.topics)

            # LLM ambiguity gate (optional)
            if self._llm_fn and r.scores.get("confidence", 0.5) < self._llm_threshold:
                try:
                    llm_result = self._llm_fn(text, {"topics": r.topics, "entities": r.entities})
                    if isinstance(llm_result, dict):
                        r.topics = llm_result.get("topics", r.topics)
                        r.scores.update(llm_result.get("scores", {}))
                except Exception:
                    pass

            all_tags = list(set((tags or []) + r.topics))
            anchor = Anchor.create(
                text=text,
                tags=all_tags,
                embedding=emb,
                source_session=source_session,
                **r.scores,
            )
            self._graph.add_anchor(anchor)
            r.links = _link_to_existing(anchor, self._graph)
            r.anchor = anchor
            r.ingested = True

            # Add edges for links
            for linked_id, sim in r.links:
                self._graph.add_edge(anchor.id, linked_id,
                                     weight=sim, edge_type="topical")

        return results
