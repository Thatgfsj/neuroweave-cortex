"""Abstract memory compression layer — multi-level summarization from episodic to meta.

Compression operates at four levels:
  RAW (0)      — original anchors, no compression
  EPISODIC (1) — cluster-level summaries within a session
  STRATEGIC (2)— cross-episode pattern extraction
  META (3)     — cross-domain principle discovery

Compressed anchors are stored as SummaryAnchor dataclasses, each with an optional
Anchor proxy for direct insertion into the StarGraph.

Architecture:
  Template-based summarization engine (module-level helpers)
  SessionCompressor      — Level 0→1, per-session clustering + summary
  MultiLevelCompressor   — full pipeline across all levels
"""

from __future__ import annotations

import enum
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .anchor import Anchor, AnchorVector, Oscillator, MemoryState
from .config import Config


# ---------------------------------------------------------------------------
# CompressionLevel enum
# ---------------------------------------------------------------------------

class CompressionLevel(enum.Enum):
    RAW = 0
    EPISODIC = 1
    STRATEGIC = 2
    META = 3


# ---------------------------------------------------------------------------
# SummaryAnchor dataclass
# ---------------------------------------------------------------------------

@dataclass
class SummaryAnchor:
    """A compressed summary of multiple source anchors at a given compression level.

    The optional `anchor` field is a proxy Anchor suitable for insertion into
    the StarGraph. It carries the centroid embedding, high stability (0.85),
    and low hippocampal dependency (0.1) so the summary behaves like a
    cortical (consolidated) memory.
    """

    id: str
    text: str
    source_anchor_ids: list[str]
    centroid_embedding: list[float]
    compression_level: CompressionLevel
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    token_count: int = 0
    anchor: Optional[Anchor] = None  # proxy for graph insertion

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(self.text.split())

    def to_anchor_proxy(self) -> Anchor:
        """Create (or return) an Anchor proxy suitable for graph insertion.

        The proxy has high stability and low hippocampal dependency,
        representing a consolidated/cortical memory.
        """
        if self.anchor is not None:
            return self.anchor

        c = Config.get()
        comp_cfg = getattr(c, 'compression', None)
        stability = getattr(comp_cfg, 'summary_initial_stability', 0.85) if comp_cfg else 0.85
        hipp_dep = getattr(comp_cfg, 'summary_hippocampal_dependency', 0.1) if comp_cfg else 0.1

        from .anchor import EmbedderRegistry
        try:
            embedder = EmbedderRegistry.get_embedder_singleton()
            freq = embedder.derive_frequency(importance=0.7, emotional_valence=0.0, text_length=len(self.text))
        except Exception:
            import math as _math
            freq = 0.7

        self.anchor = Anchor(
            id=self.id,
            text=self.text[:280],
            vector=AnchorVector(
                importance=0.7,
                frequency=0.1,
                recency=1.0,
                stability=stability,
                surprise=0.3,
                hippocampal_dependency=hipp_dep,
                confidence=self.confidence,
            ),
            embedding=list(self.centroid_embedding),
            oscillator=Oscillator(natural_frequency=freq, coupling_strength=0.4),
            source_session=f"compressed_l{self.compression_level.value}",
            tags=list(self.tags),
            state=MemoryState.DORMANT,
            created_at=self.created_at,
        )
        return self.anchor


# ---------------------------------------------------------------------------
# Template-based summarization engine (module-level helpers)
# ---------------------------------------------------------------------------

# Common English stop words for TF-IDF filtering
_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "and", "but", "or", "if", "because", "until", "while",
    "about", "up", "out", "just", "now", "also", "this", "that", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "their", "what", "which", "who", "whom",
}

# Common action/verb words for pattern detection
_ACTION_VERBS: set[str] = {
    "create", "build", "develop", "implement", "design", "deploy", "configure",
    "debug", "fix", "resolve", "optimize", "refactor", "test", "run", "execute",
    "install", "setup", "migrate", "update", "upgrade", "modify", "extend",
    "integrate", "connect", "fetch", "query", "insert", "delete", "process",
    "analyze", "evaluate", "compare", "select", "choose", "prefer", "like",
    "use", "using", "used", "write", "wrote", "written", "read", "learn",
    "understand", "discover", "find", "found", "decide", "determine", "check",
    "validate", "verify", "monitor", "track", "manage", "handle", "support",
    "enable", "disable", "add", "remove", "replace", "change", "convert",
    "transform", "generate", "compute", "calculate", "apply", "call",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering short tokens and stop words."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]


def _extract_key_terms(anchors: list[Anchor], top_k: int = 8) -> list[str]:
    """TF-IDF style key term extraction from a cluster of anchors.

    For each word that appears across the anchors, compute:
        score(word) = tf(word) * log(N / df(word))
    where tf = total frequency across cluster, df = number of anchors containing it.

    Returns the top-k terms sorted by score descending.
    """
    if not anchors:
        return []

    N = len(anchors)
    # Document frequency: how many anchors contain each term
    doc_freq: Counter = Counter()
    # Term frequency: total occurrences across all anchors
    term_freq: Counter = Counter()

    for anchor in anchors:
        tokens = _tokenize(anchor.text)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freq[t] += 1
        for t in tokens:
            term_freq[t] += 1

    # Score: tf * log(N / df)
    scored: list[tuple[str, float]] = []
    for term, tf in term_freq.items():
        df = doc_freq.get(term, 1)
        idf = math.log((N + 1) / (df + 1)) + 1.0  # smoothed IDF
        score = tf * idf
        scored.append((term, score))

    scored.sort(key=lambda x: -x[1])
    return [term for term, _ in scored[:top_k]]


def _extract_action_patterns(texts: list[str]) -> str:
    """Find common verb/action patterns across a set of texts.

    Scans for action words (verbs ending in -ing, -ed, or from _ACTION_VERBS)
    that appear across multiple texts. Returns a concise action description.
    """
    if not texts:
        return "interacted with"

    # Collect all action verbs across texts
    action_counter: Counter = Counter()
    for text in texts:
        found: set[str] = set()
        tokens = text.lower().split()
        for token in tokens:
            stripped = token.strip(".,;:!?()[]{}'\"")
            # Match -ing, -ed forms, or known action verbs
            if stripped in _ACTION_VERBS:
                found.add(stripped)
            elif stripped.endswith("ing") and len(stripped) > 4:
                found.add(stripped)
            elif stripped.endswith("ed") and len(stripped) > 4:
                found.add(stripped)
        for a in found:
            action_counter[a] += 1

    if not action_counter:
        return "discussed topics related to"

    # Take top 3 actions that appear in at least 2 texts
    common = [(a, c) for a, c in action_counter.most_common() if c >= 2]
    if not common:
        common = action_counter.most_common(2)

    if len(common) >= 3:
        return f"{common[0][0]}, {common[1][0]}, and {common[2][0]}"
    elif len(common) == 2:
        return f"{common[0][0]} and {common[1][0]}"
    else:
        return common[0][0] if common else "interacted with"


def _extract_domain(texts: list[str], key_terms: list[str]) -> str:
    """Infer a domain label from key terms and text patterns.

    Uses keyword matching against known domains.
    """
    domain_keywords = {
        "python development": {"python", "flask", "django", "fastapi", "pip", "pytest"},
        "javascript development": {"javascript", "node", "react", "vue", "npm", "typescript"},
        "devops": {"docker", "kubernetes", "ci/cd", "deploy", "pipeline", "terraform"},
        "database": {"sql", "mysql", "postgresql", "redis", "mongo", "query", "index"},
        "debugging": {"bug", "error", "fix", "debug", "trace", "exception", "crash"},
        "api design": {"api", "rest", "endpoint", "request", "response", "http"},
        "system design": {"architecture", "design", "pattern", "component", "module"},
        "security": {"auth", "token", "permission", "encrypt", "vulnerability"},
        "testing": {"test", "unit", "integration", "mock", "assert", "coverage"},
        "performance": {"optimize", "performance", "slow", "cache", "latency", "benchmark"},
        "configuration": {"config", "settings", "env", "environment", "variable"},
        "data processing": {"data", "parse", "transform", "pipeline", "csv", "json"},
    }

    all_text = " ".join(texts).lower()
    scores: dict[str, int] = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for kw in keywords if kw in all_text)
        if score > 0:
            scores[domain] = score

    if scores:
        return max(scores, key=scores.get)
    if key_terms:
        return " ".join(key_terms[:2])
    return "general"


def _extract_entities(texts: list[str], key_terms: list[str]) -> list[str]:
    """Extract named entities / capitalized terms from texts."""
    entity_counter: Counter = Counter()
    for text in texts:
        # Find capitalized multi-word sequences (potential proper nouns)
        words = text.split()
        for i, w in enumerate(words):
            clean = w.strip(".,;:!?()[]{}'\"")
            if clean and clean[0].isupper() and len(clean) > 1 and clean.lower() not in _STOP_WORDS:
                # Check if it's a real entity (not sentence-start)
                if i > 0 or not clean.isalpha():
                    entity_counter[clean] += 1

    # Also add key terms that look like entities
    for term in key_terms:
        if term[0].isupper() if term else False:
            entity_counter[term] += 2

    return [e for e, c in entity_counter.most_common(5) if c >= 2]


def _fill_template(level: CompressionLevel, domain: str, topic: str,
                   action: str, entities: list[str], context: str,
                   count: int, token_limit: int = 150) -> str:
    """Fill a template for the given compression level with token budget enforcement.

    Args:
        level: CompressionLevel (EPISODIC, STRATEGIC, or META)
        domain: Inferred domain (e.g., "python development")
        topic: Key topic (from key terms)
        action: Action pattern string
        entities: Named entities found
        context: Additional context string
        count: Number of source anchors being compressed
        token_limit: Maximum tokens in output
    """
    entity_str = ", ".join(entities[:3]) if entities else "the system"

    if level == CompressionLevel.EPISODIC:
        templates = [
            f"During {context}, {entity_str} worked on {domain}: {action} related to {topic}. "
            f"This encompasses {count} related interactions.",

            f"In {count} interactions about {domain}, the key activity was {action} around {topic}. "
            f"Involved: {entity_str}. Context: {context}.",

            f"Episode summary ({count} events): focused on {domain} with {action} patterns. "
            f"Topic: {topic}. Entities: {entity_str}.",
        ]
    elif level == CompressionLevel.STRATEGIC:
        templates = [
            f"Strategic pattern across {count} episodes: in {domain}, {action} consistently "
            f"involves {topic}. This approach is used by {entity_str}. Context: {context}.",

            f"Cross-episode insight ({count} clusters): {action} emerges as the dominant pattern "
            f"for {domain} problems involving {topic}. Key actors: {entity_str}.",

            f"Pattern analysis of {count} memories: the {domain} domain shows consistent "
            f"{action} behavior around {topic}. Entities: {entity_str}.",
        ]
    else:  # META
        templates = [
            f"Meta-pattern across domains ({count} strategies): the principle of {action} "
            f"governs outcomes in {domain} contexts involving {topic}. "
            f"Universal strategy applicable to {entity_str}.",

            f"Cross-domain abstraction ({count} inputs): {action} is a recurring meta-principle "
            f"across {domain} scenarios. Core insight: {topic}. Applies to: {entity_str}.",

            f"Deep pattern ({count} sources): {action} unifies {domain} approaches to {topic}. "
            f"This is a transferable principle for {entity_str}.",
        ]

    # Select template that fits the token budget best
    best_text = ""
    best_score = float('inf')
    for template in templates:
        tokens = len(template.split())
        # Prefer templates that use the budget well (not too short, not over)
        if tokens <= token_limit:
            score = token_limit - tokens  # closer to limit = better (more informative)
        else:
            score = (tokens - token_limit) * 10  # heavy penalty for exceeding
        if score < best_score:
            best_score = score
            best_text = template

    # Enforce token limit by truncation if needed
    words = best_text.split()
    if len(words) > token_limit:
        # Truncate to token limit, try to end at a sentence boundary
        truncated = " ".join(words[:token_limit])
        # Find last sentence-ending punctuation
        for punct in ['. ', '! ', '? ']:
            last_idx = truncated.rfind(punct)
            if last_idx > token_limit // 2:
                return truncated[:last_idx + 1]
        return truncated + "."

    return best_text


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a list of embeddings."""
    if not embeddings:
        return []
    emb_matrix = np.array(embeddings)
    return emb_matrix.mean(axis=0).tolist()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(min_len))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


def _find_clusters(anchor_ids: list[str],
                   embeddings: dict[str, list[float]],
                   similarity_threshold: float) -> list[list[str]]:
    """Find connected components in similarity graph (BFS-based clustering)."""
    n = len(anchor_ids)
    if n == 0:
        return []

    # Build adjacency
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            ei = embeddings.get(anchor_ids[i])
            ej = embeddings.get(anchor_ids[j])
            if ei and ej and _cosine_sim(ei, ej) > similarity_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited: set[int] = set()
    clusters: list[list[str]] = []

    for i in range(n):
        if i in visited:
            continue
        component: list[int] = []
        queue = [i]
        visited.add(i)
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append([anchor_ids[idx] for idx in component])

    return clusters


# ---------------------------------------------------------------------------
# SessionCompressor — Level 0→1 compression
# ---------------------------------------------------------------------------

class SessionCompressor:
    """Compresses a single session's anchors into EPISODIC summaries.

    Workflow:
    1. Filter anchors by source_session
    2. Compute pairwise cosine similarity on embeddings
    3. Find connected components (similarity > threshold)
    4. For each cluster >= min_cluster_size:
       - Compute centroid embedding
       - Generate summary text via templates
       - Create SummaryAnchor with source_anchor_ids
       - Create proxy Anchor with high stability
       - Down-weight source anchors (retention *= 0.3)
    """

    def __init__(self, min_cluster_size: int | None = None,
                 similarity_threshold: float | None = None,
                 token_limit: int | None = None,
                 source_retention_factor: float | None = None,
                 max_source_anchors: int | None = None):
        c = Config.get()
        comp_cfg = getattr(c, 'compression', None)
        self.min_cluster_size = (
            min_cluster_size if min_cluster_size is not None
            else getattr(comp_cfg, 'min_cluster_size', 3) if comp_cfg else 3
        )
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else getattr(comp_cfg, 'similarity_threshold', 0.55) if comp_cfg else 0.55
        )
        self.token_limit = (
            token_limit if token_limit is not None
            else getattr(comp_cfg, 'episodic_token_limit', 150) if comp_cfg else 150
        )
        self.source_retention_factor = (
            source_retention_factor if source_retention_factor is not None
            else getattr(comp_cfg, 'source_retention_factor', 0.3) if comp_cfg else 0.3
        )
        self.max_source_anchors = (
            max_source_anchors if max_source_anchors is not None
            else getattr(comp_cfg, 'max_source_anchors', 30) if comp_cfg else 30
        )
        self._counter = 0

    def compress(self, anchors: list[Anchor],
                 session_id: str) -> list[SummaryAnchor]:
        """Compress a session's anchors into episodic summaries.

        Args:
            anchors: All anchors (session filtering happens inside)
            session_id: Only anchors with source_session == session_id are compressed

        Returns:
            List of newly created SummaryAnchor objects
        """
        # Filter anchors by session
        session_anchors = [a for a in anchors if a.source_session == session_id]
        if len(session_anchors) < self.min_cluster_size:
            return []

        # Build embedding lookup
        anchor_map: dict[str, Anchor] = {a.id: a for a in session_anchors}
        embeddings: dict[str, list[float]] = {}
        for a in session_anchors:
            if a.embedding:
                embeddings[a.id] = a.embedding

        # Now compute embeddings for any anchor that doesn't have one
        try:
            from .anchor import EmbedderRegistry
            embedder = EmbedderRegistry.get_embedder_singleton()
        except Exception:
            embedder = None

        for a in session_anchors:
            if a.id not in embeddings:
                if embedder and a.text:
                    try:
                        emb = embedder.encode(a.text)
                        a.embedding = emb
                        embeddings[a.id] = emb
                    except Exception:
                        pass

        # Find clusters via connected components
        aid_list = list(anchor_map.keys())
        clusters = _find_clusters(aid_list, embeddings, self.similarity_threshold)

        summaries: list[SummaryAnchor] = []

        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue

            # Limit cluster size
            effective_cluster = cluster[:self.max_source_anchors]
            cluster_anchors = [anchor_map[aid] for aid in effective_cluster]

            # Compute centroid embedding
            cluster_embs = [embeddings[aid] for aid in effective_cluster if aid in embeddings]
            if not cluster_embs:
                continue
            centroid = _compute_centroid(cluster_embs)

            # Extract features for template filling
            key_terms = _extract_key_terms(cluster_anchors)
            texts = [a.text for a in cluster_anchors]
            action = _extract_action_patterns(texts)
            domain = _extract_domain(texts, key_terms)
            topic = key_terms[0] if key_terms else "general"
            entities = _extract_entities(texts, key_terms)
            context = f"session {session_id[:8]}" if session_id else "an interaction session"

            # Generate summary text
            summary_text = _fill_template(
                level=CompressionLevel.EPISODIC,
                domain=domain,
                topic=topic,
                action=action,
                entities=entities,
                context=context,
                count=len(effective_cluster),
                token_limit=self.token_limit,
            )

            # Compute intra-cluster confidence
            intra_sims: list[float] = []
            for i in range(min(len(cluster_embs), 15)):  # sample for perf
                for j in range(i + 1, min(len(cluster_embs), 15)):
                    intra_sims.append(_cosine_sim(cluster_embs[i], cluster_embs[j]))
            confidence = sum(intra_sims) / max(1, len(intra_sims))

            if confidence < self.similarity_threshold * 0.9:
                continue

            # Collect common tags
            all_tags: list[str] = []
            for a in cluster_anchors:
                all_tags.extend(a.tags)
            tag_counter = Counter(all_tags)
            common_tags = [t for t, c in tag_counter.most_common(6) if c >= 2]

            # Create SummaryAnchor
            self._counter += 1
            level_str = CompressionLevel.EPISODIC.name.lower()
            summary = SummaryAnchor(
                id=f"summary_l1_{session_id[:8]}_{self._counter}",
                text=summary_text,
                source_anchor_ids=effective_cluster,
                centroid_embedding=centroid,
                compression_level=CompressionLevel.EPISODIC,
                confidence=min(1.0, confidence + 0.1),
                tags=common_tags + [f"domain:{domain}", f"level:{level_str}"],
            )

            # Create proxy Anchor for graph insertion
            summary.to_anchor_proxy()

            # Down-weight source anchors
            for aid in effective_cluster:
                if aid in anchor_map:
                    # Apply retention factor to recency and frequency
                    anchor = anchor_map[aid]
                    anchor.vector.recency *= self.source_retention_factor
                    anchor.vector.frequency *= self.source_retention_factor
                    # Mark as having been compressed (transition toward DORMANT)
                    if anchor.state == MemoryState.ACTIVE:
                        anchor.transition('consolidate')

            summaries.append(summary)

        return summaries


# ---------------------------------------------------------------------------
# MultiLevelCompressor — full pipeline across all levels
# ---------------------------------------------------------------------------

class MultiLevelCompressor:
    """Multi-level compression pipeline: RAW → EPISODIC → STRATEGIC → META.

    Level 0→1: SessionCompressor groups anchors within sessions into episodes
    Level 1→2: Cross-episode clustering identifies strategic patterns
    Level 2→3: Cross-domain abstraction discovers meta-principles
    """

    def __init__(self):
        c = Config.get()
        comp_cfg = getattr(c, 'compression', None)

        self.episodic_token_limit = (
            getattr(comp_cfg, 'episodic_token_limit', 150) if comp_cfg else 150
        )
        self.strategic_token_limit = (
            getattr(comp_cfg, 'strategic_token_limit', 100) if comp_cfg else 100
        )
        self.meta_token_limit = (
            getattr(comp_cfg, 'meta_token_limit', 70) if comp_cfg else 70
        )
        self.min_cluster_size = (
            getattr(comp_cfg, 'min_cluster_size', 3) if comp_cfg else 3
        )
        self.similarity_threshold = (
            getattr(comp_cfg, 'similarity_threshold', 0.55) if comp_cfg else 0.55
        )
        self._counter = 0
        self._session_compressor = SessionCompressor()

    def compress_session(self, anchors: list[Anchor],
                         session_id: str) -> list[SummaryAnchor]:
        """Level 0→1: Compress raw anchors within a session into episodic summaries."""
        return self._session_compressor.compress(anchors, session_id)

    def compress_strategic(self,
                           summaries: list[SummaryAnchor]) -> list[SummaryAnchor]:
        """Level 1→2: Cluster EPISODIC summaries into STRATEGIC patterns.

        Groups similar episodic summaries and extracts recurring patterns
        that span multiple sessions/episodes.
        """
        if len(summaries) < self.min_cluster_size:
            return []

        # Build embedding lookup from centroids
        embeddings: dict[str, list[float]] = {}
        summary_map: dict[str, SummaryAnchor] = {}
        for s in summaries:
            if s.centroid_embedding:
                embeddings[s.id] = s.centroid_embedding
                summary_map[s.id] = s

        if len(summary_map) < self.min_cluster_size:
            return []

        aid_list = list(summary_map.keys())
        clusters = _find_clusters(aid_list, embeddings, self.similarity_threshold)

        strategic_summaries: list[SummaryAnchor] = []

        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue

            cluster_summaries = [summary_map[aid] for aid in cluster]

            # Collect all source texts from these summaries
            all_texts: list[str] = []
            for s in cluster_summaries:
                all_texts.append(s.text)
                # Also include key anchors from source IDs (if available via tags)
                all_texts.append(" ".join(s.tags))

            # Compute centroid of centroids
            cluster_embs = [embeddings[aid] for aid in cluster]
            centroid = _compute_centroid(cluster_embs)

            # Extract features
            key_terms = _extract_key_terms_via_texts(all_texts)
            action = _extract_action_patterns(all_texts)
            domain = _extract_domain(all_texts, key_terms)
            topic = key_terms[0] if key_terms else "pattern"
            entities = _extract_entities(all_texts, key_terms)

            # Collect all source anchor IDs recursively
            all_source_ids: list[str] = []
            for s in cluster_summaries:
                all_source_ids.extend(s.source_anchor_ids)

            # Generate summary text
            summary_text = _fill_template(
                level=CompressionLevel.STRATEGIC,
                domain=domain,
                topic=topic,
                action=action,
                entities=entities,
                context="multiple sessions",
                count=len(cluster),
                token_limit=self.strategic_token_limit,
            )

            # Intra-cluster confidence
            intra_sims: list[float] = []
            for ii in range(min(len(cluster_embs), 10)):
                for jj in range(ii + 1, min(len(cluster_embs), 10)):
                    intra_sims.append(_cosine_sim(cluster_embs[ii], cluster_embs[jj]))
            confidence = sum(intra_sims) / max(1, len(intra_sims))

            # Collect tags — inherit from child summaries
            all_tags: list[str] = []
            for s in cluster_summaries:
                all_tags.extend(s.tags)
            tag_counter = Counter(all_tags)
            common_tags = [t for t, c in tag_counter.most_common(8) if c >= 2]
            # Filter out domain/level system tags
            common_tags = [t for t in common_tags
                           if not t.startswith("domain:") and not t.startswith("level:")]

            self._counter += 1
            level_str = CompressionLevel.STRATEGIC.name.lower()
            summary = SummaryAnchor(
                id=f"summary_l2_{self._counter}",
                text=summary_text,
                source_anchor_ids=all_source_ids[:self._session_compressor.max_source_anchors],
                centroid_embedding=centroid,
                compression_level=CompressionLevel.STRATEGIC,
                confidence=min(1.0, confidence + 0.05),
                tags=common_tags + [f"domain:{domain}", f"level:{level_str}"],
            )
            summary.to_anchor_proxy()
            strategic_summaries.append(summary)

        return strategic_summaries

    def compress_meta(self,
                      strategic_summaries: list[SummaryAnchor]) -> list[SummaryAnchor]:
        """Level 2→3: Discover cross-domain META principles from STRATEGIC summaries.

        Groups strategic summaries across different domains to find universal
        principles that transcend specific domains.
        """
        if len(strategic_summaries) < 2:  # meta needs fewer — cross-domain can be 2
            return []

        # Build embedding lookup
        embeddings: dict[str, list[float]] = {}
        summary_map: dict[str, SummaryAnchor] = {}
        for s in strategic_summaries:
            if s.centroid_embedding:
                embeddings[s.id] = s.centroid_embedding
                summary_map[s.id] = s

        if len(summary_map) < 2:
            return []

        aid_list = list(summary_map.keys())
        # Use slightly lower threshold for cross-domain matching
        meta_threshold = max(0.35, self.similarity_threshold - 0.15)
        clusters = _find_clusters(aid_list, embeddings, meta_threshold)

        meta_summaries: list[SummaryAnchor] = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            cluster_summaries = [summary_map[aid] for aid in cluster]

            # Collect texts
            all_texts = [s.text for s in cluster_summaries]

            # Compute centroid
            cluster_embs = [embeddings[aid] for aid in cluster]
            centroid = _compute_centroid(cluster_embs)

            # Extract cross-domain features
            key_terms = _extract_key_terms_via_texts(all_texts)
            action = _extract_action_patterns(all_texts)
            domains = set()
            for s in cluster_summaries:
                for tag in s.tags:
                    if tag.startswith("domain:"):
                        domains.add(tag.split(":", 1)[1])
            domain_str = ", ".join(sorted(domains)[:3]) if domains else "multiple domains"
            topic = key_terms[0] if key_terms else "universal pattern"

            # Collect source IDs
            all_source_ids: list[str] = []
            for s in cluster_summaries:
                all_source_ids.extend(s.source_anchor_ids)

            summary_text = _fill_template(
                level=CompressionLevel.META,
                domain=domain_str,
                topic=topic,
                action=action,
                entities=["cross-domain applications"],
                context=f"{len(domains)} domains",
                count=len(cluster),
                token_limit=self.meta_token_limit,
            )

            # Confidence
            intra_sims: list[float] = []
            for ii in range(min(len(cluster_embs), 10)):
                for jj in range(ii + 1, min(len(cluster_embs), 10)):
                    intra_sims.append(_cosine_sim(cluster_embs[ii], cluster_embs[jj]))
            confidence = sum(intra_sims) / max(1, len(intra_sims))

            all_tags: list[str] = []
            for s in cluster_summaries:
                all_tags.extend(s.tags)
            tag_counter = Counter(all_tags)
            common_tags = [t for t, c in tag_counter.most_common(6) if c >= 2
                           and not t.startswith("domain:")
                           and not t.startswith("level:")]

            self._counter += 1
            summary = SummaryAnchor(
                id=f"summary_l3_{self._counter}",
                text=summary_text,
                source_anchor_ids=all_source_ids[:self._session_compressor.max_source_anchors],
                centroid_embedding=centroid,
                compression_level=CompressionLevel.META,
                confidence=min(1.0, confidence),
                tags=common_tags + [
                    f"domain:{domain_str.replace(', ', '+')}",
                    "level:meta",
                ],
            )
            summary.to_anchor_proxy()
            meta_summaries.append(summary)

        return meta_summaries

    def compress_pipeline(
        self,
        anchors_by_session: dict[str, list[Anchor]],
    ) -> dict[CompressionLevel, list[SummaryAnchor]]:
        """Run the full three-level compression pipeline.

        Args:
            anchors_by_session: Dict mapping session_id → list of Anchor objects

        Returns:
            Dict mapping CompressionLevel → list of SummaryAnchor created at that level
        """
        result: dict[CompressionLevel, list[SummaryAnchor]] = {
            CompressionLevel.EPISODIC: [],
            CompressionLevel.STRATEGIC: [],
            CompressionLevel.META: [],
        }

        # Level 0→1: Session-level compression
        episodic_all: list[SummaryAnchor] = []
        for session_id, anchors in anchors_by_session.items():
            episodic = self.compress_session(anchors, session_id)
            result[CompressionLevel.EPISODIC].extend(episodic)
            episodic_all.extend(episodic)

        if not episodic_all:
            return result

        # Level 1→2: Strategic compression
        strategic_all = self.compress_strategic(episodic_all)
        result[CompressionLevel.STRATEGIC] = strategic_all

        if not strategic_all:
            return result

        # Level 2→3: Meta compression
        meta_all = self.compress_meta(strategic_all)
        result[CompressionLevel.META] = meta_all

        return result

    def add_to_graph(self, graph, summaries: list[SummaryAnchor],
                     edge_type: str = "compresses") -> int:
        """Insert summary anchors into the graph with "compresses" edges to sources.

        Each summary's proxy Anchor is added to the graph, and edges are created
        connecting the summary to each of its source anchors.

        Args:
            graph: StarGraph instance
            summaries: List of SummaryAnchor objects to insert
            edge_type: Type of edge to create (default: "compresses")

        Returns:
            Number of edges created
        """
        edges_created = 0

        for summary in summaries:
            proxy = summary.to_anchor_proxy()

            # Add the proxy anchor to the graph
            graph.add_anchor(proxy)

            # Create "compresses" edges from summary to each source anchor
            for source_id in summary.source_anchor_ids:
                if source_id in graph.anchors:
                    graph.add_edge(
                        proxy.id, source_id,
                        weight=summary.confidence * 0.7,
                        edge_type=edge_type,
                    )
                    edges_created += 1

        return edges_created


# ---------------------------------------------------------------------------
# Helper: TF-IDF for SummaryAnchor texts (no Anchor objects needed)
# ---------------------------------------------------------------------------

def _extract_key_terms_via_texts(texts: list[str], top_k: int = 8) -> list[str]:
    """TF-IDF style key term extraction from raw text strings."""
    if not texts:
        return []

    N = len(texts)
    doc_freq: Counter = Counter()
    term_freq: Counter = Counter()

    for text in texts:
        tokens = _tokenize(text)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freq[t] += 1
        for t in tokens:
            term_freq[t] += 1

    scored: list[tuple[str, float]] = []
    for term, tf in term_freq.items():
        df = doc_freq.get(term, 1)
        idf = math.log((N + 1) / (df + 1)) + 1.0
        score = tf * idf
        scored.append((term, score))

    scored.sort(key=lambda x: -x[1])
    return [term for term, _ in scored[:top_k]]
