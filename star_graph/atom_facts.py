"""Atom Facts Extraction — LLM-assisted entity/event-centric memory abstraction.

Philosophy (from Synthius-Mem, 94.37% LoCoMo):
  "Don't retrieve what was said, extract what is known about the user/world."

During sleep consolidation, clusters of raw anchors are post-processed by a
lightweight LLM to extract atomic, entity-centric facts. These Atom Facts are
stored in the graph with bidirectional hyperlinks to source anchors, enabling:

- High-quality factual recall (not just semantic similarity)
- Cross-session knowledge accumulation
- Adversarial QA resilience (Cat 5 in LoCoMo)

Provider support: OpenAI-compatible, Anthropic, or local models (via Ollama).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from .anchor import Anchor, MemoryState
from .graph import StarGraph


# ── Data Structures ────────────────────────────────────────

@dataclass
class AtomFact:
    """A single atomic fact about an entity, event, or relationship.

    Example:
      "User prefers dark mode across all applications"
      "Redis connection pool was increased from 10 to 20"
      "Selenium tests use headless Chrome on CI"
    """

    id: str
    text: str                        # The atomic fact statement
    subject: str = ""                # Primary entity (e.g., "User", "Redis", "Selenium")
    predicate: str = ""              # Relationship/action (e.g., "prefers", "was_configured")
    object: str = ""                 # Target entity or value (e.g., "dark mode", "pool_size=20")
    fact_type: str = "factual"       # factual / preference / event / relationship / constraint
    confidence: float = 0.7          # LLM-reported confidence (0..1)
    source_anchor_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    embedding: list[float] | None = None

    def to_anchor(self, tags: list[str] | None = None) -> Anchor:
        """Convert to a graph anchor for storage."""
        import hashlib
        anchor_id = hashlib.blake2b(
            (self.text + str(self.created_at)).encode(), digest_size=8
        ).hexdigest()
        tags = (tags or []) + ["atom_fact", f"fact_type:{self.fact_type}",
                                f"subject:{self.subject}"]
        if self.predicate:
            tags.append(f"predicate:{self.predicate}")
        anchor = Anchor(
            id=anchor_id, text=self.text[:280],
            tags=tags,
            embedding=self.embedding,
            source_session="atom_facts",
        )
        anchor.vector.stability = 0.9       # Atom facts are highly stable
        anchor.vector.hippocampal_dependency = 0.05  # near-cortical
        anchor.vector.importance = self.confidence
        anchor.vector.confidence = self.confidence
        anchor.state = MemoryState.ACTIVE
        return anchor


@dataclass
class ExtractionResult:
    """Output from a single fact extraction run."""
    facts: list[AtomFact]
    input_anchor_count: int
    cluster_topic: str
    latency_ms: float
    provider: str
    raw_response: str = ""


# ── LLM Provider Interface ─────────────────────────────────

class LLMProvider:
    """Abstract LLM provider for fact extraction."""

    def complete(self, prompt: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.3) -> str:
        raise NotImplementedError


class OpenAIClientProvider(LLMProvider):
    """OpenAI-compatible API (GPT-4o-mini, Qwen via Ollama, Minimax, etc.)."""

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL",
                                                   "https://api.openai.com/v1")
        self.model = model or os.environ.get("ATOM_FACT_MODEL", "gpt-4o-mini")

    def complete(self, prompt: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.3) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            response = client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            return ""
        except Exception as e:
            return f"__ERROR__:{e}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API."""

    def __init__(self, api_key: str = "", model: str = "claude-haiku-4-5"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def complete(self, prompt: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.3) -> str:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                system=system_prompt or "You extract atomic facts from conversation data.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.content[0].text if response.content else ""
            return text or ""
        except ImportError:
            return ""
        except Exception as e:
            return f"__ERROR__:{e}"


class TemplateProvider(LLMProvider):
    """Fallback: template-based extraction without LLM.

    Parses the prompt format to extract anchor texts and applies
    keyword/pattern heuristics. Works offline, zero API cost.
    """

    def complete(self, prompt: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.3) -> str:
        # Parse the prompt format: "Segments:\n  [1] text\n  [2] text\n..."
        anchors = self._parse_segment_lines(prompt)
        facts: list[dict] = []

        # Also parse topic from prompt
        topic = ""
        for line in prompt.split("\n"):
            if line.startswith("Topic:"):
                topic = line.split("Topic:", 1)[1].strip()

        seen_texts = set()
        for text in anchors:
            text_lower = text.lower()
            normalized = text.strip()[:200]

            # Preference detection
            pref_patterns = ["prefer", "like", "want", "don't like", "hate", "love",
                           "always use", "never use", "favorite", "need",
                           "doesn't like", "do not like"]
            for p in pref_patterns:
                if p in text_lower and normalized not in seen_texts:
                    seen_texts.add(normalized)
                    facts.append({
                        "text": normalized,
                        "subject": "User",
                        "predicate": "prefers",
                        "object": topic,
                        "fact_type": "preference",
                        "confidence": 0.6,
                    })
                    break

            # Fix/configuration detection
            fix_patterns = ["fixed", "configured", "set up", "increased", "decreased",
                          "changed", "updated", "installed", "deployed", "improved"]
            for p in fix_patterns:
                if p in text_lower and normalized not in seen_texts:
                    seen_texts.add(normalized)
                    facts.append({
                        "text": normalized,
                        "subject": "System",
                        "predicate": p,
                        "object": topic,
                        "fact_type": "event",
                        "confidence": 0.65,
                    })
                    break

            # Result/outcome detection
            result_patterns = ["performance improved", "after", "resulted in",
                             "led to", "caused", "resolved"]
            for p in result_patterns:
                if p in text_lower and normalized not in seen_texts:
                    seen_texts.add(normalized)
                    facts.append({
                        "text": normalized,
                        "subject": "System",
                        "predicate": "resulted_in",
                        "object": topic,
                        "fact_type": "event",
                        "confidence": 0.55,
                    })
                    break

            # Meeting/relationship detection
            import re
            names = re.findall(r'\b[A-Z][a-z]+\b', text)
            if names and len(names) >= 2 and normalized not in seen_texts:
                seen_texts.add(normalized)
                facts.append({
                    "text": normalized,
                    "subject": names[0],
                    "predicate": "met_with",
                    "object": names[1] if len(names) > 1 else "",
                    "fact_type": "relationship",
                    "confidence": 0.5,
                })

        return json.dumps({"facts": facts}, ensure_ascii=False)

    @staticmethod
    def _parse_segment_lines(prompt: str) -> list[str]:
        """Extract anchor texts from the prompt's segment lines.

        Format: "  [1] anchor text here\n  [2] more text"
        """
        texts = []
        for line in prompt.split("\n"):
            line = line.strip()
            # Match numbered segments: [1], [2], etc.
            if line.startswith("[") and "]" in line[:5]:
                bracket_end = line.index("]")
                text = line[bracket_end + 1:].strip()
                if text and len(text) > 10:
                    texts.append(text)
        return texts


# ── Fact Extraction Prompt ─────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """You are an atomic fact extractor for an AI memory system. Your task is to extract entity-centric, self-contained facts from conversation segments.

Rules:
1. Each fact must be a single, atomic statement about ONE entity or ONE event.
2. DO NOT summarize conversations. Extract SPECIFIC, REUSABLE knowledge.
3. Facts should be self-contained — understandable without context.
4. Include: preferences, decisions, technical facts, relationships, constraints, past events.
5. Output ONLY valid JSON (no markdown, no commentary).

Output format:
```json
{
  "facts": [
    {
      "text": "User prefers dark mode in all applications",
      "subject": "User",
      "predicate": "prefers",
      "object": "dark mode",
      "fact_type": "preference",
      "confidence": 0.9
    }
  ]
}
```

Fact types: "preference", "event", "relationship", "factual", "constraint", "decision"
Confidence: 0.0-1.0 based on how clearly the source material supports this fact."""

_EXTRACTION_USER_TEMPLATE = """Extract atomic facts from this conversation data:

Topic: {topic}
Number of segments: {count}

Segments:
{segments}

Extract atomic facts in JSON format:"""


# ── FactExtractor ──────────────────────────────────────────

class FactExtractor:
    """LLM-assisted atomic fact extraction from anchor clusters.

    Usage:
        extractor = FactExtractor(provider="openai", model="gpt-4o-mini")
        result = extractor.extract(anchors, topic="Redis debugging")
        for fact in result.facts:
            graph.add_anchor(fact.to_anchor())
    """

    def __init__(self,
                 provider: str = "template",  # "openai", "anthropic", "template"
                 api_key: str = "",
                 base_url: str = "",
                 model: str = "",
                 min_cluster_size: int = 3,
                 max_anchors_per_batch: int = 15,
                 min_fact_confidence: float = 0.4):
        self.min_cluster_size = min_cluster_size
        self.max_anchors_per_batch = max_anchors_per_batch
        self.min_fact_confidence = min_fact_confidence

        # Initialize the LLM provider
        if provider == "openai":
            self.llm = OpenAIClientProvider(
                api_key=api_key, base_url=base_url, model=model)
            self.provider_name = "openai"
        elif provider == "anthropic":
            self.llm = AnthropicProvider(api_key=api_key, model=model)
            self.provider_name = "anthropic"
        else:
            self.llm = TemplateProvider()
            self.provider_name = "template"

        self._extraction_count: int = 0
        self._total_facts: int = 0
        self._total_latency_ms: float = 0.0

    def extract(self, anchors: list[Anchor],
                topic: str = "",
                cluster_id: str = "") -> ExtractionResult:
        """Extract atom facts from a cluster of related anchors.

        Args:
            anchors: List of related anchors (e.g., from same session/topic)
            topic: Human-readable topic label for context
            cluster_id: Cluster identifier for traceability

        Returns:
            ExtractionResult with extracted AtomFacts
        """
        if len(anchors) < self.min_cluster_size:
            return ExtractionResult(
                facts=[], input_anchor_count=len(anchors),
                cluster_topic=topic, latency_ms=0.0,
                provider=self.provider_name,
            )

        t0 = time.perf_counter()

        # Sample anchors if cluster is too large
        selected = anchors[:self.max_anchors_per_batch]

        # Build prompt
        segments_text = "\n".join(
            f"  [{i+1}] {a.text[:200]}" for i, a in enumerate(selected)
        )
        prompt = _EXTRACTION_USER_TEMPLATE.format(
            topic=topic or "general",
            count=len(selected),
            segments=segments_text,
        )

        # Call LLM
        raw_response = self.llm.complete(
            prompt=prompt,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.2,
        )

        # Parse response
        facts = self._parse_facts(raw_response, selected)

        latency = (time.perf_counter() - t0) * 1000
        self._extraction_count += 1
        self._total_facts += len(facts)
        self._total_latency_ms += latency

        return ExtractionResult(
            facts=facts, input_anchor_count=len(anchors),
            cluster_topic=topic, latency_ms=latency,
            provider=self.provider_name, raw_response=raw_response,
        )

    def _parse_facts(self, raw_response: str,
                     source_anchors: list[Anchor]) -> list[AtomFact]:
        """Parse LLM response into AtomFact objects."""
        facts: list[AtomFact] = []

        # Handle errors
        if raw_response.startswith("__ERROR__:"):
            return facts
        if not raw_response.strip():
            return facts

        # Extract JSON from response (handle markdown code blocks)
        json_text = raw_response
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        try:
            parsed = json.loads(json_text.strip())
            raw_facts = parsed.get("facts", [])
            if isinstance(raw_facts, list):
                for rf in raw_facts:
                    if not isinstance(rf, dict):
                        continue
                    text = rf.get("text", "").strip()
                    if not text or len(text) < 10:
                        continue
                    confidence = float(rf.get("confidence", 0.7))
                    if confidence < self.min_fact_confidence:
                        continue

                    source_ids = [a.id for a in source_anchors[:5]]
                    fact = AtomFact(
                        id="",  # generated in to_anchor()
                        text=text[:280],
                        subject=rf.get("subject", ""),
                        predicate=rf.get("predicate", ""),
                        object=rf.get("object", ""),
                        fact_type=rf.get("fact_type", "factual"),
                        confidence=confidence,
                        source_anchor_ids=source_ids,
                    )
                    facts.append(fact)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return facts

    def extract_from_clusters(self,
                               clusters: list[tuple[str, list[Anchor]]]
                               ) -> list[AtomFact]:
        """Extract facts from multiple clusters in one pass.

        Args:
            clusters: List of (topic, [anchors]) tuples

        Returns:
            All extracted AtomFacts across all clusters
        """
        all_facts: list[AtomFact] = []
        for topic, anchors in clusters:
            result = self.extract(anchors, topic=topic)
            all_facts.extend(result.facts)
        return all_facts

    def add_facts_to_graph(self, graph: StarGraph,
                           facts: list[AtomFact]) -> int:
        """Store atom facts as anchors in the graph with source links.

        Creates:
        - Fact anchors with high stability
        - "derived_from" edges: fact → source anchor
        - "supports" edges: source anchor → fact

        Returns number of facts added.
        """
        added = 0
        for fact in facts:
            anchor = fact.to_anchor()
            fact.id = anchor.id
            graph.add_anchor(anchor)

            # Create bidirectional hyperlinks
            for src_id in fact.source_anchor_ids:
                if src_id in graph.anchors:
                    graph.add_edge(
                        fact.id, src_id,
                        weight=fact.confidence,
                        edge_type="derived_from",
                        source_type="llm_extraction",
                        confidence=fact.confidence,
                    )
            added += 1
        return added

    @property
    def stats(self) -> dict:
        return {
            "provider": self.provider_name,
            "extractions": self._extraction_count,
            "total_facts": self._total_facts,
            "avg_latency_ms": self._total_latency_ms / max(1, self._extraction_count),
            "facts_per_extraction": self._total_facts / max(1, self._extraction_count),
        }


# ── Helper: check if LLM API is available ──────────────────

def check_llm_availability(provider: str = "openai") -> dict:
    """Check if an LLM provider is configured and reachable."""
    result = {"provider": provider, "available": False, "model": "", "error": ""}

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            result["error"] = "OPENAI_API_KEY not set"
            return result
        try:
            from openai import OpenAI
            result["available"] = True
            result["model"] = os.environ.get("ATOM_FACT_MODEL", "gpt-4o-mini")
        except ImportError:
            result["error"] = "openai package not installed"

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            result["error"] = "ANTHROPIC_API_KEY not set"
            return result
        try:
            from anthropic import Anthropic
            result["available"] = True
            result["model"] = "claude-haiku-4-5"
        except ImportError:
            result["error"] = "anthropic package not installed"

    else:
        result["available"] = True
        result["model"] = "template (offline)"
        result["note"] = "template-based extraction, no API needed"

    return result
