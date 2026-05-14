"""Causal Edge Types — richer causal relationships beyond "related".

Extends the edge type system with:
  CAUSES       — A directly causes B
  DEPENDS_ON   — B depends on A to function
  MOTIVATES    — A motivates the creation/investigation of B
  GOAL_OF      — B is a sub-goal of A
  RESULT_OF    — B is a result/consequence of A
  PRECEDES     — A temporally precedes B (but may not cause it)

Includes heuristic type inference from text patterns.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# Extended causal relation types
CAUSAL_EDGE_TYPES = {
    "causes": {
        "weight": 1.5,
        "description": "A directly causes B to happen",
        "patterns": [
            r'(?:causes?|leads? to|results? in|triggers?|produces?)\s+',
            r'(?:导致|引起|造成|触发|产生)\s*',
            r'(?:because|since|due to|as a result of)\s+',
            r'(?:因为|由于|所以|因此)\s*',
        ],
    },
    "depends_on": {
        "weight": 1.4,
        "description": "B depends on A to function or exist",
        "patterns": [
            r'(?:depends? on|requires?|needs?|relies? on)\s+',
            r'(?:依赖|需要|依靠|取决于)\s*',
            r'(?:prerequisite|必须先|前置)\s*',
        ],
    },
    "motivates": {
        "weight": 1.3,
        "description": "A motivates the creation or investigation of B",
        "patterns": [
            r'(?:motivates?|drives?|inspires?|encourages?)\s+',
            r'(?:推动|激励|促使|驱动)\s*',
            r'(?:goal|aim|purpose|objective)\s+(?:is|was)\s+to\s+',
        ],
    },
    "goal_of": {
        "weight": 1.2,
        "description": "B is a sub-goal or step toward achieving A",
        "patterns": [
            r'(?:in order to|so that|to achieve|toward)\s+',
            r'(?:为了|以便|旨在|目标是)\s*',
            r'(?:step|phase|milestone|subtask)\s+(?:of|for)\s+',
        ],
    },
    "result_of": {
        "weight": 1.25,
        "description": "B is a result, output, or consequence of A",
        "patterns": [
            r'(?:output|outcome|result|consequence|product)\s+(?:of|from)\s+',
            r'(?:结果|产出|成果|产物)\s*',
            r'(?:as a (?:result|consequence)|thus|therefore|hence)\s+',
        ],
    },
    "precedes": {
        "weight": 1.1,
        "description": "A temporally comes before B",
        "patterns": [
            r'(?:before|prior to|preceding|earlier than)\s+',
            r'(?:之前|先于|早于|在.*之前)\s*',
            r'(?:first|initially|originally)\s+.+(?:then|later|after|随后)',
        ],
    },
}

# Weight mapping for graph traversal
CAUSAL_TRAVERSAL_WEIGHTS = {k: v["weight"] for k, v in CAUSAL_EDGE_TYPES.items()}


@dataclass
class CausalChain:
    """A traced chain of causal edges through the graph."""
    anchor_ids: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    total_strength: float = 0.0
    depth: int = 0
    start_id: str = ""
    end_id: str = ""


class CausalEdgeClassifier:
    """Heuristic classifier for inferring causal edge types from text.

    Usage:
        classifier = CausalEdgeClassifier()
        edge_type, confidence = classifier.infer(text_a, text_b)
        # → ("causes", 0.75)
    """

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for etype, info in CAUSAL_EDGE_TYPES.items():
            self._compiled_patterns[etype] = [
                re.compile(p, re.IGNORECASE) for p in info["patterns"]
            ]

    def infer(self, text_a: str, text_b: str) -> tuple[str, float]:
        """Infer the best causal edge type between two texts.

        Returns (edge_type, confidence). Defaults to ("causes", 0.1) if no match.
        """
        combined = f"{text_a} {text_b}".lower()
        scores: dict[str, float] = defaultdict(float)

        for etype, patterns in self._compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                if pattern.search(combined):
                    matches += 1
            if matches > 0:
                scores[etype] = min(0.9, 0.3 + matches * 0.15)

        if not scores:
            return ("causes", 0.1)

        best_type = max(scores, key=lambda k: scores[k])
        return best_type, scores[best_type]

    def infer_from_anchors(self, anchor_a, anchor_b) -> tuple[str, float]:
        """Infer causal type between two anchors, using text + tags."""
        text_a = getattr(anchor_a, 'text', '')
        text_b = getattr(anchor_b, 'text', '')
        edge_type, confidence = self.infer(text_a, text_b)

        # Boost confidence if tags overlap
        tags_a = set(getattr(anchor_a, 'tags', []))
        tags_b = set(getattr(anchor_b, 'tags', []))
        if tags_a and tags_b and tags_a & tags_b:
            confidence = min(0.95, confidence + 0.1)

        return edge_type, confidence

    @staticmethod
    def trace_causal_chain(graph, start_id: str, max_depth: int = 5,
                          min_strength: float = 0.2) -> list[CausalChain]:
        """Trace causal chains starting from an anchor.

        Follows only causal-type edges (causes, depends_on, motivates, goal_of,
        result_of, precedes). Returns list of CausalChain objects.
        """
        if start_id not in graph.anchors:
            return []

        chains: list[CausalChain] = []
        visited: set[str] = {start_id}

        def _dfs(node_id: str, chain_ids: list[str], chain_types: list[str],
                 strength: float, depth: int):
            if depth >= max_depth:
                chains.append(CausalChain(
                    anchor_ids=list(chain_ids),
                    edge_types=list(chain_types),
                    total_strength=strength,
                    depth=depth,
                    start_id=start_id,
                    end_id=node_id,
                ))
                return

            has_causal_neighbor = False
            for neighbor_id in graph._adjacency.get(node_id, set()):
                if neighbor_id in visited:
                    continue
                edge_key = graph._key(node_id, neighbor_id)
                edge = graph.edges.get(edge_key)
                if not edge:
                    continue
                etype = getattr(edge, 'edge_type', '')
                if etype not in CAUSAL_TRAVERSAL_WEIGHTS:
                    continue
                if edge.weight < min_strength:
                    continue

                has_causal_neighbor = True
                visited.add(neighbor_id)
                _dfs(neighbor_id,
                     chain_ids + [neighbor_id],
                     chain_types + [etype],
                     strength + edge.weight * CAUSAL_TRAVERSAL_WEIGHTS[etype],
                     depth + 1)
                visited.discard(neighbor_id)

            if not has_causal_neighbor:
                chains.append(CausalChain(
                    anchor_ids=list(chain_ids),
                    edge_types=list(chain_types),
                    total_strength=strength,
                    depth=depth,
                    start_id=start_id,
                    end_id=node_id,
                ))

        _dfs(start_id, [start_id], [], 0.0, 0)
        chains.sort(key=lambda c: -c.total_strength)
        return chains

    @staticmethod
    def get_causal_weight(edge_type: str) -> float:
        """Get the traversal weight for a causal edge type."""
        return CAUSAL_TRAVERSAL_WEIGHTS.get(edge_type, 1.0)
