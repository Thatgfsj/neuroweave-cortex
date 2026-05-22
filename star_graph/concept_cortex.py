"""Concept Cortex — concept network for cognitive memory runtime.

From "sentence memory" to "concept network". Concepts are persistent cognitive
entities that sit above individual memory anchors. They form the "understanding
layer" of the cortex.

Concepts can be:
- Activated by memory recall, perception, or inference
- Fused with similar concepts
- Linked to memory anchors as evidence
- Competitive (similar concepts vie for activation)
- Hierarchical (parent/child concept relationships)
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ── Core Concept Seeds ──────────────────────────────────────────

CORE_CONCEPT_SEEDS: dict[str, str] = {
    "技术探索": "对新技术、工具、框架的持续学习和探索行为",
    "长期主义": "偏好长期投资、深度积累、系统性思维",
    "效率优先": "追求工作效率、自动化、工具优化",
    "创造欲": "通过创造、构建、开发来获得满足感",
    "安全感": "对稳定、可靠、可控环境的偏好",
    "社交认同": "通过社区、团队、分享获得自我验证",
    "知识深度": "偏好深入理解原理而非表面应用",
    "实用主义": "关注实际效果、可用性、而非理论完美",
    "控制欲": "希望对系统、代码、项目有完全掌控",
    "成长驱动": "持续学习、自我提升、能力扩展",
}


# ── Data Structures ──────────────────────────────────────────────


@dataclass
class ConceptNode:
    """A concept in the cognitive concept network.

    Concepts are NOT memory anchors. They are abstract cognitive entities
    that can be activated by memory recall, perception, or inference.
    """
    id: str
    label: str                      # human-readable: "技术探索", "Python开发"
    description: str = ""           # 1-2 sentence description
    embedding: list[float] | None = None  # centroid of all linked anchors
    activation: float = 0.0         # current activation level (0..1)
    baseline_activation: float = 0.1  # resting activation
    stability: float = 0.5          # 0..1 how stable/established
    linked_anchors: list[str] = field(default_factory=list)
    parent_concepts: list[str] = field(default_factory=list)
    child_concepts: list[str] = field(default_factory=list)
    related_concepts: dict[str, float] = field(default_factory=dict)  # concept_id → weight
    domain: str = ""
    emergence_source: str = "manual"  # manual | abstraction_chain | perception | inference
    evidence_count: int = 0
    contradiction_count: int = 0
    last_activated_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def activate(self, energy: float) -> float:
        self.activation = min(1.0, self.activation + energy)
        self.last_activated_at = time.time()
        return self.activation

    def decay(self, dt: float, decay_rate: float = 0.01) -> float:
        """Decay toward baseline."""
        diff = self.activation - self.baseline_activation
        diff *= math.exp(-decay_rate * dt)
        self.activation = self.baseline_activation + max(0, diff)
        return self.activation

    @property
    def is_active(self) -> bool:
        return self.activation >= 0.2

    def strengthen_evidence(self):
        self.evidence_count += 1
        self.stability = min(1.0, self.stability + 0.01)

    def weaken_evidence(self):
        self.contradiction_count += 1
        self.stability = max(0.1, self.stability - 0.02)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label,
            "description": self.description,
            "activation": round(self.activation, 3),
            "stability": round(self.stability, 3),
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
            "linked_anchors": len(self.linked_anchors),
            "parent_count": len(self.parent_concepts),
            "child_count": len(self.child_concepts),
            "domain": self.domain,
        }


@dataclass
class ConceptFusion:
    """Result of fusing two similar concepts."""
    source_concept_ids: list[str]
    fused_concept: ConceptNode
    similarity: float
    confidence: float
    anchor_migration_count: int = 0


@dataclass
class ConceptActivationPath:
    """Trace of how activation spread through the concept network."""
    source_concept_id: str
    target_concept_id: str
    path: list[str]            # intermediate concept IDs
    total_activation: float
    path_length: int


# ── Concept Cortex ───────────────────────────────────────────────


class ConceptCortex:
    """Manages the concept network — activation, fusion, competition, anchoring."""

    def __init__(self, graph=None, *,
                 config: dict[str, Any] | None = None,
                 embedder=None):
        self.graph = graph
        self._config = config or {}
        self._embedder = embedder

        self._max_concepts = self._config.get("max_concepts", 500)
        self._min_evidence = self._config.get("min_evidence_for_concept", 2)
        self._decay_rate = self._config.get("activation_decay_rate", 0.01)
        self._baseline = self._config.get("baseline_activation", 0.1)
        self._active_threshold = self._config.get("activation_threshold", 0.2)
        self._spread_steps = self._config.get("spread_steps", 2)
        self._spread_decay = self._config.get("spread_decay", 0.5)
        self._merge_threshold = self._config.get("merge_similarity_threshold", 0.85)

        # Concept storage
        self._concepts: dict[str, ConceptNode] = {}

        # Seed concepts if enabled
        if self._config.get("seed_concepts_enabled", True):
            self._seed_core_concepts()

    # ── Concept Lifecycle ──────────────────────────────────

    def get_or_create_concept(self, label: str, *,
                              embedding: list[float] | None = None,
                              domain: str = "",
                              description: str = "") -> ConceptNode:
        """Retrieve existing concept by label or create a new one."""
        # Normalize label
        label = label.strip().lower()
        for concept in self._concepts.values():
            if concept.label.lower() == label:
                return concept

        if len(self._concepts) >= self._max_concepts:
            self._prune_weakest()

        concept = ConceptNode(
            id=str(uuid.uuid4()),
            label=label,
            description=description or f"Concept: {label}",
            embedding=embedding,
            baseline_activation=self._baseline,
            domain=domain,
        )
        self._concepts[concept.id] = concept
        return concept

    def link_anchor_to_concept(self, anchor_id: str,
                               concept_ids: list[str],
                               strength: float = 0.5):
        """Associate an anchor with concepts."""
        for cid in concept_ids:
            if cid in self._concepts:
                concept = self._concepts[cid]
                if anchor_id not in concept.linked_anchors:
                    concept.linked_anchors.append(anchor_id)
                    concept.strengthen_evidence()
                # Boost activation
                concept.activate(strength * 0.2)

    def merge_concepts(self, cid_a: str, cid_b: str) -> ConceptFusion | None:
        """Fuse two similar concepts into one."""
        if cid_a not in self._concepts or cid_b not in self._concepts:
            return None

        a = self._concepts[cid_a]
        b = self._concepts[cid_b]

        # Compute similarity (tag overlap proxy if no embeddings)
        sim = self._concept_similarity(a, b)
        if sim < self._merge_threshold:
            return None

        # Merge: a absorbs b
        merged_anchors = list(set(a.linked_anchors + b.linked_anchors))

        fused = ConceptNode(
            id=a.id,
            label=a.label,
            description=f"{a.description}; also: {b.description}",
            embedding=a.embedding,
            activation=max(a.activation, b.activation),
            stability=(a.stability + b.stability) / 2 + 0.05,
            linked_anchors=merged_anchors,
            parent_concepts=list(set(a.parent_concepts + b.parent_concepts)),
            child_concepts=list(set(a.child_concepts + b.child_concepts)),
            domain=a.domain or b.domain,
            emergence_source="fusion",
            evidence_count=a.evidence_count + b.evidence_count,
            contradiction_count=a.contradiction_count + b.contradiction_count,
        )

        # Remap b's children
        for child_id in b.child_concepts:
            if child_id in self._concepts:
                child = self._concepts[child_id]
                if b.id in child.parent_concepts:
                    child.parent_concepts.remove(b.id)
                if fused.id not in child.parent_concepts:
                    child.parent_concepts.append(fused.id)

        # Update related concepts
        for rid, weight in b.related_concepts.items():
            fused.related_concepts[rid] = max(fused.related_concepts.get(rid, 0), weight)

        self._concepts[fused.id] = fused
        del self._concepts[cid_b]

        return ConceptFusion(
            source_concept_ids=[cid_a, cid_b],
            fused_concept=fused,
            similarity=sim,
            confidence=sim,
            anchor_migration_count=len(b.linked_anchors),
        )

    @staticmethod
    def _concept_similarity(a: ConceptNode, b: ConceptNode) -> float:
        """Compute similarity between two concepts."""
        score = 0.0
        # Label overlap
        a_words = set(a.label.lower().split())
        b_words = set(b.label.lower().split())
        if a_words and b_words:
            score += len(a_words & b_words) / len(a_words | b_words) * 0.4
        # Description overlap
        if a.description and b.description:
            a_desc_words = set(a.description.lower().split())
            b_desc_words = set(b.description.lower().split())
            if a_desc_words and b_desc_words:
                score += len(a_desc_words & b_desc_words) / len(a_desc_words | b_desc_words) * 0.3
        # Shared anchors
        shared = len(set(a.linked_anchors) & set(b.linked_anchors))
        total = len(set(a.linked_anchors) | set(b.linked_anchors))
        if total > 0:
            score += (shared / total) * 0.3
        return min(1.0, score)

    # ── Activation Dynamics ────────────────────────────────

    def activate(self, concept_ids: list[str],
                 energy: float = 0.5) -> list[str]:
        """Activate concepts and return IDs of cascade-activated concepts."""
        activated: set[str] = set()

        for cid in concept_ids:
            if cid not in self._concepts:
                continue
            self._concepts[cid].activate(energy)
            activated.add(cid)

        return list(activated)

    def spread_activation(self, seed_concept_ids: list[str],
                          steps: int | None = None,
                          decay: float | None = None) -> list[ConceptActivationPath]:
        """Spreading activation through the concept network."""
        steps = steps or self._spread_steps
        decay = decay or self._spread_decay
        paths: list[ConceptActivationPath] = []

        for seed_id in seed_concept_ids:
            if seed_id not in self._concepts:
                continue

            seed = self._concepts[seed_id]
            energy = seed.activation * decay

            # BFS through related concepts
            visited: set[str] = {seed_id}
            queue: list[tuple[str, float, list[str]]] = [(seed_id, energy, [seed_id])]

            for _ in range(steps):
                next_queue: list[tuple[str, float, list[str]]] = []
                for cid, current_energy, path in queue:
                    concept = self._concepts.get(cid)
                    if not concept:
                        continue

                    # Spread to related concepts
                    for rel_id, weight in concept.related_concepts.items():
                        if rel_id in visited or rel_id not in self._concepts:
                            continue
                        visited.add(rel_id)
                        new_energy = current_energy * weight * decay
                        if new_energy > 0.05:
                            self._concepts[rel_id].activate(new_energy)
                            new_path = list(path) + [rel_id]
                            next_queue.append((rel_id, new_energy, new_path))
                            paths.append(ConceptActivationPath(
                                source_concept_id=seed_id,
                                target_concept_id=rel_id,
                                path=new_path,
                                total_activation=new_energy,
                                path_length=len(new_path),
                            ))

                    # Spread to children
                    for child_id in concept.child_concepts:
                        if child_id in visited or child_id not in self._concepts:
                            continue
                        visited.add(child_id)
                        new_energy = current_energy * 0.7 * decay
                        if new_energy > 0.05:
                            self._concepts[child_id].activate(new_energy)
                            new_path = list(path) + [child_id]
                            next_queue.append((child_id, new_energy, new_path))
                            paths.append(ConceptActivationPath(
                                source_concept_id=seed_id,
                                target_concept_id=child_id,
                                path=new_path,
                                total_activation=new_energy,
                                path_length=len(new_path),
                            ))

                queue = next_queue
                if not queue:
                    break

        return paths

    def decay_all(self, dt: float):
        """Decay activation of all concepts toward baseline."""
        for concept in self._concepts.values():
            concept.decay(dt, self._decay_rate)

    # ── Retrieval ──────────────────────────────────────────

    def get_active_concepts(self, min_activation: float | None = None) -> list[ConceptNode]:
        """All concepts above activation threshold."""
        threshold = min_activation if min_activation is not None else self._active_threshold
        return [c for c in self._concepts.values() if c.activation >= threshold]

    def get_concepts_for_domain(self, domain: str) -> list[ConceptNode]:
        """Concepts filtered by domain."""
        return [c for c in self._concepts.values() if c.domain == domain]

    def find_related_concepts(self, concept_id: str,
                              max_results: int = 10) -> list[tuple[str, float]]:
        """Find concepts related to a given concept, ranked by association weight."""
        if concept_id not in self._concepts:
            return []
        rel = self._concepts[concept_id].related_concepts
        sorted_rel = sorted(rel.items(), key=lambda x: x[1], reverse=True)
        return sorted_rel[:max_results]

    def get_by_label(self, label: str) -> ConceptNode | None:
        """Find concept by label (case-insensitive)."""
        label_lower = label.strip().lower()
        for c in self._concepts.values():
            if c.label.lower() == label_lower:
                return c
        return None

    # ── Maintenance ────────────────────────────────────────

    def prune_weak_concepts(self, min_evidence: int | None = None) -> list[str]:
        """Remove concepts with insufficient evidential support."""
        min_ev = min_evidence if min_evidence is not None else self._min_evidence
        removed: list[str] = []

        for cid, concept in list(self._concepts.items()):
            if (concept.evidence_count < min_ev and
                    concept.emergence_source != "manual" and
                    concept.stability < 0.3):
                # Re-link anchors to parent if possible
                for parent_id in concept.parent_concepts:
                    if parent_id in self._concepts:
                        parent = self._concepts[parent_id]
                        for anchor_id in concept.linked_anchors:
                            if anchor_id not in parent.linked_anchors:
                                parent.linked_anchors.append(anchor_id)
                removed.append(cid)
                del self._concepts[cid]

        return removed

    def _prune_weakest(self):
        """Remove the weakest concept to make room."""
        if not self._concepts:
            return
        # Score: evidence + stability + activation (lower = weaker)
        def weakness(c: ConceptNode) -> float:
            return c.evidence_count * 0.3 + c.stability * 0.4 + c.activation * 0.3

        weakest = min(self._concepts.values(), key=weakness)
        del self._concepts[weakest.id]

    # ── Seeding ────────────────────────────────────────────

    def _seed_core_concepts(self):
        """Insert built-in core concept seeds."""
        for label, desc in CORE_CONCEPT_SEEDS.items():
            existing = self.get_by_label(label)
            if not existing:
                concept = ConceptNode(
                    id=str(uuid.uuid4()),
                    label=label,
                    description=desc,
                    stability=0.7,
                    emergence_source="manual",
                    baseline_activation=self._baseline,
                )
                self._concepts[concept.id] = concept

    # ── Stats ──────────────────────────────────────────────

    @property
    def concept_count(self) -> int:
        return len(self._concepts)

    @property
    def active_count(self) -> int:
        return sum(1 for c in self._concepts.values() if c.is_active)

    def get_stats(self) -> dict:
        return {
            "total_concepts": self.concept_count,
            "active_concepts": self.active_count,
            "avg_stability": round(
                sum(c.stability for c in self._concepts.values()) / max(self.concept_count, 1), 3
            ),
            "avg_activation": round(
                sum(c.activation for c in self._concepts.values()) / max(self.concept_count, 1), 3
            ),
            "total_evidence": sum(c.evidence_count for c in self._concepts.values()),
        }
