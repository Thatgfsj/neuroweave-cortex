"""Spreading Activation Engine — Phase 6 multi-source activation with path formation.

Upgrades the existing SpreadingActivation (spreading.py) with:
- Multi-seed activation: concepts, goals, emotions, query terms all as seeds
- Edge weights modulated by Hebbian learning results
- Semantic path formation (proto thought-chains before reasoning)
- Lateral inhibition between competing paths
- Concept-level integration with ConceptCortex

From "retrieval" to "association" — this is how the cortex thinks, not just searches.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .thought_object import ThoughtObject


# ── Activation Seed ─────────────────────────────────────────────


@dataclass
class ActivationSeed:
    """A seed for spreading activation — can be a concept, goal, emotion, or query term."""
    id: str
    seed_type: str = "query"         # query | concept | goal | emotion | memory | perception
    content: str = ""
    embedding: list[float] | None = None
    energy: float = 0.5              # initial activation energy
    priority: float = 0.5


# ── Activation Token ────────────────────────────────────────────


@dataclass
class ActivationToken:
    """A single activation pulse traveling through the graph."""
    id: str
    source_id: str                   # where this activation originated
    target_id: str                   # which node receives activation
    energy: float                    # activation energy (0..1, decayed by distance)
    path: list[str] = field(default_factory=list)  # node IDs traversed so far
    depth: int = 0                   # hops from source
    edge_types_used: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ── Activated Node (extended) ───────────────────────────────────


@dataclass
class ActivatedNode:
    """A node that received activation during spreading."""
    anchor_id: str
    accumulated_activation: float = 0.0
    activation_depth: int = 0
    source_seeds: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    content: str = ""
    tags: list[str] = field(default_factory=list)
    anchor: Any = None               # reference to original Anchor object

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "activation": round(self.accumulated_activation, 3),
            "depth": self.activation_depth,
            "path": self.path,
            "content": self.content[:200],
            "tags": self.tags,
        }


# ── Semantic Path ────────────────────────────────────────────────


@dataclass
class SemanticPath:
    """A formed semantic path — proto thought-chain before reasoning."""
    id: str
    seed_ids: list[str]              # initial activation seeds
    node_ids: list[str]              # activated nodes in traversal order
    total_activation: float = 0.0    # sum of activation along path
    path_confidence: float = 0.0     # 0..1 how coherent this path is
    dominant_concept: str = ""       # which concept best describes this path
    edge_types: list[str] = field(default_factory=list)
    depth: int = 0
    created_at: float = field(default_factory=time.time)


# ── Activation Result ────────────────────────────────────────────


@dataclass
class ActivationResult:
    """Result of a spreading activation query."""
    activated_nodes: list[ActivatedNode] = field(default_factory=list)
    semantic_paths: list[SemanticPath] = field(default_factory=list)
    concept_activations: dict[str, float] = field(default_factory=dict)
    total_energy_consumed: float = 0.0
    computation_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "nodes_count": len(self.activated_nodes),
            "paths_count": len(self.semantic_paths),
            "concepts_activated": len(self.concept_activations),
            "total_energy": round(self.total_energy_consumed, 3),
            "time_ms": round(self.computation_time_ms, 1),
            "top_nodes": [n.to_dict() for n in self.activated_nodes[:5]],
            "top_paths": [
                {"node_ids": p.node_ids, "confidence": round(p.path_confidence, 3),
                 "concept": p.dominant_concept}
                for p in self.semantic_paths[:3]
            ],
        }


# ── Activation Engine ────────────────────────────────────────────


class ActivationEngine:
    """Multi-source spreading activation with path formation.

    Upgrades SpreadingActivation with:
    - Multi-seed activation (goals, concepts, emotions, memories, query terms)
    - Edge weights modulated by Hebbian learning results
    - Path formation (chains of activated nodes)
    - Concept-level integration with ConceptCortex
    - Lateral inhibition between competing paths
    """

    # Default edge type weights (from plan.yml)
    _DEFAULT_EDGE_TYPE_WEIGHTS = {
        "causal": 1.5,
        "semantic_similarity": 1.2,
        "temporal_cooccurrence": 1.0,
        "contradiction": 0.5,
        "hierarchy": 0.8,
        "association": 0.7,
    }

    def __init__(self, graph=None, *,
                 config: dict[str, Any] | None = None,
                 hebbian_learner=None,
                 concept_cortex=None,
                 embedder=None):
        self.graph = graph
        self._config = config or {}
        self._hebbian_learner = hebbian_learner
        self._concept_cortex = concept_cortex
        self._embedder = embedder

        # Parameters from config
        self._max_depth = self._config.get("default_max_depth", 3)
        self._max_nodes = self._config.get("default_max_nodes", 50)
        self._default_decay = self._config.get("default_decay", 0.6)
        self._min_energy = self._config.get("min_activation_energy", 0.05)
        self._max_paths = self._config.get("max_paths", 5)
        self._min_path_length = self._config.get("min_path_length", 2)
        self._path_confidence_threshold = self._config.get("path_confidence_threshold", 0.3)
        self._inhibition_radius = self._config.get("lateral_inhibition_radius", 0.7)
        self._inhibition_strength = self._config.get("lateral_inhibition_strength", 0.3)

        # Seed type → default energy
        self._seed_energies = {
            "concept": self._config.get("concept_seed_energy", 0.7),
            "goal": self._config.get("goal_seed_energy", 0.8),
            "emotion": self._config.get("emotion_seed_energy", 0.5),
            "query": self._config.get("query_seed_energy", 0.9),
            "memory": self._config.get("query_seed_energy", 0.7),
            "perception": self._config.get("query_seed_energy", 0.7),
        }

        # Edge type weights (config overrides defaults)
        cfg_edge_weights = self._config.get("edge_type_weights", {})
        self._edge_type_weights = {**self._DEFAULT_EDGE_TYPE_WEIGHTS, **cfg_edge_weights}

    # ── Main Activation ────────────────────────────────────

    def activate(self, seeds: list[ActivationSeed], *,
                 max_depth: int | None = None,
                 max_nodes: int | None = None,
                 decay: float | None = None,
                 min_energy: float | None = None) -> ActivationResult:
        """Multi-seed spreading activation.

        Returns activated nodes and formed semantic paths.
        """
        t0 = time.perf_counter()
        max_depth = max_depth or self._max_depth
        max_nodes = max_nodes or self._max_nodes
        decay = decay or self._default_decay
        min_energy = min_energy or self._min_energy

        # Phase 1: Seed → graph activation via BFS
        activated: dict[str, ActivatedNode] = {}

        for seed in seeds:
            self._spread_from_seed(seed, activated, max_depth, decay, min_energy)

        # Sort by activation descending
        sorted_nodes = sorted(activated.values(),
                              key=lambda n: n.accumulated_activation, reverse=True)

        # Enforce max nodes
        if len(sorted_nodes) > max_nodes:
            sorted_nodes = sorted_nodes[:max_nodes]

        # Phase 2: Lateral inhibition
        sorted_nodes = self._apply_lateral_inhibition(sorted_nodes)

        # Phase 3: Form semantic paths
        paths = self._form_paths(sorted_nodes, max_paths=self._max_paths,
                                 min_path_length=self._min_path_length)

        # Compute concept activations
        concept_acts: dict[str, float] = {}
        if self._concept_cortex:
            # ConceptCortex handles its own activation internally
            pass
        for node in sorted_nodes[:10]:
            for tag in node.tags:
                concept_acts[tag] = concept_acts.get(tag, 0.0) + node.accumulated_activation * 0.1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_energy = sum(n.accumulated_activation for n in sorted_nodes)

        return ActivationResult(
            activated_nodes=sorted_nodes,
            semantic_paths=paths,
            concept_activations=concept_acts,
            total_energy_consumed=total_energy,
            computation_time_ms=elapsed_ms,
        )

    def _spread_from_seed(self, seed: ActivationSeed,
                          activated: dict[str, ActivatedNode],
                          max_depth: int, decay: float, min_energy: float):
        """BFS spreading from a single seed through the graph."""
        if not self.graph:
            # No graph — simulate with just the seed itself
            self._add_activation(activated, seed.id, seed.content, seed.energy,
                                 0, [seed.id], [], seed.id, None)
            return

        # BFS queue
        queue: list[tuple[str, float, int, list[str], list[str], Any]] = [
            (seed.id, seed.energy, 0, [seed.id], [], None)
        ]
        visited: set[str] = {seed.id}

        while queue:
            node_id, energy, depth, path, edge_types, anchor_obj = queue.pop(0)

            if energy < min_energy or depth > max_depth:
                continue

            self._add_activation(activated, node_id,
                                 getattr(anchor_obj, 'text', '') if anchor_obj else '',
                                 energy, depth, path, edge_types, seed.id, anchor_obj)

            if depth >= max_depth:
                continue

            # Get neighbors from graph
            neighbors = self._get_neighbors(node_id)
            for neighbor_id, edge_weight, edge_type in neighbors:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                # Effective weight = base weight × edge_type_modifier × hebbian_modifier
                type_mult = self._edge_type_weights.get(edge_type, 1.0)
                hebbian_mult = self._get_hebbian_modifier(node_id, neighbor_id)
                effective_weight = edge_weight * type_mult * hebbian_mult

                new_energy = energy * effective_weight * (decay ** (depth + 1))
                if new_energy >= min_energy:
                    new_path = list(path) + [neighbor_id]
                    new_edge_types = list(edge_types) + [edge_type]
                    # Get anchor reference if available
                    neighbor_anchor = self._get_anchor(neighbor_id)
                    queue.append((neighbor_id, new_energy, depth + 1,
                                  new_path, new_edge_types, neighbor_anchor))

    @staticmethod
    def _add_activation(activated: dict[str, ActivatedNode], node_id: str,
                        content: str, energy: float, depth: int,
                        path: list[str], edge_types: list[str],
                        seed_id: str, anchor_obj: Any = None):
        """Add or accumulate activation to a node."""
        if node_id in activated:
            activated[node_id].accumulated_activation += energy
            if seed_id not in activated[node_id].source_seeds:
                activated[node_id].source_seeds.append(seed_id)
        else:
            tags = list(getattr(anchor_obj, 'tags', [])) if anchor_obj else []
            activated[node_id] = ActivatedNode(
                anchor_id=node_id,
                accumulated_activation=energy,
                activation_depth=depth,
                source_seeds=[seed_id],
                path=list(path),
                content=content,
                tags=tags,
                anchor=anchor_obj,
            )

    def _get_neighbors(self, node_id: str) -> list[tuple[str, float, str]]:
        """Get neighbors from graph. Returns [(neighbor_id, weight, edge_type), ...]."""
        if not self.graph:
            return []
        try:
            neighbors = []
            if hasattr(self.graph, 'get_neighbors'):
                raw = self.graph.get_neighbors(node_id)
                for entry in raw:
                    if isinstance(entry, tuple):
                        if len(entry) == 3:
                            neighbors.append(entry)
                        elif len(entry) == 2:
                            neighbors.append((entry[0], entry[1], "association"))
                    elif hasattr(entry, 'target_id'):
                        neighbors.append((
                            entry.target_id,
                            getattr(entry, 'weight', 0.5),
                            getattr(entry, 'edge_type', 'association'),
                        ))
            return neighbors
        except Exception:
            return []

    def _get_anchor(self, node_id: str) -> Any:
        """Get anchor object by ID from graph."""
        if not self.graph:
            return None
        try:
            if hasattr(self.graph, 'get_anchor'):
                return self.graph.get_anchor(node_id)
        except Exception:
            pass
        return None

    def _get_hebbian_modifier(self, source_id: str, target_id: str) -> float:
        """Get Hebbian learning modifier for an edge (default 1.0 = no modification)."""
        if not self._hebbian_learner:
            return 1.0
        try:
            return self._hebbian_learner.get_edge_strength(source_id, target_id)
        except Exception:
            return 1.0

    # ── Lateral Inhibition ─────────────────────────────────

    def _apply_lateral_inhibition(self, nodes: list[ActivatedNode]) -> list[ActivatedNode]:
        """Suppress activation of similar nodes — winner-take-all dynamics.

        Nodes with similar tags/content compete. The weaker node gets suppressed.
        """
        if len(nodes) < 2:
            return nodes

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                similarity = self._node_similarity(a, b)

                if similarity > self._inhibition_radius:
                    # Weaker node gets suppressed
                    if a.accumulated_activation >= b.accumulated_activation:
                        suppression = self._inhibition_strength * similarity
                        b.accumulated_activation *= (1.0 - suppression)
                    else:
                        suppression = self._inhibition_strength * similarity
                        a.accumulated_activation *= (1.0 - suppression)

        # Re-sort after inhibition
        nodes.sort(key=lambda n: n.accumulated_activation, reverse=True)
        return nodes

    @staticmethod
    def _node_similarity(a: ActivatedNode, b: ActivatedNode) -> float:
        """Compute similarity between two activated nodes."""
        score = 0.0
        # Tag overlap
        a_tags = set(a.tags)
        b_tags = set(b.tags)
        if a_tags and b_tags:
            overlap = len(a_tags & b_tags) / max(len(a_tags | b_tags), 1)
            score += overlap * 0.5
        # Path overlap
        a_path = set(a.path)
        b_path = set(b.path)
        if a_path and b_path:
            overlap = len(a_path & b_path) / max(len(a_path | b_path), 1)
            score += overlap * 0.5
        return score

    # ── Path Formation ─────────────────────────────────────

    def _form_paths(self, nodes: list[ActivatedNode], *,
                    max_paths: int = 5,
                    min_path_length: int = 2) -> list[SemanticPath]:
        """Form semantic paths from activated nodes using graph connectivity."""
        paths: list[SemanticPath] = []

        # Group nodes by their paths — nodes sharing path prefixes form a semantic path
        path_groups: dict[tuple, list[ActivatedNode]] = defaultdict(list)
        for node in nodes:
            if len(node.path) >= min_path_length:
                # Use the first node in the path as the grouping key (seed)
                key = tuple(node.path[:2]) if len(node.path) >= 2 else (node.path[0],)
                path_groups[key].append(node)

        for key, group in path_groups.items():
            if len(group) < min_path_length:
                continue
            if len(paths) >= max_paths:
                break

            # Sort group by activation
            group.sort(key=lambda n: n.accumulated_activation, reverse=True)

            # Build the path
            all_node_ids: list[str] = []
            seen: set[str] = set()
            edge_types: list[str] = []
            total_activation = 0.0
            tags_counter: dict[str, float] = defaultdict(float)

            for node in group[:10]:
                for nid in node.path:
                    if nid not in seen:
                        all_node_ids.append(nid)
                        seen.add(nid)
                total_activation += node.accumulated_activation
                for tag in node.tags:
                    tags_counter[tag] += node.accumulated_activation

            # Path confidence = mean node activation × connectivity
            mean_activation = total_activation / max(len(group), 1)
            connectivity = len(group) / max(len(nodes), 1)
            confidence = mean_activation * 0.6 + connectivity * 0.4

            if confidence < self._path_confidence_threshold:
                continue

            # Dominant concept = most activated tag
            dominant = max(tags_counter, key=tags_counter.get) if tags_counter else ""

            paths.append(SemanticPath(
                id=str(uuid.uuid4()),
                seed_ids=list(key),
                node_ids=all_node_ids,
                total_activation=total_activation,
                path_confidence=confidence,
                dominant_concept=dominant,
                edge_types=edge_types,
                depth=len(all_node_ids),
            ))

        # Sort paths by confidence
        paths.sort(key=lambda p: p.path_confidence, reverse=True)
        return paths[:max_paths]

    # ── Convenience Activators ─────────────────────────────

    def activate_from_query(self, query: str, *,
                            embedding: list[float] | None = None) -> ActivationResult:
        """Activate from a natural language query."""
        seed = ActivationSeed(
            id=str(uuid.uuid4()),
            seed_type="query",
            content=query,
            embedding=embedding,
            energy=self._seed_energies.get("query", 0.9),
        )
        return self.activate([seed])

    def activate_from_concepts(self, concept_ids: list[str],
                               energy: float | None = None) -> ActivationResult:
        """Activate from concept IDs (from ConceptCortex)."""
        en = energy or self._seed_energies.get("concept", 0.7)
        seeds = [
            ActivationSeed(id=cid, seed_type="concept", energy=en)
            for cid in concept_ids
        ]
        return self.activate(seeds)

    def activate_from_goals(self, goal_ids: list[str],
                            energy: float | None = None) -> ActivationResult:
        """Activate from goal IDs (from GoalSystem)."""
        en = energy or self._seed_energies.get("goal", 0.8)
        seeds = [
            ActivationSeed(id=gid, seed_type="goal", energy=en)
            for gid in goal_ids
        ]
        return self.activate(seeds)

    def activate_from_perception(self, frame) -> ActivationResult:
        """Activate from a PerceptionFrame (from 6.1)."""
        seeds: list[ActivationSeed] = []

        for concept in getattr(frame, 'extracted_concepts', []):
            seeds.append(ActivationSeed(
                id=str(uuid.uuid4()),
                seed_type="concept",
                content=concept,
                energy=self._seed_energies.get("concept", 0.7),
            ))

        for goal in getattr(frame, 'explicit_goals', []):
            seeds.append(ActivationSeed(
                id=str(uuid.uuid4()),
                seed_type="goal",
                content=goal,
                energy=self._seed_energies.get("goal", 0.8),
            ))

        if getattr(frame, 'emotional_arousal', 0) > 0.5:
            seeds.append(ActivationSeed(
                id=str(uuid.uuid4()),
                seed_type="emotion",
                content=f"emotional:{frame.emotional_valence}",
                energy=self._seed_energies.get("emotion", 0.5),
            ))

        if not seeds:
            seeds.append(ActivationSeed(
                id=str(uuid.uuid4()),
                seed_type="query",
                content=getattr(frame, 'raw_text', ''),
                energy=self._seed_energies.get("query", 0.9),
            ))

        return self.activate(seeds)

    # ── Configuration ──────────────────────────────────────

    def configure(self, *, max_depth: int | None = None,
                  max_nodes: int | None = None,
                  decay: float | None = None,
                  min_energy: float | None = None):
        """Override default hyperparameters."""
        if max_depth is not None:
            self._max_depth = max_depth
        if max_nodes is not None:
            self._max_nodes = max_nodes
        if decay is not None:
            self._default_decay = decay
        if min_energy is not None:
            self._min_energy = min_energy
