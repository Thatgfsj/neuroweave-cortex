"""Hub Abstraction Layer — cross-cortex bridges via summary nodes.

Hubs are NOT super-nodes. They are compressed, stable summary nodes that
bridge different cortices without creating O(n²) cross-connections.

Hierarchy:
    Leaf Hub (topic-level) -> Domain Hub (cortex-level) -> Global Self Hub

A hub:
- Stores a compressed summary, not raw memories
- Has pointers to source anchors (in their respective cortices)
- Is near-immune to decay (stability > 0.9)
- Is the ONLY mechanism for cross-cortex association
- Forms the "4th dimension" connecting 3D star clusters across cortices
- Can have edges to other hubs, forming a cross-domain reasoning network

v0.6: Added HubEdge for hub-to-hub multi-hop reasoning.

Example:
    Python Memory (DevCortex) -> Python Summary Hub
    Budget Discussion (FinanceCortex) -> Budget Summary Hub
    Python Summary Hub + Budget Summary Hub -> Developer Identity Hub
    Developer Identity Hub -> Global Self Hub
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HubNode:
    """A compressed, stable summary node bridging memory clusters.

    Hubs do not store raw memory. They store:
    - A compressed summary of the memories they abstract
    - Pointers (anchor IDs) back to source memories
    - Cross-references to other hubs (forming the hub hierarchy)
    """
    id: str
    text: str                          # compressed summary text
    hub_level: str = "leaf"            # "leaf" | "domain" | "global"
    source_anchor_ids: list[str] = field(default_factory=list)
    source_cortex: str = ""            # which cortex this hub abstracts
    embedding: list[float] | None = None
    importance: float = 0.5
    stability: float = 0.9             # hubs are highly stable
    confidence: float = 0.5
    parent_hub_id: str = ""            # link to higher-level hub
    child_hub_ids: list[str] = field(default_factory=list)  # links to lower-level hubs
    cross_ref_hubs: list[str] = field(default_factory=list) # peer hubs in other cortices
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    update_count: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        """Hubs resist decay. Stability only drops after many contradictory updates."""
        return self.stability >= 0.7

    @property
    def decay_factor(self) -> float:
        """Hubs decay extremely slowly."""
        hours = (time.time() - self.last_updated_at) / 3600
        # 10x slower decay than regular anchors (half-life ~300 days vs 30)
        return math.exp(-hours / (7200 + 1e-8))

    def refresh(self, new_text: str | None = None,
                new_importance: float | None = None):
        """Update hub summary when source anchors change."""
        if new_text:
            self.text = new_text
        if new_importance is not None:
            self.importance = 0.8 * self.importance + 0.2 * new_importance
        self.last_updated_at = time.time()
        self.update_count += 1
        # Hubs become slightly more stable with each update
        self.stability = min(0.99, self.stability + 0.005)

    def add_cross_ref(self, other_hub_id: str):
        """Add a cross-reference to a hub in another cortex."""
        if other_hub_id not in self.cross_ref_hubs:
            self.cross_ref_hubs.append(other_hub_id)

    @classmethod
    def create(cls, text: str, hub_level: str = "leaf",
               source_anchor_ids: list[str] | None = None,
               source_cortex: str = "",
               importance: float = 0.5,
               **kwargs) -> HubNode:
        """Create a hub with a deterministic ID based on text content."""
        hash_input = f"{hub_level}:{text}:{source_cortex}"
        hub_id = hashlib.blake2b(
            hash_input.encode(), digest_size=10).hexdigest()
        return cls(
            id=hub_id,
            text=text,
            hub_level=hub_level,
            source_anchor_ids=source_anchor_ids or [],
            source_cortex=source_cortex,
            importance=importance,
            **kwargs,
        )


@dataclass
class HubEdge:
    """An edge between two hub nodes in the HubSphere.

    Hub edges enable multi-hop reasoning across domains:
    "Python async" (DevCortex hub) -> "performance optimization" (InfraCortex hub)
    """
    id: str
    source_hub_id: str
    target_hub_id: str
    weight: float = 0.5
    edge_type: str = "cross_domain"  # "cross_domain", "causal", "analogical", "temporal"
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    reinforcement_count: int = 0

    def reinforce(self, delta: float = 0.05):
        self.weight = min(1.0, self.weight + delta)
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + delta * 0.5)

    @classmethod
    def create(cls, source_id: str, target_id: str,
               weight: float = 0.5, edge_type: str = "cross_domain") -> HubEdge:
        import hashlib
        eid = hashlib.blake2b(
            f"{source_id}:{target_id}:{edge_type}".encode(), digest_size=8
        ).hexdigest()
        return cls(id=eid, source_hub_id=source_id, target_hub_id=target_id,
                   weight=weight, edge_type=edge_type)


class HubLayer:
    """Manages the hierarchy of hub nodes across cortices.

    Three-level hierarchy:
    - Leaf hubs: compress a single topic cluster (e.g., "Python async patterns")
    - Domain hubs: aggregate leaf hubs within a cortex (e.g., "Developer identity")
    - Global hubs: aggregate across all cortices (e.g., "Global self model")

    Usage:
        layer = HubLayer()
        leaf = layer.create_leaf("Python async patterns", source_anchors, "dev")
        domain = layer.create_domain("Developer identity", [leaf.id], "dev")
        global_hub = layer.create_global("Global self", [domain.id])
    """

    def __init__(self):
        self.hubs: dict[str, HubNode] = {}
        self.edges: dict[str, HubEdge] = {}  # hub-to-hub edges
        # Index: cortex name -> hub IDs in that cortex
        self._cortex_index: dict[str, list[str]] = {}
        # Adjacency for hub-to-hub traversal
        self._hub_adjacency: dict[str, set[str]] = {}

    # ── Hub creation ─────────────────────────────────────

    def create_leaf(self, text: str,
                    source_anchor_ids: list[str],
                    cortex_name: str,
                    importance: float = 0.5,
                    embedding: list[float] | None = None) -> HubNode:
        """Create a leaf-level hub from a topic cluster."""
        hub = HubNode.create(
            text=text,
            hub_level="leaf",
            source_anchor_ids=source_anchor_ids,
            source_cortex=cortex_name,
            importance=importance,
        )
        if embedding:
            hub.embedding = embedding
        self._add(hub, cortex_name)
        return hub

    def create_domain(self, text: str,
                      child_hub_ids: list[str],
                      cortex_name: str,
                      importance: float = 0.5) -> HubNode | None:
        """Create a domain-level hub aggregating leaf hubs.

        Only leaf hubs from the SAME cortex can be aggregated into a domain hub.
        Cross-cortex aggregation happens at the global level.
        """
        # Validate all child hubs are in the same cortex
        for child_id in child_hub_ids:
            child = self.hubs.get(child_id)
            if child is None or child.source_cortex != cortex_name:
                return None

        hub = HubNode.create(
            text=text,
            hub_level="domain",
            source_cortex=cortex_name,
            importance=importance,
        )

        # Link parents
        for child_id in child_hub_ids:
            child = self.hubs[child_id]
            child.parent_hub_id = hub.id
            hub.child_hub_ids.append(child_id)
            # Domain hubs aggregate importance from children
            hub.importance = max(hub.importance, child.importance)

        self._add(hub, cortex_name)
        return hub

    def create_global(self, text: str,
                      domain_hub_ids: list[str],
                      importance: float = 0.5) -> HubNode:
        """Create a global hub aggregating domain hubs from multiple cortices."""
        hub = HubNode.create(
            text=text,
            hub_level="global",
            importance=importance,
        )

        for child_id in domain_hub_ids:
            child = self.hubs.get(child_id)
            if child:
                child.parent_hub_id = hub.id
                hub.child_hub_ids.append(child_id)
                hub.importance = max(hub.importance, child.importance)

        self._add(hub, "__global__")
        return hub

    # ── Cross-cortex bridging ────────────────────────────

    def bridge(self, hub_a_id: str, hub_b_id: str) -> bool:
        """Create a cross-reference between hubs in different cortices.

        This is the ONLY mechanism for cross-cortex association.
        Direct anchor-to-anchor edges across cortices are forbidden.
        """
        hub_a = self.hubs.get(hub_a_id)
        hub_b = self.hubs.get(hub_b_id)
        if not hub_a or not hub_b:
            return False
        if hub_a.source_cortex == hub_b.source_cortex:
            return False  # Same cortex — not a cross-cortex bridge

        hub_a.add_cross_ref(hub_b_id)
        hub_b.add_cross_ref(hub_a_id)
        return True

    # ── Hub-to-hub edges (cross-domain reasoning network) ─

    def add_hub_edge(self, source_id: str, target_id: str,
                     weight: float = 0.5,
                     edge_type: str = "cross_domain") -> HubEdge | None:
        """Create an edge between two hub nodes.

        Hub edges form a cross-domain reasoning network:
        "Python async" hub -> "performance optimization" hub -> "cost reduction" hub
        """
        if source_id not in self.hubs or target_id not in self.hubs:
            return None
        if source_id == target_id:
            return None

        edge = HubEdge.create(source_id, target_id, weight, edge_type)
        self.edges[edge.id] = edge

        # Update adjacency
        if source_id not in self._hub_adjacency:
            self._hub_adjacency[source_id] = set()
        self._hub_adjacency[source_id].add(target_id)

        return edge

    def traverse_hubs(self, start_hub_id: str, max_hops: int = 2,
                      max_results: int = 5) -> list[HubNode]:
        """Multi-hop traversal on the hub-to-hub reasoning network.

        From a starting hub, follow edges up to max_hops to discover
        connected hubs in other domains. This enables cross-domain inference.
        """
        if start_hub_id not in self.hubs:
            return []

        visited: set[str] = {start_hub_id}
        frontier: list[str] = [start_hub_id]
        results: list[HubNode] = []

        for _ in range(max_hops):
            next_frontier: list[str] = []
            for hub_id in frontier:
                for neighbor_id in self._hub_adjacency.get(hub_id, set()):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor = self.hubs.get(neighbor_id)
                        if neighbor:
                            results.append(neighbor)
                            next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        # Sort by importance
        results.sort(key=lambda h: -h.importance)
        return results[:max_results]

    # ── Queries ──────────────────────────────────────────

    def get_hubs_for_cortex(self, cortex_name: str,
                            level: str | None = None) -> list[HubNode]:
        """Get all hubs for a cortex, optionally filtered by level."""
        ids = self._cortex_index.get(cortex_name, [])
        hubs = [self.hubs[hid] for hid in ids if hid in self.hubs]
        if level:
            hubs = [h for h in hubs if h.hub_level == level]
        return sorted(hubs, key=lambda h: -h.importance)

    def get_parent_chain(self, hub_id: str) -> list[HubNode]:
        """Get the chain of parent hubs from leaf -> global."""
        chain: list[HubNode] = []
        current = self.hubs.get(hub_id)
        while current:
            chain.append(current)
            current = self.hubs.get(current.parent_hub_id) if current.parent_hub_id else None
        return chain

    def get_cross_references(self, hub_id: str) -> list[HubNode]:
        """Get all hubs in OTHER cortices that this hub references."""
        hub = self.hubs.get(hub_id)
        if not hub:
            return []
        return [self.hubs[rid] for rid in hub.cross_ref_hubs
                if rid in self.hubs]

    # ── Internal ─────────────────────────────────────────

    def _add(self, hub: HubNode, cortex_name: str):
        self.hubs[hub.id] = hub
        if cortex_name not in self._cortex_index:
            self._cortex_index[cortex_name] = []
        if hub.id not in self._cortex_index[cortex_name]:
            self._cortex_index[cortex_name].append(hub.id)

    # ── Health ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        levels = {"leaf": 0, "domain": 0, "global": 0}
        for hub in self.hubs.values():
            if hub.hub_level in levels:
                levels[hub.hub_level] += 1
        edge_types: dict[str, int] = {}
        for e in self.edges.values():
            edge_types[e.edge_type] = edge_types.get(e.edge_type, 0) + 1
        return {
            "total_hubs": len(self.hubs),
            "by_level": levels,
            "total_edges": len(self.edges),
            "edge_types": edge_types,
            "cortices_indexed": len(self._cortex_index),
            "cross_cortex_bridges": sum(
                len(h.cross_ref_hubs) for h in self.hubs.values()) // 2,
        }
