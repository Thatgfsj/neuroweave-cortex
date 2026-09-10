"""Persistence layer — v0.4 with pluggable backends.

JSONStorage: single-file JSON (fine for <10K anchors, simple deployment).
SQLiteStorage: indexed SQLite (for 10K+ anchors, in star_graph/sqlite_storage.py).

Storage is kept as a backward-compatible alias for JSONStorage.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .anchor import Anchor, AnchorVector, AnchorPrediction, Oscillator, MemoryState
from .graph import StarGraph, Edge, RichEdge, Schema
from .storage_backend import StorageBackend


DEFAULT_PATH = Path.home() / ".star_graph" / "memory.json"


class JSONStorage(StorageBackend):
    """JSON file-backed storage implementing the StorageBackend interface."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, graph: StarGraph) -> None:
        data = {
            "version": 2,
            "saved_at": time.time(),
            "anchors": [
                {
                    "id": a.id,
                    "text": a.text,
                    "vector": a.vector.to_list(),
                    "embedding": a.embedding,
                    "prediction": {
                        "emotional_tone": a.prediction.emotional_tone,
                        "expected_duration": a.prediction.expected_duration,
                        "confidence": a.prediction.confidence,
                    } if a.prediction else None,
                    "oscillator": {
                        "natural_frequency": a.oscillator.natural_frequency,
                        "phase_offset": a.oscillator.phase_offset,
                        "coupling_strength": a.oscillator.coupling_strength,
                        "damping": a.oscillator.damping,
                    },
                    "created_at": a.created_at,
                    "last_activated_at": a.last_activated_at,
                    "source_session": a.source_session,
                    "tags": a.tags,
                    "schema_ref": a.schema_ref,
                    "replay_count": a.replay_count,
                    "source_attribution": a.source_attribution,
                    "source_trust": a.source_trust,
                    "state": a.state.value if isinstance(a.state, MemoryState) else str(a.state),
                    "activation_level": a.activation_level,
                    "salience": a.salience,
                    "exact_match_keys": a.exact_match_keys,
                    "cortex_path": a.cortex_path,
                    "community_id": a.community_id,
                    "invalid_at": a.invalid_at,
                    "conflict_candidate": a.conflict_candidate,
                }
                for a in graph.anchors.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "edge_type": e.edge_type,
                    "co_activation_count": e.co_activation_count,
                    "created_at": e.created_at,
                    "last_activated_at": e.last_activated_at,
                    "decay_rate": e.decay_rate,
                    **({
                        "relation": e.relation,
                        "confidence": e.confidence,
                        "source_type": e.source_type,
                        "reinforcement_count": e.reinforcement_count,
                        "temporal_order": e.temporal_order,
                        "causal_strength": e.causal_strength,
                        "is_stale": e.is_stale,
                    } if isinstance(e, RichEdge) else {}),
                }
                for e in graph.edges.values()
            ],
            "schemas": [
                {
                    "id": s.id,
                    "template": s.template,
                    "slots": s.slots,
                    "instance_ids": s.instance_ids,
                    "confidence": s.confidence,
                    "created_at": s.created_at,
                    "tags": s.tags,
                }
                for s in graph.schemas.values()
            ],
        }
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def load(self) -> StarGraph:
        graph = StarGraph()
        if not self.path.exists():
            return graph

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for a_data in data.get("anchors", []):
            # Oscillator
            osc_data = a_data.get("oscillator", {})
            oscillator = Oscillator(
                natural_frequency=osc_data.get("natural_frequency", 0.5),
                phase_offset=osc_data.get("phase_offset", 0.0),
                coupling_strength=osc_data.get("coupling_strength", 0.3),
                damping=osc_data.get("damping", 0.1),
            )
            # Prediction
            pred = None
            if a_data.get("prediction"):
                pred = AnchorPrediction(
                    emotional_tone=a_data["prediction"].get("emotional_tone", 0.0),
                    expected_duration=a_data["prediction"].get("expected_duration", 10.0),
                    confidence=a_data["prediction"].get("confidence", 0.5),
                )
            # State (legacy "ghost" maps to DORMANT via MemoryState._missing_)
            try:
                state = MemoryState(a_data.get("state", "active"))
            except ValueError:
                state = MemoryState.ACTIVE
            anchor = Anchor(
                id=a_data["id"],
                text=a_data["text"],
                vector=AnchorVector.from_list(a_data.get("vector", [])),
                embedding=a_data.get("embedding"),
                prediction=pred,
                oscillator=oscillator,
                created_at=a_data.get("created_at", time.time()),
                last_activated_at=a_data.get("last_activated_at", time.time()),
                source_session=a_data.get("source_session", ""),
                tags=a_data.get("tags", []),
                schema_ref=a_data.get("schema_ref"),
                replay_count=a_data.get("replay_count", 0),
                state=state,
                activation_level=a_data.get("activation_level", 1.0),
                salience=a_data.get("salience", 0.0),
                exact_match_keys=a_data.get("exact_match_keys", []),
                cortex_path=a_data.get("cortex_path", ""),
                community_id=a_data.get("community_id", ""),
                invalid_at=a_data.get("invalid_at"),
                conflict_candidate=a_data.get("conflict_candidate", False),
                source_attribution=a_data.get("source_attribution", "observation"),
                source_trust=a_data.get("source_trust", 0.6),
            )
            graph.add_anchor(anchor)

        for e_data in data.get("edges", []):
            rich_markers = ("confidence", "relation", "source_type",
                            "reinforcement_count", "temporal_order",
                            "causal_strength", "is_stale")
            if any(k in e_data for k in rich_markers):
                edge = RichEdge(
                    source=e_data["source"],
                    target=e_data["target"],
                    weight=e_data.get("weight", 0.5),
                    edge_type=e_data.get("edge_type", "topical"),
                    co_activation_count=e_data.get("co_activation_count", 0),
                    created_at=e_data.get("created_at", time.time()),
                    last_activated_at=e_data.get("last_activated_at", time.time()),
                    confidence=e_data.get("confidence", 0.5),
                    relation=e_data.get("relation", ""),
                    source_type=e_data.get("source_type", "implicit"),
                    reinforcement_count=e_data.get("reinforcement_count", 0),
                    temporal_order=e_data.get("temporal_order", ""),
                    causal_strength=e_data.get("causal_strength", 0.0),
                    is_stale=e_data.get("is_stale", False),
                    decay_rate=e_data.get("decay_rate", 0.01),
                )
            else:
                edge = Edge(
                    source=e_data["source"],
                    target=e_data["target"],
                    weight=e_data.get("weight", 0.5),
                    edge_type=e_data.get("edge_type", "topical"),
                    co_activation_count=e_data.get("co_activation_count", 0),
                    created_at=e_data.get("created_at", time.time()),
                    last_activated_at=e_data.get("last_activated_at", time.time()),
                    decay_rate=e_data.get("decay_rate", 0.001),
                )
            key = graph._key(edge.source, edge.target)
            graph.edges[key] = edge
            graph._adjacency[edge.source].add(edge.target)
            graph._adjacency[edge.target].add(edge.source)

        for s_data in data.get("schemas", []):
            schema = Schema(
                id=s_data["id"],
                template=s_data["template"],
                slots=s_data.get("slots", {}),
                instance_ids=s_data.get("instance_ids", []),
                confidence=s_data.get("confidence", 0.0),
                created_at=s_data.get("created_at", time.time()),
                tags=s_data.get("tags", []),
            )
            graph.schemas[schema.id] = schema

        self._rebuild_cortical_index(graph)
        return graph

    def _rebuild_cortical_index(self, graph: StarGraph) -> None:
        graph.cortical_index = [
            (a.embedding, a.id)
            for a in graph.anchors.values()
            if a.embedding and a.is_cortical
        ]

    @property
    def exists(self) -> bool:
        return self.path.exists()


# Backward-compatible alias
Storage = JSONStorage
