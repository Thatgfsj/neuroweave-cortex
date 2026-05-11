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

from .anchor import Anchor, AnchorVector, AnchorPrediction, Oscillator, GhostAnchor
from .graph import StarGraph, Edge, Schema
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
                }
                for e in graph.edges.values()
            ],
            "ghosts": [
                {
                    "id": g.id,
                    "residue": g.residue,
                    "original_tags": g.original_tags,
                    "pruned_at": g.pruned_at,
                    "revival_count": g.revival_count,
                    "original_importance": g.original_importance,
                }
                for g in graph.ghosts.values()
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
            )
            graph.add_anchor(anchor)

        for e_data in data.get("edges", []):
            edge = Edge(
                source=e_data["source"],
                target=e_data["target"],
                weight=e_data["weight"],
                edge_type=e_data.get("edge_type", "topical"),
                co_activation_count=e_data.get("co_activation_count", 0),
                created_at=e_data.get("created_at", time.time()),
                last_activated_at=e_data.get("last_activated_at", time.time()),
            )
            key = graph._key(edge.source, edge.target)
            graph.edges[key] = edge
            graph._adjacency[edge.source].add(edge.target)
            graph._adjacency[edge.target].add(edge.source)

        for g_data in data.get("ghosts", []):
            ghost = GhostAnchor(
                id=g_data["id"],
                residue=g_data.get("residue", []),
                original_tags=g_data.get("original_tags", []),
                pruned_at=g_data.get("pruned_at", time.time()),
                revival_count=g_data.get("revival_count", 0),
                original_importance=g_data.get("original_importance", 0.5),
            )
            graph.ghosts[ghost.id] = ghost

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
