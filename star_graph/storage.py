"""Persistence layer for the star graph.

Uses JSON for portability. For production, swap with SQLite or a vector DB.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .anchor import Anchor, AnchorVector
from .graph import StarGraph, Edge


DEFAULT_PATH = Path.home() / ".star_graph" / "memory.json"


class Storage:
    """JSON file-backed storage for the star graph."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, graph: StarGraph) -> None:
        data = {
            "version": 1,
            "saved_at": time.time(),
            "anchors": [
                {
                    "id": a.id,
                    "text": a.text,
                    "vector": a.vector.to_list(),
                    "embedding": a.embedding,
                    "created_at": a.created_at,
                    "last_activated_at": a.last_activated_at,
                    "source_session": a.source_session,
                    "tags": a.tags,
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
        }
        # Atomic write
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
            anchor = Anchor(
                id=a_data["id"],
                text=a_data["text"],
                vector=AnchorVector.from_list(a_data["vector"]),
                embedding=a_data.get("embedding"),
                created_at=a_data["created_at"],
                last_activated_at=a_data["last_activated_at"],
                source_session=a_data.get("source_session", ""),
                tags=a_data.get("tags", []),
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

        return graph

    @property
    def exists(self) -> bool:
        return self.path.exists()
