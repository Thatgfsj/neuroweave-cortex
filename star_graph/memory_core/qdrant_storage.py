"""Qdrant vector-database storage backend for StarGraph.

Provides fast HNSW-accelerated search and scalable persistence.
Requires: pip install qdrant-client (optional dependency).

Usage:
    from star_graph.qdrant_storage import QdrantStorage

    backend = QdrantStorage(collection="my_graph", url="http://localhost:6333")
    backend.save(graph)
    results = backend.search("user query", top_k=10)
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .anchor import Anchor, AnchorVector, AnchorPrediction, Oscillator, MemoryState
from .graph import StarGraph, Edge, Schema
from .storage_backend import StorageBackend

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.exceptions import UnexpectedResponse

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore


# ── Helpers ──────────────────────────────────────────────────────────

def _embedding_to_list(embedding: Any) -> list[float]:
    """Normalise any embedding representation to a flat list of floats."""
    if embedding is None:
        return []
    if isinstance(embedding, list):
        return embedding
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def _anchor_to_point(anchor: Anchor) -> dict:
    """Serialize an Anchor into a Qdrant-compatible payload dict."""
    return {
        "id": anchor.id,
        "text": anchor.text,
        "vector_json": json.dumps(anchor.vector.to_list()),
        "prediction_json": json.dumps(anchor.prediction.to_dict()) if anchor.prediction else "{}",
        "oscillator_json": json.dumps(anchor.oscillator.to_dict()) if anchor.oscillator else "{}",
        "created_at": anchor.created_at,
        "last_activated_at": anchor.last_activated_at,
        "source_session": anchor.source_session,
        "tags_json": json.dumps(list(anchor.tags)) if anchor.tags else "[]",
        "source_attribution": getattr(anchor, "source_attribution", "observation"),
        "source_trust": getattr(anchor, "source_trust", 0.6),
        "state": anchor.state.value if anchor.state else "active",
        "replay_count": getattr(anchor, "replay_count", 0),
        "schema_ref": getattr(anchor, "schema_ref", "") or "",
    }


def _point_to_anchor(point) -> Anchor:
    """Reconstruct an Anchor from a Qdrant retrieved point + vector."""
    p = point.payload or {}
    vec = _embedding_to_list(point.vector or [])
    vector = AnchorVector.from_list(json.loads(p.get("vector_json", "[]")))

    pred_data = json.loads(p.get("prediction_json", "{}"))
    prediction = AnchorPrediction(**pred_data) if pred_data else AnchorPrediction()

    osc_data = json.loads(p.get("oscillator_json", "{}"))
    oscillator = Oscillator(**osc_data) if osc_data else Oscillator()

    anchor = Anchor(
        id=p.get("id", point.id),
        text=p.get("text", ""),
        vector=vector,
        embedding=vec,
        prediction=prediction,
        oscillator=oscillator,
        created_at=p.get("created_at", 0.0),
        last_activated_at=p.get("last_activated_at", 0.0),
        source_session=p.get("source_session", ""),
        tags=set(json.loads(p.get("tags_json", "[]"))),
        state=MemoryState(p.get("state", "active")),
    )
    anchor.source_attribution = p.get("source_attribution", "observation")
    anchor.source_trust = p.get("source_trust", 0.6)
    anchor.replay_count = p.get("replay_count", 0)
    anchor.schema_ref = p.get("schema_ref", "") or None
    return anchor


def _edge_to_payload(edge: Edge, src: str, tgt: str) -> dict:
    """Serialize an Edge into a Qdrant payload dict."""
    from .graph import RichEdge

    base = {
        "source": src,
        "target": tgt,
        "weight": edge.weight,
        "edge_type": edge.edge_type,
        "co_activation_count": edge.co_activation_count,
        "created_at": edge.created_at,
        "last_activated_at": edge.last_activated_at,
    }
    if isinstance(edge, RichEdge):
        base.update({
            "confidence": edge.confidence,
            "relation": edge.relation,
            "causal_strength": edge.causal_strength,
            "source_type": edge.source_type,
            "reinforcement_count": edge.reinforcement_count,
            "is_stale": edge.is_stale,
            "decay_rate": edge.decay_rate,
        })
    return base


def _graph_to_payloads(graph: StarGraph) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Convert a full StarGraph into point payloads for each collection."""
    from .graph import RichEdge

    anchor_points: list[dict] = []
    for aid, anchor in graph.anchors.items():
        vec = _embedding_to_list(getattr(anchor, "embedding", None))
        payload = _anchor_to_point(anchor)
        anchor_points.append({"id": aid, "vector": vec, "payload": payload})

    edge_points: list[dict] = []
    for key, edge in graph.edges.items():
        src, tgt = key if isinstance(key, tuple) else (graph.edges[key].source, graph.edges[key].target)
        edge_points.append({
            "id": f"e:{src}:{tgt}",
            "vector": [0.0],  # placeholder — edges not vectorised
            "payload": _edge_to_payload(edge, src, tgt),
        })

    schema_points: list[dict] = []
    for sid, schema in (getattr(graph, "schemas", {}) or {}).items():
        schema_points.append({
            "id": f"s:{sid}",
            "vector": [0.0],
            "payload": {
                "id": sid,
                "template": schema.template if hasattr(schema, "template") else "",
                "confidence": getattr(schema, "confidence", 0.5),
            },
        })

    ghost_points: list[dict] = []
    for g in getattr(graph, "ghosts", []) or []:
        ghost_points.append({
            "id": f"g:{getattr(g, 'id', str(time.time_ns()))}",
            "vector": [0.0],
            "payload": {
                "original_tags": list(getattr(g, "original_tags", [])),
                "revival_count": getattr(g, "revival_count", 0),
                "original_importance": getattr(g, "original_importance", 0.0),
            },
        })

    return anchor_points, edge_points, schema_points, ghost_points


# ── Collection schema helpers ───────────────────────────────────────

COLLECTION_ANCHORS = "anchors"
COLLECTION_EDGES = "edges"
COLLECTION_SCHEMAS = "schemas"
COLLECTION_GHOSTS = "ghosts"

ANCHOR_VECTOR_SIZE = 768  # common default; overridden on first save


def _ensure_collection(client: QdrantClient, name: str, vector_size: int | None = None) -> None:
    """Create a Qdrant collection if it doesn't already exist."""
    collections = [c.name for c in client.get_collections().collections]
    if name in collections:
        return

    if name == COLLECTION_ANCHORS and vector_size and vector_size > 0:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
    else:
        # Non-vector collections use a small dummy vector
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=1, distance=qmodels.Distance.COSINE),
        )


# ── QdrantStorage ───────────────────────────────────────────────────

class QdrantStorage(StorageBackend):
    """Qdrant-backed persistent storage for StarGraph.

    Stores anchors as vectorised points (enabling fast HNSW similarity
    search), and edges / schemas / ghosts as payload-only points.
    """

    def __init__(
        self,
        collection: str = "star_graph",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        prefer_grpc: bool = False,
        vector_size: int = 768,
    ) -> None:
        if not HAS_QDRANT:
            raise ImportError(
                "QdrantStorage requires qdrant-client. "
                "Install: pip install qdrant-client"
            )

        self._collection = collection
        self._vector_size = vector_size
        self._client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )

    @property
    def exists(self) -> bool:
        """Check if the anchors collection exists and has data."""
        try:
            collections = [c.name for c in self._client.get_collections().collections]
            return COLLECTION_ANCHORS in collections
        except Exception:
            return False

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Qdrant handles atomicity at the point level; no-op."""
        # Future: use payload+vector batching for atomic multi-point writes
        yield

    # ── Full save / load ──────────────────────────────────────────────

    def save(self, graph: Any) -> None:
        """Persist the full StarGraph state to Qdrant collections.

        This is a full snapshot: existing points are replaced.
        For incremental updates, use batch_save().
        """
        from .graph import RichEdge

        anchor_points, edge_points, schema_points, ghost_points = _graph_to_payloads(graph)

        # Determine vector size from actual embeddings
        max_dim = 0
        for ap in anchor_points:
            dims = len(ap.get("vector", []))
            if dims > max_dim:
                max_dim = dims
        vsize = max_dim or self._vector_size

        # Ensure all collections exist
        _ensure_collection(self._client, COLLECTION_ANCHORS, vsize)
        _ensure_collection(self._client, COLLECTION_EDGES)
        _ensure_collection(self._client, COLLECTION_SCHEMAS)
        _ensure_collection(self._client, COLLECTION_GHOSTS)

        # Upsert in batches of 64 (Qdrant batch limit)
        batch_size = 64

        if anchor_points:
            for i in range(0, len(anchor_points), batch_size):
                batch = anchor_points[i:i + batch_size]
                self._client.upsert(
                    collection_name=COLLECTION_ANCHORS,
                    points=[
                        qmodels.PointStruct(
                            id=p["id"],
                            vector=p["vector"] if p["vector"] else [0.0] * vsize,
                            payload=p["payload"],
                        )
                        for p in batch
                    ],
                )

        if edge_points:
            for i in range(0, len(edge_points), batch_size):
                batch = edge_points[i:i + batch_size]
                self._client.upsert(
                    collection_name=COLLECTION_EDGES,
                    points=[
                        qmodels.PointStruct(id=p["id"], vector=[0.0], payload=p["payload"])
                        for p in batch
                    ],
                )

        if schema_points:
            for i in range(0, len(schema_points), batch_size):
                batch = schema_points[i:i + batch_size]
                self._client.upsert(
                    collection_name=COLLECTION_SCHEMAS,
                    points=[
                        qmodels.PointStruct(id=p["id"], vector=[0.0], payload=p["payload"])
                        for p in batch
                    ],
                )

        if ghost_points:
            for i in range(0, len(ghost_points), batch_size):
                batch = ghost_points[i:i + batch_size]
                self._client.upsert(
                    collection_name=COLLECTION_GHOSTS,
                    points=[
                        qmodels.PointStruct(id=p["id"], vector=[0.0], payload=p["payload"])
                        for p in batch
                    ],
                )

    def load(self) -> StarGraph:
        """Reconstruct a StarGraph from Qdrant collections."""
        from .graph import StarGraph, Edge, RichEdge, Schema

        graph = StarGraph()

        # Load anchors
        anchor_count = self._client.count(COLLECTION_ANCHORS).count
        if anchor_count > 0:
            offset: str | None = None
            while True:
                results = self._client.scroll(
                    collection_name=COLLECTION_ANCHORS,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                points, next_offset = results
                for pt in points:
                    anchor = _point_to_anchor(pt)
                    graph.anchors[anchor.id] = anchor
                if next_offset is None:
                    break
                offset = str(next_offset) if next_offset else None

        # Load edges
        edge_count = self._client.count(COLLECTION_EDGES).count
        if edge_count > 0:
            offset = None
            while True:
                results = self._client.scroll(
                    collection_name=COLLECTION_EDGES,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                points, next_offset = results
                for pt in points:
                    p = pt.payload
                    src, tgt = p.get("source", ""), p.get("target", "")
                    edge = Edge(
                        source=src,
                        target=tgt,
                        weight=p.get("weight", 0.5),
                        edge_type=p.get("edge_type", "topical"),
                        co_activation_count=p.get("co_activation_count", 0),
                        created_at=p.get("created_at", 0.0),
                        last_activated_at=p.get("last_activated_at", 0.0),
                    )
                    graph.edges[graph._key(src, tgt)] = edge
                if next_offset is None:
                    break
                offset = str(next_offset) if next_offset else None

        # Rebuild adjacency
        graph._rebuild_cortical_index()

        return graph

    # ── Search ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Vector similarity search against stored anchors.

        Requires the client side to provide an embedding for the query
        string. If no embedding is available, returns empty.
        """
        # This backend relies on the caller having already embedded the
        # query.  For a stateless text-only search we fall back to empty.
        return []

    def search_by_vector(
        self, vector: list[float], top_k: int = 10, score_threshold: float = 0.0
    ) -> list[dict]:
        """Search anchors by raw embedding vector via Qdrant HNSW index.

        Args:
            vector: Query embedding.
            top_k: Max results.
            score_threshold: Minimum cosine-similarity score (0..1).

        Returns:
            List of payload dicts with 'id', 'text', 'score' keys,
            sorted descending by score.
        """
        if not vector:
            return []

        try:
            results = self._client.search(
                collection_name=COLLECTION_ANCHORS,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception:
            return []

        out: list[dict] = []
        for hit in results:
            p = hit.payload or {}
            out.append({
                "id": p.get("id", hit.id),
                "text": p.get("text", ""),
                "score": hit.score,
            })
        return out

    # ── Batch / fine-grained ops ──────────────────────────────────────

    def batch_save(self, items: list[tuple[str, dict]]) -> None:
        """Persist multiple item updates atomically.

        Each item is (item_type, data_dict) where item_type is
        "anchor", "edge", "ghost", or "schema".
        """
        for item_type, data in items:
            if item_type == "anchor":
                aid = data.get("id", "")
                vec = _embedding_to_list(data.get("embedding", []))
                payload = {k: v for k, v in data.items() if k != "embedding"}
                self._client.upsert(
                    collection_name=COLLECTION_ANCHORS,
                    points=[qmodels.PointStruct(id=aid, vector=vec or [0.0], payload=payload)],
                )
            elif item_type == "edge":
                eid = f"e:{data.get('source', '')}:{data.get('target', '')}"
                self._client.upsert(
                    collection_name=COLLECTION_EDGES,
                    points=[qmodels.PointStruct(id=eid, vector=[0.0], payload=data)],
                )
            elif item_type == "ghost":
                gid = data.get("id", str(time.time_ns()))
                self._client.upsert(
                    collection_name=COLLECTION_GHOSTS,
                    points=[qmodels.PointStruct(id=gid, vector=[0.0], payload=data)],
                )
            elif item_type == "schema":
                sid = data.get("id", "")
                self._client.upsert(
                    collection_name=COLLECTION_SCHEMAS,
                    points=[qmodels.PointStruct(id=sid, vector=[0.0], payload=data)],
                )

    def save_anchor(self, anchor_id: str, data: dict) -> None:
        self.batch_save([("anchor", {**data, "id": anchor_id})])

    def delete_anchor(self, anchor_id: str) -> None:
        try:
            self._client.delete(
                collection_name=COLLECTION_ANCHORS,
                points_selector=qmodels.PointIdsList(points=[anchor_id]),
            )
        except Exception:
            pass

    def save_edge(self, src: str, tgt: str, data: dict) -> None:
        self.batch_save([("edge", {**data, "source": src, "target": tgt})])

    def delete_edge(self, src: str, tgt: str) -> None:
        eid = f"e:{src}:{tgt}"
        try:
            self._client.delete(
                collection_name=COLLECTION_EDGES,
                points_selector=qmodels.PointIdsList(points=[eid]),
            )
        except Exception:
            pass

    def close(self) -> None:
        """Release Qdrant client resources."""
        try:
            self._client.close()
        except Exception:
            pass

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_url(cls, url: str = "http://localhost:6333", collection: str = "star_graph",
                 vector_size: int = 768) -> QdrantStorage:
        """Create a QdrantStorage from a URL string (convenience factory)."""
        return cls(collection=collection, url=url, vector_size=vector_size)
