"""SQLite storage backend — indexed persistence for 10K+ anchor graphs.

Schema:
  anchors: id, text, vector_json, embedding_blob, prediction_json, oscillator_json,
           created_at, last_activated_at, source_session, tags_json, schema_ref,
           replay_count, state, state_history_json
  edges: source, target, weight, edge_type, co_activation_count, created_at, last_activated_at
  ghosts: id, residue_blob, original_tags_json, pruned_at, revival_count, original_importance
  schemas: id, template, slots_json, instance_ids_json, confidence, created_at, tags_json
  meta: key, value (for version, saved_at, etc.)

Indexes on: edges(source), edges(target), ghosts(pruned_at), schemas(confidence)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .anchor import (
    Anchor, AnchorVector, AnchorPrediction, Oscillator, MemoryState,
)
from .graph import StarGraph, Edge, Schema
from .storage_backend import StorageBackend


DEFAULT_SQLITE_PATH = Path.home() / ".star_graph" / "memory.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS anchors (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector_json TEXT NOT NULL DEFAULT '[]',
    embedding_blob BLOB,
    prediction_json TEXT,
    oscillator_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    last_activated_at REAL NOT NULL,
    source_session TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    schema_ref TEXT,
    replay_count INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'active',
    state_history_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    edge_type TEXT NOT NULL DEFAULT 'topical',
    co_activation_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_activated_at REAL NOT NULL,
    PRIMARY KEY (source, target)
);

CREATE TABLE IF NOT EXISTS schemas (
    id TEXT PRIMARY KEY,
    template TEXT NOT NULL,
    slots_json TEXT NOT NULL DEFAULT '{}',
    instance_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_schemas_confidence ON schemas(confidence DESC);
"""


def _embedding_to_blob(emb: list[float] | None) -> bytes | None:
    if emb is None:
        return None
    import struct
    return struct.pack(f'{len(emb)}f', *emb)


def _blob_to_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    import struct
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


class SQLiteStorage(StorageBackend):
    """SQLite-backed storage with indexed queries for larger graphs."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_SQLITE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA_SQL)
        return self._conn

    # ── Full save/load ──────────────────────────────────

    def save(self, graph: StarGraph) -> None:
        c = self.conn
        with c:
            c.execute("DELETE FROM anchors")
            c.execute("DELETE FROM edges")
            c.execute("DELETE FROM ghosts")
            c.execute("DELETE FROM schemas")

            for a in graph.anchors.values():
                c.execute(
                    """INSERT INTO anchors (id, text, vector_json, embedding_blob,
                       prediction_json, oscillator_json, created_at, last_activated_at,
                       source_session, tags_json, schema_ref, replay_count, state,
                       state_history_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        a.id, a.text,
                        json.dumps(a.vector.to_list()),
                        _embedding_to_blob(a.embedding),
                        json.dumps({"emotional_tone": a.prediction.emotional_tone,
                                     "expected_duration": a.prediction.expected_duration,
                                     "confidence": a.prediction.confidence}) if a.prediction else None,
                        json.dumps({"natural_frequency": a.oscillator.natural_frequency,
                                     "phase_offset": a.oscillator.phase_offset,
                                     "coupling_strength": a.oscillator.coupling_strength,
                                     "damping": a.oscillator.damping}),
                        a.created_at, a.last_activated_at,
                        a.source_session, json.dumps(a.tags),
                        a.schema_ref, a.replay_count,
                        a.state.value if isinstance(a.state, MemoryState) else str(a.state),
                        json.dumps([(s.value if isinstance(s, MemoryState) else str(s), ts)
                                     for s, ts in a.state_history]),
                    ),
                )

            for e in graph.edges.values():
                c.execute(
                    """INSERT INTO edges (source, target, weight, edge_type,
                       co_activation_count, created_at, last_activated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (e.source, e.target, e.weight, e.edge_type,
                     e.co_activation_count, e.created_at, e.last_activated_at),
                )

            for s in graph.schemas.values():
                c.execute(
                    """INSERT INTO schemas (id, template, slots_json, instance_ids_json,
                       confidence, created_at, tags_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (s.id, s.template, json.dumps(s.slots),
                     json.dumps(s.instance_ids), s.confidence,
                     s.created_at, json.dumps(s.tags)),
                )

            c.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("version", "2"),
            )
            c.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("saved_at", str(time.time())),
            )

    def load(self) -> StarGraph:
        graph = StarGraph()
        c = self.conn

        # Check if tables exist
        tables = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='anchors'"
        ).fetchall()
        if not tables:
            return graph

        for row in c.execute("SELECT * FROM anchors"):
            (
                aid, text, vector_json, embedding_blob, prediction_json,
                oscillator_json, created_at, last_activated_at, source_session,
                tags_json, schema_ref, replay_count, state_str, state_history_json,
            ) = row

            osc = json.loads(oscillator_json)
            oscillator = Oscillator(
                natural_frequency=osc.get("natural_frequency", 0.5),
                phase_offset=osc.get("phase_offset", 0.0),
                coupling_strength=osc.get("coupling_strength", 0.3),
                damping=osc.get("damping", 0.1),
            )

            pred = None
            if prediction_json:
                p = json.loads(prediction_json)
                pred = AnchorPrediction(
                    emotional_tone=p.get("emotional_tone", 0.0),
                    expected_duration=p.get("expected_duration", 10.0),
                    confidence=p.get("confidence", 0.5),
                )

            try:
                state = MemoryState(state_str)
            except ValueError:
                state = MemoryState.ACTIVE

            state_history = []
            if state_history_json:
                sh = json.loads(state_history_json)
                for s_val, ts in sh:
                    try:
                        s_state = MemoryState(s_val)
                    except ValueError:
                        s_state = MemoryState.ACTIVE
                    state_history.append((s_state, ts))

            anchor = Anchor(
                id=aid, text=text,
                vector=AnchorVector.from_list(json.loads(vector_json)),
                embedding=_blob_to_embedding(embedding_blob),
                prediction=pred, oscillator=oscillator,
                created_at=created_at, last_activated_at=last_activated_at,
                source_session=source_session, tags=json.loads(tags_json),
                schema_ref=schema_ref, replay_count=replay_count,
                state=state, state_history=state_history,
            )
            graph.add_anchor(anchor)

        for row in c.execute("SELECT * FROM edges"):
            src, tgt, weight, etype, coact, cat, lat = row
            edge = Edge(
                source=src, target=tgt, weight=weight, edge_type=etype,
                co_activation_count=coact, created_at=cat, last_activated_at=lat,
            )
            key = graph._key(src, tgt)
            graph.edges[key] = edge
            graph._adjacency[src].add(tgt)
            graph._adjacency[tgt].add(src)

        for row in c.execute("SELECT * FROM schemas"):
            sid, template, slots_json, inst_json, conf, cat, tags_json = row
            schema = Schema(
                id=sid, template=template,
                slots=json.loads(slots_json),
                instance_ids=json.loads(inst_json),
                confidence=conf, created_at=cat,
                tags=json.loads(tags_json),
            )
            graph.schemas[sid] = schema

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
        if not self.path.exists():
            return False
        try:
            c = self.conn
            tables = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='anchors'"
            ).fetchall()
            return len(tables) > 0
        except Exception:
            return False

    # ── Fine-grained ops ────────────────────────────────

    def save_anchor(self, anchor_id: str, data: dict) -> None:
        c = self.conn
        with c:
            c.execute(
                """INSERT OR REPLACE INTO anchors
                   (id, text, vector_json, embedding_blob, created_at,
                    last_activated_at, source_session, tags_json, oscillator_json,
                    replay_count, state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (anchor_id, data.get("text", ""),
                 json.dumps(data.get("vector", [])),
                 _embedding_to_blob(data.get("embedding")),
                 data.get("created_at", time.time()),
                 data.get("last_activated_at", time.time()),
                 data.get("source_session", ""),
                 json.dumps(data.get("tags", [])),
                 json.dumps(data.get("oscillator", {})),
                 data.get("replay_count", 0),
                 str(data.get("state", "active"))),
            )

    def delete_anchor(self, anchor_id: str) -> None:
        c = self.conn
        with c:
            c.execute("DELETE FROM anchors WHERE id = ?", (anchor_id,))
            c.execute("DELETE FROM edges WHERE source = ? OR target = ?",
                      (anchor_id, anchor_id))

    def save_edge(self, source: str, target: str, data: dict) -> None:
        c = self.conn
        key = (source, target) if source < target else (target, source)
        with c:
            c.execute(
                """INSERT OR REPLACE INTO edges
                   (source, target, weight, edge_type, co_activation_count,
                    created_at, last_activated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (key[0], key[1], data.get("weight", 0.5),
                 data.get("edge_type", "topical"),
                 data.get("co_activation_count", 0),
                 data.get("created_at", time.time()),
                 data.get("last_activated_at", time.time())),
            )

    def delete_edge(self, source: str, target: str) -> None:
        key = (source, target) if source < target else (target, source)
        with self.conn:
            self.conn.execute(
                "DELETE FROM edges WHERE source = ? AND target = ?",
                (key[0], key[1]),
            )

    def save_ghost(self, ghost_id: str, data: dict) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO ghosts
                   (id, residue_blob, original_tags_json, pruned_at,
                    revival_count, original_importance)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ghost_id, _embedding_to_blob(data.get("residue")),
                 json.dumps(data.get("original_tags", [])),
                 data.get("pruned_at", time.time()),
                 data.get("revival_count", 0),
                 data.get("original_importance", 0.5)),
            )

    def delete_ghost(self, ghost_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM ghosts WHERE id = ?", (ghost_id,))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
