"""SQLite storage backend — indexed persistence for 10K+ anchor graphs.

Schema:
  anchors: id, text, vector_json, embedding_blob, prediction_json, oscillator_json,
           created_at, last_activated_at, source_session, tags_json, schema_ref,
           replay_count, state, state_history_json
  edges: source, target, weight, edge_type, co_activation_count, created_at, last_activated_at
  ghosts: id, residue_blob, original_tags_json, pruned_at, revival_count, original_importance,
          semantic_shadow, emotion_trace, residual_edges_json, reactivation_probability,
          partial_recall_count, last_resonated_at
  schemas: id, template, slots_json, instance_ids_json, confidence, created_at, tags_json
  meta: key, value (for version, saved_at, etc.)

Indexes on: edges(source), edges(target), ghosts(pruned_at), schemas(confidence)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

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
    source_attribution TEXT NOT NULL DEFAULT 'observation',
    source_trust REAL NOT NULL DEFAULT 0.6,
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

CREATE TABLE IF NOT EXISTS ghosts (
    id TEXT PRIMARY KEY,
    residue_blob BLOB,
    original_tags_json TEXT NOT NULL DEFAULT '[]',
    pruned_at REAL NOT NULL,
    revival_count INTEGER NOT NULL DEFAULT 0,
    original_importance REAL NOT NULL DEFAULT 0.5,
    semantic_shadow TEXT NOT NULL DEFAULT '',
    emotion_trace REAL NOT NULL DEFAULT 0.0,
    residual_edges_json TEXT NOT NULL DEFAULT '{}',
    reactivation_probability REAL NOT NULL DEFAULT 0.5,
    partial_recall_count INTEGER NOT NULL DEFAULT 0,
    last_resonated_at REAL NOT NULL DEFAULT 0.0
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
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA_SQL)
            # Migrate old databases: add missing columns if needed (idempotent)
            for stmt in (
                "ALTER TABLE anchors ADD COLUMN source_attribution TEXT NOT NULL DEFAULT 'observation'",
                "ALTER TABLE anchors ADD COLUMN source_trust REAL NOT NULL DEFAULT 0.6",
                "ALTER TABLE ghosts ADD COLUMN semantic_shadow TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE ghosts ADD COLUMN emotion_trace REAL NOT NULL DEFAULT 0.0",
                "ALTER TABLE ghosts ADD COLUMN residual_edges_json TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE ghosts ADD COLUMN reactivation_probability REAL NOT NULL DEFAULT 0.5",
                "ALTER TABLE ghosts ADD COLUMN partial_recall_count INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE ghosts ADD COLUMN last_resonated_at REAL NOT NULL DEFAULT 0.0",
            ):
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists
        return self._conn

    # ── Full save/load ──────────────────────────────────

    def save(self, graph: StarGraph) -> None:
        c = self.conn
        with c:
            c.execute("DELETE FROM anchors")
            c.execute("DELETE FROM edges")
            c.execute("DELETE FROM ghosts")
            c.execute("DELETE FROM schemas")

            c.executemany(
                """INSERT INTO anchors (id, text, vector_json, embedding_blob,
                   prediction_json, oscillator_json, created_at, last_activated_at,
                   source_session, tags_json, schema_ref, replay_count,
                   source_attribution, source_trust, state,
                   state_history_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
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
                        a.source_attribution, a.source_trust,
                        a.state.value if isinstance(a.state, MemoryState) else str(a.state),
                        json.dumps([(s.value if isinstance(s, MemoryState) else str(s), ts)
                                     for s, ts in a.state_history]),
                    )
                    for a in graph.anchors.values()
                ],
            )

            c.executemany(
                """INSERT INTO edges (source, target, weight, edge_type,
                   co_activation_count, created_at, last_activated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (e.source, e.target, e.weight, e.edge_type,
                     e.co_activation_count, e.created_at, e.last_activated_at)
                    for e in graph.edges.values()
                ],
            )

            c.executemany(
                """INSERT INTO schemas (id, template, slots_json, instance_ids_json,
                   confidence, created_at, tags_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (s.id, s.template, json.dumps(s.slots),
                     json.dumps(s.instance_ids), s.confidence,
                     s.created_at, json.dumps(s.tags))
                    for s in graph.schemas.values()
                ],
            )

            # Persist ghost traces — previously deleted but never re-written,
            # which silently dropped all ghosts on every save.
            c.executemany(
                """INSERT INTO ghosts (id, residue_blob, original_tags_json, pruned_at,
                   revival_count, original_importance, semantic_shadow, emotion_trace,
                   residual_edges_json, reactivation_probability,
                   partial_recall_count, last_resonated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        g.id,
                        _embedding_to_blob(g.compressed_embedding),
                        json.dumps(list(g.original_tags)),
                        g.pruned_at,
                        g.revival_count,
                        g.original_importance,
                        g.semantic_shadow,
                        g.emotion_trace,
                        json.dumps(dict(g.residual_edges)),
                        g.reactivation_probability,
                        g.partial_recall_count,
                        g.last_resonated_at,
                    )
                    for g in graph._ghost_subsystem.ghosts.values()
                ],
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

        # Detect available columns for backward compat
        anchor_cols = [col[1] for col in c.execute("PRAGMA table_info(anchors)").fetchall()]
        has_attribution = 'source_attribution' in anchor_cols

        # Select named columns in a known order — positional SELECT * breaks
        # whenever the physical column order differs between schema versions.
        wanted_cols = ["id", "text", "vector_json", "embedding_blob", "prediction_json",
                       "oscillator_json", "created_at", "last_activated_at",
                       "source_session", "tags_json", "schema_ref", "replay_count",
                       "source_attribution", "source_trust", "state",
                       "state_history_json"]
        select_cols = [col for col in wanted_cols if col in anchor_cols]

        for row in c.execute(f"SELECT {', '.join(select_cols)} FROM anchors"):
            r = dict(zip(select_cols, row))

            aid = r["id"]
            text = r["text"]
            vector_json = r.get("vector_json", "[]")
            embedding_blob = r.get("embedding_blob")
            prediction_json = r.get("prediction_json")
            oscillator_json = r.get("oscillator_json", "{}")
            created_at = r.get("created_at", 0.0)
            last_activated_at = r.get("last_activated_at", 0.0)
            source_session = r.get("source_session", "")
            tags_json = r.get("tags_json", "[]")
            schema_ref = r.get("schema_ref")
            replay_count = r.get("replay_count", 0)
            state_str = r.get("state", "active")
            state_history_json = r.get("state_history_json", "[]")
            source_attribution = r.get("source_attribution", "observation")
            source_trust = float(r.get("source_trust", 0.6))

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
                source_attribution=source_attribution,
                source_trust=source_trust,
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

        # Restore ghost traces (older DBs may lack the extended columns)
        ghost_cols = [col[1] for col in c.execute("PRAGMA table_info(ghosts)").fetchall()]
        if ghost_cols:
            from ..ghost import GhostNode

            base_cols = ["id", "residue_blob", "original_tags_json", "pruned_at",
                         "revival_count", "original_importance"]
            extra_cols = ["semantic_shadow", "emotion_trace", "residual_edges_json",
                          "reactivation_probability", "partial_recall_count",
                          "last_resonated_at"]
            select_cols = [col for col in base_cols + extra_cols if col in ghost_cols]
            for row in c.execute(f"SELECT {', '.join(select_cols)} FROM ghosts"):
                g = dict(zip(select_cols, row))
                ghost = GhostNode(
                    id=g["id"],
                    compressed_embedding=_blob_to_embedding(g.get("residue_blob")) or [],
                    residual_edges=json.loads(g.get("residual_edges_json", "{}")),
                    emotion_trace=float(g.get("emotion_trace", 0.0)),
                    pruned_at=float(g.get("pruned_at", 0.0)),
                    original_tags=json.loads(g.get("original_tags_json", "[]")),
                    original_importance=float(g.get("original_importance", 0.5)),
                    semantic_shadow=g.get("semantic_shadow", ""),
                    reactivation_probability=float(g.get("reactivation_probability", 0.5)),
                    revival_count=int(g.get("revival_count", 0)),
                    partial_recall_count=int(g.get("partial_recall_count", 0)),
                    last_resonated_at=float(g.get("last_resonated_at", 0.0)),
                )
                graph._ghost_subsystem.ghosts[ghost.id] = ghost

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
        # Read-only check — must not create the file, connection cache, or
        # tables as a side effect (self.conn would do all three).
        if not self.path.exists():
            return False
        try:
            ro = sqlite3.connect(str(self.path))
            try:
                tables = ro.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='anchors'"
                ).fetchall()
                return len(tables) > 0
            finally:
                ro.close()
        except Exception:
            return False

    # ── Transaction context manager ──────────────────────

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """SQLite transaction context — auto-commit or rollback."""
        c = self.conn
        with c:  # sqlite3 connection as context manager = transaction
            yield

    # ── Batch save ───────────────────────────────────────

    def batch_save(self, items: list[tuple[str, dict]]) -> None:
        """Efficiently batch-write multiple items in a single transaction."""
        c = self.conn
        with c:
            for item_type, data in items:
                if item_type == "anchor":
                    self.save_anchor(data.get("id", ""), data)
                elif item_type == "edge":
                    self.save_edge(data.get("source", ""),
                                   data.get("target", ""), data)
                elif item_type == "ghost":
                    self.save_ghost(data.get("id", ""), data)

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
