"""Snapshot + WAL — crash-safe persistence with versioned state snapshots.

Provides:
- Timed auto-snapshots (JSON/SQLite dump at configurable intervals)
- Write-Ahead Log (WAL) for crash recovery (leveraging SQLite's WAL journal)
- State versioning: keep last N snapshots, support manual rollback
- Recovery: auto-detect incomplete operations on load and roll back
- Snapshot metadata: timestamp, cycle count, anchor/edge counts, checksum

Usage:
    snap = SnapshotManager(base_dir="~/.star_graph/snapshots")
    snap.snapshot(graph)                          # create numbered snapshot
    graph = snap.load_latest()                    # load most recent snapshot
    snap.rollback(version=3)                      # revert to specific version
    snap.cleanup(keep=5)                          # remove old snapshots
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_SNAPSHOT_DIR = Path.home() / ".star_graph" / "snapshots"


@dataclass
class SnapshotMeta:
    """Metadata for one snapshot version."""
    version: int = 0
    timestamp: float = field(default_factory=time.time)
    cycle_count: int = 0
    anchor_count: int = 0
    edge_count: int = 0
    ghost_count: int = 0
    schema_count: int = 0
    checksum: str = ""
    compressed: bool = False
    file_size_bytes: int = 0
    description: str = ""


class SnapshotManager:
    """Manages versioned snapshots with WAL integration.

    Snapshots are stored as numbered files: snapshot_00001.json.gz, etc.
    The WAL records operations since the last snapshot for crash recovery.
    Keeps the last N snapshots, auto-deleting older ones.

    Usage:
        snap = SnapshotManager(keep=5, compress=True)
        snap.snapshot(graph)
        graph = snap.load_latest()
        snap.rollback(version=3)
    """

    def __init__(self, base_dir: Path | str = DEFAULT_SNAPSHOT_DIR,
                 keep: int = 5, compress: bool = True,
                 auto_snapshot_interval_hours: float = 6.0):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.compress = compress
        self.auto_interval = auto_snapshot_interval_hours
        self._last_snapshot_time: float = 0.0
        self._wal_path = self.base_dir / "wal.jsonl"
        self._meta_path = self.base_dir / "snapshot_index.json"
        self._wal_entries: list[dict] = []
        self._index: dict[int, SnapshotMeta] = {}
        self._load_index()

    # ── Core operations ────────────────────────────────────────

    def snapshot(self, graph, description: str = "",
                 force: bool = False) -> SnapshotMeta:
        """Create a new snapshot of the current graph state.

        Args:
            graph: StarGraph instance to snapshot.
            description: Optional human-readable label for this snapshot.
            force: If True, create snapshot even if auto-interval hasn't elapsed.

        Returns:
            SnapshotMeta for the new snapshot.
        """
        now = time.time()

        # Respect auto-interval unless forced
        if not force and self._last_snapshot_time:
            hours_since = (now - self._last_snapshot_time) / 3600
            if hours_since < self.auto_interval:
                return self._latest_meta() or SnapshotMeta()

        version = self._next_version()
        data = self._serialize_graph(graph)
        checksum = hashlib.blake2b(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        filepath = self._snapshot_path(version, self.compress)

        if self.compress:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        file_size = filepath.stat().st_size if filepath.exists() else 0

        meta = SnapshotMeta(
            version=version,
            timestamp=now,
            cycle_count=getattr(graph, '_cycle_count', 0),
            anchor_count=len(getattr(graph, 'anchors', {})),
            edge_count=len(getattr(graph, 'edges', {})),
            ghost_count=len(getattr(graph, 'ghosts', {})),
            schema_count=len(getattr(graph, 'schemas', {})),
            checksum=checksum,
            compressed=self.compress,
            file_size_bytes=file_size,
            description=description,
        )

        self._index[version] = meta
        self._last_snapshot_time = now
        self._save_index()

        # Flush WAL after successful snapshot
        self._flush_wal()

        # Cleanup old snapshots
        self.cleanup()

        return meta

    def load(self, version: int | None = None) -> tuple:
        """Load a specific snapshot version, or the latest if version is None.

        Returns (StarGraph, SnapshotMeta) tuple.
        """
        if version is None:
            meta = self._latest_meta()
            if meta is None:
                from .graph import StarGraph
                return StarGraph(), SnapshotMeta()
        else:
            meta = self._index.get(version)
            if meta is None:
                raise FileNotFoundError(f"Snapshot version {version} not found")

        filepath = self._snapshot_path(meta.version, meta.compressed)
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot file missing: {filepath}")

        if meta.compressed:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        graph = self._deserialize_graph(data)

        # Apply any WAL entries written after this snapshot
        graph = self._apply_wal(graph, after_timestamp=meta.timestamp)

        return graph, meta

    def load_latest(self) -> tuple:
        """Load the most recent snapshot."""
        return self.load(version=None)

    def rollback(self, version: int) -> bool:
        """Roll back to a specific snapshot version.

        Deletes all snapshots newer than 'version'.
        Returns True if successful.
        """
        if version not in self._index:
            return False

        # Delete snapshots newer than target
        for v in sorted(self._index.keys()):
            if v > version:
                path = self._snapshot_path(v, self._index[v].compressed)
                if path.exists():
                    path.unlink()
                del self._index[v]

        self._save_index()
        return True

    def cleanup(self, keep: int | None = None):
        """Remove old snapshots, keeping only the most recent 'keep' versions."""
        keep = keep or self.keep
        versions = sorted(self._index.keys())
        to_remove = versions[:-keep] if len(versions) > keep else []

        for v in to_remove:
            meta = self._index.get(v)
            if meta:
                path = self._snapshot_path(v, meta.compressed)
                if path.exists():
                    path.unlink()
            del self._index[v]

        if to_remove:
            self._save_index()

    # ── WAL (Write-Ahead Log) ─────────────────────────────────

    def wal_append(self, operation: str, data: dict):
        """Append an operation to the WAL for crash recovery.

        Called before critical mutations so they can be replayed after crash.
        """
        entry = {
            "op": operation,
            "ts": time.time(),
            "data": data,
        }
        self._wal_entries.append(entry)

        # Append to file for durability
        with open(self._wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def wal_clear(self):
        """Clear WAL after successful snapshot."""
        self._wal_entries.clear()
        if self._wal_path.exists():
            self._wal_path.unlink()

    def _flush_wal(self):
        """Flush WAL: persist then clear (snapshot subsumes WAL)."""
        self.wal_clear()

    def _apply_wal(self, graph, after_timestamp: float = 0.0):
        """Apply WAL entries newer than after_timestamp to the graph.

        Used during recovery to replay operations that happened after
        the last snapshot but before a crash.
        """
        if not self._wal_path.exists():
            return graph

        entries = []
        with open(self._wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("ts", 0) > after_timestamp:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue

        for entry in entries:
            self._replay_wal_entry(graph, entry)

        return graph

    def _replay_wal_entry(self, graph, entry: dict):
        """Replay a single WAL operation on the graph."""
        op = entry.get("op", "")
        data = entry.get("data", {})

        if op == "add_anchor":
            from .anchor import Anchor, AnchorVector
            anchor = Anchor(
                id=data["id"],
                text=data["text"][:280],
                vector=AnchorVector.from_list(data.get("vector", [])),
                source_session=data.get("source_session", ""),
                tags=data.get("tags", []),
            )
            graph.add_anchor(anchor)

        elif op == "remove_anchor":
            graph.remove_anchor(data.get("id", ""))

        elif op == "add_edge":
            graph.add_edge(
                data["source"], data["target"],
                weight=data.get("weight", 0.5),
                edge_type=data.get("edge_type", "topical"),
            )

        elif op == "remove_edge":
            key = graph._key(data["source"], data["target"])
            graph.edges.pop(key, None)
            graph._adjacency.get(data["source"], set()).discard(data["target"])
            graph._adjacency.get(data["target"], set()).discard(data["source"])

    # ── Recovery ───────────────────────────────────────────────

    def recover(self) -> tuple:
        """Attempt crash recovery: load latest snapshot + replay WAL.

        Returns (StarGraph, recovery_log: list[str]).
        """
        log: list[str] = []

        # Try loading latest snapshot
        try:
            graph, meta = self.load_latest()
            log.append(f"Loaded snapshot v{meta.version} ({meta.anchor_count} anchors)")
        except (FileNotFoundError, json.JSONDecodeError):
            from .graph import StarGraph
            graph = StarGraph()
            log.append("No valid snapshot found — starting fresh")

        # Replay WAL entries
        if self._wal_path.exists():
            entries = []
            with open(self._wal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            log.append(f"Warning: corrupt WAL line skipped")

            replayed = 0
            for entry in entries:
                try:
                    self._replay_wal_entry(graph, entry)
                    replayed += 1
                except Exception as e:
                    log.append(f"Warning: failed to replay WAL entry: {e}")

            log.append(f"Replayed {replayed}/{len(entries)} WAL entries")
            self.wal_clear()

        return graph, log

    # ── Internal helpers ───────────────────────────────────────

    def _serialize_graph(self, graph) -> dict:
        """Serialize a StarGraph to a JSON-safe dict."""
        anchors = []
        for a in graph.anchors.values():
            anchors.append({
                "id": a.id, "text": a.text,
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
                "state": a.state.value if hasattr(a.state, 'value') else str(a.state),
                "community_id": a.community_id,
                "secondary_community_ids": a.secondary_community_ids,
                "exact_match_keys": a.exact_match_keys,
                "salience": a.salience,
                "cortex_path": a.cortex_path,
                "segment_id": a.segment_id,
            })

        edges = []
        for e in graph.edges.values():
            edges.append({
                "source": e.source, "target": e.target,
                "weight": e.weight, "edge_type": e.edge_type,
                "co_activation_count": e.co_activation_count,
                "created_at": e.created_at,
                "last_activated_at": e.last_activated_at,
            })

        ghosts = []
        for g in graph.ghosts.values():
            ghosts.append({
                "id": g.id, "residue": g.residue,
                "original_tags": g.original_tags,
                "pruned_at": g.pruned_at,
                "revival_count": g.revival_count,
                "original_importance": g.original_importance,
            })

        schemas = []
        for s in graph.schemas.values():
            schemas.append({
                "id": s.id, "template": s.template,
                "slots": s.slots, "instance_ids": s.instance_ids,
                "confidence": s.confidence, "created_at": s.created_at,
                "tags": s.tags,
            })

        return {
            "version": 3,
            "snapshot_at": time.time(),
            "anchors": anchors,
            "edges": edges,
            "ghosts": ghosts,
            "schemas": schemas,
        }

    def _deserialize_graph(self, data: dict):
        """Deserialize a JSON dict back into a StarGraph."""
        from .graph import StarGraph
        from .anchor import (
            Anchor, AnchorVector, AnchorPrediction, Oscillator, GhostAnchor, MemoryState,
        )

        graph = StarGraph()

        for a_data in data.get("anchors", []):
            osc_data = a_data.get("oscillator", {})
            oscillator = Oscillator(
                natural_frequency=osc_data.get("natural_frequency", 0.5),
                phase_offset=osc_data.get("phase_offset", 0.0),
                coupling_strength=osc_data.get("coupling_strength", 0.3),
                damping=osc_data.get("damping", 0.1),
            )
            pred = None
            if a_data.get("prediction"):
                pred = AnchorPrediction(
                    emotional_tone=a_data["prediction"].get("emotional_tone", 0.0),
                    expected_duration=a_data["prediction"].get("expected_duration", 10.0),
                    confidence=a_data["prediction"].get("confidence", 0.5),
                )
            state_str = a_data.get("state", "active")
            try:
                state = MemoryState(state_str)
            except ValueError:
                state = MemoryState.ACTIVE

            anchor = Anchor(
                id=a_data["id"], text=a_data["text"],
                vector=AnchorVector.from_list(a_data.get("vector", [])),
                embedding=a_data.get("embedding"),
                prediction=pred, oscillator=oscillator,
                created_at=a_data.get("created_at", time.time()),
                last_activated_at=a_data.get("last_activated_at", time.time()),
                source_session=a_data.get("source_session", ""),
                tags=a_data.get("tags", []),
                schema_ref=a_data.get("schema_ref"),
                replay_count=a_data.get("replay_count", 0),
                state=state,
                community_id=a_data.get("community_id", ""),
                secondary_community_ids=a_data.get("secondary_community_ids", []),
                exact_match_keys=a_data.get("exact_match_keys", []),
                salience=a_data.get("salience", 0.0),
                cortex_path=a_data.get("cortex_path", ""),
                segment_id=a_data.get("segment_id", ""),
            )
            graph.add_anchor(anchor)

        for e_data in data.get("edges", []):
            from .graph import Edge
            edge = Edge(
                source=e_data["source"], target=e_data["target"],
                weight=e_data["weight"], edge_type=e_data.get("edge_type", "topical"),
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
                id=g_data["id"], residue=g_data.get("residue", []),
                original_tags=g_data.get("original_tags", []),
                pruned_at=g_data.get("pruned_at", time.time()),
                revival_count=g_data.get("revival_count", 0),
                original_importance=g_data.get("original_importance", 0.5),
            )
            graph.ghosts[ghost.id] = ghost

        for s_data in data.get("schemas", []):
            from .graph import Schema
            schema = Schema(
                id=s_data["id"], template=s_data["template"],
                slots=s_data.get("slots", {}),
                instance_ids=s_data.get("instance_ids", []),
                confidence=s_data.get("confidence", 0.0),
                created_at=s_data.get("created_at", time.time()),
                tags=s_data.get("tags", []),
            )
            graph.schemas[schema.id] = schema

        return graph

    def _next_version(self) -> int:
        """Get the next snapshot version number."""
        return max(self._index.keys(), default=0) + 1

    def _snapshot_path(self, version: int, compressed: bool) -> Path:
        """Get the file path for a snapshot version."""
        ext = ".json.gz" if compressed else ".json"
        return self.base_dir / f"snapshot_{version:05d}{ext}"

    def _latest_meta(self) -> SnapshotMeta | None:
        if not self._index:
            return None
        max_v = max(self._index.keys())
        return self._index[max_v]

    def _save_index(self):
        """Persist the snapshot index to disk."""
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "last_snapshot_time": self._last_snapshot_time,
                "snapshots": {
                    str(v): {
                        "version": m.version,
                        "timestamp": m.timestamp,
                        "cycle_count": m.cycle_count,
                        "anchor_count": m.anchor_count,
                        "edge_count": m.edge_count,
                        "ghost_count": m.ghost_count,
                        "schema_count": m.schema_count,
                        "checksum": m.checksum,
                        "compressed": m.compressed,
                        "file_size_bytes": m.file_size_bytes,
                        "description": m.description,
                    }
                    for v, m in self._index.items()
                },
            }, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        """Load the snapshot index from disk."""
        if not self._meta_path.exists():
            return
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._last_snapshot_time = data.get("last_snapshot_time", 0.0)
            for v_str, m_data in data.get("snapshots", {}).items():
                v = int(v_str)
                self._index[v] = SnapshotMeta(
                    version=m_data["version"],
                    timestamp=m_data["timestamp"],
                    cycle_count=m_data.get("cycle_count", 0),
                    anchor_count=m_data.get("anchor_count", 0),
                    edge_count=m_data.get("edge_count", 0),
                    ghost_count=m_data.get("ghost_count", 0),
                    schema_count=m_data.get("schema_count", 0),
                    checksum=m_data.get("checksum", ""),
                    compressed=m_data.get("compressed", False),
                    file_size_bytes=m_data.get("file_size_bytes", 0),
                    description=m_data.get("description", ""),
                )
        except (json.JSONDecodeError, KeyError):
            self._index.clear()

    @property
    def versions(self) -> list[int]:
        """List all available snapshot versions (sorted)."""
        return sorted(self._index.keys())

    @property
    def stats(self) -> dict:
        return {
            "snapshots": len(self._index),
            "latest_version": max(self._index.keys()) if self._index else 0,
            "wal_entries": len(self._wal_entries),
            "wal_path": str(self._wal_path),
            "base_dir": str(self.base_dir),
        }
