"""Memory Shard Manager — domain + time + size file partitioning.

Routes anchors to the right shard file by domain (cortex type), then by time
bucket (quarter/week), with size-based rotation to keep files manageable.

Directory layout:
    memory/
    ├── procedural/python/2026_Q2_01.mem
    ├── episodic/2026_05_week2.mem
    ├── semantic/user_preferences.mem
    └── reflection/strategy.mem
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DOMAIN_DIRS: dict[str, str] = {
    "procedural": "procedural",
    "episodic": "episodic",
    "semantic": "semantic",
    "reflection": "reflection",
    "hippocampus": "hippocampus",
    "default": "general",
}


def _time_bucket(ts: float, granularity: str = "quarter") -> str:
    """Convert timestamp to a time bucket label."""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    if granularity == "quarter":
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}_Q{quarter}"
    elif granularity == "week":
        iso = dt.isocalendar()
        return f"{iso[0]}_{iso[1]:02d}_week{iso[1]}"
    else:  # month
        return f"{dt.year}_{dt.month:02d}"


@dataclass
class ShardInfo:
    """Metadata about a single shard file."""
    path: str
    domain: str
    subdomain: str = ""
    time_bucket: str = ""
    anchor_count: int = 0
    size_bytes: int = 0
    is_active: bool = True


class MemoryShardManager:
    """Manages partitioned memory files across domains and time buckets.

    Usage:
        shards = MemoryShardManager(base_dir="memory")
        shards.save_anchors(anchors, domain="episodic")
        ...
        all_anchors = shards.load_all()
    """

    def __init__(self,
                 base_dir: str = "memory",
                 max_file_size_mb: int = 50,
                 time_granularity: str = "quarter"):
        self.base_dir = Path(base_dir)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.time_granularity = time_granularity
        self._shards: dict[str, ShardInfo] = {}
        self._current_files: dict[str, str] = {}  # domain_key → current file path
        self._ensure_dirs()

    # ── Public API ─────────────────────────────────────────

    def route_anchor(self, anchor, domain: str = "default",
                     subdomain: str = "") -> str:
        """Determine which shard file an anchor belongs to. Returns file path."""
        domain_dir = DOMAIN_DIRS.get(domain, "general")
        ts = getattr(anchor, 'created_at', time.time())
        bucket = _time_bucket(ts, self.time_granularity)

        key = f"{domain_dir}/{subdomain}/{bucket}" if subdomain else f"{domain_dir}/{bucket}"
        if key not in self._current_files:
            existing = self._find_existing_shard(domain_dir, subdomain, bucket)
            if existing:
                self._current_files[key] = existing
            else:
                self._current_files[key] = self._next_shard_path(domain_dir, subdomain, bucket)

        current_path = self._current_files[key]
        # Rotate if oversized
        if os.path.exists(current_path) and os.path.getsize(current_path) > self.max_file_size:
            self._current_files[key] = self._next_shard_path(domain_dir, subdomain, bucket)
            current_path = self._current_files[key]

        return current_path

    def save_shard(self, file_path: str, anchors_data: list[dict]) -> None:
        """Write a batch of anchor data to a shard file (append or create)."""
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        existing = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []

        # Merge: update existing anchors by ID, append new ones
        existing_ids = {a.get("id"): i for i, a in enumerate(existing)}
        for anchor_data in anchors_data:
            aid = anchor_data.get("id", "")
            if aid and aid in existing_ids:
                existing[existing_ids[aid]] = anchor_data
            else:
                existing.append(anchor_data)

        tmp = file_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        os.replace(tmp, file_path)

    def load_shard(self, file_path: str) -> list[dict]:
        """Load all anchors from a shard file."""
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def load_all(self) -> list[dict]:
        """Load all anchors from all shard files across all domains."""
        all_anchors: list[dict] = []
        for domain_dir in DOMAIN_DIRS.values():
            domain_path = self.base_dir / domain_dir
            if not domain_path.exists():
                continue
            for root, _, files in os.walk(domain_path):
                for fname in sorted(files):
                    if fname.endswith(".mem"):
                        file_path = os.path.join(root, fname)
                        all_anchors.extend(self.load_shard(file_path))
        return all_anchors

    def list_shards(self) -> list[ShardInfo]:
        """List all shard files with metadata."""
        shards: list[ShardInfo] = []
        for domain_dir in DOMAIN_DIRS.values():
            domain_path = self.base_dir / domain_dir
            if not domain_path.exists():
                continue
            for root, _, files in os.walk(domain_path):
                for fname in sorted(files):
                    if fname.endswith(".mem"):
                        fp = os.path.join(root, fname)
                        rel = os.path.relpath(fp, self.base_dir)
                        parts = rel.replace("\\", "/").split("/")
                        try:
                            data = json.load(open(fp, "r", encoding="utf-8"))
                            count = len(data) if isinstance(data, list) else 0
                        except Exception:
                            count = 0
                        shards.append(ShardInfo(
                            path=fp,
                            domain=parts[0] if len(parts) > 0 else "",
                            subdomain=parts[1] if len(parts) > 2 else "",
                            time_bucket=parts[-1].replace(".mem", "") if len(parts) > 1 else "",
                            anchor_count=count,
                            size_bytes=os.path.getsize(fp),
                        ))
        return sorted(shards, key=lambda s: s.path)

    @property
    def stats(self) -> dict:
        shards = self.list_shards()
        total_anchors = sum(s.anchor_count for s in shards)
        total_size = sum(s.size_bytes for s in shards)
        return {
            "shard_count": len(shards),
            "total_anchors": total_anchors,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "domains": list(set(s.domain for s in shards)),
        }

    # ── Internal ───────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        for d in DOMAIN_DIRS.values():
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)

    def _find_existing_shard(self, domain_dir: str, subdomain: str,
                             bucket: str) -> str | None:
        search_dir = self.base_dir / domain_dir
        if subdomain:
            search_dir = search_dir / subdomain
        if not search_dir.exists():
            return None
        for fname in sorted(os.listdir(search_dir), reverse=True):
            if bucket in fname and fname.endswith(".mem"):
                fp = os.path.join(str(search_dir), fname)
                if os.path.getsize(fp) < self.max_file_size:
                    return fp
        return None

    def _next_shard_path(self, domain_dir: str, subdomain: str,
                         bucket: str) -> str:
        search_dir = self.base_dir / domain_dir
        if subdomain:
            search_dir = search_dir / subdomain
        os.makedirs(str(search_dir), exist_ok=True)
        # Find highest existing index
        idx = 0
        prefix = f"{bucket}_"
        for fname in os.listdir(str(search_dir)):
            if fname.startswith(prefix) and fname.endswith(".mem"):
                try:
                    num = int(fname[len(prefix):-4])
                    idx = max(idx, num)
                except ValueError:
                    pass
        return os.path.join(str(search_dir), f"{prefix}{idx + 1:02d}.mem")
