"""Tests for shard module — ShardInfo, MemoryShardManager, _time_bucket."""

import os
import json
import time
import tempfile

import pytest

from star_graph.shard import (
    ShardInfo,
    MemoryShardManager,
    _time_bucket,
    DOMAIN_DIRS,
)
from star_graph.anchor import Anchor


class TestTimeBucket:
    def test_quarter(self):
        import datetime
        dt = datetime.datetime(2026, 3, 15)
        ts = dt.timestamp()
        bucket = _time_bucket(ts, "quarter")
        assert bucket == "2026_Q1"

    def test_quarter_q4(self):
        import datetime
        dt = datetime.datetime(2026, 10, 1)
        ts = dt.timestamp()
        bucket = _time_bucket(ts, "quarter")
        assert bucket == "2026_Q4"

    def test_week(self):
        import datetime
        dt = datetime.datetime(2026, 1, 15)
        ts = dt.timestamp()
        bucket = _time_bucket(ts, "week")
        assert "2026_" in bucket

    def test_month(self):
        import datetime
        dt = datetime.datetime(2026, 5, 10)
        ts = dt.timestamp()
        bucket = _time_bucket(ts, "month")
        assert bucket == "2026_05"


class TestShardInfo:
    def test_defaults(self):
        si = ShardInfo(path="memory/episodic/test.mem", domain="episodic")
        assert si.path == "memory/episodic/test.mem"
        assert si.domain == "episodic"
        assert si.anchor_count == 0
        assert si.is_active is True

    def test_with_all_fields(self):
        si = ShardInfo(
            path="/test.mem", domain="procedural",
            subdomain="python", time_bucket="2026_Q1",
            anchor_count=100, size_bytes=50000,
            is_active=False,
        )
        assert si.subdomain == "python"
        assert si.anchor_count == 100
        assert si.is_active is False


class TestDomainDirs:
    def test_known_domains(self):
        assert "episodic" in DOMAIN_DIRS
        assert "semantic" in DOMAIN_DIRS
        assert "procedural" in DOMAIN_DIRS
        assert "reflection" in DOMAIN_DIRS


class TestMemoryShardManager:
    def test_init_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            assert msm.max_file_size > 0
            assert msm.time_granularity == "quarter"

    def test_init_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(
                base_dir=tmp, max_file_size_mb=10,
                time_granularity="week",
            )
            assert msm.max_file_size == 10 * 1024 * 1024
            assert msm.time_granularity == "week"

    def test_route_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            a = Anchor(id="a1", text="test")
            a.created_at = 1710000000.0  # fixed timestamp
            path = msm.route_anchor(a, domain="episodic")
            assert path.endswith(".mem")
            assert "episodic" in path

    def test_route_anchor_default_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            a = Anchor(id="a1", text="test")
            path = msm.route_anchor(a)
            assert path.endswith(".mem")

    def test_route_anchor_with_subdomain(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            a = Anchor(id="a1", text="test")
            a.created_at = 1710000000.0
            path = msm.route_anchor(a, domain="procedural", subdomain="python")
            assert "procedural" in path
            assert "python" in path

    def test_save_and_load_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            fp = os.path.join(tmp, "episodic", "2024_Q1", "test.mem")
            data = [
                {"id": "a1", "text": "first"},
                {"id": "a2", "text": "second"},
            ]
            msm.save_shard(fp, data)
            loaded = msm.load_shard(fp)
            assert len(loaded) == 2

    def test_save_shard_merges_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            fp = os.path.join(tmp, "test.mem")
            msm.save_shard(fp, [{"id": "a1", "text": "first"}])
            msm.save_shard(fp, [{"id": "a1", "text": "updated"}])
            loaded = msm.load_shard(fp)
            assert len(loaded) == 1
            assert loaded[0]["text"] == "updated"

    def test_save_shard_appends_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            fp = os.path.join(tmp, "test.mem")
            msm.save_shard(fp, [{"id": "a1", "text": "first"}])
            msm.save_shard(fp, [{"id": "a2", "text": "second"}])
            loaded = msm.load_shard(fp)
            assert len(loaded) == 2

    def test_load_shard_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            result = msm.load_shard("/nonexistent/path.mem")
            assert result == []

    def test_load_shard_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            fp = os.path.join(tmp, "corrupt.mem")
            with open(fp, "w") as f:
                f.write("not json")
            result = msm.load_shard(fp)
            assert result == []

    def test_load_all_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            result = msm.load_all()
            assert result == []

    def test_list_shards_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            shards = msm.list_shards()
            assert shards == []

    def test_list_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            fp = os.path.join(tmp, "episodic", "2024_Q1_01.mem")
            msm.save_shard(fp, [{"id": "a1", "text": "test"}])
            shards = msm.list_shards()
            assert len(shards) >= 1

    def test_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            s = msm.stats
            assert s["shard_count"] == 0
            assert s["total_anchors"] == 0

    def test_stats_with_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            fp = os.path.join(tmp, "episodic", "test.mem")
            msm.save_shard(fp, [{"id": "a1"}, {"id": "a2"}])
            s = msm.stats
            assert s["shard_count"] >= 1
            assert s["total_anchors"] >= 2

    def test_save_shard_handles_corrupt_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            fp = os.path.join(tmp, "corrupt.mem")
            with open(fp, "w") as f:
                f.write("garbage{{{")
            # Should handle gracefully
            msm.save_shard(fp, [{"id": "a1", "text": "recovered"}])
            loaded = msm.load_shard(fp)
            assert len(loaded) == 1

    # ── Internal methods ──────────────────────────────────

    def test_ensure_dirs_creates_domain_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            for domain_dir in DOMAIN_DIRS.values():
                assert os.path.isdir(os.path.join(tmp, domain_dir))

    def test_find_existing_shard_nonexistent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            result = msm._find_existing_shard("nope", "", "2024_Q1")
            assert result is None

    def test_find_existing_shard_finds_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
            fp = os.path.join(tmp, "episodic", "2024_Q1_01.mem")
            with open(fp, "w") as f:
                json.dump([{"id": "a1"}], f)
            result = msm._find_existing_shard("episodic", "", "2024_Q1")
            assert result is not None
            assert "2024_Q1" in result

    def test_find_existing_shard_with_subdomain(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "procedural", "python"), exist_ok=True)
            fp = os.path.join(tmp, "procedural", "python", "2024_Q2_01.mem")
            with open(fp, "w") as f:
                json.dump([{"id": "a1"}], f)
            result = msm._find_existing_shard("procedural", "python", "2024_Q2")
            assert result is not None
            assert "python" in result

    def test_find_existing_shard_skips_oversized(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp, max_file_size_mb=0.001)  # 1KB max
            os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
            fp = os.path.join(tmp, "episodic", "2024_Q1_01.mem")
            with open(fp, "w") as f:
                json.dump([{"id": f"a{i}", "text": "x" * 500} for i in range(10)], f)
            result = msm._find_existing_shard("episodic", "", "2024_Q1")
            assert result is None  # file too big

    def test_next_shard_path_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            path = msm._next_shard_path("episodic", "", "2024_Q1")
            assert path.endswith("2024_Q1_01.mem")
            assert "episodic" in path

    def test_next_shard_path_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
            with open(os.path.join(tmp, "episodic", "2024_Q1_01.mem"), "w") as f:
                f.write("[]")
            with open(os.path.join(tmp, "episodic", "2024_Q1_02.mem"), "w") as f:
                f.write("[]")
            path = msm._next_shard_path("episodic", "", "2024_Q1")
            assert path.endswith("2024_Q1_03.mem")

    def test_next_shard_path_with_subdomain(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            path = msm._next_shard_path("procedural", "python", "2024_Q1")
            assert "procedural" in path
            assert "python" in path
            assert "2024_Q1" in path

    def test_next_shard_path_handles_non_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
            with open(os.path.join(tmp, "episodic", "2024_Q1_abc.mem"), "w") as f:
                f.write("[]")
            path = msm._next_shard_path("episodic", "", "2024_Q1")
            assert path.endswith("2024_Q1_01.mem")  # ignores non-numeric

    # ── Route anchor with rotation ────────────────────────

    def test_route_anchor_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp, max_file_size_mb=0.001)  # 1KB
            a = Anchor(id="a1", text="test")
            a.created_at = 1710000000.0
            # First route creates the file
            path1 = msm.route_anchor(a, domain="episodic")
            # Write enough data to exceed 1KB
            os.makedirs(os.path.dirname(path1), exist_ok=True)
            with open(path1, "w") as f:
                json.dump([{"id": f"x{i}", "text": "padding" * 100} for i in range(20)], f)
            # Second route should rotate to new file
            path2 = msm.route_anchor(a, domain="episodic")
            assert path1 != path2

    # ── load_all comprehensive ────────────────────────────

    def test_load_all_multiple_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            msm.save_shard(
                os.path.join(tmp, "episodic", "test.mem"),
                [{"id": "e1", "text": "episodic"}],
            )
            msm.save_shard(
                os.path.join(tmp, "semantic", "test.mem"),
                [{"id": "s1", "text": "semantic"}],
            )
            all_data = msm.load_all()
            assert len(all_data) >= 2

    def test_load_all_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            all_data = msm.load_all()
            assert all_data == []

    def test_load_all_skips_non_mem_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
            with open(os.path.join(tmp, "episodic", "notes.txt"), "w") as f:
                f.write("not a mem file")
            all_data = msm.load_all()
            assert all_data == []  # .txt file skipped

    # ── list_shards comprehensive ─────────────────────────

    def test_list_shards_with_subdomain(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "procedural", "python"), exist_ok=True)
            fp = os.path.join(tmp, "procedural", "python", "test.mem")
            with open(fp, "w") as f:
                json.dump([{"id": "a1"}], f)
            shards = msm.list_shards()
            assert len(shards) == 1
            assert shards[0].subdomain == "python"

    def test_list_shards_handles_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
            fp = os.path.join(tmp, "episodic", "corrupt.mem")
            with open(fp, "w") as f:
                f.write("{invalid")
            shards = msm.list_shards()
            assert len(shards) == 1
            assert shards[0].anchor_count == 0

    def test_list_shards_handles_non_list_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
            fp = os.path.join(tmp, "episodic", "test.mem")
            with open(fp, "w") as f:
                json.dump({"not": "a list"}, f)
            shards = msm.list_shards()
            assert len(shards) == 1
            assert shards[0].anchor_count == 0

    # ── stats comprehensive ───────────────────────────────

    def test_stats_multiple_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            msm.save_shard(
                os.path.join(tmp, "episodic", "test.mem"),
                [{"id": "a1"}, {"id": "a2"}],
            )
            msm.save_shard(
                os.path.join(tmp, "semantic", "test.mem"),
                [{"id": "b1"}],
            )
            s = msm.stats
            assert s["shard_count"] == 2
            assert s["total_anchors"] == 3
            assert s["total_size_mb"] >= 0
            assert "episodic" in s["domains"]

    # ── Route anchor edge cases ───────────────────────────

    def test_route_anchor_unknown_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            a = Anchor(id="a1", text="test")
            a.created_at = 1710000000.0
            path = msm.route_anchor(a, domain="unknown_domain")
            assert "general" in path

    def test_route_anchor_uses_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            msm = MemoryShardManager(base_dir=tmp)
            a = Anchor(id="a1", text="test")
            a.created_at = 1710000000.0
            path1 = msm.route_anchor(a, domain="episodic")
            # Reset internal state to force re-find
            msm._current_files.clear()
            path2 = msm.route_anchor(a, domain="episodic")
            assert path1 == path2  # should find existing
