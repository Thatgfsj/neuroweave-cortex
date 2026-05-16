"""Tests for ThermalStore — 3-tier hot/cold/archive auto storage."""

import os
import time
import tempfile
import pytest
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor
from star_graph.thermal_store import ThermalStore


def _make_anchor(aid: str, text: str, last_activated: float | None = None) -> Anchor:
    a = Anchor.create(text=text)
    a.id = aid
    if last_activated is not None:
        a.last_activated_at = last_activated
    return a


class TestThermalStoreInit:
    def test_default_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(storage_dir=tmpdir)
            assert store.hot_to_cold_hours == 72.0
            assert store.cold_to_archive_hours == 720.0

    def test_custom_thresholds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, hot_to_cold_hours=1.0,
                                cold_to_archive_hours=10.0,
                                promote_accesses=3)
            assert store.hot_to_cold_hours == 1.0
            assert store.promote_accesses == 3

    def test_empty_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(storage_dir=tmpdir)
            s = store.stats
            assert s["cold_count"] == 0
            assert s["archive_count"] == 0


class TestThermalStoreTouch:
    def test_touch_records_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            store.touch("anchor_1")
            assert "anchor_1" in store._access_log

    def test_touch_multiple_times(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, promote_accesses=3)
            for _ in range(3):
                store.touch("a1")
            assert len(store._access_log["a1"]) == 3


class TestThermalStoreDemote:
    def test_demote_old_anchor_to_cold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, hot_to_cold_hours=-1.0)  # any idle → demote
            g = StarGraph()
            a = _make_anchor("a1", "test anchor", last_activated=0.0)
            g.add_anchor(a)
            result = store.demote_scan(g)
            assert result["hot_to_cold"] >= 1

    def test_no_demote_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, hot_to_cold_hours=1000.0)
            g = StarGraph()
            a = _make_anchor("a2", "recent anchor",
                            last_activated=time.time())
            g.add_anchor(a)
            result = store.demote_scan(g)
            assert result["hot_to_cold"] == 0

    def test_demote_empty_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            g = StarGraph()
            result = store.demote_scan(g)
            assert result["hot_to_cold"] == 0


class TestThermalStoreLoad:
    def test_load_cold_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            assert store.load_cold("nonexistent") is None

    def test_load_archive_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            assert store.load_archive("nonexistent") is None

    def test_thaw_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            g = StarGraph()
            result = store.thaw_anchor("nonexistent", g)
            assert result is None

    def test_load_cold_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            data = {"id": "a1", "text": "cold data"}
            store._cold_store.offload("a1", data)
            store._cold_ids.add("a1")
            result = store.load_cold("a1")
            assert result is not None
            assert result["text"] == "cold data"

    def test_load_archive_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            data = {"id": "a1", "text": "archive data"}
            store._archive_store.offload("a1", data)
            store._archive_ids.add("a1")
            result = store.load_archive("a1")
            assert result is not None
            assert result["text"] == "archive data"

    def test_thaw_from_cold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            g = StarGraph()
            data = {
                "id": "a1", "text": "thawed", "embedding": [0.1, 0.2],
                "tags": ["test"], "source_session": "s1",
                "created_at": time.time(), "last_activated_at": time.time(),
                "importance": 0.7, "emotional_valence": 0.3,
                "community_id": "c1",
            }
            store._cold_store.offload("a1", data)
            store._cold_ids.add("a1")
            anchor = store.thaw_anchor("a1", g)
            assert anchor is not None
            assert anchor.text == "thawed"
            assert "a1" in g.anchors
            assert "a1" not in store._cold_ids

    def test_thaw_from_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            g = StarGraph()
            data = {
                "id": "a1", "text": "archive thaw", "embedding": None,
                "tags": [], "source_session": "",
                "created_at": time.time(), "last_activated_at": time.time(),
                "importance": 0.5, "emotional_valence": 0.0,
                "community_id": "",
            }
            store._archive_store.offload("a1", data)
            store._archive_ids.add("a1")
            anchor = store.thaw_anchor("a1", g)
            assert anchor is not None
            assert "a1" in g.anchors
            assert "a1" not in store._archive_ids


class TestThermalStorePromotion:
    def test_promote_archive_to_cold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            data = {"id": "a1", "text": "test", "embedding": None,
                    "tags": [], "source_session": "",
                    "created_at": time.time(), "last_activated_at": time.time(),
                    "importance": 0.5, "emotional_valence": 0.0,
                    "community_id": ""}
            store._archive_store.offload("a1", data)
            store._archive_ids.add("a1")
            store._promote_archive_to_cold("a1")
            assert "a1" in store._cold_ids
            assert "a1" not in store._archive_ids
            assert store._total_promotions == 1

    def test_promote_archive_to_cold_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            store._archive_ids.add("a1")
            store._promote_archive_to_cold("a1")
            assert "a1" not in store._cold_ids

    def test_promote_cold_to_hot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            data = {"id": "a1", "text": "test"}
            store._cold_store.offload("a1", data)
            store._cold_ids.add("a1")
            store._promote_cold_to_hot("a1")
            assert "a1" not in store._cold_ids
            assert store._total_promotions == 1

    def test_promote_cold_to_hot_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            store._cold_ids.add("a1")
            store._promote_cold_to_hot("a1")
            assert "a1" not in store._cold_ids

    def test_touch_triggers_promote_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, promote_accesses=2)
            data = {"id": "a1", "text": "test", "embedding": None,
                    "tags": [], "source_session": "",
                    "created_at": time.time(), "last_activated_at": time.time(),
                    "importance": 0.5, "emotional_valence": 0.0,
                    "community_id": ""}
            store._archive_store.offload("a1", data)
            store._archive_ids.add("a1")
            store.touch("a1")
            store.touch("a1")
            assert "a1" in store._cold_ids
            assert "a1" not in store._archive_ids

    def test_touch_triggers_promote_cold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, promote_accesses=2)
            data = {"id": "a1", "text": "test"}
            store._cold_store.offload("a1", data)
            store._cold_ids.add("a1")
            store.touch("a1")
            store.touch("a1")
            assert "a1" not in store._cold_ids


class TestThermalStoreAdvanced:
    def test_demote_cold_to_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, hot_to_cold_hours=-1.0,
                                cold_to_archive_hours=-1.0)
            g = StarGraph()
            a = _make_anchor("a1", "test", last_activated=0.0)
            g.add_anchor(a)
            result = store.demote_scan(g)
            assert result["hot_to_cold"] >= 1
            assert result["cold_to_archive"] >= 1

    def test_demote_with_now_param(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, hot_to_cold_hours=1.0)
            g = StarGraph()
            a = _make_anchor("a1", "test", last_activated=0.0)
            g.add_anchor(a)
            result = store.demote_scan(g, now=time.time() + 100 * 3600)
            assert result["hot_to_cold"] >= 1

    def test_demote_skip_missing_anchor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, hot_to_cold_hours=-1.0)
            g = StarGraph()
            a = _make_anchor("a1", "test", last_activated=0.0)
            g.add_anchor(a)
            g.anchors.pop("a1", None)  # remove before demote
            result = store.demote_scan(g)
            assert result["hot_to_cold"] == 0

    def test_demote_cold_skip_missing_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, hot_to_cold_hours=-1.0,
                                cold_to_archive_hours=-1.0)
            g = StarGraph()
            a = _make_anchor("a1", "test", last_activated=0.0)
            g.add_anchor(a)
            store.demote_scan(g)  # hot → cold
            # Corrupt the cold store
            store._cold_store._store.clear()
            result = store.demote_scan(g)  # try cold → archive
            assert result["cold_to_archive"] == 0

    def test_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            store._cold_store.offload("a1", {"id": "a1", "text": "test"})
            store.flush()
            # Verify file was written
            assert os.path.exists(store._cold_store._path)

    def test_serialize_anchor(self):
        a = Anchor(id="a1", text="test text", tags=["tag1"],
                  source_session="s1")
        a.vector.importance = 0.9
        a.vector.emotional_valence = 0.4
        data = ThermalStore._serialize_anchor(a)
        assert data["id"] == "a1"
        assert data["text"] == "test text"
        assert data["importance"] == 0.9
        assert data["emotional_valence"] == 0.4
        assert "tag1" in data["tags"]
        assert data["source_session"] == "s1"

    def test_stats_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir)
            store._cold_ids.add("c1")
            store._cold_ids.add("c2")
            store._archive_ids.add("a1")
            s = store.stats
            assert s["cold_count"] == 2
            assert s["archive_count"] == 1

    def test_touch_cleans_old_window_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ThermalStore(tmpdir, promote_window_hours=0.0)
            store.touch("a1")
            time.sleep(0.1)
            store.touch("a1")
            # Old entry outside zero window should be cleaned
            assert len(store._access_log["a1"]) <= 1
