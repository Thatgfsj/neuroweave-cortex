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
