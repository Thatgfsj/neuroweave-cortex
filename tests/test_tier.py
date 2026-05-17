"""Tests for tier module — MemoryTier, TierEntry, ShortTermMemory, CoreMemory, TieredStorage."""

import os
import time
import tempfile

import pytest

from star_graph.tier import (
    MemoryTier,
    TIER_DECAY_HALF_LIFE,
    TIER_MAX_ITEMS,
    TIER_PROMOTION_THRESHOLD,
    TierEntry,
    ShortTermMemory,
    CoreMemory,
    TieredStorage,
    offload_anchor_to_cold,
    _cosine_sim,
)
from star_graph.anchor import Anchor


class TestMemoryTier:
    def test_values(self):
        assert MemoryTier.STM.value == "stm"
        assert MemoryTier.MTM.value == "mtm"
        assert MemoryTier.LTM.value == "ltm"
        assert MemoryTier.CORE.value == "core"

    def test_decay_half_life(self):
        assert TIER_DECAY_HALF_LIFE[MemoryTier.STM] < 1.0
        assert TIER_DECAY_HALF_LIFE[MemoryTier.CORE] > 365

    def test_max_items(self):
        assert TIER_MAX_ITEMS[MemoryTier.STM] == 100
        assert TIER_MAX_ITEMS[MemoryTier.MTM] == 2000
        assert TIER_MAX_ITEMS[MemoryTier.LTM] == 500
        assert TIER_MAX_ITEMS[MemoryTier.CORE] == 50

    def test_promotion_thresholds(self):
        assert TIER_PROMOTION_THRESHOLD[(MemoryTier.STM, MemoryTier.MTM)] == 0.3
        assert TIER_PROMOTION_THRESHOLD[(MemoryTier.MTM, MemoryTier.LTM)] == 0.5
        assert TIER_PROMOTION_THRESHOLD[(MemoryTier.LTM, MemoryTier.CORE)] == 0.7


class TestTierEntry:
    def test_defaults(self):
        e = TierEntry(id="e1", text="test", tier=MemoryTier.STM)
        assert e.id == "e1"
        assert e.text == "test"
        assert e.importance == 0.5
        assert e.stability == 0.3
        assert e.reinforcement == 0.0
        assert e.access_count == 0

    def test_age_hours(self):
        e = TierEntry(id="e1", text="test", tier=MemoryTier.STM,
                     created_at=time.time() - 3600)
        assert e.age_hours == pytest.approx(1.0, abs=0.1)

    def test_idle_hours(self):
        e = TierEntry(id="e1", text="test", tier=MemoryTier.STM,
                     last_accessed_at=time.time() - 7200)
        assert e.idle_hours == pytest.approx(2.0, abs=0.1)

    def test_retention_score_decays_with_age(self):
        e = TierEntry(id="e1", text="test", tier=MemoryTier.STM,
                     created_at=0.0, importance=0.5, stability=0.3)
        assert e.retention_score < 0.15  # very old STM

    def test_retention_score_with_reinforcement(self):
        e1 = TierEntry(id="e1", text="test", tier=MemoryTier.LTM,
                      created_at=time.time(), importance=0.5,
                      stability=0.5, reinforcement=0.0)
        e2 = TierEntry(id="e2", text="test", tier=MemoryTier.LTM,
                      created_at=time.time(), importance=0.5,
                      stability=0.5, reinforcement=1.0)
        assert e2.retention_score > e1.retention_score

    def test_access(self):
        e = TierEntry(id="e1", text="test", tier=MemoryTier.STM)
        old_reinf = e.reinforcement
        e.access()
        assert e.access_count == 1
        assert e.reinforcement > old_reinf

    def test_custom_fields(self):
        e = TierEntry(
            id="e1", text="custom", tier=MemoryTier.LTM,
            embedding=[0.1, 0.2], tags=["tag1", "tag2"],
            importance=0.8, emotional_valence=0.5,
            source_session="s1", metadata={"key": "val"},
            promoted_from="old_id",
        )
        assert e.embedding == [0.1, 0.2]
        assert "tag1" in e.tags
        assert e.metadata["key"] == "val"
        assert e.promoted_from == "old_id"


class TestShortTermMemory:
    def test_init_default(self):
        stm = ShortTermMemory()
        assert stm.max_items == 100
        assert stm.ttl_hours == 2.0
        assert len(stm) == 0

    def test_init_custom(self):
        stm = ShortTermMemory(max_items=10, ttl_hours=1.0)
        assert stm.max_items == 10
        assert stm.ttl_hours == 1.0

    def test_add_entry(self):
        stm = ShortTermMemory()
        entry = stm.add("test message", tags=["test"])
        assert entry.text == "test message"
        assert entry.tier == MemoryTier.STM
        assert "test" in entry.tags
        assert len(stm) == 1

    def test_add_entry_with_all_fields(self):
        stm = ShortTermMemory()
        entry = stm.add(
            "test", embedding=[0.5, 0.5], tags=["a", "b"],
            importance=0.9, emotional_valence=0.3,
            source_session="s1",
        )
        assert entry.embedding == [0.5, 0.5]
        assert entry.importance == 0.9
        assert entry.emotional_valence == 0.3
        assert entry.source_session == "s1"

    def test_add_evicts_oldest_when_full(self):
        stm = ShortTermMemory(max_items=3)
        e1 = stm.add("first")
        e2 = stm.add("second")
        e3 = stm.add("third")
        e4 = stm.add("fourth")
        assert len(stm) == 3
        assert e1.id not in stm._by_id
        assert e4.id in stm._by_id

    def test_search_by_embedding(self):
        stm = ShortTermMemory()
        stm.add("python coding", embedding=[1.0, 0.0, 0.0])
        stm.add("cooking recipes", embedding=[0.0, 1.0, 0.0])
        results = stm.search([1.0, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert "python" in results[0][0].text

    def test_search_empty(self):
        stm = ShortTermMemory()
        results = stm.search([0.5, 0.5])
        assert results == []

    def test_get(self):
        stm = ShortTermMemory()
        entry = stm.add("test")
        found = stm.get(entry.id)
        assert found is entry

    def test_get_nonexistent(self):
        stm = ShortTermMemory()
        assert stm.get("nope") is None

    def test_evict_expired(self):
        stm = ShortTermMemory(ttl_hours=0.0)
        entry = stm.add("old message")
        expired = stm.evict_expired()
        assert len(expired) == 1
        assert expired[0].id == entry.id

    def test_evict_expired_none_fresh(self):
        stm = ShortTermMemory(ttl_hours=999.0)
        stm.add("fresh message")
        expired = stm.evict_expired()
        assert len(expired) == 0

    def test_all_entries(self):
        stm = ShortTermMemory()
        stm.add("first")
        stm.add("second")
        all_e = stm.all_entries
        assert len(all_e) == 2

    def test_all_entries_empty(self):
        stm = ShortTermMemory()
        assert stm.all_entries == []


class TestCoreMemory:
    def test_init_default(self):
        core = CoreMemory()
        assert core.max_entries == 50
        assert len(core) == 0

    def test_init_custom(self):
        core = CoreMemory(max_entries=10)
        assert core.max_entries == 10

    def test_set_new_entry(self):
        core = CoreMemory()
        entry = core.set("key1", "value1", confidence=0.7, tags=["test"])
        assert entry.tier == MemoryTier.CORE
        assert entry.text == "value1"
        assert entry.metadata["key"] == "key1"
        assert entry.metadata["confidence"] == 0.7
        assert len(core) == 1

    def test_set_reinforces_existing(self):
        core = CoreMemory()
        e1 = core.set("key1", "value1")
        old_stability = e1.stability
        e2 = core.set("key1", "value2")
        assert e1.id == e2.id  # same entry
        assert e2.text == "value2"
        assert e2.stability > old_stability
        assert e2.reinforcement > 0
        assert e2.access_count == 1

    def test_set_evicts_lowest_confidence(self):
        core = CoreMemory(max_entries=2)
        core.set("k1", "v1", confidence=0.3)
        core.set("k2", "v2", confidence=0.8)
        core.set("k3", "v3", confidence=0.5)
        assert len(core) <= 2
        # k1 (confidence 0.3) should be evicted
        assert core.get("k1") is None

    def test_get(self):
        core = CoreMemory()
        core.set("key1", "value1")
        found = core.get("key1")
        assert found is not None
        assert found.text == "value1"

    def test_get_nonexistent(self):
        core = CoreMemory()
        assert core.get("nope") is None

    def test_search(self):
        core = CoreMemory()
        core.set("lang", "User prefers Python for backend development", confidence=0.8)
        core.set("editor", "User uses VS Code", confidence=0.6)
        results = core.search("Python backend", top_k=2)
        assert len(results) >= 1

    def test_search_no_match(self):
        core = CoreMemory()
        core.set("key1", "unrelated text here", confidence=0.5)
        results = core.search("xyzzy nothing")
        assert results == []

    def test_to_dict(self):
        core = CoreMemory()
        core.set("k1", "v1")
        core.set("k2", "v2")
        d = core.to_dict()
        assert d == {"k1": "v1", "k2": "v2"}

    def test_all_entries(self):
        core = CoreMemory()
        core.set("k1", "v1")
        core.set("k2", "v2")
        entries = core.all_entries
        assert len(entries) == 2


class TestTieredStorage:
    def test_init_default(self):
        ts = TieredStorage()
        assert ts._path == ""
        assert ts.size == 0

    def test_init_with_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            ts = TieredStorage(path=path)
            assert ts._path == path

    def test_offload_and_load(self):
        ts = TieredStorage()
        ts.offload("a1", {"id": "a1", "text": "test data"})
        assert ts.size == 1
        data = ts.load("a1")
        assert data is not None
        assert data["text"] == "test data"
        assert "offloaded_at" in data

    def test_remove(self):
        ts = TieredStorage()
        ts.offload("a1", {"id": "a1", "text": "test"})
        ts.remove("a1")
        assert ts.size == 0
        assert ts.load("a1") is None

    def test_remove_nonexistent(self):
        ts = TieredStorage()
        ts.remove("nope")  # should not raise

    def test_contains(self):
        ts = TieredStorage()
        ts.offload("a1", {"id": "a1"})
        assert ts.contains("a1") is True
        assert ts.contains("nope") is False

    def test_ids(self):
        ts = TieredStorage()
        ts.offload("a1", {"id": "a1"})
        ts.offload("a2", {"id": "a2"})
        ids = ts.ids()
        assert set(ids) == {"a1", "a2"}

    def test_flush_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            ts = TieredStorage(path=path)
            ts.offload("a1", {"id": "a1", "text": "persisted"})
            ts.flush()
            assert os.path.exists(path)
            # Load from disk into a new storage
            ts2 = TieredStorage(path=path)
            data = ts2.load("a1")
            assert data is not None
            assert data["text"] == "persisted"

    def test_flush_no_path(self):
        ts = TieredStorage()
        ts.offload("a1", {"id": "a1"})
        ts.flush()  # should not raise (no path)

    def test_flush_not_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            ts = TieredStorage(path=path)
            ts.flush()  # should not raise, nothing to flush

    def test_compact(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            path = os.path.join(tmp, "store.json")
            ts = TieredStorage(path=path)
            ts.offload("a1", {"id": "a1"})
            ts.offload("a2", {"id": "a2"})
            ts.remove("a1")
            count = ts.compact()
            assert count == 1  # before compact: 1 item remaining

    def test_reload_discards_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            ts = TieredStorage(path=path)
            ts.offload("a1", {"id": "a1", "text": "saved"})
            ts.flush()
            ts.offload("a2", {"id": "a2", "text": "unsaved"})
            ts.reload()
            assert ts.load("a1") is not None
            assert ts.load("a2") is None  # lost on reload

    def test_load_from_disk_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            ts1 = TieredStorage(path=path)
            ts1.offload("a1", {"id": "a1", "text": "auto load"})
            ts1.flush()
            # New storage should auto-load
            ts2 = TieredStorage(path=path)
            data = ts2.load("a1")
            assert data is not None

    def test_load_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            with open(path, "w") as f:
                f.write("not json{{")
            ts = TieredStorage(path=path)
            data = ts.load("anything")
            assert data is None

    def test_stats(self):
        ts = TieredStorage()
        ts.offload("a1", {"id": "a1", "text": "hello world"})
        s = ts.stats
        assert s["cold_anchors"] == 1
        assert s["total_text_bytes"] > 0
        assert s["dirty"] is True

    def test_stats_empty(self):
        ts = TieredStorage()
        s = ts.stats
        assert s["cold_anchors"] == 0


class TestOffloadAnchorToCold:
    def test_offload(self):
        anchor = Anchor(id="a1", text="test anchor", tags=["t1"],
                       source_session="s1")
        anchor.vector.importance = 0.8
        anchor.vector.emotional_valence = 0.4
        cold = TieredStorage()
        data = offload_anchor_to_cold(anchor, cold)
        assert data["text"] == "test anchor"
        assert data["importance"] == 0.8
        assert cold.contains("a1") is True


class TestCosineSim:
    def test_identical(self):
        result = _cosine_sim([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result == pytest.approx(1.0, abs=0.001)

    def test_orthogonal(self):
        result = _cosine_sim([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(0.0, abs=0.001)

    def test_opposite(self):
        result = _cosine_sim([1.0, 0.0], [-1.0, 0.0])
        assert result == pytest.approx(-1.0, abs=0.001)

    def test_zero_vector(self):
        result = _cosine_sim([0.0, 0.0], [1.0, 2.0])
        assert result == 0.0
