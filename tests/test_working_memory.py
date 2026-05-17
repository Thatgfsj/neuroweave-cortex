"""Tests for working_memory module — WorkingMemory and WorkingMemoryEntry."""

import time

import pytest

from star_graph.working_memory import WorkingMemory, WorkingMemoryEntry, _extract_keys


class TestWorkingMemoryEntry:
    def test_default_values(self):
        e = WorkingMemoryEntry(text="hello")
        assert e.text == "hello"
        assert e.importance == 0.5
        assert e.tags == []
        assert e.access_count == 0

    def test_age_seconds(self):
        e = WorkingMemoryEntry(text="hello")
        assert e.age_seconds >= 0

    def test_idle_seconds(self):
        e = WorkingMemoryEntry(text="hello")
        assert e.idle_seconds >= 0

    def test_priority(self):
        e = WorkingMemoryEntry(text="hello", importance=0.8)
        p = e.priority
        assert 0.0 < p < 1.0

    def test_touch_increases_access(self):
        e = WorkingMemoryEntry(text="hello")
        old_count = e.access_count
        e.touch()
        assert e.access_count == old_count + 1

    def test_high_importance_higher_priority(self):
        e_high = WorkingMemoryEntry(text="important", importance=0.9)
        e_low = WorkingMemoryEntry(text="trivial", importance=0.1)
        assert e_high.priority > e_low.priority


class TestExtractKeys:
    def test_possessive_pattern(self):
        keys = _extract_keys("Alice's birthday is tomorrow", [])
        assert "alice-birthday" in keys

    def test_kv_pattern(self):
        keys = _extract_keys("color is blue", [])
        assert "color-blue" in keys

    def test_tags_as_keys(self):
        keys = _extract_keys("some text", ["important", "project-alpha"])
        assert "tag:important" in keys
        # tag normalization replaces spaces with underscores, hyphens are preserved
        assert any(k.startswith("tag:project") for k in keys)

    def test_fallback_first_two_words(self):
        keys = _extract_keys("hello world", [])
        assert "hello-world" in keys

    def test_deduplication(self):
        keys = _extract_keys("hello world hello world hello world", [])
        assert len(keys) == len(set(keys))

    def test_max_five_keys(self):
        keys = _extract_keys(
            "Alice's birthday is tomorrow, color is blue, mood is happy, "
            "Bob's car is red, status is active, energy is high",
            ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"],
        )
        assert len(keys) <= 5


class TestWorkingMemory:
    def test_init_defaults(self):
        wm = WorkingMemory()
        assert wm.max_capacity == 15
        assert wm.ttl_seconds == 3600
        assert wm.size == 0

    def test_add_single_item(self):
        wm = WorkingMemory()
        entry = wm.add("hello world")
        assert wm.size == 1
        assert entry.text == "hello world"

    def test_add_with_embedding(self):
        wm = WorkingMemory()
        entry = wm.add("test", embedding=[0.1, 0.2, 0.3])
        assert entry.embedding == [0.1, 0.2, 0.3]

    def test_get_all(self):
        wm = WorkingMemory()
        wm.add("first")
        wm.add("second")
        entries = wm.get_all()
        assert len(entries) == 2

    def test_get_all_max_items(self):
        wm = WorkingMemory()
        for i in range(5):
            wm.add(f"item {i}")
        entries = wm.get_all(max_items=3)
        assert len(entries) == 3

    def test_is_full(self):
        wm = WorkingMemory(max_capacity=2)
        wm.add("first")
        assert not wm.is_full
        wm.add("second")
        assert wm.is_full

    def test_eviction_on_overflow(self):
        wm = WorkingMemory(max_capacity=2)
        wm.add("low", importance=0.1)
        wm.add("mid", importance=0.5)
        wm.add("high", importance=0.9)
        # The lowest priority should be evicted
        assert wm.size <= 2

    def test_get_relevant_with_text(self):
        wm = WorkingMemory()
        wm.add("redis timeout fix")
        wm.add("unrelated topic")
        results = wm.get_relevant(query_text="redis timeout")
        assert len(results) > 0

    def test_get_relevant_with_embedding(self):
        wm = WorkingMemory()
        wm.add("test a", embedding=[1.0, 0.0, 0.0])
        wm.add("test b", embedding=[0.0, 1.0, 0.0])
        results = wm.get_relevant(query_embedding=[1.0, 0.0, 0.0])
        assert len(results) > 0
        # First result should be "test a"
        assert results[0][0].text == "test a"

    def test_get_relevant_min_score(self):
        wm = WorkingMemory()
        wm.add("unrelated text")
        results = wm.get_relevant(query_text="redis timeout", min_score=0.9)
        assert len(results) == 0

    def test_get_exact(self):
        wm = WorkingMemory()
        wm.add("Alice's birthday is today", importance=0.8)
        results = wm.get_exact("alice-birthday")
        assert len(results) == 1
        assert "Alice" in results[0].text

    def test_get_exact_nonexistent(self):
        wm = WorkingMemory()
        results = wm.get_exact("nonexistent-key")
        assert results == []

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("item1")
        wm.add("item2")
        wm.clear()
        assert wm.size == 0

    def test_clear_session(self):
        wm = WorkingMemory()
        wm.add("session 1 item", source_session="s1")
        wm.add("session 2 item", source_session="s2")
        wm.clear_session("s1")
        assert wm.size == 1
        entries = wm.get_all()
        assert entries[0].source_session == "s2"

    def test_summary_empty(self):
        wm = WorkingMemory()
        assert "empty" in wm.summary

    def test_summary_with_items(self):
        wm = WorkingMemory()
        wm.add("hello world from the test suite")
        summary = wm.summary
        assert "1/" in summary

    def test_expired_items_removed(self):
        wm = WorkingMemory(ttl_seconds=-1)  # everything expired
        wm.add("expired item")
        assert wm.size == 0

    def test_promote(self):
        from star_graph.runtime import MemoryRuntime
        rt = MemoryRuntime()
        wm = rt.working_memory
        entry = wm.add("learned important fact", importance=0.8)
        aid = wm.promote(entry, rt)
        assert aid
        # Item should be removed from working memory
        assert wm.size == 0
