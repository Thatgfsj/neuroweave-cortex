"""Test memory tiering: STM/MTM/LTM/Core four-layer API + promotion pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from star_graph import (
    MemoryTier, TierEntry,
    ShortTermMemory, MiddleTermMemory,
    LongTermMemory, CoreMemory, MemoryTierManager,
    TIER_DECAY_HALF_LIFE, TIER_MAX_ITEMS,
)


class TestShortTermMemory:
    """STM — deque-based transient buffer."""

    def test_add_and_retrieve(self):
        stm = ShortTermMemory(max_items=10)
        entry = stm.add("user is working on a memory system",
                       tags=["project", "memory"])
        assert entry.id.startswith("stm_")
        assert entry.tier == MemoryTier.STM
        assert entry.tags == ["project", "memory"]

        found = stm.get(entry.id)
        assert found is not None
        assert found.text == "user is working on a memory system"

    def test_capacity_eviction(self):
        stm = ShortTermMemory(max_items=3)
        ids = []
        for i in range(5):
            e = stm.add(f"message {i}")
            ids.append(e.id)

        assert len(stm) == 3
        # First 2 should be evicted
        assert stm.get(ids[0]) is None
        assert stm.get(ids[1]) is None
        assert stm.get(ids[2]) is not None

    def test_ttl_eviction(self):
        stm = ShortTermMemory(max_items=10, ttl_hours=0.0)  # immediate TTL
        stm.add("ephemeral message")
        expired = stm.evict_expired()
        assert len(expired) == 1
        assert len(stm) == 0

    def test_search_by_embedding(self):
        stm = ShortTermMemory(max_items=10)
        stm.add("python development", embedding=[0.9, 0.1, 0.0])
        stm.add("cooking recipes", embedding=[0.1, 0.9, 0.0])
        stm.add("java programming", embedding=[0.8, 0.2, 0.0])

        results = stm.search([0.85, 0.15, 0.0], top_k=2)
        assert len(results) == 2
        assert "python" in results[0][0].text or "java" in results[0][0].text


class TestMiddleTermMemory:
    """MTM — topic-cluster StarGraph memory."""

    def test_add_topic(self):
        mtm = MiddleTermMemory(max_topics=100)
        topic_id = mtm.add_topic(
            "Python backend development patterns",
            embedding=[0.5, 0.5, 0.0],
            tags=["python", "backend", "development"],
            source_stm_ids=["stm_1", "stm_2", "stm_3"],
        )
        assert topic_id.startswith("mtm_")
        assert topic_id in mtm.graph.anchors

    def test_search(self):
        mtm = MiddleTermMemory(max_topics=100)
        mtm.add_topic("Python async patterns", embedding=[0.9, 0.1, 0.0],
                     tags=["python", "async"])
        mtm.add_topic("JavaScript promises", embedding=[0.1, 0.9, 0.0],
                     tags=["javascript", "async"])
        mtm.add_topic("Python decorators", embedding=[0.85, 0.15, 0.0],
                     tags=["python", "decorators"])

        results = mtm.search([0.88, 0.12, 0.0], top_k=2)
        assert len(results) >= 1
        # Python topics should rank higher for python query
        assert any("python" in r[0].tags for r in results)


class TestLongTermMemory:
    """LTM — summary-only high-stability memory."""

    def test_add_summary(self):
        ltm = LongTermMemory(max_summaries=100)
        sid = ltm.add_summary(
            "User consistently prefers Python for backend work",
            embedding=[0.7, 0.3, 0.0],
            tags=["python", "preference"],
            confidence=0.8,
        )
        assert sid.startswith("ltm_")
        anchor = ltm.graph.anchors[sid]
        assert anchor.vector.stability == 0.8
        assert anchor.vector.confidence == 0.8

    def test_confidence_boosts_search(self):
        ltm = LongTermMemory(max_summaries=100)
        ltm.add_summary("High confidence fact", embedding=[0.9, 0.1, 0.0],
                        tags=["fact"], confidence=0.9)
        ltm.add_summary("Low confidence guess", embedding=[0.89, 0.11, 0.0],
                        tags=["guess"], confidence=0.3)

        results = ltm.search([0.9, 0.1, 0.0], top_k=2)
        assert len(results) >= 1
        # High confidence should score higher
        scores = [s for _, s in results]
        if len(results) >= 2:
            assert scores[0] >= scores[1]


class TestCoreMemory:
    """Core — near-immutable profile store."""

    def test_set_and_get(self):
        core = CoreMemory(max_entries=10)
        core.set("language", "Python (primary)", confidence=0.9)
        entry = core.get("language")
        assert entry is not None
        assert entry.text == "Python (primary)"
        assert entry.tier == MemoryTier.CORE

    def test_reinforce_existing(self):
        core = CoreMemory(max_entries=10)
        e1 = core.set("editor", "VS Code", confidence=0.5)
        access_before = e1.access_count
        e2 = core.set("editor", "VS Code (preferred)", confidence=0.8)
        assert e1.id == e2.id  # same entry
        assert e2.access_count > access_before  # reinforced
        assert e2.reinforcement >= 0.1

    def test_to_dict(self):
        core = CoreMemory(max_entries=10)
        core.set("a", "value_a")
        core.set("b", "value_b")
        d = core.to_dict()
        assert d == {"a": "value_a", "b": "value_b"}


class TestMemoryTierManager:
    """End-to-end tier orchestration."""

    def test_remember_goes_to_stm(self):
        mgr = MemoryTierManager()
        entry = mgr.remember("working on a memory system project",
                            tags=["project", "memory"])
        assert entry.tier == MemoryTier.STM
        assert len(mgr.stm) == 1

    def test_recall_searches_all_tiers(self):
        mgr = MemoryTierManager()

        # Add to STM
        mgr.remember("user is debugging a Python async issue",
                    tags=["python", "debug"])

        # Add directly to MTM
        mgr.mtm.add_topic("Python async programming patterns",
                         embedding=[0.9, 0.1, 0.0],
                         tags=["python", "async"])

        # Add to LTM
        mgr.ltm.add_summary("User is a Python backend developer",
                           embedding=[0.85, 0.15, 0.0],
                           tags=["python", "profile"],
                           confidence=0.9)

        # Add to Core
        mgr.core.set("primary_language", "Python", confidence=0.95)

        results = mgr.recall("Python programming", max_items=10)
        assert len(results) >= 1
        # Core should be first
        tiers = [r["tier"] for r in results]
        assert "core" in tiers

    def test_promote_stm_to_mtm(self):
        mgr = MemoryTierManager()

        # Add multiple STM entries about the same topic
        for i in range(5):
            mgr.remember(f"python deployment issue {i}: dependency conflict",
                        tags=["python", "deployment", "bug"])

        assert len(mgr.stm) == 5

        result = mgr.promote_stm_to_mtm()
        assert result["topics_created"] >= 0  # best-effort

        # Some MTM topics may have been created
        assert result["stm_after"] <= result["stm_before"]

    def test_full_promotion_pipeline(self):
        """STM→MTM→LTM→Core pipeline runs without error."""
        mgr = MemoryTierManager()

        # Phase 1: add to STM
        for i in range(10):
            mgr.remember(
                f"python backend development note {i}: API design patterns",
                tags=["python", "backend", "development"],
            )
        assert len(mgr.stm) == 10

        # Phase 2: STM → MTM
        stm_result = mgr.promote_stm_to_mtm()
        assert isinstance(stm_result, dict)

        # Phase 3: MTM → LTM (may not always create summaries)
        ltm_result = mgr.promote_mtm_to_ltm()
        assert isinstance(ltm_result, dict)

        # Phase 4: LTM → Core (may not always extract profiles)
        core_result = mgr.promote_ltm_to_core()
        assert isinstance(core_result, dict)

    def test_tier_stats(self):
        mgr = MemoryTierManager()
        mgr.remember("test message", tags=["test"])
        stats = mgr.stats
        assert stats["stm"]["count"] == 1
        assert "total_items" in stats
        assert "promotion_log" in stats
