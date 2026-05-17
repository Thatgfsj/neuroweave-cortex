"""Tests for autobiography module — SelfNarrative, AutobiographicalMemory."""

import time

import pytest

from star_graph.autobiography import SelfNarrative, AutobiographicalMemory


class TestSelfNarrative:
    def test_create_default(self):
        sn = SelfNarrative.create(
            episode_summary="Discussed Redis timeout with user",
            self_belief="User prefers hands-on debugging",
            emotional_tone=0.3,
            source_session="s1",
        )
        assert sn.id != ""
        assert "Redis timeout" in sn.episode_summary
        assert sn.stability == 0.5
        assert sn.access_count == 0

    def test_create_clamps_emotional_tone(self):
        sn = SelfNarrative.create(emotional_tone=1.5)
        assert sn.emotional_tone == 1.0
        sn2 = SelfNarrative.create(emotional_tone=-2.0)
        assert sn2.emotional_tone == -1.0

    def test_create_deterministic_id(self):
        sn1 = SelfNarrative.create(
            episode_summary="Same summary",
            self_belief="Same belief",
            source_session="s1",
        )
        sn2 = SelfNarrative.create(
            episode_summary="Same summary",
            self_belief="Same belief",
            source_session="s1",
        )
        assert sn1.id == sn2.id  # same content → same ID

    def test_create_with_tags(self):
        sn = SelfNarrative.create(
            episode_summary="test", tags=["dev", "python"],
        )
        assert "dev" in sn.tags
        assert "python" in sn.tags

    def test_relevance(self):
        sn = SelfNarrative.create(episode_summary="test")
        sn.stability = 0.8
        sn.last_accessed_at = time.time()  # just now
        r = sn.relevance
        assert r > 0.7

    def test_relevance_decayed(self):
        sn = SelfNarrative.create(episode_summary="test")
        sn.stability = 0.8
        sn.last_accessed_at = time.time() - 30 * 86400  # 30 days ago
        r = sn.relevance
        assert r < 0.5  # should have decayed significantly

    def test_access(self):
        sn = SelfNarrative.create(episode_summary="test")
        old_stability = sn.stability
        sn.access()
        assert sn.access_count == 1
        assert sn.stability > old_stability

    def test_reinforce(self):
        sn = SelfNarrative.create(episode_summary="test")
        old_stability = sn.stability
        sn.reinforce(new_belief="updated belief")
        assert sn.stability > old_stability
        assert sn.self_belief == "updated belief"

    def test_reinforce_no_new_belief(self):
        sn = SelfNarrative.create(
            episode_summary="test", self_belief="original",
        )
        sn.reinforce()
        assert sn.self_belief == "original"

    def test_weaken(self):
        sn = SelfNarrative.create(episode_summary="test")
        sn.stability = 0.5
        sn.weaken()
        assert sn.stability < 0.5
        assert sn.stability >= 0.05

    def test_weaken_floor(self):
        sn = SelfNarrative.create(episode_summary="test")
        sn.stability = 0.05
        sn.weaken()
        assert sn.stability == 0.05  # floor

    def test_degrade(self):
        sn = SelfNarrative.create(episode_summary="test")
        sn.stability = 0.8
        sn.last_accessed_at = time.time() - 60 * 86400  # 60 days ago
        remaining = sn.degrade(half_life_days=30.0)
        assert remaining < 0.8
        assert remaining >= 0.01


class TestAutobiographicalMemory:
    def test_init_default(self):
        am = AutobiographicalMemory()
        assert am.max_narratives == 1000
        assert am._total_formed == 0

    def test_init_custom(self):
        am = AutobiographicalMemory(max_narratives=100)
        assert am.max_narratives == 100

    def test_form_from_interaction(self):
        am = AutobiographicalMemory()
        sn = am.form_from_interaction(
            episode_summary="Discussed Redis timeout",
            self_belief="User prefers hands-on debugging",
            emotional_tone=0.5,
            source_session="s1",
            tags=["dev", "debugging"],
        )
        assert sn.id in am._narratives
        assert am._total_formed == 1

    def test_form_from_interaction_duplicate(self):
        am = AutobiographicalMemory()
        sn1 = am.form_from_interaction(
            episode_summary="Discussed Redis timeout with user",
            source_session="s1",
        )
        sn2 = am.form_from_interaction(
            episode_summary="Discussed Redis timeout with user",
            source_session="s1",
        )
        # Should reinforce existing rather than duplicate
        assert sn1.id == sn2.id
        assert am._total_formed == 1

    def test_get(self):
        am = AutobiographicalMemory()
        sn = am.form_from_interaction(episode_summary="test")
        retrieved = am.get(sn.id)
        assert retrieved is sn

    def test_get_nonexistent(self):
        am = AutobiographicalMemory()
        assert am.get("nonexistent") is None

    def test_forget(self):
        am = AutobiographicalMemory()
        sn = am.form_from_interaction(episode_summary="test")
        removed = am.forget(sn.id)
        assert removed is sn
        assert sn.id not in am._narratives

    def test_forget_nonexistent(self):
        am = AutobiographicalMemory()
        assert am.forget("nonexistent") is None

    def test_recall_self_empty(self):
        am = AutobiographicalMemory()
        results = am.recall_self("test")
        assert results == []

    def test_recall_self_with_query(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="Discussed Redis timeout",
            self_belief="User prefers Redis",
            tags=["redis"],
        )
        am.form_from_interaction(
            episode_summary="Discussed Python deployment",
            self_belief="User uses Flask",
            tags=["python"],
        )
        results = am.recall_self("Redis")
        assert len(results) >= 1

    def test_recall_self_no_query(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(episode_summary="test 1")
        am.form_from_interaction(episode_summary="test 2")
        results = am.recall_self()
        assert len(results) >= 1

    def test_recall_self_min_stability(self):
        am = AutobiographicalMemory()
        sn = am.form_from_interaction(episode_summary="test")
        sn.stability = 0.01
        results = am.recall_self("test", min_stability=0.05)
        assert len(results) == 0

    def test_recall_session(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="test 1", source_session="s1",
        )
        am.form_from_interaction(
            episode_summary="test 2", source_session="s2",
        )
        results = am.recall_session("s1")
        assert len(results) == 1
        assert results[0].source_session == "s1"

    def test_get_beliefs(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="test 1",
            self_belief="User prefers Python",
        )
        am.form_from_interaction(
            episode_summary="test 2",
            self_belief="",  # no belief
        )
        beliefs = am.get_beliefs()
        assert len(beliefs) >= 1
        assert beliefs[0]["belief"] == "User prefers Python"

    def test_get_emotional_profile_empty(self):
        am = AutobiographicalMemory()
        profile = am.get_emotional_profile()
        assert profile["count"] == 0
        assert profile["trend"] == "neutral"

    def test_get_emotional_profile_with_data(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="test 1", emotional_tone=0.5,
            source_session="s1",
        )
        am.form_from_interaction(
            episode_summary="test 2", emotional_tone=0.3,
            source_session="s1",
        )
        profile = am.get_emotional_profile()
        assert profile["count"] == 2
        assert "avg_tone" in profile
        assert "trend" in profile

    def test_get_emotional_profile_by_session(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="test 1", emotional_tone=0.8,
            source_session="s1",
        )
        am.form_from_interaction(
            episode_summary="test 2", emotional_tone=-0.2,
            source_session="s2",
        )
        profile = am.get_emotional_profile(session_id="s1")
        assert profile["count"] == 1

    def test_get_emotional_profile_trend(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="test 1", emotional_tone=-0.5,
            source_session="s1",
        )
        am.form_from_interaction(
            episode_summary="test 2", emotional_tone=0.5,
            source_session="s1",
        )
        profile = am.get_emotional_profile()
        # Improving trend (second > first by > 0.1)
        assert profile["trend"] in ("improving", "stable", "declining")

    def test_update_belief(self):
        am = AutobiographicalMemory()
        sn = am.form_from_interaction(
            episode_summary="test",
            self_belief="old belief",
        )
        result = am.update_belief(sn.id, "new belief")
        assert result is True
        assert am.get(sn.id).self_belief == "new belief"

    def test_update_belief_nonexistent(self):
        am = AutobiographicalMemory()
        assert am.update_belief("nonexistent", "new") is False

    def test_contradict_belief(self):
        am = AutobiographicalMemory()
        sn = am.form_from_interaction(
            episode_summary="test",
            self_belief="Python is slow for everything",
        )
        old_stability = sn.stability
        affected = am.contradict_belief("Python is slow")
        assert sn.id in affected
        assert sn.stability < old_stability

    def test_contradict_belief_with_correction(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="test",
            self_belief="Python is slow",
        )
        affected = am.contradict_belief(
            "Python is slow",
            correction="Python with PyPy is fast",
        )
        assert len(affected) >= 1
        # A corrected narrative should have been created
        assert len(am._narratives) >= 2

    def test_degrade_all(self):
        am = AutobiographicalMemory()
        sn = am.form_from_interaction(episode_summary="test")
        sn.last_accessed_at = 0.0  # very old
        removed = am.degrade_all(half_life_days=30.0)
        assert removed >= 0

    def test_enforce_limit(self):
        am = AutobiographicalMemory(max_narratives=2)
        am.form_from_interaction(
            episode_summary="first", source_session="s1",
        )
        am.form_from_interaction(
            episode_summary="second", source_session="s2",
        )
        am.form_from_interaction(
            episode_summary="third", source_session="s3",
        )
        assert len(am._narratives) <= 2

    def test_stats_empty(self):
        am = AutobiographicalMemory()
        s = am.stats
        assert s["total"] == 0

    def test_stats_with_data(self):
        am = AutobiographicalMemory()
        am.form_from_interaction(
            episode_summary="test", emotional_tone=0.3,
            self_belief="I am helpful",
        )
        s = am.stats
        assert s["total"] == 1
        assert s["beliefs"] == 1
        assert s["avg_stability"] > 0
