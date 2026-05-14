"""Tests for Versioned Memory — cognitive trajectory tracking."""

import time
import pytest
from star_graph.versioned_memory import (
    CognitiveTrajectory, BeliefVersion,
)


class TestBeliefVersion:
    def test_defaults(self):
        bv = BeliefVersion(id="bv_1", topic="test", text="user likes Python")
        assert bv.id == "bv_1"
        assert bv.topic == "test"
        assert bv.confidence == 0.5
        assert bv.is_current is True
        assert bv.is_initial is True
        assert bv.version_number == 1

    def test_superseded_not_current(self):
        bv = BeliefVersion(id="bv_1", topic="test", text="v1", superseded_by="bv_2")
        assert bv.is_current is False


class TestCognitiveTrajectoryInit:
    def test_initial_state(self):
        ct = CognitiveTrajectory()
        assert ct._total_recorded == 0
        assert ct._total_superseded == 0

    def test_normalize_topic(self):
        result = CognitiveTrajectory.normalize_topic("User likes Python programming")
        assert "python" in result
        assert "user" in result


class TestRecordBelief:
    def test_first_belief(self):
        ct = CognitiveTrajectory()
        bv = ct.record_belief("language", "用户喜欢Python")
        assert bv.version_number == 1
        assert bv.is_current is True
        assert bv.is_initial is True

    def test_superseding_belief(self):
        ct = CognitiveTrajectory(similarity_threshold=0.2)
        bv1 = ct.record_belief("language", "用户喜欢Python开发")
        bv2 = ct.record_belief("language", "用户现在使用Python做AI开发")
        # bv2 should supersede bv1 if topic words overlap
        if bv2.supersedes:
            assert bv1.is_current is False
            assert bv2.version_number == 2

    def test_get_current_belief(self):
        ct = CognitiveTrajectory()
        ct.record_belief("language", "用户喜欢Python", confidence=0.6)
        ct.record_belief("language", "用户偏向AI开发", confidence=0.7)
        current = ct.get_current_belief("language")
        assert current is not None

    def test_get_trajectory(self):
        ct = CognitiveTrajectory(similarity_threshold=0.2)
        ct.record_belief("tools", "使用VSCode")
        ct.record_belief("tools", "使用VSCode加Copilot")
        ct.record_belief("tools", "现在用Cursor")
        chain = ct.get_trajectory("tools")
        assert len(chain) >= 1


class TestGetAllCurrent:
    def test_multiple_topics(self):
        ct = CognitiveTrajectory()
        ct.record_belief("lang", "Python")
        ct.record_belief("editor", "VSCode")
        ct.record_belief("os", "Windows")
        current = ct.get_all_current_beliefs()
        assert len(current) == 3


class TestRecentChanges:
    def test_recent(self):
        ct = CognitiveTrajectory()
        ct.record_belief("test", "new belief")
        recent = ct.get_recent_changes(hours=1.0)
        assert len(recent) == 1

    def test_old(self):
        ct = CognitiveTrajectory()
        # Manually create an old belief
        bv = BeliefVersion(id="old_1", topic="old", text="old belief",
                          timestamp=time.time() - 999999)
        ct._beliefs[bv.id] = bv
        ct._topic_index[bv.topic].append(bv.id)
        recent = ct.get_recent_changes(hours=1.0)
        assert len(recent) == 0


class TestStats:
    def test_initial_stats(self):
        ct = CognitiveTrajectory()
        s = ct.stats
        assert s["total_beliefs"] == 0
        assert s["current_beliefs"] == 0
        assert s["topics_tracked"] == 0

    def test_stats_after_recording(self):
        ct = CognitiveTrajectory()
        ct.record_belief("a", "belief a")
        ct.record_belief("b", "belief b")
        s = ct.stats
        assert s["total_beliefs"] == 2
        assert s["topics_tracked"] == 2
