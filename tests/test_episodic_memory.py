"""Tests for Episodic Memory — time + context + state event streams."""

import time
import pytest
from star_graph.episodic_memory import (
    EpisodicMemory, EpisodeNode, SessionSummary,
)


class TestEpisodeNode:
    def test_defaults(self):
        ep = EpisodeNode(id="ep_1", session_id="s1")
        assert ep.id == "ep_1"
        assert ep.session_id == "s1"
        assert ep.emotional_valence == 0.0
        assert ep.importance == 0.5
        assert ep.is_first_in_session is True

    def test_linked_episodes(self):
        ep1 = EpisodeNode(id="ep_1", session_id="s1")
        ep2 = EpisodeNode(id="ep_2", session_id="s1", prev_episode_id="ep_1")
        ep1.next_episode_id = "ep_2"
        assert ep1.is_first_in_session is True
        assert ep2.is_first_in_session is False


class TestEpisodicMemoryInit:
    def test_initial_state(self):
        em = EpisodicMemory()
        assert em._total_recorded == 0
        assert len(em._episodes) == 0
        assert len(em._session_index) == 0

    def test_custom_max(self):
        em = EpisodicMemory(max_episodes_per_session=100)
        assert em.max_episodes_per_session == 100


class TestRecordEpisode:
    def test_record_basic(self):
        em = EpisodicMemory()
        ep = em.record_episode(
            session_id="s1",
            summary="Discussed Redis bug",
            emotional_valence=-0.3,
            tags=["redis", "debug"],
        )
        assert ep.session_id == "s1"
        assert ep.emotional_valence == -0.3
        assert "redis" in ep.tags

    def test_records_chain(self):
        em = EpisodicMemory()
        ep1 = em.record_episode(session_id="s1", summary="First")
        ep2 = em.record_episode(session_id="s1", summary="Second")
        # ep2 should link back to ep1
        assert ep2.prev_episode_id == ep1.id
        assert em._episodes[ep1.id].next_episode_id == ep2.id

    def test_enforces_max_per_session(self):
        em = EpisodicMemory(max_episodes_per_session=3)
        for i in range(5):
            em.record_episode(session_id="s1", summary=f"Episode {i}")
        session_eps = em.recall_session("s1")
        assert len(session_eps) <= 3


class TestRecallSession:
    def test_empty_session(self):
        em = EpisodicMemory()
        assert em.recall_session("nonexistent") == []

    def test_chronological_order(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="A", timestamp=100)
        em.record_episode(session_id="s1", summary="B", timestamp=200)
        em.record_episode(session_id="s1", summary="C", timestamp=300)
        eps = em.recall_session("s1")
        assert len(eps) == 3
        assert eps[0].summary == "A"
        assert eps[2].summary == "C"


class TestContextualRecall:
    def test_time_filter(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="old",
                         timestamp=time.time() - 999999)
        em.record_episode(session_id="s1", summary="recent")
        results = em.contextual_recall(time_range_hours=1.0)
        assert len(results) == 1
        assert results[0].summary == "recent"

    def test_tag_filter(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="redis fix", tags=["redis"])
        em.record_episode(session_id="s1", summary="python refactor", tags=["python"])
        results = em.contextual_recall(tags=["redis"])
        assert len(results) == 1
        assert "redis" in results[0].summary

    def test_importance_filter(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="low", importance=0.2)
        em.record_episode(session_id="s1", summary="high", importance=0.8)
        results = em.contextual_recall(min_importance=0.5)
        assert len(results) == 1
        assert results[0].summary == "high"

    def test_query_filter(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="redis timeout issue",
                         action="fixed connection pool")
        em.record_episode(session_id="s1", summary="python linting",
                         action="added ruff config")
        results = em.contextual_recall(query="redis")
        assert len(results) == 1


class TestSummarization:
    def test_summarize_session(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="redis bug", importance=0.7,
                         tags=["redis", "debug"])
        em.record_episode(session_id="s1", summary="python refactor", importance=0.5,
                         tags=["python"])
        summary = em.summarize_session("s1")
        assert summary is not None
        assert summary.episode_count == 2
        assert "redis" in summary.key_topics or "python" in summary.key_topics

    def test_summarize_empty(self):
        em = EpisodicMemory()
        assert em.summarize_session("nonexistent") is None

    def test_get_summary(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="test")
        em.summarize_session("s1")
        s = em.get_session_summary("s1")
        assert s is not None
        assert s.session_id == "s1"


class TestStats:
    def test_initial_stats(self):
        em = EpisodicMemory()
        s = em.stats
        assert s["total_episodes"] == 0
        assert s["total_sessions"] == 0

    def test_stats_after_recording(self):
        em = EpisodicMemory()
        em.record_episode(session_id="s1", summary="a")
        em.record_episode(session_id="s1", summary="b")
        em.record_episode(session_id="s2", summary="c")
        s = em.stats
        assert s["total_episodes"] == 3
        assert s["total_sessions"] == 2
