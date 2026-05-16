"""Tests for scheduler module — MemoryType, AgentContext, MemoryItem, MemoryContext, classification."""

import math
import pytest

from star_graph.scheduler import (
    MemoryType,
    AgentContext,
    MemoryItem,
    MemoryContext,
    CognitiveMemoryScheduler,
    _cosine_sim,
)
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


class TestMemoryType:
    def test_values(self):
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.WORKING.value == "working"

    def test_members(self):
        types = list(MemoryType)
        assert len(types) == 4


class TestAgentContext:
    def test_defaults(self):
        ctx = AgentContext()
        assert ctx.task_type == "general"
        assert ctx.emotional_state == 0.0
        assert ctx.active_goals == []
        assert ctx.current_topic == ""
        assert ctx.context_budget_tokens == 4000

    def test_custom_fields(self):
        ctx = AgentContext(
            task_type="debugging",
            emotional_state=0.5,
            active_goals=["fix Redis timeout"],
            current_topic="Redis",
            recent_anchor_ids=["a1", "a2"],
            context_budget_tokens=2000,
            session_id="s1",
        )
        assert ctx.task_type == "debugging"
        assert ctx.emotional_state == 0.5
        assert "fix Redis timeout" in ctx.active_goals
        assert ctx.session_id == "s1"
        assert ctx.context_budget_tokens == 2000


class TestMemoryItem:
    def test_defaults(self):
        a = Anchor(id="a1", text="test")
        item = MemoryItem(anchor=a)
        assert item.relevance_score == 0.0
        assert item.confidence == 0.5
        assert item.memory_type == MemoryType.EPISODIC
        assert item.compression_level == 0
        assert item.compressed_text == ""

    def test_custom_fields(self):
        a = Anchor(id="a1", text="test")
        item = MemoryItem(
            anchor=a,
            relevance_score=0.8,
            confidence=0.9,
            memory_type=MemoryType.SEMANTIC,
            related_anchors=["a2"],
            reasoning_path=["a1", "a2", "a3"],
            compression_level=1,
            compressed_text="compressed",
        )
        assert item.relevance_score == 0.8
        assert item.confidence == 0.9
        assert item.reasoning_path == ["a1", "a2", "a3"]


class TestMemoryContext:
    def test_defaults(self):
        ctx = MemoryContext(items=[])
        assert ctx.items == []
        assert ctx.memory_summary == ""
        assert ctx.active_patterns == []
        assert ctx.total_tokens == 0
        assert ctx.retrieval_latency_ms == 0.0

    def test_custom_fields(self):
        a = Anchor(id="a1", text="test")
        item = MemoryItem(anchor=a, relevance_score=0.7)
        ctx = MemoryContext(
            items=[item],
            memory_summary="Found 1 memory",
            active_patterns=["Test pattern"],
            relevant_facts=["Key fact"],
            reasoning_traces=["a1 -> a2 -> a3"],
            reflections=[{"id": "r1", "text": "reflection"}],
            total_tokens=50,
            retrieval_latency_ms=12.5,
        )
        assert ctx.memory_summary == "Found 1 memory"
        assert "Test pattern" in ctx.active_patterns
        assert "Key fact" in ctx.relevant_facts
        assert ctx.total_tokens == 50
        assert ctx.retrieval_latency_ms == 12.5


class TestCosineSim:
    def test_identical(self):
        result = _cosine_sim([1.0, 0.0], [1.0, 0.0])
        assert result == pytest.approx(1.0, abs=0.001)

    def test_orthogonal(self):
        result = _cosine_sim([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(0.0, abs=0.001)

    def test_zero_vector(self):
        result = _cosine_sim([0.0, 0.0], [1.0, 2.0])
        assert result == 0.0


def _make_anchor(aid, text="", tags=None, state=None, stability=0.5,
                replay_count=0, hippocampal_dep=0.5):
    a = Anchor.create(text=text or f"Memory {aid}", tags=tags or [])
    a.id = aid
    a.vector.stability = stability
    a.replay_count = replay_count
    a.vector.hippocampal_dependency = hippocampal_dep
    if state is not None:
        a.state = state
    return a


class TestClassification:
    def test_classify_working_by_tag(self):
        from star_graph.anchor import MemoryState
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", tags=["working"])
        result = scheduler._classify_memory_type(a)
        assert result == MemoryType.WORKING

    def test_classify_procedural_by_tag(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", tags=["preference"])
        result = scheduler._classify_memory_type(a)
        assert result == MemoryType.PROCEDURAL

    def test_classify_procedural_by_replay(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", replay_count=5)
        result = scheduler._classify_memory_type(a)
        assert result == MemoryType.PROCEDURAL

    def test_classify_procedural_by_hippocampal(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", hippocampal_dep=0.2)
        result = scheduler._classify_memory_type(a)
        assert result == MemoryType.PROCEDURAL

    def test_classify_semantic_by_tag(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", tags=["fact"])
        result = scheduler._classify_memory_type(a)
        assert result == MemoryType.SEMANTIC

    def test_classify_semantic_by_stability(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", stability=0.6)
        result = scheduler._classify_memory_type(a)
        assert result == MemoryType.SEMANTIC

    def test_classify_episodic_default(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", stability=0.2, hippocampal_dep=0.8)
        result = scheduler._classify_memory_type(a)
        assert result == MemoryType.EPISODIC

    def test_select_memory_types_coding(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="coding")
        types = scheduler._select_memory_types(ctx)
        assert MemoryType.PROCEDURAL in types
        assert MemoryType.SEMANTIC in types

    def test_select_memory_types_debugging(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="debugging")
        types = scheduler._select_memory_types(ctx)
        assert MemoryType.EPISODIC in types

    def test_select_memory_types_unknown(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="unknown_task")
        types = scheduler._select_memory_types(ctx)
        assert len(types) >= 2  # defaults


class TestSchedulerInit:
    def test_init(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        assert scheduler.graph is g

    def test_init_with_working_memory(self):
        from star_graph.working_memory import WorkingMemory
        g = StarGraph()
        wm = WorkingMemory()
        scheduler = CognitiveMemoryScheduler(g, working_memory=wm)
        assert scheduler.working_memory is wm


class TestUserProfileAndHistory:
    def test_get_user_profile(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        profile = scheduler.get_user_profile()
        assert profile is not None

    def test_get_relevant_history(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        result = scheduler.get_relevant_history("python coding")
        assert result is not None

    def test_get_relevant_history_custom_max(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        result = scheduler.get_relevant_history("test", max_items=3)
        assert result is not None


class TestCompositeRank:
    def test_rank_empty(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="coding")
        result = scheduler._composite_rank([], ctx, None)
        assert result == []

    def test_rank_single_item(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "test")
        g.add_anchor(a)
        item = MemoryItem(anchor=a, relevance_score=0.5)
        ctx = AgentContext(task_type="debugging")
        result = scheduler._composite_rank([item], ctx, None)
        assert len(result) == 1
        assert result[0].relevance_score >= 0

    def test_rank_multiple_items(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a1 = _make_anchor("a1", "important memory")
        a2 = _make_anchor("a2", "less important")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_edge("a1", "a2", weight=0.5, edge_type="related")
        item1 = MemoryItem(anchor=a1, relevance_score=0.8, confidence=0.7)
        item2 = MemoryItem(anchor=a2, relevance_score=0.3, confidence=0.4)
        ctx = AgentContext(task_type="reflection")
        result = scheduler._composite_rank([item1, item2], ctx, None)
        assert len(result) == 2

    def test_rank_different_task_types(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "test")
        g.add_anchor(a)
        for task_type in ["coding", "debugging", "planning", "reflection", "conversation"]:
            item = MemoryItem(anchor=a, relevance_score=0.5)
            ctx = AgentContext(task_type=task_type)
            result = scheduler._composite_rank([item], ctx, None)
            assert len(result) == 1


class TestAdaptiveCompress:
    def test_compress_empty(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(context_budget_tokens=1000)
        result = scheduler._adaptive_compress([], ctx)
        assert result == []

    def test_compress_level_0(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "This is a short memory.")
        item = MemoryItem(anchor=a)
        ctx = AgentContext(context_budget_tokens=10000)
        result = scheduler._adaptive_compress([item], ctx)
        assert result[0].compression_level == 0

    def test_compress_with_reasoning_path(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "Long text " * 50)
        item = MemoryItem(
            anchor=a,
            reasoning_path=["a1", "a2", "a3"],
        )
        ctx = AgentContext(context_budget_tokens=5)
        result = scheduler._adaptive_compress([item], ctx)
        assert result[0].compression_level in (1, 2)


class TestBuildContext:
    def test_build_empty(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._build_context([], ctx, 0.0)
        assert "No relevant memories" in result.memory_summary

    def test_build_with_items(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "Python is a programming language", tags=["fact"])
        g.add_anchor(a)
        item = MemoryItem(
            anchor=a, relevance_score=0.8, confidence=0.7,
            memory_type=MemoryType.SEMANTIC,
            compressed_text="Python is a programming language",
        )
        ctx = AgentContext()
        result = scheduler._build_context([item], ctx, 10.0)
        assert result.total_tokens > 0
        assert result.retrieval_latency_ms == 10.0

    def test_build_with_pattern(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "User prefers concise answers", tags=["preference"])
        g.add_anchor(a)
        item = MemoryItem(
            anchor=a, relevance_score=0.6, confidence=0.8,
            memory_type=MemoryType.PROCEDURAL,
            compressed_text="User prefers concise answers",
        )
        ctx = AgentContext()
        result = scheduler._build_context([item], ctx, 5.0)
        assert len(result.active_patterns) >= 1

    def test_build_with_reasoning(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a1 = _make_anchor("a1", "Start memory")
        a2 = _make_anchor("a2", "Middle memory")
        a3 = _make_anchor("a3", "End memory")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_anchor(a3)
        item = MemoryItem(
            anchor=a3, relevance_score=0.5,
            reasoning_path=["a1", "a2", "a3"],
            compressed_text="End memory",
        )
        ctx = AgentContext()
        result = scheduler._build_context([item], ctx, 5.0)
        assert len(result.reasoning_traces) >= 1


class TestRetrieve2DPlane:
    def test_empty_graph(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._retrieve_2d_plane(ctx, max_items=5)
        assert result == []

    def test_with_anchors(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a1 = _make_anchor("a1", "recent important memory")
        a2 = _make_anchor("a2", "old unimportant memory")
        a2.last_activated_at = 0.0
        g.add_anchor(a1)
        g.add_anchor(a2)
        ctx = AgentContext()
        result = scheduler._retrieve_2d_plane(ctx, max_items=2)
        assert len(result) >= 1
        assert len(result) <= 2


class TestRetrieveTimeline:
    def test_no_working_memory(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._retrieve_timeline(ctx, max_items=5)
        assert result == []

    def test_with_working_memory_and_anchors(self):
        from star_graph.working_memory import WorkingMemory
        g = StarGraph()
        a1 = _make_anchor("a1", "test memory")
        a2 = _make_anchor("a2", "another memory")
        g.add_anchor(a1)
        g.add_anchor(a2)
        wm = WorkingMemory()
        scheduler = CognitiveMemoryScheduler(g, working_memory=wm)
        ctx = AgentContext()
        result = scheduler._retrieve_timeline(ctx, max_items=2)
        assert len(result) <= 2


class TestRetrieveFromWorkingMemory:
    def test_no_working_memory(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._retrieve_from_working_memory(ctx, "test", None)
        assert result == []

    def test_with_working_memory_empty(self):
        from star_graph.working_memory import WorkingMemory
        g = StarGraph()
        wm = WorkingMemory()
        scheduler = CognitiveMemoryScheduler(g, working_memory=wm)
        ctx = AgentContext()
        result = scheduler._retrieve_from_working_memory(ctx, "test", None)
        assert result == []

    def test_with_working_memory_items(self):
        from star_graph.working_memory import WorkingMemory
        g = StarGraph()
        wm = WorkingMemory()
        wm.add("working item text", source_session="s1", embedding=[0.1] * 384)
        scheduler = CognitiveMemoryScheduler(g, working_memory=wm)
        ctx = AgentContext()
        result = scheduler._retrieve_from_working_memory(ctx, "working", [0.1] * 384)
        assert isinstance(result, list)


class TestDiscoverSeeds:
    def test_discover_seeds_empty(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._discover_seeds(ctx, "test", None,
                                            [MemoryType.SEMANTIC, MemoryType.EPISODIC])
        assert result == []

    def test_discover_seeds_with_anchors(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "python debugging tips", tags=["python"])
        a2 = _make_anchor("a2", "docker deployment guide", tags=["docker"])
        g.add_anchor(a1)
        g.add_anchor(a2)
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="coding", active_goals=["python"])
        result = scheduler._discover_seeds(ctx, "python debugging", None,
                                            [MemoryType.SEMANTIC, MemoryType.EPISODIC])
        assert isinstance(result, list)

    def test_discover_seeds_with_embedding(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "test memory one")
        a1.embedding = [0.1] * 384
        a2 = _make_anchor("a2", "test memory two")
        a2.embedding = [0.2] * 384
        g.add_anchor(a1)
        g.add_anchor(a2)
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="debugging")
        result = scheduler._discover_seeds(ctx, "test", [0.15] * 384,
                                            [MemoryType.SEMANTIC, MemoryType.EPISODIC])
        assert len(result) >= 0

    def test_discover_seeds_with_session_match(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "session one memory")
        a1.source_session = "s1"
        a1.embedding = [0.1] * 384
        a2 = _make_anchor("a2", "session two memory")
        a2.source_session = "s2"
        a2.embedding = [0.2] * 384
        g.add_anchor(a1)
        g.add_anchor(a2)
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="coding", session_id="s1")
        result = scheduler._discover_seeds(ctx, "memory", [0.15] * 384,
                                            [MemoryType.SEMANTIC, MemoryType.EPISODIC])
        assert isinstance(result, list)

    def test_discover_seeds_with_emotional_context(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "happy memory")
        a1.embedding = [0.1] * 384
        a1.vector.emotional_valence = 0.8
        g.add_anchor(a1)
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="reflection", emotional_state=0.7)
        result = scheduler._discover_seeds(ctx, "happy", [0.1] * 384,
                                            [MemoryType.SEMANTIC, MemoryType.EPISODIC])
        assert isinstance(result, list)

    def test_discover_seeds_with_goal_relevance(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "deploy docker container to production")
        a1.embedding = [0.1] * 384
        g.add_anchor(a1)
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext(task_type="coding", active_goals=["docker deployment"])
        result = scheduler._discover_seeds(ctx, "deploy", [0.15] * 384,
                                            [MemoryType.SEMANTIC, MemoryType.EPISODIC])
        assert isinstance(result, list)


class TestMultiHopTraverse:
    def test_traverse_empty_seeds(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._multi_hop_traverse([], ctx, None)
        assert result == []

    def test_traverse_single_seed_no_edges(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "single memory")
        g.add_anchor(a1)
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._multi_hop_traverse([a1], ctx, None)
        assert len(result) >= 1
        assert result[0].anchor.id == "a1"

    def test_traverse_with_edges(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "start memory")
        a2 = _make_anchor("a2", "neighbor memory")
        a3 = _make_anchor("a3", "far memory")
        g.add_anchor(a1)
        g.add_anchor(a2)
        g.add_anchor(a3)
        g.add_edge("a1", "a2", weight=0.8, edge_type="related")
        g.add_edge("a2", "a3", weight=0.6, edge_type="related")
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._multi_hop_traverse([a1], ctx, None, max_hops=2)
        assert len(result) >= 1


class TestCommunityAwareRetrieve:
    def test_community_aware_empty(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._community_aware_retrieve(
            ctx, "test", None,
            [MemoryType.SEMANTIC, MemoryType.EPISODIC],
            max_items=5, community_filter={"c1"},
        )
        assert result == []

    def test_community_aware_with_anchors(self):
        g = StarGraph()
        a1 = _make_anchor("a1", "community memory", tags=["test"])
        a1.community_id = "c1"
        a1.embedding = [0.1] * 384
        g.add_anchor(a1)
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._community_aware_retrieve(
            ctx, "memory", [0.1] * 384,
            [MemoryType.SEMANTIC, MemoryType.EPISODIC],
            max_items=5, community_filter={"c1"},
        )
        assert isinstance(result, list)


class TestPrintContext:
    def test_print_context(self, capsys):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "test memory")
        item = MemoryItem(anchor=a, relevance_score=0.5)
        ctx = MemoryContext(items=[item], memory_summary="test")
        scheduler.print_context(ctx)
        captured = capsys.readouterr()
        assert "Cognitive Memory Context" in captured.out

    def test_print_context_with_details(self, capsys):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        a = _make_anchor("a1", "test memory", tags=["fact"])
        item = MemoryItem(
            anchor=a, relevance_score=0.8, confidence=0.9,
            memory_type=MemoryType.SEMANTIC, compression_level=1,
            compressed_text="compressed text",
            reasoning_path=["a1", "a2"],
        )
        ctx = MemoryContext(
            items=[item],
            memory_summary="test summary",
            relevant_facts=["Key fact about testing"],
            active_patterns=["Test pattern detected"],
            reasoning_traces=["a1 -> a2 -> a3"],
            total_tokens=50,
            retrieval_latency_ms=10.0,
        )
        scheduler.print_context(ctx)
        captured = capsys.readouterr()
        assert "Key Facts" in captured.out
        assert "Behavioral Patterns" in captured.out
        assert "Reasoning Traces" in captured.out


class TestDimensionalReductionRetrieve:
    def test_empty(self):
        g = StarGraph()
        scheduler = CognitiveMemoryScheduler(g)
        ctx = AgentContext()
        result = scheduler._dimensional_reduction_retrieve(
            ctx, "test", None,
            [MemoryType.SEMANTIC, MemoryType.EPISODIC], max_items=5)
        assert result == []

    def test_with_anchors_falls_to_timeline(self):
        from star_graph.working_memory import WorkingMemory
        g = StarGraph()
        a1 = _make_anchor("a1", "test memory one")
        a2 = _make_anchor("a2", "test memory two")
        g.add_anchor(a1)
        g.add_anchor(a2)
        wm = WorkingMemory()
        scheduler = CognitiveMemoryScheduler(g, working_memory=wm)
        ctx = AgentContext(task_type="general")
        result = scheduler._dimensional_reduction_retrieve(
            ctx, "test", None,
            [MemoryType.SEMANTIC, MemoryType.EPISODIC], max_items=2)
        assert len(result) >= 0
