"""Tests for GoalTree — hierarchical goal decomposition with progress tracking."""

import time
import pytest
from star_graph.goal_tree import GoalTree, GoalNode, GoalStatus
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


def _make_graph_with_texts(texts: list[str]) -> StarGraph:
    g = StarGraph()
    for text in texts:
        a = Anchor.create(text=text)
        g.add_anchor(a)
    return g


class TestGoalStatus:
    def test_constants(self):
        assert GoalStatus.ACTIVE == "active"
        assert GoalStatus.ACHIEVED == "achieved"
        assert GoalStatus.ABANDONED == "abandoned"
        assert GoalStatus.BLOCKED == "blocked"
        assert GoalStatus.ARCHIVED == "archived"


class TestGoalNode:
    def test_defaults(self):
        g = GoalNode(id="g1", description="test goal")
        assert g.id == "g1"
        assert g.description == "test goal"
        assert g.status == GoalStatus.ACTIVE
        assert g.progress == 0.0
        assert g.priority == 0.5
        assert g.confidence == 0.5
        assert g.parent_id == ""
        assert g.children == []
        assert g.depth == 0

    def test_is_leaf(self):
        g = GoalNode(id="g1", description="test")
        assert g.is_leaf is True
        g.children = ["child1"]
        assert g.is_leaf is False

    def test_is_root(self):
        g = GoalNode(id="g1", description="test")
        assert g.is_root is True
        g.parent_id = "parent1"
        assert g.is_root is False

    def test_staleness_hours(self):
        g = GoalNode(id="g1", description="test", last_progress_at=time.time() - 7200)
        assert 1.5 <= g.staleness_hours <= 2.5


class TestGoalTreeInit:
    def test_initial_state(self):
        gt = GoalTree()
        assert gt._total_detected == 0
        assert gt._total_achieved == 0
        assert gt._total_abandoned == 0
        assert len(gt._goals) == 0


class TestGoalDetection:
    def test_detect_need_to(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["I need to implement the login page"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1
        assert any("login" in goal.description.lower() for goal in new)

    def test_detect_must(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["we must fix the authentication bug"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1

    def test_detect_todo(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["TODO: refactor the database layer"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1
        assert any("refactor" in goal.description.lower() for goal in new)

    def test_detect_fixme(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["FIXME: memory leak in connection pool"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1

    def test_detect_explicit_goal(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["goal: improve test coverage to 90%"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1

    def test_detect_working_on(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["working on implementing the search feature"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1

    def test_no_duplicate_detection(self):
        gt = GoalTree()
        g = _make_graph_with_texts([
            "I need to fix the bug",
            "I need to fix the bug",  # duplicate
        ])
        new1 = gt.detect_from_graph(g)
        # Second detection on same graph should find nothing new
        new2 = gt.detect_from_graph(g)
        assert len(new2) == 0

    def test_goal_has_tags(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["I need to fix the critical bug in production"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1
        assert "bugfix" in new[0].tags

    def test_goal_has_priority(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["urgent: need to fix the critical production bug asap"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1
        assert new[0].priority >= 0.7

    def test_learning_goal_lower_priority(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["maybe I should learn about rust someday later"])
        new = gt.detect_from_graph(g)
        assert len(new) >= 1
        assert new[0].priority <= 0.5


class TestGoalManagement:
    def test_add_subgoal(self):
        gt = GoalTree()
        gt._goals["goal_parent"] = GoalNode(id="goal_parent", description="parent")
        sub = gt.add_subgoal("goal_parent", "child task")
        assert sub is not None
        assert sub.parent_id == "goal_parent"
        assert sub.depth == 1
        assert sub.id in gt._goals["goal_parent"].children

    def test_add_subgoal_nonexistent_parent(self):
        gt = GoalTree()
        sub = gt.add_subgoal("nonexistent", "child task")
        assert sub is None

    def test_mark_progress(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="test")
        ok = gt.mark_progress("g1", 0.3)
        assert ok is True
        assert gt._goals["g1"].progress == 0.3

    def test_mark_progress_nonexistent(self):
        gt = GoalTree()
        ok = gt.mark_progress("nonexistent", 0.3)
        assert ok is False

    def test_mark_progress_auto_achieve(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="test")
        gt.mark_progress("g1", 1.0)
        assert gt._goals["g1"].status == GoalStatus.ACHIEVED
        assert gt._goals["g1"].progress == 1.0

    def test_mark_achieved(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="test")
        ok = gt.mark_achieved("g1")
        assert ok is True
        assert gt._goals["g1"].status == GoalStatus.ACHIEVED
        assert gt._goals["g1"].achieved_at > 0

    def test_mark_abandoned(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="test")
        ok = gt.mark_abandoned("g1")
        assert ok is True
        assert gt._goals["g1"].status == GoalStatus.ABANDONED

    def test_mark_blocked(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="test")
        ok = gt.mark_blocked("g1")
        assert ok is True
        assert gt._goals["g1"].status == GoalStatus.BLOCKED


class TestProgressPropagation:
    def test_parent_updated_from_children(self):
        gt = GoalTree()
        parent = GoalNode(id="goal_p", description="parent")
        child1 = GoalNode(id="goal_c1", description="child1", parent_id="goal_p", progress=1.0)
        child2 = GoalNode(id="goal_c2", description="child2", parent_id="goal_p", progress=0.0)
        parent.children = ["goal_c1", "goal_c2"]
        gt._goals["goal_p"] = parent
        gt._goals["goal_c1"] = child1
        gt._goals["goal_c2"] = child2
        gt.propagate_progress()
        assert gt._goals["goal_p"].progress == 0.5

    def test_achieved_propagates_to_parent(self):
        gt = GoalTree()
        parent = GoalNode(id="goal_p", description="parent")
        child = GoalNode(id="goal_c1", description="child", parent_id="goal_p", progress=0.5)
        parent.children = ["goal_c1"]
        gt._goals["goal_p"] = parent
        gt._goals["goal_c1"] = child
        gt.mark_achieved("goal_c1")
        assert gt._goals["goal_p"].progress == 1.0


class TestArchival:
    def test_archive_stale(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="stale goal",
                                    last_progress_at=time.time() - 999999)
        count = gt.archive_stale(hours=1.0)
        assert count >= 1
        assert gt._goals["g1"].status == GoalStatus.ARCHIVED

    def test_archive_stale_fresh_goal(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="fresh goal")
        count = gt.archive_stale(hours=999.0)
        assert count == 0
        assert gt._goals["g1"].status == GoalStatus.ACTIVE

    def test_archive_achieved_old(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="old achieved",
                                    status=GoalStatus.ACHIEVED,
                                    achieved_at=time.time() - 999999)
        count = gt.archive_achieved(age_hours=1.0)
        assert count >= 1
        assert gt._goals["g1"].status == GoalStatus.ARCHIVED

    def test_archive_achieved_recent(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="recent achieved",
                                    status=GoalStatus.ACHIEVED,
                                    achieved_at=time.time())
        count = gt.archive_achieved(age_hours=999.0)
        assert count == 0


class TestQueryAPI:
    def test_get_active_goals(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="active 1", priority=0.8)
        gt._goals["g2"] = GoalNode(id="g2", description="active 2", priority=0.3)
        gt._goals["g3"] = GoalNode(id="g3", description="achieved", status=GoalStatus.ACHIEVED)
        active = gt.get_active_goals()
        assert len(active) == 2
        assert active[0].id == "g1"  # higher priority first

    def test_get_blocked_goals(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="active")
        gt._goals["g2"] = GoalNode(id="g2", description="blocked", status=GoalStatus.BLOCKED)
        blocked = gt.get_blocked_goals()
        assert len(blocked) == 1

    def test_get_recently_achieved(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="recent",
                                    status=GoalStatus.ACHIEVED, achieved_at=time.time())
        gt._goals["g2"] = GoalNode(id="g2", description="old",
                                    status=GoalStatus.ACHIEVED, achieved_at=time.time() - 999999)
        recent = gt.get_recently_achieved(hours=1.0)
        assert len(recent) == 1
        assert recent[0].id == "g1"

    def test_get_root_goals(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="root")
        gt._goals["g2"] = GoalNode(id="g2", description="child", parent_id="g1")
        roots = gt.get_root_goals()
        assert len(roots) == 1
        assert roots[0].id == "g1"

    def test_get_children(self):
        gt = GoalTree()
        parent = GoalNode(id="goal_p", description="parent")
        child = GoalNode(id="goal_c1", description="child", parent_id="goal_p")
        parent.children = ["goal_c1"]
        gt._goals["goal_p"] = parent
        gt._goals["goal_c1"] = child
        children = gt.get_children("goal_p")
        assert len(children) == 1
        assert children[0].id == "goal_c1"

    def test_get_stale_goals(self):
        gt = GoalTree()
        gt._goals["g1"] = GoalNode(id="g1", description="stale", last_progress_at=time.time() - 999999)
        gt._goals["g2"] = GoalNode(id="g2", description="fresh")
        stale = gt.get_stale_goals(hours=1.0)
        assert len(stale) == 1
        assert stale[0].id == "g1"


class TestStats:
    def test_initial_stats(self):
        gt = GoalTree()
        s = gt.stats
        assert s["total_goals"] == 0
        assert s["active"] == 0
        assert s["achieved"] == 0
        assert s["total_detected"] == 0

    def test_stats_after_detection(self):
        gt = GoalTree()
        g = _make_graph_with_texts(["TODO: implement feature X", "need to fix bug Y"])
        gt.detect_from_graph(g)
        s = gt.stats
        assert s["total_goals"] >= 1
        assert s["total_detected"] >= 1
        assert "active_goals" in s
