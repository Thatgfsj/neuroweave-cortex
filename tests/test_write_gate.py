"""Tests for MemoryWriteGate — pre-write quality filter."""

import pytest
from star_graph.write_gate import MemoryWriteGate, GateDecision, GateResult


class TestGateDecisionEnum:
    def test_decisions_exist(self):
        assert GateDecision.ACCEPT.value == "accept"
        assert GateDecision.REJECT.value == "reject"
        assert GateDecision.MERGE.value == "merge"
        assert GateDecision.DEFER.value == "defer"


class TestGateResult:
    def test_default_construction(self):
        result = GateResult(GateDecision.ACCEPT)
        assert result.decision == GateDecision.ACCEPT
        assert result.score == 0.0

    def test_full_construction(self):
        result = GateResult(GateDecision.MERGE, reason="similar",
                           score=0.85, merge_target_id="abc123")
        assert result.merge_target_id == "abc123"
        assert result.score == 0.85


class TestWriteGateInit:
    def test_default_init(self):
        gate = MemoryWriteGate()
        assert gate.min_importance > 0
        assert gate.min_text_length > 0
        assert gate.duplicate_threshold > 0.5

    def test_stats(self):
        gate = MemoryWriteGate()
        stats = gate.stats
        assert "min_importance" in stats
        assert "duplicate_threshold" in stats
        assert "merge_threshold" in stats


class TestWriteGateRejections:
    def test_reject_empty_text(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("")
        assert result.decision == GateDecision.REJECT
        assert "empty" in result.reason

    def test_reject_whitespace_only(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("   \n  \t  ")
        assert result.decision == GateDecision.REJECT

    def test_reject_too_short(self):
        gate = MemoryWriteGate()
        gate.min_text_length = 10
        result = gate.evaluate("hi")
        assert result.decision == GateDecision.REJECT
        assert "too_short" in result.reason

    def test_reject_pure_emoji(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("😀👍💪")
        assert result.decision == GateDecision.REJECT
        # Rejected either as noise or too_short (emoji are narrow chars, len < 8)
        assert "noise" in result.reason or "too_short" in result.reason

    def test_reject_ok_only(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("ok")
        assert result.decision == GateDecision.REJECT

    def test_reject_thanks_only(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("thanks")
        assert result.decision in (GateDecision.REJECT, GateDecision.DEFER)

    def test_reject_ha_reaction(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("hahaha")
        assert result.decision == GateDecision.REJECT


class TestWriteGateAccept:
    def test_accept_substantive_text(self):
        gate = MemoryWriteGate()
        result = gate.evaluate(
            "Fixed the Redis connection pool timeout by increasing max connections to 20",
            importance=0.6,
        )
        assert result.decision == GateDecision.ACCEPT

    def test_accept_preference_statement(self):
        gate = MemoryWriteGate()
        result = gate.evaluate(
            "User prefers dark mode and uses vim keybindings in VS Code",
            importance=0.7,
            tags=["preference", "user", "style"],
        )
        assert result.decision == GateDecision.ACCEPT

    def test_accept_technical_knowledge(self):
        gate = MemoryWriteGate()
        result = gate.evaluate(
            "The deployment pipeline uses GitHub Actions with a three-stage "
            "approval gate for production releases",
            importance=0.8,
            tags=["knowledge", "deploy", "workflow"],
        )
        assert result.decision == GateDecision.ACCEPT

    def test_accept_with_high_importance(self):
        gate = MemoryWriteGate()
        result = gate.evaluate(
            "Important project decision: migrate from MySQL to PostgreSQL",
            importance=0.9,
            tags=["project", "decision"],
        )
        assert result.decision == GateDecision.ACCEPT


class TestWriteGateEmotionalNoise:
    def test_defer_high_emotion_short_text(self):
        gate = MemoryWriteGate()
        result = gate.evaluate(
            "I'm so frustrated with this stupid bug!!!",
            importance=0.3,
            emotional_valence=-0.9,
        )
        assert result.decision in (GateDecision.DEFER, GateDecision.REJECT)

    def test_accept_moderate_emotion_substantive(self):
        gate = MemoryWriteGate()
        result = gate.evaluate(
            "I'm frustrated because the deployment keeps failing due to a "
            "missing environment variable in the staging config",
            importance=0.6,
            emotional_valence=-0.5,
        )
        # Should pass because it's substantive despite emotion
        assert result.decision in (GateDecision.ACCEPT, GateDecision.DEFER)


class TestWriteGateImportanceThreshold:
    def test_reject_very_low_importance(self):
        gate = MemoryWriteGate()
        gate.min_importance = 0.2
        result = gate.evaluate("Some random thought", importance=0.01)
        assert result.decision == GateDecision.REJECT

    def test_defer_borderline_importance(self):
        gate = MemoryWriteGate()
        gate.min_importance = 0.2
        result = gate.evaluate("A fairly short note", importance=0.12)
        assert result.decision == GateDecision.DEFER


class TestWriteGateDuplicateCheck:
    def test_no_duplicate_in_empty_graph(self):
        gate = MemoryWriteGate()
        from star_graph.graph import StarGraph
        g = StarGraph()
        result = gate.evaluate(
            "Python flask web development",
            embedding=[0.1] * 128,
            graph=g,
            importance=0.6,
        )
        assert result.decision == GateDecision.ACCEPT

    def test_duplicate_detection_brute_force(self):
        gate = MemoryWriteGate()
        gate.duplicate_threshold = 0.9
        gate.merge_threshold = 0.7
        from star_graph.graph import StarGraph
        from star_graph.anchor import Anchor
        g = StarGraph()
        emb1 = [0.5] * 128
        a = Anchor.create(text="Python flask web development", embedding=emb1)
        a.id = "dup_test_1"
        g.add_anchor(a)
        # Very similar embedding should trigger duplicate
        result = gate.evaluate(
            "Python flask web development tutorial",
            embedding=[0.5] * 128,
            graph=g,
            importance=0.6,
        )
        assert result.decision in (GateDecision.REJECT, GateDecision.MERGE)

    def test_different_content_not_duplicate(self):
        gate = MemoryWriteGate()
        from star_graph.graph import StarGraph
        from star_graph.anchor import Anchor
        g = StarGraph()
        a = Anchor.create(text="Python flask web development",
                         embedding=[0.3] * 128)
        a.id = "diff_test_1"
        g.add_anchor(a)
        result = gate.evaluate(
            "Docker kubernetes deployment pipeline CI/CD",
            embedding=[-0.5] * 128,
            graph=g,
            importance=0.6,
        )
        assert result.decision in (GateDecision.ACCEPT, GateDecision.DEFER)


class TestWriteGateDebounce:
    def test_debounce_rejects_duplicate_within_window(self):
        gate = MemoryWriteGate()
        gate._debounce_window = 3600.0  # 1 hour
        text = "A unique test message for debounce checking"
        r1 = gate.evaluate(text, importance=0.6)
        assert r1.decision == GateDecision.ACCEPT
        r2 = gate.evaluate(text, importance=0.6)
        assert r2.decision == GateDecision.REJECT
        assert "debounce" in r2.reason

    def test_different_texts_not_debounced(self):
        gate = MemoryWriteGate()
        gate._debounce_window = 3600.0
        gate.evaluate("First unique message", importance=0.6)
        r2 = gate.evaluate("Second different message", importance=0.6)
        assert r2.decision == GateDecision.ACCEPT


class TestWriterGateNoisePatterns:
    def test_bot_command_flagged(self):
        gate = MemoryWriteGate()
        # Pure bot command without additional context → noise
        result = gate.evaluate("/status")
        # May be rejected as too_short or noise
        assert result.decision != GateDecision.ACCEPT

    def test_single_letter_noise(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("abc")
        assert result.decision in (GateDecision.REJECT, GateDecision.DEFER)

    def test_greeting_only(self):
        gate = MemoryWriteGate()
        result = gate.evaluate("hello")
        assert result.decision in (GateDecision.REJECT, GateDecision.DEFER)
