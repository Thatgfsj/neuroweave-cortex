"""Tests for atom_facts module — AtomFact dataclass."""

import pytest

from star_graph.atom_facts import AtomFact
from star_graph.anchor import Anchor, MemoryState


class TestAtomFact:
    def test_default_values(self):
        af = AtomFact(id="af1", text="User prefers dark mode")
        assert af.id == "af1"
        assert af.text == "User prefers dark mode"
        assert af.fact_type == "factual"
        assert af.confidence == 0.7
        assert af.source_anchor_ids == []

    def test_with_all_fields(self):
        af = AtomFact(
            id="af1", text="Redis pool increased from 10 to 20",
            subject="Redis", predicate="was_configured",
            object="pool_size=20", fact_type="event",
            confidence=0.9, source_anchor_ids=["a1", "a2"],
        )
        assert af.subject == "Redis"
        assert af.predicate == "was_configured"
        assert af.fact_type == "event"

    def test_to_anchor(self):
        af = AtomFact(
            id="af1", text="User uses dark mode",
            subject="User", predicate="prefers", object="dark mode",
            fact_type="preference", confidence=0.9,
        )
        anchor = af.to_anchor()
        assert isinstance(anchor, Anchor)
        assert anchor.text.startswith("User uses dark mode")
        # Should have atom_fact and fact_type tags
        tags_joined = " ".join(anchor.tags)
        assert "atom_fact" in tags_joined

    def test_to_anchor_with_custom_tags(self):
        af = AtomFact(
            id="af1", text="Selenium uses headless Chrome",
            subject="Selenium", predicate="configured_with",
            object="headless Chrome", fact_type="factual",
        )
        anchor = af.to_anchor(tags=["ci", "testing"])
        assert "ci" in anchor.tags
        assert "testing" in anchor.tags
        assert "atom_fact" in anchor.tags
