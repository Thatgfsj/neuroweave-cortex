"""Tests for PersonalityModel — deep user trait extraction."""

import pytest
from star_graph.personality import PersonalityModel, PersonalityProfile
from star_graph.graph import StarGraph
from star_graph.anchor import Anchor


class TestPersonalityProfile:
    def test_defaults(self):
        p = PersonalityProfile()
        assert p.openness == 0.5
        assert p.conscientiousness == 0.5
        assert p.extraversion == 0.5
        assert p.agreeableness == 0.5
        assert p.neuroticism == 0.5
        assert p.planner_vs_doer == 0.5
        assert p.learning_style == "balanced"
        assert p.formality == 0.5
        assert p.confidence == 0.3
        assert p.evidence_count == 0
        assert p.version == 0

    def test_custom_values(self):
        p = PersonalityProfile(openness=0.8, conscientiousness=0.3)
        assert p.openness == 0.8
        assert p.conscientiousness == 0.3


class TestPersonalityModelInit:
    def test_initial_state(self):
        pm = PersonalityModel()
        assert pm.profile.evidence_count == 0
        assert pm._total_messages == 0

    def test_trait_scores_default(self):
        pm = PersonalityModel()
        scores = pm.trait_scores
        assert all(v == 0.5 for v in scores.values())
        assert set(scores.keys()) == {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}


class TestIngestAnchor:
    def test_increments_evidence(self):
        pm = PersonalityModel()
        a = Anchor.create(text="I need to plan and organize the project carefully")
        pm.ingest_anchor(a)
        assert pm.profile.evidence_count == 1
        assert pm._total_messages == 1

    def test_openness_signal(self):
        pm = PersonalityModel()
        a = Anchor.create(text="I want to explore new ideas and experiment with creative solutions")
        pm.ingest_anchor(a)
        assert pm.profile.openness > 0.5

    def test_conscientiousness_signal(self):
        pm = PersonalityModel()
        # Use high importance to get stronger signal
        a = Anchor.create(text="plan organize schedule todo test review refactor clean document")
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        assert pm.profile.conscientiousness > 0.5

    def test_extraversion_signal(self):
        pm = PersonalityModel()
        a = Anchor.create(text="let's discuss and share with the team, collaborate together")
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        assert pm.profile.extraversion > 0.5

    def test_agreeableness_signal(self):
        pm = PersonalityModel()
        a = Anchor.create(text="I agree, that's a good point, makes sense, very helpful, thanks")
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        assert pm.profile.agreeableness > 0.5

    def test_neuroticism_signal(self):
        pm = PersonalityModel()
        a = Anchor.create(text="I'm anxious and worried, feeling stressed and frustrated about this")
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        assert pm.profile.neuroticism > 0.5

    def test_learning_style_reading(self):
        pm = PersonalityModel()
        a = Anchor.create(text="I read the documentation and articles about this")
        pm.ingest_anchor(a)
        assert pm.profile.learning_style == "reading"

    def test_learning_style_doing(self):
        pm = PersonalityModel()
        a = Anchor.create(text="let me try to build and implement the code")
        pm.ingest_anchor(a)
        assert pm.profile.learning_style == "doing"

    def test_learning_style_asking(self):
        pm = PersonalityModel()
        a = Anchor.create(text="I want to ask how to do this, can you help?")
        pm.ingest_anchor(a)
        assert pm.profile.learning_style == "asking"

    def test_value_efficiency(self):
        pm = PersonalityModel()
        a = Anchor.create(text="need to make this fast quick efficient and optimize performance")
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        assert pm.profile.values.get("efficiency", 0.0) > 0.0

    def test_value_simplicity(self):
        pm = PersonalityModel()
        a = Anchor.create(text="keep it simple clean minimal and straightforward")
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        assert pm.profile.values.get("simplicity", 0.0) > 0.0

    def test_code_detection(self):
        pm = PersonalityModel()
        a = Anchor.create(text="def my_function(): import os; print('hello')")
        pm.ingest_anchor(a)
        assert pm._code_count == 1
        assert pm._question_count == 0

    def test_question_detection(self):
        pm = PersonalityModel()
        a = Anchor.create(text="how to implement this? what is the best approach?")
        pm.ingest_anchor(a)
        assert pm._question_count == 1

    def test_expertise_from_tags(self):
        pm = PersonalityModel()
        a = Anchor.create(text="some content", tags=["python", "machine-learning", "nlp"])
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        assert "python" in pm.profile.expertise_areas
        assert "machine-learning" in pm.profile.expertise_areas


class TestExtractFromGraph:
    def test_empty_graph(self):
        pm = PersonalityModel()
        g = StarGraph()
        profile = pm.extract_from_graph(g)
        assert profile.evidence_count == 0
        assert profile.confidence <= 0.3

    def test_extract_with_anchors(self):
        pm = PersonalityModel()
        g = StarGraph()
        for text in [
            "I plan to explore new creative ideas",
            "let's collaborate and discuss with the team",
            "need to test and review the code carefully",
        ]:
            a = Anchor.create(text=text)
            g.add_anchor(a)
        profile = pm.extract_from_graph(g)
        assert profile.evidence_count == 3
        assert profile.confidence >= 0.2

    def test_formality_computation(self):
        pm = PersonalityModel()
        g = StarGraph()
        a = Anchor.create(text="please could you help me, thank you I appreciate it")
        g.add_anchor(a)
        profile = pm.extract_from_graph(g)
        assert profile.formality >= 0.5


class TestStats:
    def test_initial_stats(self):
        pm = PersonalityModel()
        s = pm.stats
        assert s["evidence_count"] == 0
        assert s["confidence"] == 0.3
        assert "traits" in s
        assert "expertise_areas" in s
        assert "top_expertise" in s
        assert "learning_style" in s

    def test_stats_after_ingest(self):
        pm = PersonalityModel()
        a = Anchor.create(text="I plan and test everything carefully", tags=["python", "testing"])
        a.vector.importance = 1.0
        pm.ingest_anchor(a)
        s = pm.stats
        assert s["evidence_count"] == 1
        assert s["traits"]["conscientiousness"] > 0.5


class TestTopExpertise:
    def test_empty(self):
        pm = PersonalityModel()
        assert pm.top_expertise() == []

    def test_sorted_by_level(self):
        pm = PersonalityModel()
        pm.profile.expertise_areas = {"python": 0.5, "rust": 0.8, "go": 0.3}
        top = pm.top_expertise(2)
        assert len(top) == 2
        assert top[0] == ("rust", 0.8)
        assert top[1] == ("python", 0.5)

    def test_top_n(self):
        pm = PersonalityModel()
        pm.profile.expertise_areas = {"a": 0.9, "b": 0.7, "c": 0.5}
        assert len(pm.top_expertise(1)) == 1
        assert len(pm.top_expertise(5)) == 3
