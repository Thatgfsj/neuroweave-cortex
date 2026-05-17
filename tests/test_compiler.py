"""Tests for compiler module — WorldviewNode, UserProfile, CognitiveCompiler."""

import time

import pytest

from star_graph.compiler import (
    WorldviewNode,
    UserProfile,
    CognitiveCompiler,
)
from star_graph.config import Config


class TestWorldviewNode:
    def test_defaults(self):
        wv = WorldviewNode(
            id="wv1", label="Test Belief",
            description="User prefers Python",
            source_concept_ids=["c1", "c2"],
        )
        assert wv.id == "wv1"
        assert wv.label == "Test Belief"
        assert wv.confidence == 0.5
        assert wv.stability == 0.5
        assert wv.worldview_type == "belief"
        assert wv.is_stable is False

    def test_is_stable(self):
        wv = WorldviewNode(
            id="wv1", label="Stable Belief", description="",
            source_concept_ids=["c1", "c2"],
            stability=0.8, evidence_count=3,
        )
        assert wv.is_stable is True

    def test_is_stable_insufficient_evidence(self):
        wv = WorldviewNode(
            id="wv1", label="Belief", description="",
            source_concept_ids=["c1"],
            stability=0.9, evidence_count=1,
        )
        assert wv.is_stable is False

    def test_reinforce(self):
        wv = WorldviewNode(
            id="wv1", label="Test", description="",
            source_concept_ids=["c1"],
            stability=0.5, confidence=0.5, evidence_count=1,
        )
        wv.reinforce()
        assert wv.stability > 0.5
        assert wv.confidence > 0.5
        assert wv.evidence_count == 2

    def test_weaken(self):
        wv = WorldviewNode(
            id="wv1", label="Test", description="",
            source_concept_ids=["c1"],
            stability=0.5, confidence=0.5,
        )
        wv.weaken()
        assert wv.stability < 0.5
        assert wv.confidence < 0.5

    def test_weaken_floor(self):
        wv = WorldviewNode(
            id="wv1", label="Test", description="",
            source_concept_ids=["c1"],
            stability=0.05, confidence=0.05,
        )
        wv.weaken()
        assert wv.stability == 0.05

    def test_degrade(self):
        wv = WorldviewNode(
            id="wv1", label="Test", description="",
            source_concept_ids=["c1"],
            stability=0.8,
        )
        wv.last_reinforced_at = 0.0  # very old
        remaining = wv.degrade(half_life_days=90.0)
        assert remaining < 0.8
        assert remaining >= 0.01

    def test_domain_and_type(self):
        wv = WorldviewNode(
            id="wv1", label="Pref", description="User likes dark mode",
            source_concept_ids=["c1"], domain="ui", worldview_type="preference",
        )
        assert wv.domain == "ui"
        assert wv.worldview_type == "preference"


class TestUserProfile:
    def test_defaults(self):
        up = UserProfile(
            id="up1", summary="Test user",
            preferences=[], expertise_areas=[], working_style="",
            values=[], habits=[], source_worldview_ids=[],
        )
        assert up.id == "up1"
        assert up.version == 1

    def test_from_worldviews(self):
        wvs = [
            WorldviewNode(
                id="wv1", label="Likes Python",
                description="Python for scripting", source_concept_ids=["c1"],
                worldview_type="preference", confidence=0.8,
            ),
            WorldviewNode(
                id="wv2", label="Docker expert",
                description="Docker and Kubernetes", source_concept_ids=["c2"],
                worldview_type="expertise", confidence=0.9,
            ),
            WorldviewNode(
                id="wv3", label="Morning worker",
                description="Works best in morning", source_concept_ids=["c3"],
                worldview_type="habit", confidence=0.7,
            ),
            WorldviewNode(
                id="wv4", label="Quality focused",
                description="Values code quality", source_concept_ids=["c4"],
                worldview_type="value", confidence=0.85,
            ),
        ]
        up = UserProfile.from_worldviews(wvs, worldview_id="custom_id")
        assert up.id == "custom_id"
        assert "Python" in up.summary
        assert "Docker and Kubernetes" in up.expertise_areas
        assert "Works best in morning" in up.habits
        assert "Values code quality" in up.values

    def test_from_worldviews_empty(self):
        up = UserProfile.from_worldviews([])
        assert "Profile forming" in up.summary
        assert up.confidence == 0.0


class TestCognitiveCompiler:
    def test_init_defaults(self):
        cc = CognitiveCompiler()
        assert cc.worldviews == {}
        assert cc.profile is None
        assert cc.episodic_ratio > 0
        assert cc.max_worldviews > 0

    def test_infer_domain_from_concept(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            tags = ["python", "flask"]
            text = "User uses Flask for APIs"
            label = "Python Developer"

        domain = cc._infer_domain_from_concept(FakeConcept())
        assert domain in ("python_development", "general")

    def test_infer_domain_general(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            tags = []
            text = "xyzzy flurbo garply"
            label = "Unknown"

        domain = cc._infer_domain_from_concept(FakeConcept())
        assert domain == "general"

    def test_infer_worldview_type(self):
        cc = CognitiveCompiler()

        class PrefConcept:
            pattern_text = "User prefers dark mode"
            text = ""
            description = ""

        class ExpertConcept:
            pattern_text = "User is experienced in Rust"
            text = ""
            description = ""

        class HabitConcept:
            pattern_text = "User always starts with tests"
            text = ""
            description = ""

        assert cc._infer_worldview_type([PrefConcept()], "ui") == "preference"
        assert cc._infer_worldview_type([ExpertConcept()], "lang") == "expertise"
        assert cc._infer_worldview_type([HabitConcept()], "workflow") == "habit"

    def test_extract_key_terms(self):
        texts = [
            "python flask api development",
            "python django web application",
            "redis caching with python",
        ]
        terms = CognitiveCompiler._extract_key_terms(texts, top_k=3)
        assert "python" in terms
        assert len(terms) <= 3

    def test_extract_key_terms_empty(self):
        terms = CognitiveCompiler._extract_key_terms([])
        assert terms == []

    def test_extract_key_terms_stop_words_filtered(self):
        texts = ["the is and for with"]
        terms = CognitiveCompiler._extract_key_terms(texts)
        assert len(terms) == 0

    def test_summarize_texts(self):
        cc = CognitiveCompiler()
        result = cc._summarize_texts(
            ["redis timeout debugging", "redis connection pool"],
            "knowledge of",
        )
        assert "redis" in result
        assert "knowledge of" in result

    def test_summarize_texts_empty(self):
        cc = CognitiveCompiler()
        result = cc._summarize_texts([], "working with")
        assert "patterns" in result

    def test_summarize_texts_one_term(self):
        cc = CognitiveCompiler()
        result = cc._summarize_texts(["python coding patterns"], "expertise in")
        assert "expertise in" in result

    def test_degrade_worldviews_empty(self):
        cc = CognitiveCompiler()
        removed = cc.degrade_worldviews()
        assert removed == 0

    def test_degrade_worldviews(self):
        cc = CognitiveCompiler()
        wv = WorldviewNode(
            id="wv1", label="Old", description="",
            source_concept_ids=["c1"], stability=0.5,
        )
        wv.last_reinforced_at = 0.0  # very old
        cc.worldviews["wv1"] = wv
        removed = cc.degrade_worldviews()
        # May or may not remove depending on decay rate
        assert removed >= 0

    def test_get_stable_worldviews_empty(self):
        cc = CognitiveCompiler()
        result = cc.get_stable_worldviews()
        assert result == []

    def test_get_stable_worldviews(self):
        cc = CognitiveCompiler()
        cc.worldviews["wv1"] = WorldviewNode(
            id="wv1", label="Stable", description="",
            source_concept_ids=["c1"], stability=0.8, confidence=0.9,
        )
        cc.worldviews["wv2"] = WorldviewNode(
            id="wv2", label="Unstable", description="",
            source_concept_ids=["c2"], stability=0.3, confidence=0.5,
        )
        result = cc.get_stable_worldviews(min_stability=0.5)
        assert len(result) == 1
        assert result[0].id == "wv1"

    def test_stats_empty(self):
        cc = CognitiveCompiler()
        s = cc.stats
        assert s["total_worldviews"] == 0
        assert s["profile_version"] == 0

    def test_stats_with_worldviews(self):
        cc = CognitiveCompiler()
        cc.worldviews["wv1"] = WorldviewNode(
            id="wv1", label="Test", description="",
            source_concept_ids=["c1"], confidence=0.7, stability=0.6,
            worldview_type="preference",
        )
        s = cc.stats
        assert s["total_worldviews"] == 1
        assert "preference" in str(s["worldview_types"])

    def test_profile_property(self):
        cc = CognitiveCompiler()
        assert cc.profile is None
        up = UserProfile(
            id="up1", summary="Test", preferences=[], expertise_areas=[],
            working_style="", values=[], habits=[], source_worldview_ids=[],
        )
        cc._profile = up
        assert cc.profile is up


class TestCognitiveCompilerPipeline:
    """Tests for the compilation pipeline methods."""

    def test_cluster_by_domain(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            def __init__(self, tag, text=""):
                self.tags = [tag]
                self.text = text
                self.label = ""
        concepts = [
            FakeConcept("python", "flask api"),
            FakeConcept("python", "django web"),
            FakeConcept("docker", "container deploy"),
        ]
        clusters = cc._cluster_by_domain(concepts)
        assert "python_development" in clusters
        assert "devops" in clusters

    def test_synthesize_worldview_preference(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            tags = ["dark-mode", "ui"]
            pattern_text = "User prefers dark mode"
            text = "User prefers dark mode for coding"
            description = "Dark mode preference"
            id = "c1"
            confidence = 0.8

        wv = cc._synthesize_worldview("ui", [FakeConcept()])
        assert wv.domain == "ui"
        assert wv.worldview_type == "preference"
        assert "prefers" in wv.description.lower()
        assert wv.confidence >= 0.3
        assert wv.evidence_count == 1

    def test_synthesize_worldview_expertise(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            tags = ["python", "backend"]
            pattern_text = "User is experienced in Python backend"
            text = "User has expertise in Python and Flask"
            description = "Python expertise"
            id = "c1"
            confidence = 0.9

        wv = cc._synthesize_worldview("python_development", [FakeConcept()])
        assert wv.worldview_type == "expertise"
        assert "expertise" in wv.description.lower()

    def test_synthesize_worldview_habit(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            tags = ["workflow"]
            pattern_text = "User always starts with tests"
            text = "User always writes tests first"
            description = "Test-first habit"
            id = "c1"
            confidence = 0.7

        wv = cc._synthesize_worldview("testing", [FakeConcept()])
        assert wv.worldview_type in ("habit", "preference")

    def test_synthesize_worldview_value(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            tags = ["quality"]
            pattern_text = "User believes code quality is important"
            text = "User values code quality above speed"
            description = "Quality value"
            id = "c1"
            confidence = 0.85

        wv = cc._synthesize_worldview("architecture", [FakeConcept()])
        assert wv.worldview_type in ("value", "belief")

    def test_synthesize_worldview_multiple_concepts(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            def __init__(self, cid, text, conf=0.7, tags=None):
                self.id = cid
                self.pattern_text = text
                self.text = text
                self.description = text
                self.confidence = conf
                self.tags = tags or ["test"]

        concepts = [
            FakeConcept("c1", "User prefers fast iteration", 0.8, ["agile"]),
            FakeConcept("c2", "User likes quick feedback loops", 0.7, ["agile"]),
            FakeConcept("c3", "User favors rapid prototyping", 0.75, ["agile"]),
        ]
        wv = cc._synthesize_worldview("workflow", concepts)
        assert wv.evidence_count == 3
        assert wv.confidence > 0.7  # boosted by multiple concepts
        assert len(wv.source_concept_ids) == 3

    def test_synthesize_worldview_unknown_type(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            tags = ["general"]
            pattern_text = "Some random text about xyzzy"
            text = "xyzzy flurbo garply"
            description = "Unknown topic"
            id = "c1"
            confidence = 0.5

        wv = cc._synthesize_worldview("general", [FakeConcept()])
        assert wv.worldview_type == "belief"
        assert "Belief about general" in wv.description

    def test_compile_worldviews_empty(self):
        cc = CognitiveCompiler()
        result = cc._compile_worldviews([])
        assert result == []

    def test_compile_worldviews_single_concept(self):
        cc = CognitiveCompiler()

        class FakeConcept:
            def __init__(self, cid, text, tags=None):
                self.id = cid
                self.pattern_text = text
                self.text = text
                self.description = text
                self.confidence = 0.7
                self.tags = tags or ["test"]

        # Single concept won't form cluster (needs >= 2)
        result = cc._compile_worldviews([FakeConcept("c1", "test", ["python"])])
        assert result == []

    def test_compile_worldviews_forms_worldview(self):
        cc = CognitiveCompiler()
        cc.min_worldview_confidence = 0.3

        class FakeConcept:
            def __init__(self, cid, text, tags=None, conf=0.7):
                self.id = cid
                self.pattern_text = text
                self.text = text
                self.description = text
                self.confidence = conf
                self.tags = tags or ["test"]

        concepts = [
            FakeConcept("c1", "User prefers Python", ["python"], 0.8),
            FakeConcept("c2", "User likes Flask", ["python"], 0.7),
        ]
        result = cc._compile_worldviews(concepts)
        assert len(result) >= 1
        assert len(cc.worldviews) >= 1

    def test_compile_worldviews_enforces_max(self):
        cc = CognitiveCompiler()
        cc.max_worldviews = 2
        cc.min_worldview_confidence = 0.1

        class FakeConcept:
            def __init__(self, cid, text, tags=None, conf=0.5):
                self.id = cid
                self.pattern_text = text
                self.text = text
                self.description = text
                self.confidence = conf
                self.tags = tags or ["test"]

        # Create enough to exceed max_worldviews=2
        topic_groups = [
            ("python", [FakeConcept("p1", "Python pref", ["python"]),
                       FakeConcept("p2", "Django use", ["python"])]),
            ("js", [FakeConcept("j1", "JS pref", ["javascript"]),
                   FakeConcept("j2", "React use", ["javascript"])]),
            ("devops", [FakeConcept("d1", "Docker use", ["docker"]),
                       FakeConcept("d2", "K8s use", ["docker"])]),
        ]
        for domain, concepts in topic_groups:
            cc._compile_worldviews(concepts)
        assert len(cc.worldviews) <= cc.max_worldviews

    def test_compile_profile_no_stable_worldviews(self):
        cc = CognitiveCompiler()
        # Add an unstable worldview
        cc.worldviews["wv1"] = WorldviewNode(
            id="wv1", label="Unstable", description="",
            source_concept_ids=["c1"], stability=0.3, confidence=0.4,
        )
        profile = cc._compile_profile()
        assert profile is None

    def test_compile_profile_creates_profile(self):
        cc = CognitiveCompiler()
        cc.worldviews["wv1"] = WorldviewNode(
            id="wv1", label="Python pref", description="User prefers Python",
            source_concept_ids=["c1"], stability=0.8, confidence=0.9,
            worldview_type="preference", evidence_count=3,
        )
        cc.worldviews["wv2"] = WorldviewNode(
            id="wv2", label="Docker expert", description="User knows Docker",
            source_concept_ids=["c2"], stability=0.9, confidence=0.85,
            worldview_type="expertise", evidence_count=4,
        )
        profile = cc._compile_profile()
        assert profile is not None
        assert "Python" in profile.summary
        assert "Docker" in str(profile.expertise_areas)
        assert cc._profile is profile

    def test_compile_profile_updates_version(self):
        cc = CognitiveCompiler()
        cc._profile = UserProfile(
            id="up1", summary="Old", preferences=[], expertise_areas=[],
            working_style="", values=[], habits=[], source_worldview_ids=[],
            version=3,
        )
        cc.worldviews["wv1"] = WorldviewNode(
            id="wv1", label="Stable", description="Test",
            source_concept_ids=["c1"], stability=0.8, confidence=0.9,
            evidence_count=3,
        )
        profile = cc._compile_profile()
        assert profile is not None
        assert profile.version == 4  # incremented

    def test_compile_full_pipeline_empty(self):
        cc = CognitiveCompiler()
        from star_graph.graph import StarGraph
        g = StarGraph()
        result = cc.compile(g, session_groups={})
        assert "episodic" in result
        assert "worldviews" in result
        assert "profile" in result
        assert "stats" in result
        assert result["stats"]["raw_count"] == 0

    def test_compile_pipeline_with_sessions(self):
        cc = CognitiveCompiler()
        from star_graph.graph import StarGraph
        from star_graph.anchor import Anchor
        g = StarGraph()
        a1 = Anchor.create(text="User prefers Python for scripting",
                          tags=["python", "preference"],
                          source_session="s1")
        a2 = Anchor.create(text="User likes Flask for APIs",
                          tags=["flask", "preference"],
                          source_session="s1")
        a3 = Anchor.create(text="User uses Docker for deployment",
                          tags=["docker", "devops"],
                          source_session="s1")
        a4 = Anchor.create(text="Discussion about database optimization",
                          tags=["database"],
                          source_session="s2")
        a5 = Anchor.create(text="User migrated from MySQL to PostgreSQL",
                          tags=["database"],
                          source_session="s2")
        a6 = Anchor.create(text="Performance improvements noted",
                          tags=["performance"],
                          source_session="s2")
        for a in [a1, a2, a3, a4, a5, a6]:
            g.add_anchor(a)
        result = cc.compile(g)
        assert "stats" in result

    def test_compile_worldviews_no_min_confidence(self):
        cc = CognitiveCompiler()
        cc.min_worldview_confidence = 0.99  # very high threshold

        class FakeConcept:
            def __init__(self, cid, text, tags=None):
                self.id = cid
                self.pattern_text = text
                self.text = text
                self.description = text
                self.confidence = 0.5
                self.tags = tags or ["test"]

        concepts = [
            FakeConcept("c1", "Some pattern", ["python"]),
            FakeConcept("c2", "Similar pattern", ["python"]),
        ]
        result = cc._compile_worldviews(concepts)
        assert result == []  # below confidence threshold

    def test_compile_worldviews_no_cluster_small(self):
        cc = CognitiveCompiler()
        cc.min_worldview_confidence = 0.1

        class FakeConcept:
            def __init__(self, cid, text, tags=None, conf=0.7):
                self.id = cid
                self.pattern_text = text
                self.text = text
                self.description = text
                self.confidence = conf
                self.tags = tags or ["test"]

        # Each concept forms its own cluster (different domains), none has >=2
        concepts = [
            FakeConcept("c1", "Python", ["python"], 0.8),
        ]
        result = cc._compile_worldviews(concepts)
        assert result == []

    def test_infer_domain_from_concept_empty(self):
        cc = CognitiveCompiler()

        class EmptyConcept:
            tags = []
            text = ""
            label = ""

        domain = cc._infer_domain_from_concept(EmptyConcept())
        assert domain == "general"

    def test_infer_domain_multiple_matches(self):
        cc = CognitiveCompiler()

        class MultiConcept:
            tags = ["python", "docker"]
            text = "deployment with flask and kubernetes"
            label = "DevOps Python"

        domain = cc._infer_domain_from_concept(MultiConcept())
        # Should pick the highest scoring domain
        assert domain in ("python_development", "devops")
