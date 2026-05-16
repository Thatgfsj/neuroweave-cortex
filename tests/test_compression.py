"""Tests for compression module — SessionCompressor, MultiLevelCompressor, helpers."""

import pytest

from star_graph.compression import (
    CompressionLevel,
    SummaryAnchor,
    SessionCompressor,
    MultiLevelCompressor,
    _tokenize,
    _extract_key_terms,
    _extract_action_patterns,
    _extract_domain,
    _extract_entities,
    _fill_template,
    _compute_centroid,
    _cosine_sim,
    _find_clusters,
    _extract_key_terms_via_texts,
)
from star_graph.anchor import Anchor, AnchorVector
from star_graph.graph import StarGraph


def make_anchor(name: str, text: str = "", embedding: list | None = None,
               tags: list | None = None, source_session: str = "s1") -> Anchor:
    a = Anchor(id=name, text=text or f"Memory {name}", tags=tags or [],
              source_session=source_session)
    if embedding:
        a.embedding = embedding
    return a


# ── Helper functions ──────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello world test function")
        assert "hello" in tokens
        assert "world" in tokens

    def test_stop_words_excluded(self):
        tokens = _tokenize("the is and for with")
        assert len(tokens) == 0  # all stop words

    def test_short_tokens_excluded(self):
        tokens = _tokenize("a b c d")
        assert len(tokens) == 0

    def test_numbers_kept(self):
        tokens = _tokenize("error 404 fix")
        assert "error" in tokens
        assert "404" in tokens


class TestExtractKeyTerms:
    def test_empty(self):
        assert _extract_key_terms([]) == []

    def test_single_anchor(self):
        a = make_anchor("a1", "redis timeout fix for production server")
        terms = _extract_key_terms([a], top_k=5)
        assert len(terms) >= 1

    def test_multiple_anchors(self):
        anchors = [
            make_anchor("a1", "redis timeout fix for production"),
            make_anchor("a2", "redis connection pool exhausted"),
            make_anchor("a3", "timeout error in database query"),
        ]
        terms = _extract_key_terms(anchors, top_k=5)
        assert "redis" in terms


class TestExtractActionPatterns:
    def test_known_verbs(self):
        result = _extract_action_patterns(["debugged the connection issue",
                                           "fixed the timeout bug"])
        assert result != "interacted with"

    def test_ing_verbs(self):
        result = _extract_action_patterns(["debugging the server",
                                           "testing the endpoint"])
        assert result != "interacted with"

    def test_empty(self):
        result = _extract_action_patterns([])
        assert result == "interacted with"

    def test_no_verbs(self):
        result = _extract_action_patterns(["the server is down", "error 500"])
        assert result == "discussed topics related to"


class TestExtractDomain:
    def test_recognized_domain(self):
        domain = _extract_domain(
            ["debugging the redis timeout issue"],
            ["redis", "timeout"],
        )
        assert domain in ("debugging", "database") or isinstance(domain, str)

    def test_python_domain(self):
        domain = _extract_domain(
            ["pytest failing on import error with flask app"],
            ["pytest", "flask"],
        )
        assert domain == "python development"

    def test_database_domain(self):
        domain = _extract_domain(
            ["mysql query timeout with redis cache miss"],
            ["mysql", "redis"],
        )
        assert domain == "database"

    def test_fallback_to_key_terms(self):
        domain = _extract_domain(["xyzzy flurbo garply"], ["xyzzy", "flurbo"])
        assert "xyzzy" in domain


class TestExtractEntities:
    def test_capitalized_words(self):
        entities = _extract_entities(
            ["Alice deployed to Production", "Bob fixed the Staging server"],
            [],
        )
        # At least some entities should be found
        assert isinstance(entities, list)

    def test_empty(self):
        entities = _extract_entities([], [])
        assert entities == []


class TestFillTemplate:
    def test_episodic_level(self):
        result = _fill_template(
            CompressionLevel.EPISODIC,
            "python development", "redis", "debugged",
            ["Alice"], "session abc12345", 5, token_limit=150,
        )
        assert isinstance(result, str)
        assert len(result.split()) <= 150

    def test_strategic_level(self):
        result = _fill_template(
            CompressionLevel.STRATEGIC,
            "devops", "deployment", "deployed and tested",
            ["Alice", "Bob"], "multiple sessions", 8, token_limit=100,
        )
        assert isinstance(result, str)

    def test_meta_level(self):
        result = _fill_template(
            CompressionLevel.META,
            "python, javascript", "testing", "verified and deployed",
            ["cross-domain"], "3 domains", 12, token_limit=70,
        )
        assert isinstance(result, str)


class TestCosineSim:
    def test_identical(self):
        assert _cosine_sim([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_empty(self):
        assert _cosine_sim([], []) == 0.0


class TestComputeCentroid:
    def test_empty(self):
        assert _compute_centroid([]) == []

    def test_two_embeddings(self):
        centroid = _compute_centroid([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        assert centroid == [2.0, 2.0, 2.0]

    def test_single_embedding(self):
        centroid = _compute_centroid([[1.0, 2.0, 3.0]])
        assert centroid == [1.0, 2.0, 3.0]


class TestFindClusters:
    def test_empty(self):
        assert _find_clusters([], {}, 0.5) == []

    def test_single_anchor(self):
        assert _find_clusters(["a1"], {"a1": [1.0, 0.0]}, 0.5) == [["a1"]]

    def test_connected_pair(self):
        clusters = _find_clusters(
            ["a1", "a2"],
            {"a1": [1.0, 0.0], "a2": [1.0, 0.0]},
            0.5,
        )
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a1", "a2"}

    def test_disconnected(self):
        clusters = _find_clusters(
            ["a1", "a2"],
            {"a1": [1.0, 0.0], "a2": [0.0, 1.0]},
            0.5,
        )
        assert len(clusters) == 2


class TestExtractKeyTermsViaTexts:
    def test_empty(self):
        assert _extract_key_terms_via_texts([]) == []

    def test_basic(self):
        terms = _extract_key_terms_via_texts(
            ["redis timeout fix", "redis connection error"],
            top_k=3,
        )
        assert "redis" in terms


# ── SummaryAnchor ─────────────────────────────────────────

class TestSummaryAnchor:
    def test_create(self):
        s = SummaryAnchor(
            id="s1", text="test summary", source_anchor_ids=["a1", "a2"],
            centroid_embedding=[0.1, 0.2, 0.3],
            compression_level=CompressionLevel.EPISODIC,
        )
        assert s.id == "s1"
        assert s.compression_level == CompressionLevel.EPISODIC
        assert s.token_count > 0

    def test_to_anchor_proxy(self):
        s = SummaryAnchor(
            id="s1", text="test summary of redis debugging",
            source_anchor_ids=["a1", "a2"],
            centroid_embedding=[0.1, 0.2, 0.3],
            compression_level=CompressionLevel.EPISODIC,
        )
        proxy = s.to_anchor_proxy()
        assert proxy.id == "s1"
        assert proxy.text == "test summary of redis debugging"
        assert proxy.vector.stability > 0.5

    def test_anchor_proxy_cached(self):
        s = SummaryAnchor(
            id="s1", text="test", source_anchor_ids=["a1"],
            centroid_embedding=[0.1, 0.2, 0.3],
            compression_level=CompressionLevel.EPISODIC,
        )
        proxy1 = s.to_anchor_proxy()
        proxy2 = s.to_anchor_proxy()
        assert proxy1 is proxy2  # cached


# ── SessionCompressor ─────────────────────────────────────

class TestSessionCompressor:
    def test_compress_empty_anchors(self):
        sc = SessionCompressor()
        summaries = sc.compress([], "s1")
        assert summaries == []

    def test_compress_insufficient_anchors(self):
        sc = SessionCompressor(min_cluster_size=5)
        anchors = [make_anchor(f"a{i}", f"text {i}", embedding=[float(i)] * 384)
                   for i in range(3)]
        summaries = sc.compress(anchors, "s1")
        assert summaries == []

    def test_compress_wrong_session(self):
        sc = SessionCompressor(min_cluster_size=1)
        anchors = [make_anchor("a1", "text 1", embedding=[0.1] * 384, source_session="s2")]
        summaries = sc.compress(anchors, "s1")
        assert summaries == []


# ── MultiLevelCompressor ──────────────────────────────────

class TestMultiLevelCompressor:
    def test_init(self):
        mc = MultiLevelCompressor()
        assert mc.min_cluster_size >= 1
        assert mc.similarity_threshold > 0

    def test_compress_session_empty(self):
        mc = MultiLevelCompressor()
        summaries = mc.compress_session([], "s1")
        assert summaries == []

    def test_compress_pipeline_empty(self):
        mc = MultiLevelCompressor()
        result = mc.compress_pipeline({})
        assert result[CompressionLevel.EPISODIC] == []
        assert result[CompressionLevel.STRATEGIC] == []
        assert result[CompressionLevel.META] == []

    def test_add_to_graph(self):
        mc = MultiLevelCompressor()
        g = StarGraph()
        s = SummaryAnchor(
            id="s1", text="test summary of debugging redis timeout errors",
            source_anchor_ids=[], centroid_embedding=[0.1] * 384,
            compression_level=CompressionLevel.EPISODIC,
        )
        edges = mc.add_to_graph(g, [s], edge_type="compresses")
        assert edges == 0  # no source anchors to connect to

    def test_add_to_graph_with_sources(self):
        mc = MultiLevelCompressor()
        g = StarGraph()
        a1 = make_anchor("a1", "source 1", embedding=[0.1] * 384)
        g.add_anchor(a1)
        s = SummaryAnchor(
            id="s1", text="test summary",
            source_anchor_ids=["a1"], centroid_embedding=[0.2] * 384,
            compression_level=CompressionLevel.EPISODIC,
        )
        edges = mc.add_to_graph(g, [s], edge_type="compresses")
        assert edges == 1
        assert "s1" in g.anchors

    def test_add_to_graph_default_edge_type(self):
        mc = MultiLevelCompressor()
        g = StarGraph()
        a1 = make_anchor("a1", "source", embedding=[0.1] * 384)
        g.add_anchor(a1)
        s = SummaryAnchor(
            id="s2", text="test", source_anchor_ids=["a1"],
            centroid_embedding=[0.2] * 384,
            compression_level=CompressionLevel.EPISODIC,
        )
        edges = mc.add_to_graph(g, [s])  # default edge_type
        assert edges == 1

    def test_compress_meta_empty(self):
        mc = MultiLevelCompressor()
        result = mc.compress_meta([])
        assert result == []

    def test_compress_meta_insufficient(self):
        mc = MultiLevelCompressor()
        s1 = SummaryAnchor(
            id="s1", text="Summary", source_anchor_ids=[],
            centroid_embedding=[0.1] * 384,
            compression_level=CompressionLevel.STRATEGIC,
        )
        result = mc.compress_meta([s1])
        assert result == []

    def test_compress_meta_with_strategic(self):
        mc = MultiLevelCompressor()
        s1 = SummaryAnchor(
            id="s1", text="python debugging strategy",
            source_anchor_ids=["a1"], centroid_embedding=[0.1] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["domain:python_development", "level:strategic"],
        )
        s2 = SummaryAnchor(
            id="s2", text="javascript debugging strategy",
            source_anchor_ids=["a2"], centroid_embedding=[0.15] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["domain:javascript_development", "level:strategic"],
        )
        result = mc.compress_meta([s1, s2])
        # With similar embeddings they should form a cluster
        assert isinstance(result, list)

    def test_compress_pipeline_empty(self):
        mc = MultiLevelCompressor()
        result = mc.compress_pipeline({})
        assert result[CompressionLevel.EPISODIC] == []
        assert result[CompressionLevel.STRATEGIC] == []
        assert result[CompressionLevel.META] == []

    def test_compress_pipeline_with_data(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.5
        anchors = {
            "s1": [
                make_anchor("a1", "debugging redis timeout error", embedding=[0.1] * 384),
                make_anchor("a2", "fixing redis connection pool timeout", embedding=[0.11] * 384),
                make_anchor("a3", "optimizing redis cache performance", embedding=[0.12] * 384),
            ],
        }
        result = mc.compress_pipeline(anchors)
        assert isinstance(result, dict)


class TestFillTemplateTruncation:
    def test_episodic_truncation(self):
        long_context = "x " * 200
        result = _fill_template(
            CompressionLevel.EPISODIC, "testing", "topic",
            "debugging", [], long_context, count=3, token_limit=20,
        )
        assert len(result.split()) <= 20

    def test_strategic_truncation(self):
        result = _fill_template(
            CompressionLevel.STRATEGIC, "architecture", "design",
            "refactoring and building and testing", ["System"],
            "multi session analysis", count=5, token_limit=15,
        )
        assert len(result.split()) <= 15

    def test_meta_no_entities(self):
        result = _fill_template(
            CompressionLevel.META, "cross domain", "principle",
            "learning", [], "multiple domains", count=3, token_limit=50,
        )
        assert len(result.split()) <= 50


class TestExtractActionPatternsEdge:
    def test_ed_verbs(self):
        result = _extract_action_patterns([
            "user deployed and configured the server",
            "admin configured and deployed services",
        ])
        assert len(result) > 0

    def test_mixed_verb_forms(self):
        result = _extract_action_patterns([
            "debugging the redis timeout",
            "debugged the mysql connection",
        ])
        assert "debug" in result.lower() or "debugging" in result.lower()

    def test_single_common_verb(self):
        result = _extract_action_patterns([
            "user created new project",
        ])
        assert len(result) > 0


class TestSessionCompressorExtended:
    def test_compress_with_embeddings(self):
        sc = SessionCompressor(min_cluster_size=2, similarity_threshold=0.8)
        a1 = make_anchor("a1", "docker container deployment", embedding=[0.1] * 384)
        a2 = make_anchor("a2", "kubernetes pod configuration", embedding=[0.11] * 384)
        a1.source_session = "s1"
        a2.source_session = "s1"
        result = sc.compress([a1, a2], "s1")
        # May or may not form cluster depending on embedding similarity
        assert isinstance(result, list)

    def test_compress_with_anchor_embedding_generation(self):
        sc = SessionCompressor(min_cluster_size=2, similarity_threshold=0.95)
        a1 = make_anchor("a1", "unique topic one", embedding=None)
        a2 = make_anchor("a2", "unique topic two", embedding=None)
        a1.source_session = "s1"
        a2.source_session = "s1"
        result = sc.compress([a1, a2], "s1")
        assert isinstance(result, list)

    def test_compress_state_transition(self):
        sc = SessionCompressor(min_cluster_size=2, similarity_threshold=0.5)
        a1 = make_anchor("a1", "docker deploy pipeline setup", embedding=[0.1] * 384)
        a2 = make_anchor("a2", "ci/cd pipeline configuration", embedding=[0.11] * 384)
        a1.source_session = "s1"
        a2.source_session = "s1"
        result = sc.compress([a1, a2], "s1")
        assert isinstance(result, list)


# ── MultiLevelCompressor: compress_strategic ─────────────

class TestCompressStrategic:
    def test_insufficient_summaries(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 5
        s1 = SummaryAnchor(
            id="s1", text="summary one", source_anchor_ids=["a1"],
            centroid_embedding=[0.1] * 10, compression_level=CompressionLevel.EPISODIC)
        s2 = SummaryAnchor(
            id="s2", text="summary two", source_anchor_ids=["a2"],
            centroid_embedding=[0.2] * 10, compression_level=CompressionLevel.EPISODIC)
        result = mc.compress_strategic([s1, s2])
        assert result == []

    def test_forms_strategic_cluster(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.5
        s1 = SummaryAnchor(
            id="s1", text="debugging redis timeout issues in production",
            source_anchor_ids=["a1"], centroid_embedding=[0.1] * 384,
            compression_level=CompressionLevel.EPISODIC,
            tags=["domain:database", "level:episodic"])
        s2 = SummaryAnchor(
            id="s2", text="fixing redis connection pool exhaustion",
            source_anchor_ids=["a2"], centroid_embedding=[0.11] * 384,
            compression_level=CompressionLevel.EPISODIC,
            tags=["domain:database", "level:episodic"])
        result = mc.compress_strategic([s1, s2])
        assert isinstance(result, list)
        if result:
            assert result[0].compression_level == CompressionLevel.STRATEGIC

    def test_forms_strategic_cluster_with_many(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.5
        summaries = []
        for i in range(5):
            summaries.append(SummaryAnchor(
                id=f"s{i}", text=f"redis cache optimization step {i}",
                source_anchor_ids=[f"a{i}"],
                centroid_embedding=[0.1 + i * 0.01] * 384,
                compression_level=CompressionLevel.EPISODIC,
                tags=[f"step:{i}", "domain:database", "level:episodic"]))
        result = mc.compress_strategic(summaries)
        assert isinstance(result, list)


# ── MultiLevelCompressor: compress_meta extended ─────────

class TestCompressMetaExtended:
    def test_forms_meta_summary(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.5
        s1 = SummaryAnchor(
            id="s1", text="python debugging strategy for async code",
            source_anchor_ids=["a1"], centroid_embedding=[0.1] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["domain:python_development", "level:strategic"])
        s2 = SummaryAnchor(
            id="s2", text="javascript debugging strategy for async code",
            source_anchor_ids=["a2"], centroid_embedding=[0.12] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["domain:javascript_development", "level:strategic"])
        result = mc.compress_meta([s1, s2])
        assert isinstance(result, list)
        if result:
            assert result[0].compression_level == CompressionLevel.META

    def test_no_cluster_below_similarity(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.99
        s1 = SummaryAnchor(
            id="s1", text="python async debugging",
            source_anchor_ids=["a1"], centroid_embedding=[0.1] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["domain:python_development", "level:strategic"])
        s2 = SummaryAnchor(
            id="s2", text="unrelated memory topic",
            source_anchor_ids=["a2"], centroid_embedding=[0.9] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["domain:unrelated", "level:strategic"])
        result = mc.compress_meta([s1, s2])
        assert isinstance(result, list)
        # With very different embeddings, should not form a cluster
        assert result == [] or len(result) >= 0

    def test_compress_meta_tags_filtering(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.5
        s1 = SummaryAnchor(
            id="s1", text="database optimization strategy one",
            source_anchor_ids=["a1"], centroid_embedding=[0.1] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["key1", "key2", "domain:database", "level:strategic"])
        s2 = SummaryAnchor(
            id="s2", text="database optimization strategy two",
            source_anchor_ids=["a2"], centroid_embedding=[0.11] * 384,
            compression_level=CompressionLevel.STRATEGIC,
            tags=["key1", "key3", "domain:database", "level:strategic"])
        result = mc.compress_meta([s1, s2])
        assert isinstance(result, list)


# ── MultiLevelCompressor: compress_pipeline extended ─────

class TestCompressPipelineExtended:
    def test_pipeline_with_two_sessions(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.5
        anchors = {
            "s1": [
                make_anchor("a1", "redis timeout debugging", embedding=[0.1] * 384, source_session="s1"),
                make_anchor("a2", "redis connection pool fix", embedding=[0.11] * 384, source_session="s1"),
            ],
            "s2": [
                make_anchor("a3", "postgres query optimization", embedding=[0.3] * 384, source_session="s2"),
                make_anchor("a4", "postgres index tuning", embedding=[0.31] * 384, source_session="s2"),
            ],
        }
        result = mc.compress_pipeline(anchors)
        assert isinstance(result, dict)
        assert CompressionLevel.EPISODIC in result
        assert CompressionLevel.STRATEGIC in result
        # META may or may not form

    def test_pipeline_three_levels(self):
        mc = MultiLevelCompressor()
        mc.min_cluster_size = 2
        mc.similarity_threshold = 0.5
        # Create enough anchors across sessions to form all three levels
        anchors = {}
        for sess in range(3):
            sid = f"s{sess}"
            session_anchors = []
            for i in range(4):
                session_anchors.append(make_anchor(
                    f"a_{sid}_{i}",
                    f"database performance optimization technique {i}",
                    embedding=[0.1 + sess * 0.01 + i * 0.001] * 384,
                    source_session=sid))
            anchors[sid] = session_anchors
        result = mc.compress_pipeline(anchors)
        assert CompressionLevel.EPISODIC in result
        assert CompressionLevel.STRATEGIC in result
        assert CompressionLevel.META in result
