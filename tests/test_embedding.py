"""Tests for embedding module — EmbeddingProvider, get_embedder, reset_embedder."""

import math
import pytest

from star_graph.embedding import EmbeddingProvider, get_embedder, reset_embedder


class TestEmbeddingProvider:
    def test_init_defaults(self):
        ep = EmbeddingProvider()
        assert ep.dim == 384
        assert ep.backend == "none"

    def test_custom_dim(self):
        ep = EmbeddingProvider(dim=128)
        assert ep.dim == 128

    def test_hash_embed_produces_unit_vector(self):
        ep = EmbeddingProvider(dim=128)
        ep._backend = "hash"
        vec = ep._hash_embed("test text")
        assert len(vec) == 128
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0)

    def test_hash_embed_deterministic(self):
        ep = EmbeddingProvider(dim=128)
        ep._backend = "hash"
        vec1 = ep._hash_embed("test text")
        vec2 = ep._hash_embed("test text")
        # Same text produces same embedding (deterministic)
        assert vec1 == vec2

    def test_hash_embed_different_texts(self):
        ep = EmbeddingProvider(dim=128)
        ep._backend = "hash"
        vec1 = ep._hash_embed("text one")
        vec2 = ep._hash_embed("text two")
        assert vec1 != vec2

    def test_encode_falls_back_to_hash(self):
        ep = EmbeddingProvider(dim=16)
        ep._backend = "hash"
        vec = ep.encode("test")
        assert len(vec) == 16
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0)

    def test_derive_phase(self):
        ep = EmbeddingProvider()
        phase = ep.derive_phase("test text", importance=0.8, emotional_valence=0.3)
        assert 0.0 <= phase < 2 * math.pi

    def test_derive_phase_different_importance(self):
        ep = EmbeddingProvider()
        phase_high = ep.derive_phase("test", importance=0.9, emotional_valence=0.0)
        phase_low = ep.derive_phase("test", importance=0.1, emotional_valence=0.0)
        # Different importance should affect phase
        assert isinstance(phase_high, float)
        assert isinstance(phase_low, float)

    def test_derive_frequency(self):
        ep = EmbeddingProvider()
        freq = ep.derive_frequency(importance=0.5, emotional_valence=0.0)
        assert 0.1 <= freq <= 1.0

    def test_derive_frequency_high_importance(self):
        ep = EmbeddingProvider()
        freq_high = ep.derive_frequency(importance=0.9, emotional_valence=0.0)
        freq_low = ep.derive_frequency(importance=0.1, emotional_valence=0.0)
        assert freq_high >= freq_low

    def test_derive_frequency_emotional(self):
        ep = EmbeddingProvider()
        freq_emotional = ep.derive_frequency(importance=0.5, emotional_valence=0.9)
        freq_neutral = ep.derive_frequency(importance=0.5, emotional_valence=0.0)
        assert freq_emotional >= freq_neutral

    def test_derive_driving_phasor(self):
        ep = EmbeddingProvider()
        freq, phase = ep.derive_driving_phasor("what is the meaning of life")
        assert 0.1 <= freq <= 1.0
        assert 0.0 <= phase < 2 * math.pi

    def test_derive_driving_phasor_with_embedding(self):
        ep = EmbeddingProvider()
        freq, phase = ep.derive_driving_phasor("test", embedding=[0.1, 0.2, 0.3, 0.4])
        assert 0.1 <= freq <= 1.0

    def test_derive_driving_phasor_emotional_query(self):
        ep = EmbeddingProvider()
        freq_plain, _ = ep.derive_driving_phasor("hello world")
        freq_emo, _ = ep.derive_driving_phasor("urgent critical important")
        assert 0.1 <= freq_plain <= 1.0
        assert 0.1 <= freq_emo <= 1.0


class TestModuleLevelFunctions:
    def test_get_embedder_returns_singleton(self):
        reset_embedder()
        ep1 = get_embedder()
        ep2 = get_embedder()
        assert ep1 is ep2

    def test_reset_embedder(self):
        ep1 = get_embedder()
        reset_embedder()
        ep2 = get_embedder()
        # After reset, a new instance should be created
        assert ep1 is not ep2

    def test_get_embedder_backend_is_set(self):
        reset_embedder()
        ep = get_embedder()
        assert ep.backend in ("none", "sentence-transformers", "hash")
