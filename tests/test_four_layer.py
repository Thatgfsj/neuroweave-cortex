"""Tests for FourLayerCompressor — message→event→semantic→personality compression."""

import time
import pytest
from star_graph.four_layer import (
    FourLayerCompressor, CompressLayer, LayerConfig, CompressedMemory,
)


class TestCompressLayerEnum:
    def test_layer_values(self):
        assert CompressLayer.MESSAGE.value == 0
        assert CompressLayer.EVENT.value == 1
        assert CompressLayer.SEMANTIC.value == 2
        assert CompressLayer.PERSONALITY.value == 3

    def test_layer_order(self):
        layers = list(CompressLayer)
        assert layers[0] == CompressLayer.MESSAGE
        assert layers[-1] == CompressLayer.PERSONALITY


class TestLayerConfig:
    def test_default_config(self):
        cfg = LayerConfig()
        assert cfg.max_items == 500
        assert cfg.ttl_hours == 2.0
        assert cfg.compression_ratio == 5  # N items → 1 compressed

    def test_custom_config(self):
        cfg = LayerConfig(max_items=100, ttl_hours=1.0, compression_ratio=0.5)
        assert cfg.max_items == 100


class TestCompressedMemory:
    def test_create_entry(self):
        entry = CompressedMemory(
            id="test_1",
            text="Hello world",
            layer=CompressLayer.MESSAGE,
            importance=0.7,
            tags=["test"],
        )
        assert entry.id == "test_1"
        assert entry.layer == CompressLayer.MESSAGE
        assert entry.importance == 0.7
        assert entry.tags == ["test"]
        assert entry.source_ids == []  # default

    def test_entry_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        entry = CompressedMemory(
            id="emb_test",
            text="Test with embedding",
            layer=CompressLayer.EVENT,
            embedding=embedding,
        )
        assert entry.embedding == embedding

    def test_entry_source_ids(self):
        entry = CompressedMemory(
            id="src_test",
            text="Test with sources",
            layer=CompressLayer.SEMANTIC,
            source_ids=["a1", "a2", "a3"],
        )
        assert len(entry.source_ids) == 3


class TestFourLayerCompressorInit:
    def test_default_init(self):
        compressor = FourLayerCompressor()
        assert CompressLayer.MESSAGE in compressor.layers
        assert CompressLayer.EVENT in compressor.layers
        assert CompressLayer.SEMANTIC in compressor.layers
        assert CompressLayer.PERSONALITY in compressor.layers

    def test_layer_configs(self):
        compressor = FourLayerCompressor()
        cfg = compressor.layer_configs[CompressLayer.MESSAGE]
        assert cfg.ttl_hours > 0
        cfg_p = compressor.layer_configs[CompressLayer.PERSONALITY]
        import math
        assert math.isinf(cfg_p.ttl_hours)  # personality never decays

    def test_counter_starts_at_zero(self):
        compressor = FourLayerCompressor()
        assert compressor._counter == 0


class TestIngestMessage:
    def test_ingest_single_message(self):
        compressor = FourLayerCompressor()
        entry = compressor.ingest_message("Test message", importance=0.5)
        assert entry is not None
        assert entry.layer == CompressLayer.MESSAGE
        assert "msg_" in entry.id

    def test_ingest_multiple_messages(self):
        compressor = FourLayerCompressor()
        for i in range(5):
            compressor.ingest_message(f"Message {i}", importance=0.5)
        layer0 = compressor.layers[CompressLayer.MESSAGE]
        assert len(layer0) == 5

    def test_ingest_with_tags(self):
        compressor = FourLayerCompressor()
        entry = compressor.ingest_message("Tagged message", tags=["test", "debug"])
        assert entry is not None
        assert "test" in entry.tags
        assert "debug" in entry.tags

    def test_ingest_with_embedding(self):
        compressor = FourLayerCompressor()
        emb = [0.1] * 128
        entry = compressor.ingest_message("Embedded message", embedding=emb)
        assert entry is not None
        assert entry.embedding is not None
        assert len(entry.embedding) == 128


class TestCompressLayer0:
    def test_compress_empty(self):
        compressor = FourLayerCompressor()
        count = compressor.compress_layer0()
        assert count == 0

    def test_compress_with_messages(self):
        compressor = FourLayerCompressor()
        for i in range(10):
            compressor.ingest_message(f"Message about Python development {i}",
                                     embedding=[0.5] * 16, importance=0.6)
        events = compressor.compress_layer0()
        # With similar embeddings, should create some events
        assert isinstance(events, int)


class TestCompressLayer1:
    def test_compress_empty_layer1(self):
        compressor = FourLayerCompressor()
        count = compressor.compress_layer1()
        assert count == 0

    def test_compress_after_layer0(self):
        compressor = FourLayerCompressor()
        for i in range(10):
            compressor.ingest_message(
                f"Deployment pipeline configuration step {i}",
                embedding=[0.5 + i * 0.01] * 16,
                importance=0.6,
            )
        compressor.compress_layer0()
        # Now compress layer1
        count = compressor.compress_layer1()
        assert isinstance(count, int)


class TestCompressLayer2:
    def test_compress_empty_layer2(self):
        compressor = FourLayerCompressor()
        count = compressor.compress_layer2()
        assert count == 0


class TestDecayAll:
    def test_decay_empty(self):
        compressor = FourLayerCompressor()
        result = compressor.decay_all()
        assert result["total"] == 0

    def test_decay_with_messages(self):
        compressor = FourLayerCompressor()
        compressor.ingest_message("Old message", importance=0.5)
        # Set TTL very low — but also need to set last_accessed_at to the past
        entry = next(iter(compressor.layers[CompressLayer.MESSAGE].values()))
        entry.last_accessed_at = 0.0  # force expiry
        compressor.layer_configs[CompressLayer.MESSAGE].ttl_hours = 0.001
        result = compressor.decay_all()
        assert isinstance(result, dict)
        assert "removed" in result
        # Entry should be expired
        assert result["removed"][CompressLayer.MESSAGE] >= 1


class TestGetForRetrieval:
    def test_empty_retrieval(self):
        compressor = FourLayerCompressor()
        result = compressor.get_for_retrieval("test query")
        assert isinstance(result, dict)
        # Empty when no content exists

    def test_retrieval_with_content(self):
        compressor = FourLayerCompressor()
        for i in range(5):
            compressor.ingest_message(
                f"Test retrieval message {i}",
                embedding=[0.5] * 16,
                importance=0.6,
            )
        result = compressor.get_for_retrieval("test retrieval", max_per_layer=10)
        assert len(result["message"]) > 0

    def test_retrieval_respects_max_per_layer(self):
        compressor = FourLayerCompressor()
        for i in range(10):
            compressor.ingest_message(
                f"Message {i} for max test",
                embedding=[0.5] * 16,
                importance=0.6,
            )
        result = compressor.get_for_retrieval("Message", max_per_layer=3)
        assert len(result["message"]) <= 3


class TestCompressionPipeline:
    def test_full_pipeline(self):
        """Test the full compression pipeline: ingest → compress → retrieve."""
        compressor = FourLayerCompressor()
        # Ingest messages
        for i in range(15):
            compressor.ingest_message(
                f"User working on Python Flask web application step {i}",
                embedding=[0.5 + (i % 5) * 0.1] * 16,
                importance=0.6,
                tags=["python", "flask"],
            )
        # Run compression
        events = compressor.compress_layer0()
        semantics = compressor.compress_layer1()
        personalities = compressor.compress_layer2()

        assert isinstance(events, int)
        assert isinstance(semantics, int)
        assert isinstance(personalities, int)

        # Should be able to retrieve
        result = compressor.get_for_retrieval("python flask", max_per_layer=5)
        total = sum(len(v) for v in result.values())
        assert total > 0
