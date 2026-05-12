"""Tests for v1.0-6: multimodal memory (CLIP-style joint embedding)."""

import os
import tempfile

import pytest

from star_graph.multimodal import (
    MultimodalEmbeddingProvider,
    MultimodalAnchor,
    CrossModalRetriever,
    CrossModalResult,
)
from star_graph.anchor import Anchor


# Create a small test image
def _make_test_image(path: str):
    """Write a minimal 16x16 red PNG to the given path."""
    try:
        from PIL import Image
        img = Image.new("RGB", (16, 16), color=(255, 0, 0))
        img.save(path, "PNG")
    except ImportError:
        # Write a minimal valid PNG binary (1x1 red pixel)
        import struct, zlib
        def chunk(chunk_type, data):
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xff\x00\x00"  # filter=0, R=255, G=0, B=0
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend)


# ═══════════════════════════════════════════════════════════════
# Multimodal Embedding Provider
# ═══════════════════════════════════════════════════════════════

class TestMultimodalProvider:
    def test_encode_text_works(self):
        provider = MultimodalEmbeddingProvider()
        emb = provider.encode_text("Hello world")
        assert len(emb) > 0
        assert len(emb) == provider.text_dim
        assert any(x != 0 for x in emb)

    def test_encode_image_fallback(self):
        provider = MultimodalEmbeddingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.png")
            _make_test_image(path)
            emb = provider.encode_image(path)
            assert len(emb) == provider.image_dim

    def test_encode_image_nonexistent_raises(self):
        provider = MultimodalEmbeddingProvider()
        with pytest.raises(FileNotFoundError):
            provider.encode_image("/nonexistent/path.png")

    def test_has_cross_modal_is_bool(self):
        provider = MultimodalEmbeddingProvider()
        assert isinstance(provider.has_cross_modal, bool)

    def test_clip_available_is_bool(self):
        provider = MultimodalEmbeddingProvider()
        assert isinstance(provider.clip_available, bool)

    def test_cross_modal_similarity(self):
        provider = MultimodalEmbeddingProvider()
        text_emb = provider.encode_text("a cat")
        sim = provider.cross_modal_similarity(text_emb, [0.1] * provider.image_dim)
        # With CLIP: meaningful similarity in [0, 1]; without CLIP: returns 0
        assert 0.0 <= sim <= 1.0

    def test_image_similarity(self):
        provider = MultimodalEmbeddingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "img1.png")
            p2 = os.path.join(tmp, "img2.png")
            _make_test_image(p1)
            _make_test_image(p2)
            emb1 = provider.encode_image(p1)
            emb2 = provider.encode_image(p2)
            sim = provider.image_similarity(emb1, emb2)
            # Same images should have high similarity
            assert 0.0 <= sim <= 1.0

    def test_encode_text_batch(self):
        provider = MultimodalEmbeddingProvider()
        texts = ["hello", "world", "test"]
        embs = provider.encode_text_batch(texts)
        assert len(embs) == 3
        assert all(len(e) == provider.text_dim for e in embs)


# ═══════════════════════════════════════════════════════════════
# Multimodal Anchor
# ═══════════════════════════════════════════════════════════════

class TestMultimodalAnchor:
    def test_from_image_creates_anchor(self):
        provider = MultimodalEmbeddingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "photo.png")
            _make_test_image(path)
            anchor = MultimodalAnchor.from_image(
                path, provider, caption="A red square",
                tags=["test", "image"],
            )
            assert anchor.id
            assert anchor.image_path == path
            assert anchor.modality == "mixed"
            assert anchor.image_caption == "A red square"
            assert anchor.has_image
            assert anchor.is_multimodal
            assert anchor.image_embedding is not None

    def test_from_image_without_caption(self):
        provider = MultimodalEmbeddingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unnamed.png")
            _make_test_image(path)
            anchor = MultimodalAnchor.from_image(path, provider)
            assert anchor.modality == "image"
            assert anchor.text  # uses filename as text

    def test_from_text_and_image(self):
        provider = MultimodalEmbeddingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "diagram.png")
            _make_test_image(path)
            anchor = MultimodalAnchor.from_text_and_image(
                "Architecture diagram of the system",
                path, provider, tags=["architecture", "diagram"],
            )
            assert anchor.modality == "mixed"
            assert anchor.text == "Architecture diagram of the system"
            assert anchor.image_path == path
            assert anchor.has_image

    def test_cross_modal_score_no_image(self):
        provider = MultimodalEmbeddingProvider()
        anchor = MultimodalAnchor(
            id="test", text="text only",
            embedding=[0.1] * 384,
            modality="text",
        )
        score = anchor.cross_modal_score([0.1] * 384, provider)
        assert score == 0.0

    def test_cross_modal_score_with_image(self):
        provider = MultimodalEmbeddingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "img.png")
            _make_test_image(path)
            anchor = MultimodalAnchor.from_image(path, provider, caption="test")
            query_emb = provider.encode_text("test")
            score = anchor.cross_modal_score(query_emb, provider)
            assert 0.0 <= score <= 1.0

    def test_is_multimodal_false_for_text_only(self):
        anchor = MultimodalAnchor(
            id="t1", text="text", embedding=[0.1] * 384, modality="text",
        )
        assert not anchor.is_multimodal
        assert not anchor.has_image

    def test_has_image_false_when_empty(self):
        anchor = MultimodalAnchor(
            id="t2", text="text", embedding=[0.1] * 384, modality="text",
            image_path="",
        )
        assert not anchor.has_image


# ═══════════════════════════════════════════════════════════════
# Cross-Modal Retriever
# ═══════════════════════════════════════════════════════════════

class TestCrossModalRetriever:
    def test_retrieve_text_only(self):
        provider = MultimodalEmbeddingProvider()
        retriever = CrossModalRetriever(provider)

        # Create anchors with the same provider for consistent embedding space
        a1 = Anchor.create("The sky is blue", tags=["nature"])
        a1.embedding = provider.encode_text("The sky is blue")
        a2 = Anchor.create("Python is a programming language", tags=["tech"])
        a2.embedding = provider.encode_text("Python is a programming language")
        a3 = Anchor.create("Blue ocean waves", tags=["nature"])
        a3.embedding = provider.encode_text("Blue ocean waves")

        anchors = {"a1": a1, "a2": a2, "a3": a3}

        results = retriever.retrieve(anchors, query_text="blue", top_k=3)
        assert len(results) >= 1
        assert results[0].combined_score > 0
        assert results[0].text_score > 0

    def test_retrieve_empty_query(self):
        provider = MultimodalEmbeddingProvider()
        retriever = CrossModalRetriever(provider)
        anchors = {"a1": Anchor.create("test")}
        results = retriever.retrieve(anchors, query_text="", top_k=5)
        assert results == []

    def test_retrieve_with_multimodal_anchors(self):
        provider = MultimodalEmbeddingProvider()
        retriever = CrossModalRetriever(provider)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sunset.png")
            _make_test_image(path)
            ma = MultimodalAnchor.from_image(path, provider, caption="sunset photo")

        a1 = Anchor.create("A beautiful sunset over the ocean")
        a1.embedding = provider.encode_text("A beautiful sunset over the ocean")
        anchors = {
            "a1": a1,
            "ma1": ma,
        }

        results = retriever.retrieve(anchors, query_text="sunset", top_k=5)
        assert len(results) >= 1

    def test_image_search(self):
        provider = MultimodalEmbeddingProvider()
        retriever = CrossModalRetriever(provider)

        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "img1.png")
            p2 = os.path.join(tmp, "img2.png")
            _make_test_image(p1)
            _make_test_image(p2)
            ma1 = MultimodalAnchor.from_image(p1, provider, caption="first")
            ma2 = MultimodalAnchor.from_image(p2, provider, caption="second")

            anchors = {"ma1": ma1, "ma2": ma2}
            results = retriever.image_search(anchors, p1, top_k=2)
            assert len(results) >= 1
            assert results[0].image_score > 0

    def test_text_to_image(self):
        provider = MultimodalEmbeddingProvider()
        retriever = CrossModalRetriever(provider)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cat.png")
            _make_test_image(path)
            ma = MultimodalAnchor.from_image(path, provider, caption="a cat")

        anchors = {"ma": ma}
        results = retriever.text_to_image(anchors, "cat", top_k=5)
        assert len(results) >= 1

    def test_no_multimodal_anchors_fallback(self):
        """Retriever should still work with only text anchors."""
        provider = MultimodalEmbeddingProvider()
        retriever = CrossModalRetriever(provider)
        a1 = Anchor.create("test")
        a1.embedding = provider.encode_text("test")
        anchors = {"a1": a1}

        results = retriever.text_to_image(anchors, "test", top_k=5)
        # Text anchors come back with text_score, but image_score is 0
        assert all(r.image_score == 0.0 for r in results)
        # Text similarity works
        if len(results) >= 1:
            assert results[0].text_score > 0


# ═══════════════════════════════════════════════════════════════
# MemoryManager Integration
# ═══════════════════════════════════════════════════════════════

class TestManagerMultimodal:
    def test_remember_image(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_img.png")
            _make_test_image(path)
            anchor = mgr.remember_image(path, caption="test image", tags=["test"])
            assert anchor.id in mgr.graph.anchors
            assert isinstance(anchor, MultimodalAnchor)
            assert anchor.has_image

    def test_remember_text_and_image(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "diagram.png")
            _make_test_image(path)
            anchor = mgr.remember_text_and_image(
                "System architecture v2", path, tags=["architecture"],
            )
            assert anchor.modality == "mixed"
            assert anchor.has_image

    def test_cross_modal_recall(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        mgr.remember("Ocean waves crash on the shore", tags=["nature"])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ocean.png")
            _make_test_image(path)
            mgr.remember_image(path, caption="ocean photo", tags=["nature"])

        results = mgr.cross_modal_recall("ocean", max_items=5)
        assert len(results) >= 1

    def test_image_search(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "img1.png")
            p2 = os.path.join(tmp, "img2.png")
            _make_test_image(p1)
            _make_test_image(p2)
            mgr.remember_image(p1, caption="photo one", tags=["test"])
            mgr.remember_image(p2, caption="photo two", tags=["test"])

            results = mgr.image_search(p1, max_items=3)
            assert isinstance(results, list)

    def test_text_to_image(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sunset.png")
            _make_test_image(path)
            mgr.remember_image(path, caption="sunset photo", tags=["photo"])

        results = mgr.text_to_image("sunset", max_items=3)
        assert isinstance(results, list)

    def test_multimodal_provider_lazy_init(self):
        from star_graph import MemoryManager

        mgr = MemoryManager()
        provider = mgr.multimodal
        assert provider is not None
        # Second access returns same instance
        assert mgr.multimodal is provider


# ═══════════════════════════════════════════════════════════════
# CrossModalResult
# ═══════════════════════════════════════════════════════════════

class TestCrossModalResult:
    def test_repr(self):
        anchor = MultimodalAnchor(
            id="test", text="test", embedding=[0.1] * 384, modality="mixed",
        )
        result = CrossModalResult(
            anchor=anchor, text_score=0.8, image_score=0.3,
            combined_score=0.55, modality="text",
        )
        rep = repr(result)
        assert "CrossModalResult" in rep
        assert "0.55" in rep
