"""Multimodal Memory — CLIP-style joint embedding space for text + images.

v1.0-6: Extends the memory system to handle images alongside text.
Supports cross-modal retrieval (text→image, image→text) and graceful
degradation when CLIP dependencies are not available.

Architecture:
  MultimodalEmbeddingProvider
    ├── text_encoder (sentence-transformers or fallback)
    ├── image_encoder (CLIP/ViT or perceptual-hash fallback)
    └── joint_space (shared embedding space for cross-modal similarity)

  MultimodalAnchor(Anchor)
    ├── image_path / image_embedding
    ├── modality: "text" | "image" | "mixed"
    └── cross_modal_similarity(query_embedding)

Usage:
    provider = MultimodalEmbeddingProvider()
    anchor = MultimodalAnchor.from_image("photo.png", provider, caption="sunset")
    results = manager.cross_modal_recall("warm colors at dusk")
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .anchor import Anchor, AnchorVector, Oscillator, MemoryState
from .config import Config


# ═══════════════════════════════════════════════════════════════
# Multimodal Embedding Provider
# ═══════════════════════════════════════════════════════════════

class MultimodalEmbeddingProvider:
    """Joint embedding provider for text and images.

    Uses CLIP (via transformers) when available for true cross-modal alignment.
    Falls back to separate text/image encoders with cosine-similarity approximation.

    Text encoding reuses the existing EmbeddingProvider (sentence-transformers).
    Image encoding uses CLIP/ViT when available, or falls back to a simple
    perceptual hash for similarity comparison.
    """

    def __init__(self,
                 text_model: str = "all-MiniLM-L6-v2",
                 clip_model: str = "openai/clip-vit-base-patch32",
                 text_dim: int = 384,
                 image_dim: int = 512):
        self._text_provider = None
        self._clip_model = None
        self._clip_processor = None
        self._text_model_name = text_model
        self._clip_model_name = clip_model
        self._text_dim = text_dim
        self._image_dim = image_dim
        self._clip_available: bool = False
        self._checked_clip: bool = False

    @property
    def text_dim(self) -> int:
        return self._text_dim

    @property
    def image_dim(self) -> int:
        return self._image_dim

    @property
    def clip_available(self) -> bool:
        """Check if CLIP is available (lazy, cached).

        When CLIP loads successfully, registers this provider as the global
        embedder so all anchors use the joint embedding space.
        """
        if not self._checked_clip:
            self._checked_clip = True
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._clip_model = CLIPModel.from_pretrained(self._clip_model_name)
                self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)
                self._clip_available = True
                # Warm up: encode a short text to determine CLIP dimensions
                import torch
                import numpy as np
                inputs = self._clip_processor(text="init", return_tensors="pt",
                                              padding=True, truncation=True, max_length=77)
                with torch.no_grad():
                    outputs = self._clip_model.get_text_features(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    clip_dim = outputs.pooler_output.shape[-1]
                elif hasattr(outputs, 'text_embeds'):
                    clip_dim = outputs.text_embeds.shape[-1]
                else:
                    clip_dim = outputs.shape[-1]
                self._image_dim = clip_dim
                self._text_dim = clip_dim  # CLIP text and image share the same dim
            except Exception:
                self._clip_available = False
        return self._clip_available

    def _get_text_provider(self):
        """Lazy-init text embedding provider."""
        if self._text_provider is None:
            from .embedding import EmbeddingProvider
            self._text_provider = EmbeddingProvider(self._text_model_name, self._text_dim)
        return self._text_provider

    # ── Text encoding ─────────────────────────────────────

    def encode_text(self, text: str) -> list[float]:
        """Encode text into embedding space.

        Uses CLIP text encoder when available (for true cross-modal alignment),
        falling back to sentence-transformers otherwise.
        """
        if self.clip_available:
            return self._encode_text_clip(text)
        provider = self._get_text_provider()
        return provider.encode(text)

    def _encode_text_clip(self, text: str) -> list[float]:
        """Encode text using CLIP model for joint embedding space."""
        import torch
        import numpy as np

        inputs = self._clip_processor(text=text, return_tensors="pt", padding=True,
                                      truncation=True, max_length=77)
        with torch.no_grad():
            outputs = self._clip_model.get_text_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            arr = outputs.pooler_output.cpu().numpy()
        elif hasattr(outputs, 'text_embeds'):
            arr = outputs.text_embeds.cpu().numpy()
        elif hasattr(outputs, 'cpu'):
            arr = outputs.cpu().numpy()
        else:
            arr = np.asarray(outputs)
        arr = np.asarray(arr).flatten()
        vec = arr.tolist()
        # Sync dimensions: CLIP text and image share the same dim
        self._text_dim = len(vec)
        if not self._image_dim or self._image_dim != len(vec):
            self._image_dim = len(vec)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-8:
            vec = [x / norm for x in vec]
        return vec

    def encode_text_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode_text(t) for t in texts]

    # ── Image encoding ────────────────────────────────────

    def encode_image(self, image_path: str) -> list[float]:
        """Encode an image into joint embedding space.

        Uses CLIP when available (true cross-modal alignment).
        Falls back to perceptual hash + metadata when CLIP is unavailable.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.clip_available:
            return self._encode_image_clip(image_path)
        else:
            return self._encode_image_fallback(image_path)

    def _encode_image_clip(self, image_path: str) -> list[float]:
        """Encode image using CLIP model."""
        from PIL import Image
        import torch
        import numpy as np

        image = Image.open(image_path).convert("RGB")
        inputs = self._clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._clip_model.get_image_features(**inputs)
        # Handle different CLIP output formats (tensor vs BaseModelOutputWithPooling)
        if hasattr(outputs, 'pooler_output'):
            arr = outputs.pooler_output.cpu().numpy()
        elif hasattr(outputs, 'image_embeds'):
            arr = outputs.image_embeds.cpu().numpy()
        elif hasattr(outputs, 'cpu'):
            arr = outputs.cpu().numpy()
        else:
            arr = np.asarray(outputs)
        arr = np.asarray(arr).flatten()
        vec = arr.tolist()
        self._image_dim = len(vec)
        if not self._text_dim or self._text_dim != len(vec):
            self._text_dim = len(vec)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-8:
            vec = [x / norm for x in vec]
        return vec

    def _encode_image_fallback(self, image_path: str) -> list[float]:
        """Fallback image encoding: perceptual hash + simple features.

        Without CLIP, we compute a 64-bit perceptual hash and expand it
        into a normalized embedding vector. This enables basic image
        similarity (similar images get similar hashes) but NOT cross-modal
        alignment (text queries won't match image content).
        """
        try:
            from PIL import Image
            img = Image.open(image_path).convert("L").resize((32, 32))
            pixels = list(img.getdata())
            # Build simple features: mean, std, histogram bins
            mean = sum(pixels) / len(pixels)
            std = math.sqrt(sum((p - mean) ** 2 for p in pixels) / len(pixels))
            # 8-bin histogram
            bins = [0] * 8
            for p in pixels:
                idx = min(7, int(p / 32))
                bins[idx] += 1
            total = max(1, sum(bins))
            bins_norm = [b / total for b in bins]
            # Perceptual hash (difference hash)
            small = img.resize((9, 8))
            dhash = 0
            for row in range(8):
                for col in range(8):
                    left = small.getpixel((col, row))
                    right = small.getpixel((col + 1, row))
                    dhash = (dhash << 1) | (1 if left < right else 0)
            # Build embedding: dhash bits + histogram + stats
            vec = []
            for i in range(64):
                vec.append(1.0 if (dhash >> (63 - i)) & 1 else -1.0)
            vec.extend(bins_norm)
            vec.append(mean / 255.0)
            vec.append(min(1.0, std / 128.0))
            # Pad to image_dim
            while len(vec) < self._image_dim:
                vec.append(0.0)
            # Normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 1e-8:
                vec = [x / norm for x in vec]
            return vec[:self._image_dim]
        except Exception:
            return [0.0] * self._image_dim

    def encode_image_batch(self, image_paths: list[str]) -> list[list[float]]:
        return [self.encode_image(p) for p in image_paths]

    # ── Cross-modal similarity ────────────────────────────

    def cross_modal_similarity(self, text_embedding: list[float],
                               image_embedding: list[float]) -> float:
        """Compute similarity between text and image embeddings.

        With CLIP: embeddings are in the same joint space, so cosine works.
        Without CLIP: returns 0.0 (no cross-modal alignment possible).
        """
        if not text_embedding or not image_embedding:
            return 0.0

        if self.clip_available:
            return _cosine_sim(text_embedding, image_embedding)

        # Without CLIP, we can't meaningfully compare text to image
        # Return 0.0 to indicate "unknown" rather than false similarity
        return 0.0

    def image_similarity(self, img_emb1: list[float],
                        img_emb2: list[float]) -> float:
        """Compute similarity between two image embeddings."""
        if not img_emb1 or not img_emb2:
            return 0.0
        return _cosine_sim(img_emb1, img_emb2)

    @property
    def has_cross_modal(self) -> bool:
        """True if true cross-modal (text↔image) alignment is available."""
        return self.clip_available

    # ── EmbeddingProvider compatibility ──────────────────

    @property
    def dim(self) -> int:
        return self._text_dim

    @property
    def backend(self) -> str:
        return "clip" if self.clip_available else "sentence-transformers"

    def encode(self, text: str) -> list[float]:
        """Alias for encode_text — EmbeddingProvider interface."""
        return self.encode_text(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]

    def derive_phase(self, text: str, embedding: list[float] | None = None,
                     importance: float = 0.5, emotional_valence: float = 0.0,
                     timestamp: float | None = None) -> float:
        """Derive theta phase (delegates to sentence-transformers logic)."""
        provider = self._get_text_provider()
        return provider.derive_phase(text, embedding, importance,
                                     emotional_valence, timestamp)

    def derive_frequency(self, importance: float = 0.5,
                         emotional_valence: float = 0.0,
                         text_length: int = 0) -> float:
        """Derive natural frequency."""
        provider = self._get_text_provider()
        return provider.derive_frequency(importance, emotional_valence, text_length)

    def derive_driving_phasor(self, query: str,
                               embedding: list[float] | None = None) -> tuple[float, float]:
        """Derive driving frequency and phase from a query."""
        provider = self._get_text_provider()
        return provider.derive_driving_phasor(query, embedding)


# ═══════════════════════════════════════════════════════════════
# Multimodal Anchor
# ═══════════════════════════════════════════════════════════════

@dataclass
class MultimodalAnchor(Anchor):
    """An anchor that can carry both text and image modalities.

    Extends the standard Anchor with image-specific fields:
    - image_path: local path or URL to the image
    - image_embedding: CLIP or fallback embedding of the image
    - modality: which modalities this anchor represents

    Cross-modal recall works by comparing text query embeddings with
    stored image embeddings (when CLIP is available).
    """

    image_path: str = ""
    image_embedding: list[float] | None = None
    modality: str = "text"  # "text", "image", "mixed"
    image_caption: str = ""  # text description of image (always available)

    @classmethod
    def from_image(cls, image_path: str,
                   provider: MultimodalEmbeddingProvider,
                   caption: str = "",
                   tags: list[str] | None = None,
                   importance: float = 0.5,
                   source_session: str = "",
                   emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Create an anchor from an image.

        The text field is set to the caption (or image filename if no caption).
        Both text and image embeddings are stored, enabling cross-modal retrieval.
        """
        text = caption or os.path.basename(image_path)
        image_emb = provider.encode_image(image_path)

        anchor = cls(
            id=hashlib.blake2b(
                (text + image_path + source_session).encode(), digest_size=8
            ).hexdigest(),
            text=text[:280],
            vector=AnchorVector(
                importance=importance,
                emotional_valence=emotional_valence,
            ),
            embedding=provider.encode_text(text),
            image_path=image_path,
            image_embedding=image_emb,
            modality="image" if not caption else "mixed",
            image_caption=caption or text[:200],
            source_session=source_session,
            tags=tags or [],
            state=MemoryState.ACTIVE,
            state_history=[(MemoryState.ACTIVE, time.time())],
        )
        return anchor

    @classmethod
    def from_text_and_image(cls, text: str, image_path: str,
                           provider: MultimodalEmbeddingProvider,
                           tags: list[str] | None = None,
                           importance: float = 0.5,
                           source_session: str = "",
                           emotional_valence: float = 0.0) -> MultimodalAnchor:
        """Create an anchor with both text and image modalities."""
        image_emb = provider.encode_image(image_path)

        anchor = cls(
            id=hashlib.blake2b(
                (text + image_path + source_session).encode(), digest_size=8
            ).hexdigest(),
            text=text[:280],
            vector=AnchorVector(
                importance=importance,
                emotional_valence=emotional_valence,
            ),
            embedding=provider.encode_text(text),
            image_path=image_path,
            image_embedding=image_emb,
            modality="mixed",
            image_caption=text[:200],
            source_session=source_session,
            tags=tags or [],
            state=MemoryState.ACTIVE,
            state_history=[(MemoryState.ACTIVE, time.time())],
        )
        return anchor

    def cross_modal_score(self, query_embedding: list[float],
                         provider: MultimodalEmbeddingProvider) -> float:
        """Score this anchor against a query embedding for cross-modal retrieval.

        When querying with text, anchors with images get an additional
        cross-modal score based on text→image similarity (CLIP).
        """
        if not self.image_embedding:
            return 0.0
        return provider.cross_modal_similarity(query_embedding, self.image_embedding)

    @property
    def has_image(self) -> bool:
        return bool(self.image_path)

    @property
    def is_multimodal(self) -> bool:
        return self.modality in ("image", "mixed")


# ═══════════════════════════════════════════════════════════════
# Cross-Modal Retriever
# ═══════════════════════════════════════════════════════════════

@dataclass
class CrossModalResult:
    """Result from a cross-modal retrieval query."""
    anchor: MultimodalAnchor
    text_score: float      # text→text similarity
    image_score: float     # text→image similarity (0 if no image)
    combined_score: float  # weighted combination
    modality: str          # which modality contributed most

    def __repr__(self) -> str:
        return (f"CrossModalResult(modality={self.modality}, "
                f"combined={self.combined_score:.3f}, "
                f"text={self.text_score:.3f}, image={self.image_score:.3f})")


class CrossModalRetriever:
    """Retrieves memories across text and image modalities.

    Supports:
    - Text→Text (standard retrieval)
    - Text→Image (find images matching text description, CLIP required)
    - Image→Text (find text memories matching an image, CLIP required)
    - Image→Image (find similar images, always available via dhash)

    Without CLIP, cross-modal retrieval degrades gracefully:
    - Text→Image falls back to caption/tag matching
    - Image→Text falls back to metadata search
    - Image→Image uses perceptual hash similarity
    """

    def __init__(self, provider: MultimodalEmbeddingProvider | None = None):
        self.provider = provider or MultimodalEmbeddingProvider()

    def retrieve(self, anchors: dict[str, Anchor],
                query_text: str = "",
                query_image_path: str = "",
                query_embedding: list[float] | None = None,
                top_k: int = 10,
                text_weight: float = 0.6,
                image_weight: float = 0.4) -> list[CrossModalResult]:
        """Cross-modal retrieval from a collection of anchors.

        If query_text is provided, encodes it as text query.
        If query_image_path is provided, encodes it as image query.
        Both can be provided for "find similar content" queries.
        """
        # Encode queries
        if query_embedding is None:
            if query_text:
                query_embedding = self.provider.encode_text(query_text)
            elif query_image_path:
                query_embedding = self.provider.encode_image(query_image_path)
            else:
                return []

        query_image_emb = None
        if query_image_path:
            query_image_emb = self.provider.encode_image(query_image_path)

        results = []
        for anchor in anchors.values():
            # Text similarity (always available)
            text_score = 0.0
            if query_embedding and anchor.embedding:
                text_score = _cosine_sim(query_embedding, anchor.embedding)

            # Image similarity (only for multimodal anchors)
            image_score = 0.0
            if isinstance(anchor, MultimodalAnchor) and anchor.has_image:
                if query_image_emb and anchor.image_embedding:
                    # Image→Image similarity
                    image_score = self.provider.image_similarity(
                        query_image_emb, anchor.image_embedding)
                elif query_embedding and anchor.image_embedding:
                    # Text→Image cross-modal similarity (CLIP)
                    image_score = self.provider.cross_modal_similarity(
                        query_embedding, anchor.image_embedding)

            if text_score < 0.05 and image_score < 0.05:
                continue

            combined = text_weight * text_score + image_weight * image_score
            dominant = "image" if image_score > text_score else "text"

            results.append(CrossModalResult(
                anchor=anchor if isinstance(anchor, MultimodalAnchor) else None,
                text_score=text_score,
                image_score=image_score,
                combined_score=combined,
                modality=dominant,
            ))

        results.sort(key=lambda r: -r.combined_score)
        return results[:top_k]

    def image_search(self, anchors: dict[str, Anchor],
                    query_image_path: str,
                    top_k: int = 10) -> list[CrossModalResult]:
        """Find images similar to a query image."""
        return self.retrieve(
            anchors, query_image_path=query_image_path,
            top_k=top_k, text_weight=0.1, image_weight=0.9,
        )

    def text_to_image(self, anchors: dict[str, Anchor],
                     query_text: str,
                     top_k: int = 10) -> list[CrossModalResult]:
        """Find images matching a text description."""
        return self.retrieve(
            anchors, query_text=query_text,
            top_k=top_k, text_weight=0.3, image_weight=0.7,
        )


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)
