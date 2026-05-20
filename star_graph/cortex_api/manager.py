"""MemoryManager — thin facade for the cognitive memory runtime.

Single entry point for AI agents. Delegates to MemoryRuntime (lifecycle/subsystems)
and RetrievalPipeline (all retrieval paths).

Usage:
    manager = MemoryManager()
    manager.remember("User prefers concise answers", tags=["preference", "style"])
    manager.remember("Debugged Redis timeout — pool size was 10, fixed to 20",
                     tags=["debug", "redis"])

    # Context-aware recall
    ctx = AgentContext(task_type="debugging", active_goals=["fix Redis"])
    memories = manager.recall("Redis connection issues", context=ctx)

    # Cognitive maintenance
    manager.micro_consolidate()   # light online update
    report = manager.sleep()      # full consolidation

    # Persistence
    manager.save("agent_memory.db")
    manager.load("agent_memory.db")

    # Health
    manager.print_health()
"""

from __future__ import annotations

from ..graph import StarGraph
from ..config import Config
from .runtime import MemoryRuntime, ManagerStats
from ..retrieval_pipeline import RetrievalPipeline


class MemoryManager:
    """Thin facade composing MemoryRuntime + RetrievalPipeline.

    All methods not defined here are auto-delegated to the runtime
    or retrieval pipeline via __getattr__.

    Inheritable by wrappers (e.g. AsyncMemoryManager) that need to
    intercept or wrap individual methods.
    """

    def __init__(self, graph: StarGraph | None = None,
                 config: Config | None = None,
                 storage_path: str = ""):
        self._rt = MemoryRuntime(graph=graph, config=config, storage_path=storage_path)
        self._rp = RetrievalPipeline(self._rt)

    # ── Explicit delegations for IDE discoverability ──────────
    # Properties that callers access directly

    @property
    def graph(self):
        return self._rt.graph

    @graph.setter
    def graph(self, value):
        self._rt.graph = value

    @property
    def cfg(self):
        return self._rt.cfg

    @property
    def storage_path(self):
        return self._rt.storage_path

    @storage_path.setter
    def storage_path(self, value):
        self._rt.storage_path = value

    @property
    def stats(self) -> ManagerStats:
        return self._rt.stats

    # ── Auto-delegation ──────────────────────────────────────

    def __getattr__(self, name: str):
        """Delegate missing attributes to runtime, then retrieval pipeline."""
        # Avoid infinite recursion: __getattr__ is only called when normal lookup fails
        rt = self.__dict__.get('_rt')
        if rt is not None and hasattr(rt, name):
            return getattr(rt, name)
        rp = self.__dict__.get('_rp')
        if rp is not None and hasattr(rp, name):
            return getattr(rp, name)
        raise AttributeError(
            f"MemoryManager has no attribute '{name}' "
            f"(not found on MemoryRuntime or RetrievalPipeline)"
        )

    # ── Sync/Async unification ──────────────────────────────

    def to_async(self):
        """Return an AsyncMemoryManager wrapping this manager.

        Provides the same API but with async/await support for all operations.
        """
        from .async_manager import AsyncMemoryManager
        return AsyncMemoryManager(self)

    async def async_remember(self, text: str, **kwargs):
        """Async wrapper for remember()."""
        import asyncio
        return await asyncio.to_thread(self.remember, text, **kwargs)

    async def async_recall(self, query: str = "", **kwargs):
        """Async wrapper for recall()."""
        import asyncio
        return await asyncio.to_thread(self.recall, query, **kwargs)

    async def async_sleep(self, **kwargs):
        """Async wrapper for sleep()."""
        import asyncio
        return await asyncio.to_thread(self.sleep, **kwargs)

    async def async_micro_consolidate(self, **kwargs):
        """Async wrapper for micro_consolidate()."""
        import asyncio
        return await asyncio.to_thread(self.micro_consolidate, **kwargs)

    def export_markdown(self, output_dir: str = "memories",
                        organize_by: str = "both",
                        single_file: bool = False) -> str:
        """Export all memories to operator-editable Markdown files.

        Returns the absolute path to the output directory.
        """
        from ..markdown_export import export_to_markdown
        return export_to_markdown(
            self.graph,
            output_dir=output_dir,
            organize_by=organize_by,
            single_file=single_file,
        )

    def batch_remember(self, text: str, *, tags=None, importance=0.5,
                       emotional_valence=0.0, **kwargs) -> str | None:
        """Enqueue a memory for deferred batch embedding.

        Returns anchor_id immediately, embedding computed later via
        batch flush (size >= 32 or >30s timer). Call flush_vectorizer()
        to force immediate flush.

        Passes embedding=[] to skip Anchor.create() eager encoding;
        the BatchVectorizer handles encoding on flush.
        """
        from ..anchor import Anchor
        anchor = Anchor.create(
            text=text, tags=tags or [],
            importance=importance,
            emotional_valence=emotional_valence,
            embedding=[],  # defer: BatchVectorizer handles encoding
            **kwargs,
        )
        self.graph.add_anchor(anchor)
        self._rt.batch_vectorizer.enqueue(text, anchor.id)
        return anchor.id

    def flush_vectorizer(self) -> int:
        """Force-flush pending batch embeddings. Returns count written."""
        bv = self._rt.batch_vectorizer
        return bv.flush() if bv else 0

    def shutdown_batch_vectorizer(self) -> None:
        """Flush pending and stop background timer."""
        if self._rt._batch_vectorizer:
            self._rt._batch_vectorizer.shutdown()

    def remember_image(self, image_path: str, caption: str = "",
                       tags: list[str] | None = None, importance: float = 0.5,
                       emotional_valence: float = 0.0,
                       source_session: str = "") -> Anchor | None:
        """Store a memory from an image. Uses CLIP when available for cross-modal retrieval."""
        from ..multimodal import MultimodalEmbeddingProvider, MultimodalAnchor
        provider = MultimodalEmbeddingProvider()
        anchor = MultimodalAnchor.from_image(
            image_path, provider, caption=caption,
            tags=tags or [], importance=importance,
            emotional_valence=emotional_valence, source_session=source_session,
        )
        self.graph.add_anchor(anchor)
        return anchor

    def remember_audio(self, audio_path: str, transcript: str = "",
                       tags: list[str] | None = None, importance: float = 0.5,
                       emotional_valence: float = 0.0,
                       source_session: str = "") -> Anchor | None:
        """Store a memory from an audio file. Uses Whisper for transcription when available."""
        from ..multimodal import AudioEncoder, AudioAnchor
        encoder = AudioEncoder()
        anchor = AudioAnchor.from_audio(
            audio_path, encoder, transcript=transcript,
            tags=tags or [], importance=importance,
            emotional_valence=emotional_valence, source_session=source_session,
        )
        self.graph.add_anchor(anchor)
        return anchor

    def zero_llm_ingest(self, texts: list[str], *, tags: list[str] | None = None,
                        source_session: str = "",
                        llm_fn: callable | None = None) -> list:
        """Pure algorithmic ingestion — zero LLM cost per item.

        Pipeline: security filter → embed → dedup → entity extract → classify → score → link.
        Only invokes llm_fn for ambiguous low-confidence items.
        """
        from ..zero_llm_pipeline import ZeroLLMPipeline
        pipeline = ZeroLLMPipeline(
            self.graph, self._rt._get_embedder(), llm_fn=llm_fn)
        return pipeline.ingest(texts, tags=tags, source_session=source_session)

    def forget_with_certificate(self, anchor_id: str, query: str = "",
                                reason: str = "user_request") -> dict:
        """Forget a memory and generate an Ed25519-signed deletion certificate.

        Returns dict with 'deleted' bool, 'certificate' JWS string, and 'cert_path'.
        GDPR Article 17 compliant — the certificate proves provable deletion.
        """
        from ..forget_certificate import ForgetCertificate

        anchor = self._rt.forget(anchor_id, create_ghost=True)
        if anchor is None:
            return {"deleted": False, "error": "anchor not found", "certificate": "", "cert_path": ""}

        import os as _os
        cert = ForgetCertificate.generate(
            memory_id=anchor_id,
            query=query or anchor.text,
            reason=reason,
        )

        cert_dir = "certificates"
        _os.makedirs(cert_dir, exist_ok=True)
        cert_path = cert.save(f"{cert_dir}/forget-{anchor_id[:16]}.jws")

        return {
            "deleted": True,
            "certificate": cert.to_jws(),
            "cert_path": cert_path,
            "memory_id": anchor_id,
        }

    async def async_stats(self, **kwargs):
        """Async wrapper for stats()."""
        import asyncio
        return await asyncio.to_thread(lambda: self.stats, **kwargs)
