"""RetrievalPipeline — all retrieval orchestration for the cognitive memory system.

Handles multi-path recall: exact cache, raw buffer, graph dimensional descent,
dual-channel (System-1 + System-2), cross-modal, and cascade recall.

Takes a MemoryRuntime reference for subsystem access.

All retrieval methods live in the RetrievalCore mixin base class.
This module provides the concrete pipeline class with initialization and
System-2 keyword configuration.
"""

from __future__ import annotations

from .retrieval_core import RetrievalCore


class RetrievalPipeline(RetrievalCore):
    """All retrieval orchestration for a MemoryRuntime.

    Instantiated by MemoryManager with a reference to the runtime.
    Inherits all retrieval methods from RetrievalCore.
    """

    # ── Structural intent keywords for System-2 auto-trigger ──
    _SYSTEM2_KEYWORDS = {
        'all', 'every', 'list', 'enumerate', 'each',
        'which', 'select', 'what kind', 'what type',
        'before', 'after', 'last', 'first', 'previous', 'next',
        'earlier', 'later', 'since', 'until',
        'how many', 'what caused', 'why did', 'steps', 'sequence',
        'summarize', 'summary', 'overview', 'history', 'timeline', 'pattern',
        'across sessions', 'previously', 'past conversations',
    }

    def __init__(self, runtime):
        self._rt = runtime  # MemoryRuntime instance
