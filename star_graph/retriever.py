"""Retriever — re-exported from retrieval_engine/ (Phase 1 architecture slimdown)."""
from .retrieval_engine.retriever import *  # noqa: F401, F403
from .retrieval_engine.retriever import (  # noqa: F401, F403
    _clip01, _tokenize_terms, _has_temporal_signal, _matched_terms,
    _trace_reason, _build_retrieval_trace, _recall_at_k, _cosine_sim,
)  # noqa: F401 — used by tests — used by tests
