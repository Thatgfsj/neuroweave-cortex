"""StarGraph — re-exported from memory_core/ (Phase 1 architecture slimdown)."""
from .memory_core.graph import *  # noqa: F401, F403
from .memory_core.graph import _cosine_sim_simple  # noqa: F401 — used by tests

# Re-export all public names from the moved module
__all__ = [n for n in dir() if not n.startswith('_')] if False else None  # propagate all
