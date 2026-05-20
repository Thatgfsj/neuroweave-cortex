"""StarGraph — re-exported from memory_core/ (Phase 1 architecture slimdown)."""
from .memory_core.graph import *

# Re-export all public names from the moved module
__all__ = [n for n in dir() if not n.startswith('_')] if False else None  # propagate all
