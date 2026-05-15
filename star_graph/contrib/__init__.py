"""NeuroWeave Cortex contrib — optional plugin modules.

These modules are non-core and can be imported on demand.
They have no internal dependencies on other star_graph modules.
All are available through star_graph.contrib.<name> or lazily via star_graph.__init__.
"""

# Benchmark
from .benchmark import (
    BenchmarkSuite, BenchmarkScenario, BenchmarkResult,
    ScenarioResult, Category, run_benchmark, compare_systems,
)

# Symbolic filter
from .symbolic_filter import SymbolicFilter, FilterResult

# Streaming memory buffer
from .streaming import StreamItem, StreamStats, StreamingMemoryBuffer

# Snapshot manager
from .snapshot import SnapshotManager, SnapshotMeta


def __getattr__(name: str):
    """Lazy-load mcp_server (requires optional 'mcp' package)."""
    if name == "mcp_server":
        try:
            from .mcp_server import server as mcp_server
        except ImportError:
            return None
        globals()[name] = mcp_server
        return mcp_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
