"""Extras — optional, feature-gated modules.

These modules are not part of the core cognitive memory pipeline. They are
gated behind feature flags and may have heavier dependencies.

Feature flags (set in config):
  - extras.resonance: emotional resonance dynamics
  - extras.autobiography: self-model and autobiographical memory
  - extras.streaming: real-time streaming buffer
  - extras.benchmark: LoCoMo / LongMemEval benchmarking
  - extras.mcp: MCP server integration
"""

# Lazy imports to avoid pulling heavy deps at module load
__all__ = [
    "resonance",
    "autobiography",
    "streaming",
    "benchmark",
    "mcp_server",
    "snapshot",
    "symbolic_filter",
]

_ENABLED = {
    "resonance": True,
    "autobiography": True,
    "streaming": True,
    "benchmark": False,
    "mcp_server": False,
    "symbolic_filter": True,
}

def is_enabled(name: str) -> bool:
    return _ENABLED.get(name, False)

def enable(name: str) -> None:
    _ENABLED[name] = True

def disable(name: str) -> None:
    _ENABLED[name] = False
