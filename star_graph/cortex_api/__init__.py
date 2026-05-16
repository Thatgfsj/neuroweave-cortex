"""Cortex API — high-level agent-facing memory interface (lazy-loaded)."""


def __getattr__(name: str):
    _registry = {
        "MemoryManager": ("star_graph.manager", "MemoryManager"),
        "AsyncMemoryManager": ("star_graph.async_manager", "AsyncMemoryManager"),
        "MemoryRuntime": ("star_graph.runtime", "MemoryRuntime"),
        "RuntimeCore": ("star_graph.runtime_core", "RuntimeCore"),
        "RuntimeLifecycle": ("star_graph.runtime_lifecycle", "RuntimeLifecycle"),
        "AgentContext": ("star_graph.scheduler", "AgentContext"),
        "MemoryContext": ("star_graph.scheduler", "MemoryContext"),
        "MemoryItem": ("star_graph.scheduler", "MemoryItem"),
        "MemoryType": ("star_graph.scheduler", "MemoryType"),
        "WorkingMemory": ("star_graph.working_memory", "WorkingMemory"),
        "WorkingMemoryEntry": ("star_graph.working_memory", "WorkingMemoryEntry"),
    }
    if name in _registry:
        mod_name, attr = _registry[name]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'star_graph.cortex_api' has no attribute '{name}'")


__all__ = [
    "MemoryManager", "AsyncMemoryManager",
    "MemoryRuntime", "RuntimeCore", "RuntimeLifecycle",
    "AgentContext", "MemoryContext", "MemoryItem", "MemoryType",
    "WorkingMemory", "WorkingMemoryEntry",
]
