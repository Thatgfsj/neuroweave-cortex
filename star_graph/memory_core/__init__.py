"""Memory Core — foundational memory primitives (lazy-loaded)."""


def __getattr__(name: str):
    _registry = {
        "Anchor": ("star_graph.anchor", "Anchor"),
        "AnchorVector": ("star_graph.anchor", "AnchorVector"),
        "MemoryState": ("star_graph.anchor", "MemoryState"),
        "StarGraph": ("star_graph.graph", "StarGraph"),
        "Edge": ("star_graph.graph", "Edge"),
        "Constellation": ("star_graph.graph", "Constellation"),
        "Schema": ("star_graph.graph", "Schema"),
        "JSONStorage": ("star_graph.storage", "JSONStorage"),
        "StorageBackend": ("star_graph.storage_backend", "StorageBackend"),
        "SQLiteStorage": ("star_graph.sqlite_storage", "SQLiteStorage"),
        "MemoryTier": ("star_graph.tier", "MemoryTier"),
        "TierEntry": ("star_graph.tier", "TierEntry"),
        "ShortTermMemory": ("star_graph.tier", "ShortTermMemory"),
        "MiddleTermMemory": ("star_graph.tier", "MiddleTermMemory"),
        "LongTermMemory": ("star_graph.tier", "LongTermMemory"),
        "CoreMemory": ("star_graph.tier", "CoreMemory"),
        "MemoryTierManager": ("star_graph.tier", "MemoryTierManager"),
        "TieredStorage": ("star_graph.tier", "TieredStorage"),
        "offload_anchor_to_cold": ("star_graph.tiered", "offload_anchor_to_cold"),
        "ANNIndex": ("star_graph.index", "ANNIndex"),
    }
    if name in _registry:
        mod_name, attr = _registry[name]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'star_graph.memory_core' has no attribute '{name}'")


__all__ = list(__getattr__.__kwdefaults__["_registry"].keys()) if False else [
    "Anchor", "AnchorVector", "MemoryState",
    "StarGraph", "Edge", "Constellation", "Schema",
    "JSONStorage", "StorageBackend", "SQLiteStorage",
    "MemoryTier", "TierEntry", "ShortTermMemory", "MiddleTermMemory",
    "LongTermMemory", "CoreMemory", "MemoryTierManager", "TieredStorage",
    "offload_anchor_to_cold",
    "ANNIndex",
]
