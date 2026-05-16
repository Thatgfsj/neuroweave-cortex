"""Retrieval Engine — multi-path recall orchestration (lazy-loaded)."""


def __getattr__(name: str):
    _registry = {
        "Retriever": ("star_graph.retriever", "Retriever"),
        "RetrievalResult": ("star_graph.retriever", "RetrievalResult"),
        "BM25Index": ("star_graph.bm25", "BM25Index"),
        "DualChannelRetriever": ("star_graph.dual_channel", "DualChannelRetriever"),
        "DualChannelOutput": ("star_graph.dual_channel", "DualChannelOutput"),
        "CognitiveCacheManager": ("star_graph.cognitive_cache", "CognitiveCacheManager"),
        "QueryCacheEntry": ("star_graph.cognitive_cache", "QueryCacheEntry"),
        "ExactMatchCache": ("star_graph.cognitive_cache", "ExactMatchCache"),
        "ExactMatchEntry": ("star_graph.cognitive_cache", "ExactMatchEntry"),
        "extract_entity_keys": ("star_graph.cognitive_cache", "extract_entity_keys"),
        "RetrievalPipeline": ("star_graph.retrieval_pipeline", "RetrievalPipeline"),
        "RetrievalCore": ("star_graph.retrieval_core", "RetrievalCore"),
        "RetrievalBudget": ("star_graph.retrieval_budget", "RetrievalBudget"),
    }
    if name in _registry:
        mod_name, attr = _registry[name]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'star_graph.retrieval_engine' has no attribute '{name}'")


__all__ = [
    "Retriever", "RetrievalResult",
    "BM25Index",
    "DualChannelRetriever", "DualChannelOutput",
    "CognitiveCacheManager", "QueryCacheEntry", "ExactMatchCache",
    "ExactMatchEntry", "extract_entity_keys",
    "RetrievalPipeline", "RetrievalCore",
    "RetrievalBudget",
]
