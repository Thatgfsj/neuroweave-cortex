"""NeuroWeave Cortex (NWC) — v1.6.0 cognitive architecture.

A graph-first, cortex-partitioned cognitive memory runtime for AI agents.

Architecture:
  Layer 3 (Behavior):  Cortex routing, memory gating, working memory,
                        spreading activation, retrieval budget control
  Layer 2 (Cognitive): Hub abstraction, cascade recall, time spine,
                        sleep consolidation, evolution, ghost revival,
                        personality modeling, goal tracking
  Layer 1 (Storage):   CRUD, persistence, indexing, ANN lookup,
                        thermal store (hot/cold/archive), edge decay

All imports are lazy via PEP 562 __getattr__ — modules load only on first access.
"""

from __future__ import annotations

__version__ = "1.6.0"

import sys
from typing import Any

# ── Lazy import registry ────────────────────────────────────────────────
# Format: {name: (module_name, attr_name_or_None_to_import_module)}

_LAZY: dict[str, tuple[str, str | None]] = {
    # anchor / graph core
    "Anchor":                  ("star_graph.anchor", "Anchor"),
    "AnchorVector":            ("star_graph.anchor", "AnchorVector"),
    "Oscillator":              ("star_graph.anchor", "Oscillator"),
    "MemoryState":             ("star_graph.anchor", "MemoryState"),
    "ThermalState":            ("star_graph.anchor", "ThermalState"),
    "EmbedderRegistry":        ("star_graph.anchor", "EmbedderRegistry"),
    "StarGraph":               ("star_graph.graph", "StarGraph"),
    "Edge":                    ("star_graph.graph", "Edge"),
    "RichEdge":                ("star_graph.graph", "RichEdge"),
    "Constellation":           ("star_graph.graph", "Constellation"),
    "Schema":                  ("star_graph.graph", "Schema"),
    "ReflectionNode":          ("star_graph.graph", "ReflectionNode"),

    # config
    "Config":                  ("star_graph.config", "Config"),
    "config":                  ("star_graph.config", "config"),
    "override":                ("star_graph.config", "override"),
    "reload_defaults":         ("star_graph.config", "reload_defaults"),
    "load_config":             ("star_graph.config", "load_config"),

    # embedding
    "EmbeddingProvider":       ("star_graph.embedding", "EmbeddingProvider"),
    "get_embedder":            ("star_graph.embedding", "get_embedder"),

    # runtime / manager (primary entry points)
    "MemoryManager":           ("star_graph.manager", "MemoryManager"),
    "MemoryRuntime":           ("star_graph.runtime", "MemoryRuntime"),
    "ManagerStats":            ("star_graph.manager_stats", "ManagerStats"),
    "RetrievalPipeline":       ("star_graph.retrieval_pipeline", "RetrievalPipeline"),

    # scheduler
    "CognitiveMemoryScheduler": ("star_graph.scheduler", "CognitiveMemoryScheduler"),
    "AgentContext":            ("star_graph.scheduler", "AgentContext"),
    "MemoryType":              ("star_graph.scheduler", "MemoryType"),
    "MemoryItem":              ("star_graph.scheduler", "MemoryItem"),
    "MemoryContext":           ("star_graph.scheduler", "MemoryContext"),

    # sleep
    "SleepCycle":              ("star_graph.sleep", "SleepCycle"),
    "SleepReport":             ("star_graph.sleep_report", "SleepReport"),
    "PhaseMetrics":            ("star_graph.sleep_report", "PhaseMetrics"),

    # retrieval
    "RetrievalResult":         ("star_graph.retriever", "RetrievalResult"),
    "RetrievalTrace":          ("star_graph.retriever", "RetrievalTrace"),
    "RetrievalTraceEntry":     ("star_graph.retriever", "RetrievalTraceEntry"),
    "Retriever":               ("star_graph.retriever", "Retriever"),
    "OscillationResonanceRetriever": ("star_graph.retriever", "OscillationResonanceRetriever"),
    "VectorSimilarityRetriever": ("star_graph.retriever", "VectorSimilarityRetriever"),
    "HybridFusionRetriever":   ("star_graph.retriever", "HybridFusionRetriever"),
    "ExplainableScore":        ("star_graph.retriever", "ExplainableScore"),
    "personalized_pagerank":   ("star_graph.retriever", "personalized_pagerank"),
    "compare_retrievers":      ("star_graph.retriever", "compare_retrievers"),

    # index
    "ANNIndex":                ("star_graph.index", "ANNIndex"),

    # seed
    "seed_everything":         ("star_graph.seed", "seed_everything"),
    "is_deterministic":        ("star_graph.seed", "is_deterministic"),

    # ghost
    "GhostNode":               ("star_graph.ghost", "GhostNode"),
    "NegativeGhost":           ("star_graph.ghost", "NegativeGhost"),
    "GhostSubsystem":          ("star_graph.ghost", "GhostSubsystem"),

    # abstraction
    "AbstractNode":            ("star_graph.abstraction", "AbstractNode"),
    "AbstractionEngine":       ("star_graph.abstraction", "AbstractionEngine"),
    "PatternMemory":           ("star_graph.abstraction", "PatternMemory"),
    "AbstractiveMemoryEngine": ("star_graph.abstraction", "AbstractiveMemoryEngine"),

    # community
    "Community":               ("star_graph.community", "Community"),
    "CommunityHealth":         ("star_graph.community", "CommunityHealth"),
    "CommunityDetection":      ("star_graph.community", "CommunityDetection"),

    # raw buffer
    "RawBuffer":               ("star_graph.raw_buffer", "RawBuffer"),
    "RawChunk":                ("star_graph.raw_buffer", "RawChunk"),

    # dual channel
    "DualChannelRetriever":    ("star_graph.dual_channel", "DualChannelRetriever"),
    "DualChannelOutput":       ("star_graph.dual_channel", "DualChannelOutput"),
    "ChannelResult":           ("star_graph.dual_channel", "ChannelResult"),

    # other subsystems
    "FactExtractor":           ("star_graph.atom_facts", "FactExtractor"),
    "AtomFact":                ("star_graph.atom_facts", "AtomFact"),
    "ExtractionResult":        ("star_graph.atom_facts", "ExtractionResult"),
    "check_llm_availability":  ("star_graph.atom_facts", "check_llm_availability"),
    "CognitiveMetrics":        ("star_graph.metrics", "CognitiveMetrics"),
    "MemoryCompetition":       ("star_graph.competition", "MemoryCompetition"),
    "OnlineConsolidator":      ("star_graph.online", "OnlineConsolidator"),

    # storage
    "Storage":                 ("star_graph.storage", "Storage"),
    "JSONStorage":             ("star_graph.storage", "JSONStorage"),
    "StorageBackend":          ("star_graph.storage_backend", "StorageBackend"),
    "SQLiteStorage":           ("star_graph.sqlite_storage", "SQLiteStorage"),

    # evolution
    "MemoryEvolutionEngine":   ("star_graph.evolution", "MemoryEvolutionEngine"),
    "EvolutionEvent":          ("star_graph.evolution", "EvolutionEvent"),
    "BeliefTransition":        ("star_graph.evolution", "BeliefTransition"),

    # working memory
    "WorkingMemory":           ("star_graph.working_memory", "WorkingMemory"),
    "WorkingMemoryEntry":      ("star_graph.working_memory", "WorkingMemoryEntry"),

    # cortex
    "MemoryCortex":            ("star_graph.cortex", "MemoryCortex"),
    "CortexConfig":            ("star_graph.cortex", "CortexConfig"),
    "CORTEX_HIERARCHY":        ("star_graph.cortex", "CORTEX_HIERARCHY"),
    "HIERARCHY_WEIGHTS":       ("star_graph.cortex", "HIERARCHY_WEIGHTS"),
    "HIERARCHY_DECAY_DAYS":    ("star_graph.cortex", "HIERARCHY_DECAY_DAYS"),

    # router / gate
    "CortexRouter":            ("star_graph.router", "CortexRouter"),
    "RouteResult":             ("star_graph.router", "RouteResult"),
    "MemoryGate":              ("star_graph.gate", "MemoryGate"),
    "GateScore":               ("star_graph.gate", "GateScore"),

    # timespine
    "TimeSpine":               ("star_graph.timespine", "TimeSpine"),
    "TimeBucket":              ("star_graph.timespine", "TimeBucket"),
    "MemoryCluster":           ("star_graph.timespine", "MemoryCluster"),

    # cascade / spreading
    "CascadeRecall":           ("star_graph.cascade", "CascadeRecall"),
    "CausalChain":             ("star_graph.cascade", "CausalChain"),
    "SpreadingActivation":     ("star_graph.spreading", "SpreadingActivation"),
    "ActivatedNode":           ("star_graph.spreading", "ActivatedNode"),

    # cognitive cache
    "QueryCache":              ("star_graph.cognitive_cache", "QueryCache"),
    "SessionCache":            ("star_graph.cognitive_cache", "SessionCache"),
    "TopicCache":              ("star_graph.cognitive_cache", "TopicCache"),
    "ActivationCache":         ("star_graph.cognitive_cache", "ActivationCache"),
    "CognitiveCacheManager":   ("star_graph.cognitive_cache", "CognitiveCacheManager"),
    "QueryCacheEntry":         ("star_graph.cognitive_cache", "QueryCacheEntry"),

    # compiler / reflection
    "CognitiveCompiler":       ("star_graph.compiler", "CognitiveCompiler"),
    "WorldviewNode":           ("star_graph.compiler", "WorldviewNode"),
    "UserProfile":             ("star_graph.compiler", "UserProfile"),
    "SelfReflectionLoop":      ("star_graph.reflection_loop", "SelfReflectionLoop"),
    "SelfCorrectionReport":    ("star_graph.reflection_loop", "SelfCorrectionReport"),

    # topology
    "topology_rank":           ("star_graph.topology", "topology_rank"),
    "graph_first_recall":      ("star_graph.topology", "graph_first_recall"),
    "EDGE_TYPE_RICHNESS_WEIGHTS": ("star_graph.topology", "EDGE_TYPE_RICHNESS_WEIGHTS"),

    # domain / write / edge
    "DomainRouter":            ("star_graph.domain_router", "DomainRouter"),
    "DomainNode":              ("star_graph.domain_router", "DomainNode"),
    "DEFAULT_DOMAIN_TREE":     ("star_graph.domain_router", "DEFAULT_DOMAIN_TREE"),
    "MemoryWriteGate":         ("star_graph.write_gate", "MemoryWriteGate"),
    "GateDecision":            ("star_graph.write_gate", "GateDecision"),
    "GateResult":              ("star_graph.write_gate", "GateResult"),
    "EdgeBudgetManager":       ("star_graph.edge_management", "EdgeBudgetManager"),
    "EDGE_TYPE_RETENTION_PRIORITY": ("star_graph.edge_management", "EDGE_TYPE_RETENTION_PRIORITY"),

    # four-layer compression
    "FourLayerCompressor":     ("star_graph.four_layer", "FourLayerCompressor"),
    "CompressLayer":           ("star_graph.four_layer", "CompressLayer"),
    "LayerConfig":             ("star_graph.four_layer", "LayerConfig"),
    "CompressedMemory":        ("star_graph.four_layer", "CompressedMemory"),

    # thermal / edge decay
    "ThermalStore":            ("star_graph.thermal_store", "ThermalStore"),
    "EdgeDecayManager":        ("star_graph.edge_management", "EdgeDecayManager"),

    # self-org / personality / goals
    "SelfOrganization":        ("star_graph.self_org", "SelfOrganization"),
    "EmergentTopic":           ("star_graph.self_org", "EmergentTopic"),
    "PersonalityModel":        ("star_graph.personality", "PersonalityModel"),
    "PersonalityProfile":      ("star_graph.personality", "PersonalityProfile"),
    "GoalTree":                ("star_graph.goal_tree", "GoalTree"),
    "GoalNode":                ("star_graph.goal_tree", "GoalNode"),
    "GoalStatus":              ("star_graph.goal_tree", "GoalStatus"),

    # retrieval budget
    "RetrievalBudget":         ("star_graph.retrieval_budget", "RetrievalBudget"),
    "BudgetState":             ("star_graph.retrieval_budget", "BudgetState"),

    # versioned memory
    "CognitiveTrajectory":     ("star_graph.versioned_memory", "CognitiveTrajectory"),
    "BeliefVersion":           ("star_graph.versioned_memory", "BeliefVersion"),

    # cluster memory
    "ClusterRouter":           ("star_graph.cluster_memory", "ClusterRouter"),
    "ClusterCentroid":         ("star_graph.cluster_memory", "ClusterCentroid"),

    # causal edges
    "CausalEdgeClassifier":    ("star_graph.causal_edges", "CausalEdgeClassifier"),
    "CAUSAL_EDGE_TYPES":       ("star_graph.causal_edges", "CAUSAL_EDGE_TYPES"),

    # episodic memory
    "EpisodicMemory":          ("star_graph.episodic_memory", "EpisodicMemory"),
    "EpisodeNode":             ("star_graph.episodic_memory", "EpisodeNode"),
    "SessionSummary":          ("star_graph.episodic_memory", "SessionSummary"),

    # hubs
    "HubLayer":                ("star_graph.hub", "HubLayer"),
    "HubNode":                 ("star_graph.hub", "HubNode"),
    "HubEdge":                 ("star_graph.hub", "HubEdge"),
    "HubShard":                ("star_graph.hub", "HubShard"),

    # hippocampus / shard
    "HippocampusBuffer":       ("star_graph.hippocampus", "HippocampusBuffer"),
    "HippocampusItem":         ("star_graph.hippocampus", "HippocampusItem"),
    "MemoryShardManager":      ("star_graph.shard", "MemoryShardManager"),
    "ShardInfo":               ("star_graph.shard", "ShardInfo"),
    "DOMAIN_DIRS":             ("star_graph.shard", "DOMAIN_DIRS"),

    # tier
    "MemoryTier":              ("star_graph.tier", "MemoryTier"),
    "TierEntry":               ("star_graph.tier", "TierEntry"),
    "ShortTermMemory":         ("star_graph.tier", "ShortTermMemory"),
    "MiddleTermMemory":        ("star_graph.tier", "MiddleTermMemory"),
    "LongTermMemory":          ("star_graph.tier", "LongTermMemory"),
    "CoreMemory":              ("star_graph.tier", "CoreMemory"),
    "MemoryTierManager":       ("star_graph.tier", "MemoryTierManager"),
    "TIER_DECAY_HALF_LIFE":    ("star_graph.tier", "TIER_DECAY_HALF_LIFE"),
    "TIER_MAX_ITEMS":          ("star_graph.tier", "TIER_MAX_ITEMS"),

    # brain sphere
    "BrainSphere":             ("star_graph.brain_sphere", "BrainSphere"),
    "HubCenter":               ("star_graph.brain_sphere", "HubCenter"),

    # autobiography
    "AutobiographicalMemory":  ("star_graph.autobiography", "AutobiographicalMemory"),
    "SelfNarrative":           ("star_graph.autobiography", "SelfNarrative"),

    # math utils
    "cosine_sim":              ("star_graph.math_utils", "cosine_sim"),
    "safe_div":                ("star_graph.math_utils", "safe_div"),
    "clamp":                   ("star_graph.math_utils", "clamp"),
    "sigmoid":                 ("star_graph.math_utils", "sigmoid"),

    # bm25
    "BM25Index":               ("star_graph.bm25", "BM25Index"),
    "reciprocal_rank_fusion":  ("star_graph.bm25", "reciprocal_rank_fusion"),

    # logger
    "get_logger":              ("star_graph.logger", "get_logger"),
    "init_logging":            ("star_graph.logger", "init_logging"),

    # compression
    "CompressionLevel":        ("star_graph.compression", "CompressionLevel"),
    "SummaryAnchor":           ("star_graph.compression", "SummaryAnchor"),
    "SessionCompressor":       ("star_graph.compression", "SessionCompressor"),
    "MultiLevelCompressor":    ("star_graph.compression", "MultiLevelCompressor"),

    # exact cache
    "ExactMatchCache":         ("star_graph.exact_cache", "ExactMatchCache"),
    "ExactMatchEntry":         ("star_graph.exact_cache", "ExactMatchEntry"),
    "extract_entity_keys":     ("star_graph.exact_cache", "extract_entity_keys"),

    # micro sleep
    "MicroSleepScheduler":     ("star_graph.micro_sleep", "MicroSleepScheduler"),
    "MicroSleepProgress":      ("star_graph.micro_sleep", "MicroSleepProgress"),
    "MicroSleepResult":        ("star_graph.micro_sleep", "MicroSleepResult"),

    # cost estimator
    "SleepCostEstimator":      ("star_graph.cost_estimator", "SleepCostEstimator"),
    "CostEstimate":            ("star_graph.cost_estimator", "CostEstimate"),

    # snapshot
    "SnapshotManager":         ("star_graph.contrib.snapshot", "SnapshotManager"),
    "SnapshotMeta":            ("star_graph.contrib.snapshot", "SnapshotMeta"),

    # async
    "AsyncMemoryManager":      ("star_graph.async_manager", "AsyncMemoryManager"),
    "AsyncManagerStats":       ("star_graph.async_manager", "AsyncManagerStats"),

    # tracing
    "MemoryTracer":            ("star_graph.tracing", "MemoryTracer"),
    "TraceSpan":               ("star_graph.tracing", "TraceSpan"),
    "Trace":                   ("star_graph.tracing", "Trace"),
    "get_tracer":              ("star_graph.tracing", "get_tracer"),
    "trace_recall":            ("star_graph.tracing", "trace_recall"),

    # survival
    "SurvivalFunction":        ("star_graph.survival", "SurvivalFunction"),
    "EbbinghausSurvival":      ("star_graph.survival", "EbbinghausSurvival"),
    "PowerLawSurvival":        ("star_graph.survival", "PowerLawSurvival"),
    "ExponentialSurvival":     ("star_graph.survival", "ExponentialSurvival"),
    "CustomSurvival":          ("star_graph.survival", "CustomSurvival"),
    "SurvivalRegistry":        ("star_graph.survival", "SurvivalRegistry"),
    "SurvivalState":           ("star_graph.survival", "SurvivalState"),
    "derive_strength":         ("star_graph.survival", "derive_strength"),

    # resonance
    "Resonator":               ("star_graph.resonance", "Resonator"),

    # symbolic filter
    "SymbolicFilter":          ("star_graph.contrib.symbolic_filter", "SymbolicFilter"),
    "FilterResult":            ("star_graph.contrib.symbolic_filter", "FilterResult"),

    # streaming
    "StreamItem":              ("star_graph.contrib.streaming", "StreamItem"),
    "StreamStats":             ("star_graph.contrib.streaming", "StreamStats"),
    "StreamingMemoryBuffer":   ("star_graph.contrib.streaming", "StreamingMemoryBuffer"),

    # benchmark
    "BenchmarkSuite":          ("star_graph.contrib.benchmark", "BenchmarkSuite"),
    "BenchmarkScenario":       ("star_graph.contrib.benchmark", "BenchmarkScenario"),
    "BenchmarkResult":         ("star_graph.contrib.benchmark", "BenchmarkResult"),
    "ScenarioResult":          ("star_graph.contrib.benchmark", "ScenarioResult"),
    "Category":                ("star_graph.contrib.benchmark", "Category"),
    "run_benchmark":           ("star_graph.contrib.benchmark", "run_benchmark"),
    "compare_systems":         ("star_graph.contrib.benchmark", "compare_systems"),

    # layers
    "enforce_layer_boundaries": ("star_graph.layers", "enforce_layer_boundaries"),
    "layer_summary":           ("star_graph.layers", "layer_summary"),
    "get_layer":               ("star_graph.layers", "get_layer"),
    "check_import":            ("star_graph.layers", "check_import"),

    # multimodal (optional — only loaded on access)
    "MultimodalEmbeddingProvider": ("star_graph.multimodal", "MultimodalEmbeddingProvider"),
    "MultimodalAnchor":        ("star_graph.multimodal", "MultimodalAnchor"),
    "CrossModalRetriever":     ("star_graph.multimodal", "CrossModalRetriever"),
    "CrossModalResult":        ("star_graph.multimodal", "CrossModalResult"),

    # MCP server (optional)
    "mcp_server":              ("star_graph.contrib.mcp_server", "server"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = __import__(mod_name, fromlist=[attr] if attr else [])
        except ImportError:
            if name == "mcp_server":
                return None
            if name.startswith("Multimodal") or name.startswith("CrossModal"):
                return None
            raise
        if attr is not None:
            obj = getattr(mod, attr)
        else:
            obj = mod
        # Cache the resolved object in the module globals so __getattr__
        # is only called once per name.
        globals()[name] = obj
        return obj
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> list[str]:
    """Support autocomplete in REPLs."""
    base = list(globals().keys())
    return sorted(base + list(_LAZY.keys()))
