"""NeuroWeave Cortex (NWC) — v1.2.0 cognitive architecture. 6-layer memory system.

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

__version__ = "1.2.11"

import sys
from typing import Any

# ── Sub-packages (Phase 1: 5-core + abstraction + extras) ──────────────
from . import memory_core       # noqa: F401 — memory primitives
from . import retrieval_engine  # noqa: F401 — retrieval orchestration
from . import embedding_provider # noqa: F401 — embedding abstraction
from . import consolidation     # noqa: F401 — sleep-cycle maintenance
from . import cortex_api        # noqa: F401 — agent-facing API
from . import extras            # noqa: F401 — feature-gated modules

# ── Lazy import registry ────────────────────────────────────────────────
# Format: {name: (module_name, attr_name_or_None_to_import_module)}

_LAZY: dict[str, tuple[str, str | None]] = {
    # anchor / graph core
    "Anchor":                  ("star_graph.memory_core.anchor", "Anchor"),
    "AnchorVector":            ("star_graph.memory_core.anchor", "AnchorVector"),
    "Oscillator":              ("star_graph.memory_core.anchor", "Oscillator"),
    "MemoryState":             ("star_graph.memory_core.anchor", "MemoryState"),
    "ThermalState":            ("star_graph.memory_core.anchor", "ThermalState"),
    "EmbedderRegistry":        ("star_graph.memory_core.anchor", "EmbedderRegistry"),
    "StarGraph":               ("star_graph.memory_core.graph", "StarGraph"),
    "Edge":                    ("star_graph.memory_core.graph", "Edge"),
    "RichEdge":                ("star_graph.memory_core.graph", "RichEdge"),
    "Constellation":           ("star_graph.memory_core.graph", "Constellation"),
    "Schema":                  ("star_graph.memory_core.graph", "Schema"),
    "ReflectionNode":          ("star_graph.memory_core.graph", "ReflectionNode"),

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
    "MemoryManager":           ("star_graph.cortex_api.manager", "MemoryManager"),
    "MemoryRuntime":           ("star_graph.cortex_api.runtime", "MemoryRuntime"),
    "ManagerStats":            ("star_graph.cortex_api.manager_stats", "ManagerStats"),
    "RetrievalPipeline":       ("star_graph.retrieval_engine.retrieval_pipeline", "RetrievalPipeline"),

    # scheduler
    "CognitiveMemoryScheduler": ("star_graph.cortex_api.scheduler", "CognitiveMemoryScheduler"),
    "AgentContext":            ("star_graph.cortex_api.scheduler", "AgentContext"),
    "MemoryType":              ("star_graph.cortex_api.scheduler", "MemoryType"),
    "MemoryItem":              ("star_graph.cortex_api.scheduler", "MemoryItem"),
    "MemoryContext":           ("star_graph.cortex_api.scheduler", "MemoryContext"),

    # sleep
    "SleepCycle":              ("star_graph.consolidation.sleep", "SleepCycle"),
    "SleepReport":             ("star_graph.consolidation.sleep_report", "SleepReport"),
    "PhaseMetrics":            ("star_graph.consolidation.sleep_report", "PhaseMetrics"),

    # retrieval
    "RetrievalResult":         ("star_graph.retrieval_engine.retriever", "RetrievalResult"),
    "RetrievalTrace":          ("star_graph.retrieval_engine.retriever", "RetrievalTrace"),
    "RetrievalTraceEntry":     ("star_graph.retrieval_engine.retriever", "RetrievalTraceEntry"),
    "Retriever":               ("star_graph.retrieval_engine.retriever", "Retriever"),
    "OscillationResonanceRetriever": ("star_graph.retrieval_engine.retriever", "OscillationResonanceRetriever"),
    "VectorSimilarityRetriever": ("star_graph.retrieval_engine.retriever", "VectorSimilarityRetriever"),
    "HybridFusionRetriever":   ("star_graph.retrieval_engine.retriever", "HybridFusionRetriever"),
    "ExplainableScore":        ("star_graph.retrieval_engine.retriever", "ExplainableScore"),
    "personalized_pagerank":   ("star_graph.retrieval_engine.retriever", "personalized_pagerank"),
    "compare_retrievers":      ("star_graph.retrieval_engine.retriever", "compare_retrievers"),

    # index
    "ANNIndex":                ("star_graph.memory_core.index", "ANNIndex"),

    # seed
    "seed_everything":         ("star_graph.seed", "seed_everything"),
    "is_deterministic":        ("star_graph.seed", "is_deterministic"),

    # ghost
    "GhostNode":               ("star_graph.consolidation.ghost", "GhostNode"),
    "NegativeGhost":           ("star_graph.consolidation.ghost", "NegativeGhost"),
    "GhostSubsystem":          ("star_graph.consolidation.ghost", "GhostSubsystem"),

    # abstraction
    "AbstractNode":            ("star_graph.consolidation.abstraction", "AbstractNode"),
    "AbstractionEngine":       ("star_graph.consolidation.abstraction", "AbstractionEngine"),
    "PatternMemory":           ("star_graph.consolidation.abstraction", "PatternMemory"),
    "AbstractiveMemoryEngine": ("star_graph.consolidation.abstraction", "AbstractiveMemoryEngine"),

    # community
    "Community":               ("star_graph.consolidation.community", "Community"),
    "CommunityHealth":         ("star_graph.consolidation.community", "CommunityHealth"),
    "CommunityDetection":      ("star_graph.consolidation.community", "CommunityDetection"),

    # raw buffer
    "RawBuffer":               ("star_graph.raw_buffer", "RawBuffer"),
    "RawChunk":                ("star_graph.raw_buffer", "RawChunk"),

    # dual channel
    "DualChannelRetriever":    ("star_graph.retrieval_engine.dual_channel", "DualChannelRetriever"),
    "DualChannelOutput":       ("star_graph.retrieval_engine.dual_channel", "DualChannelOutput"),
    "ChannelResult":           ("star_graph.retrieval_engine.dual_channel", "ChannelResult"),

    # other subsystems
    "FactExtractor":           ("star_graph.atom_facts", "FactExtractor"),
    "AtomFact":                ("star_graph.atom_facts", "AtomFact"),
    "ExtractionResult":        ("star_graph.atom_facts", "ExtractionResult"),
    "check_llm_availability":  ("star_graph.atom_facts", "check_llm_availability"),
    "CognitiveMetrics":        ("star_graph.metrics", "CognitiveMetrics"),
    "MemoryCompetition":       ("star_graph.consolidation.competition", "MemoryCompetition"),
    "OnlineConsolidator":      ("star_graph.online", "OnlineConsolidator"),

    # storage
    "Storage":                 ("star_graph.memory_core.storage", "Storage"),
    "JSONStorage":             ("star_graph.memory_core.storage", "JSONStorage"),
    "StorageBackend":          ("star_graph.memory_core.storage_backend", "StorageBackend"),
    "SQLiteStorage":           ("star_graph.memory_core.sqlite_storage", "SQLiteStorage"),

    # evolution
    "MemoryEvolutionEngine":   ("star_graph.consolidation.evolution", "MemoryEvolutionEngine"),
    "EvolutionEvent":          ("star_graph.consolidation.evolution", "EvolutionEvent"),
    "BeliefTransition":        ("star_graph.consolidation.evolution", "BeliefTransition"),

    # working memory
    "WorkingMemory":           ("star_graph.cortex_api.working_memory", "WorkingMemory"),
    "WorkingMemoryEntry":      ("star_graph.cortex_api.working_memory", "WorkingMemoryEntry"),

    # cortex
    "MemoryCortex":            ("star_graph.cortex", "MemoryCortex"),
    "CortexConfig":            ("star_graph.cortex", "CortexConfig"),
    "CORTEX_HIERARCHY":        ("star_graph.cortex", "CORTEX_HIERARCHY"),
    "HIERARCHY_WEIGHTS":       ("star_graph.cortex", "HIERARCHY_WEIGHTS"),
    "HIERARCHY_DECAY_DAYS":    ("star_graph.cortex", "HIERARCHY_DECAY_DAYS"),

    # router / gate
    "CortexRouter":            ("star_graph.router", "CortexRouter"),
    "RouteResult":             ("star_graph.router", "RouteResult"),
    "MemoryGate":              ("star_graph.cortex_api.gate", "MemoryGate"),
    "GateScore":               ("star_graph.cortex_api.gate", "GateScore"),

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
    "QueryCache":              ("star_graph.retrieval_engine.cognitive_cache", "QueryCache"),
    "SessionCache":            ("star_graph.retrieval_engine.cognitive_cache", "SessionCache"),
    "TopicCache":              ("star_graph.retrieval_engine.cognitive_cache", "TopicCache"),
    "ActivationCache":         ("star_graph.retrieval_engine.cognitive_cache", "ActivationCache"),
    "CognitiveCacheManager":   ("star_graph.retrieval_engine.cognitive_cache", "CognitiveCacheManager"),
    "QueryCacheEntry":         ("star_graph.retrieval_engine.cognitive_cache", "QueryCacheEntry"),

    # compiler / reflection
    "CognitiveCompiler":       ("star_graph.compiler", "CognitiveCompiler"),
    "WorldviewNode":           ("star_graph.compiler", "WorldviewNode"),
    "UserProfile":             ("star_graph.compiler", "UserProfile"),
    "SelfReflectionLoop":      ("star_graph.consolidation.reflection_loop", "SelfReflectionLoop"),
    "SelfCorrectionReport":    ("star_graph.consolidation.reflection_loop", "SelfCorrectionReport"),

    # topology
    "topology_rank":           ("star_graph.topology", "topology_rank"),
    "graph_first_recall":      ("star_graph.topology", "graph_first_recall"),
    "EDGE_TYPE_RICHNESS_WEIGHTS": ("star_graph.topology", "EDGE_TYPE_RICHNESS_WEIGHTS"),

    # domain / write / edge
    "DomainRouter":            ("star_graph.consolidation.domain_router", "DomainRouter"),
    "DomainNode":              ("star_graph.consolidation.domain_router", "DomainNode"),
    "DEFAULT_DOMAIN_TREE":     ("star_graph.consolidation.domain_router", "DEFAULT_DOMAIN_TREE"),
    "MemoryWriteGate":         ("star_graph.cortex_api.write_gate", "MemoryWriteGate"),
    "GateDecision":            ("star_graph.cortex_api.write_gate", "GateDecision"),
    "GateResult":              ("star_graph.cortex_api.write_gate", "GateResult"),
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
    "SelfOrganization":        ("star_graph.consolidation.self_org", "SelfOrganization"),
    "EmergentTopic":           ("star_graph.consolidation.self_org", "EmergentTopic"),
    "PersonalityModel":        ("star_graph.personality", "PersonalityModel"),
    "PersonalityProfile":      ("star_graph.personality", "PersonalityProfile"),
    "GoalTree":                ("star_graph.goal_tree", "GoalTree"),
    "GoalNode":                ("star_graph.goal_tree", "GoalNode"),
    "GoalStatus":              ("star_graph.goal_tree", "GoalStatus"),

    # retrieval budget
    "RetrievalBudget":         ("star_graph.retrieval_engine.retrieval_budget", "RetrievalBudget"),
    "BudgetState":             ("star_graph.retrieval_engine.retrieval_budget", "BudgetState"),

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
    "HubLayer":                ("star_graph.consolidation.hub", "HubLayer"),
    "HubNode":                 ("star_graph.consolidation.hub", "HubNode"),
    "HubEdge":                 ("star_graph.consolidation.hub", "HubEdge"),
    "HubShard":                ("star_graph.consolidation.hub", "HubShard"),

    # hippocampus / shard
    "HippocampusBuffer":       ("star_graph.hippocampus", "HippocampusBuffer"),
    "HippocampusItem":         ("star_graph.hippocampus", "HippocampusItem"),
    "MemoryShardManager":      ("star_graph.shard", "MemoryShardManager"),
    "ShardInfo":               ("star_graph.shard", "ShardInfo"),
    "DOMAIN_DIRS":             ("star_graph.shard", "DOMAIN_DIRS"),

    # tier
    "MemoryTier":              ("star_graph.memory_core.tier", "MemoryTier"),
    "TierEntry":               ("star_graph.memory_core.tier", "TierEntry"),
    "ShortTermMemory":         ("star_graph.memory_core.tier", "ShortTermMemory"),
    "MiddleTermMemory":        ("star_graph.memory_core.tier", "MiddleTermMemory"),
    "LongTermMemory":          ("star_graph.memory_core.tier", "LongTermMemory"),
    "CoreMemory":              ("star_graph.memory_core.tier", "CoreMemory"),
    "MemoryTierManager":       ("star_graph.memory_core.tier", "MemoryTierManager"),
    "TIER_DECAY_HALF_LIFE":    ("star_graph.memory_core.tier", "TIER_DECAY_HALF_LIFE"),
    "TIER_MAX_ITEMS":          ("star_graph.memory_core.tier", "TIER_MAX_ITEMS"),

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
    "BM25Index":               ("star_graph.retrieval_engine.bm25", "BM25Index"),
    "reciprocal_rank_fusion":  ("star_graph.retrieval_engine.bm25", "reciprocal_rank_fusion"),

    # batch_vectorizer
    "BatchVectorizer":         ("star_graph.batch_vectorizer", "BatchVectorizer"),

    # zero_llm_pipeline
    "ZeroLLMPipeline":         ("star_graph.zero_llm_pipeline", "ZeroLLMPipeline"),

    # logger
    "get_logger":              ("star_graph.logger", "get_logger"),
    "init_logging":            ("star_graph.logger", "init_logging"),

    # compression
    "CompressionLevel":        ("star_graph.consolidation.compression", "CompressionLevel"),
    "SummaryAnchor":           ("star_graph.consolidation.compression", "SummaryAnchor"),
    "SessionCompressor":       ("star_graph.consolidation.compression", "SessionCompressor"),
    "MultiLevelCompressor":    ("star_graph.consolidation.compression", "MultiLevelCompressor"),

    # exact cache
    "ExactMatchCache":         ("star_graph.retrieval_engine.exact_cache", "ExactMatchCache"),
    "ExactMatchEntry":         ("star_graph.retrieval_engine.exact_cache", "ExactMatchEntry"),
    "extract_entity_keys":     ("star_graph.retrieval_engine.exact_cache", "extract_entity_keys"),

    # micro sleep
    "MicroSleepScheduler":     ("star_graph.consolidation.micro_sleep", "MicroSleepScheduler"),
    "MicroSleepProgress":      ("star_graph.consolidation.micro_sleep", "MicroSleepProgress"),
    "MicroSleepResult":        ("star_graph.consolidation.micro_sleep", "MicroSleepResult"),

    # cost estimator
    "SleepCostEstimator":      ("star_graph.cost_estimator", "SleepCostEstimator"),
    "CostEstimate":            ("star_graph.cost_estimator", "CostEstimate"),

    # snapshot
    "SnapshotManager":         ("star_graph.contrib.snapshot", "SnapshotManager"),
    "SnapshotMeta":            ("star_graph.contrib.snapshot", "SnapshotMeta"),

    # async
    "AsyncMemoryManager":      ("star_graph.cortex_api.async_manager", "AsyncMemoryManager"),
    "AsyncManagerStats":       ("star_graph.cortex_api.async_manager", "AsyncManagerStats"),

    # tracing
    "MemoryTracer":            ("star_graph.tracing", "MemoryTracer"),
    "TraceSpan":               ("star_graph.tracing", "TraceSpan"),
    "Trace":                   ("star_graph.tracing", "Trace"),
    "get_tracer":              ("star_graph.tracing", "get_tracer"),
    "trace_recall":            ("star_graph.tracing", "trace_recall"),

    # survival
    "SurvivalFunction":        ("star_graph.consolidation.survival", "SurvivalFunction"),
    "EbbinghausSurvival":      ("star_graph.consolidation.survival", "EbbinghausSurvival"),
    "PowerLawSurvival":        ("star_graph.consolidation.survival", "PowerLawSurvival"),
    "ExponentialSurvival":     ("star_graph.consolidation.survival", "ExponentialSurvival"),
    "CustomSurvival":          ("star_graph.consolidation.survival", "CustomSurvival"),
    "SurvivalRegistry":        ("star_graph.consolidation.survival", "SurvivalRegistry"),
    "SurvivalState":           ("star_graph.consolidation.survival", "SurvivalState"),
    "derive_strength":         ("star_graph.consolidation.survival", "derive_strength"),

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

    # ── Phase 6: Cognitive Cortex — Thought Object base ──
    "ThoughtObject":           ("star_graph.thought_object", "ThoughtObject"),
    "new_thought":             ("star_graph.thought_object", "new_thought"),
    "thought_from_anchor":     ("star_graph.thought_object", "thought_from_anchor"),

    # ── Phase 6: Cognitive Cortex — Cognitive Workspace (6.2) ──
    "CognitiveWorkspace":      ("star_graph.cognitive_workspace", "CognitiveWorkspace"),
    "WorkspaceItem":           ("star_graph.cognitive_workspace", "WorkspaceItem"),
    "WorkspaceState":          ("star_graph.cognitive_workspace", "WorkspaceState"),
    "ReasoningChain":          ("star_graph.cognitive_workspace", "ReasoningChain"),
    "ReasoningStep":           ("star_graph.cognitive_workspace", "ReasoningStep"),

    # ── Phase 6: Cognitive Cortex — Activation Engine (6.4) ──
    "ActivationEngine":        ("star_graph.activation_engine", "ActivationEngine"),
    "ActivationSeed":          ("star_graph.activation_engine", "ActivationSeed"),
    "ActivationToken":         ("star_graph.activation_engine", "ActivationToken"),
    "ActivatedNode":           ("star_graph.activation_engine", "ActivatedNode"),
    "SemanticPath":            ("star_graph.activation_engine", "SemanticPath"),
    "ActivationResult":        ("star_graph.activation_engine", "ActivationResult"),

    # ── Phase 6: Cognitive Cortex — Perception Layer (6.1) ──
    "PerceptionLayer":         ("star_graph.perception", "PerceptionLayer"),
    "PerceptionFrame":         ("star_graph.perception", "PerceptionFrame"),
    "IntentSignal":            ("star_graph.perception", "IntentSignal"),

    # ── Phase 6: Cognitive Cortex — Concept Cortex (6.3) ──
    "ConceptCortex":           ("star_graph.concept_cortex", "ConceptCortex"),
    "ConceptNode":             ("star_graph.concept_cortex", "ConceptNode"),
    "ConceptFusion":           ("star_graph.concept_cortex", "ConceptFusion"),
    "ConceptActivationPath":   ("star_graph.concept_cortex", "ConceptActivationPath"),
    "CORE_CONCEPT_SEEDS":      ("star_graph.concept_cortex", "CORE_CONCEPT_SEEDS"),

    # ── Phase 6: Cognitive Cortex — Goal System (6.5) ──
    "GoalSystem":              ("star_graph.goal_system", "GoalSystem"),
    "GoalFrame":               ("star_graph.goal_system", "GoalFrame"),
    "GoalConflict":            ("star_graph.goal_system", "GoalConflict"),
    "GoalDrivenInference":     ("star_graph.goal_system", "GoalDrivenInference"),

    # ── Phase 6: Cognitive Cortex — Salience Engine (6.6) ──
    "SalienceEngine":          ("star_graph.salience", "SalienceEngine"),
    "SalienceSignal":          ("star_graph.salience", "SalienceSignal"),
    "SalienceComponents":      ("star_graph.salience", "SalienceComponents"),
    "AttentionFocus":          ("star_graph.salience", "AttentionFocus"),

    # ── Phase 6: Cognitive Cortex — Cognitive Compression (6.7) ──
    "CognitiveCompressor":     ("star_graph.cognitive_compression", "CognitiveCompressor"),
    "CognitiveCompressionResult": ("star_graph.cognitive_compression", "CognitiveCompressionResult"),
    "WorldModelBelief":        ("star_graph.cognitive_compression", "WorldModelBelief"),
    "CompressionStage":        ("star_graph.cognitive_compression", "CompressionStage"),

    # ── Phase 6: Cognitive Cortex — Self Model (6.8) ──
    "SelfModel":               ("star_graph.self_model", "SelfModel"),
    "CognitiveState":          ("star_graph.self_model", "CognitiveState"),
    "BiasDetection":           ("star_graph.self_model", "BiasDetection"),
    "SelfModelConfig":         ("star_graph.self_model", "SelfModelConfig"),

    # ── Phase 6: Cognitive Cortex — Autonomous Reasoning (6.9) ──
    "AutonomousReasoningLoop": ("star_graph.autonomous_reasoning", "AutonomousReasoningLoop"),
    "ReasoningTrigger":        ("star_graph.autonomous_reasoning", "ReasoningTrigger"),
    "ReasoningTrace":          ("star_graph.autonomous_reasoning", "ReasoningTrace"),
    "CognitiveUpdate":         ("star_graph.autonomous_reasoning", "CognitiveUpdate"),

    # ── Phase 6: Cognitive Cortex — Memory Lifecycle (6.10) ──
    "MemoryLifecycleManager":  ("star_graph.memory_lifecycle", "MemoryLifecycleManager"),
    "LifecycleStage":          ("star_graph.memory_lifecycle", "LifecycleStage"),
    "LifecycleTransition":     ("star_graph.memory_lifecycle", "LifecycleTransition"),
    "LifecycleState":          ("star_graph.memory_lifecycle", "LifecycleState"),

    # ── Phase 7: Memory Evolution — Importance Engine (7.1) ──
    "ImportanceEngine":        ("star_graph.importance_engine", "ImportanceEngine"),
    "ImportanceSignal":        ("star_graph.importance_engine", "ImportanceSignal"),
    "ImportanceResult":        ("star_graph.importance_engine", "ImportanceResult"),
    "ImportanceLevel":         ("star_graph.importance_engine", "ImportanceLevel"),

    # ── Phase 7: Memory Evolution — Belief System (7.2) ──
    "BeliefSystem":            ("star_graph.belief_system", "BeliefSystem"),
    "Belief":                  ("star_graph.belief_system", "Belief"),
    "BeliefMerge":             ("star_graph.belief_system", "BeliefMerge"),

    # ── Phase 7: Memory Evolution — Personality Formation (7.3) ──
    "PersonalityFormationEngine": ("star_graph.personality_formation", "PersonalityFormationEngine"),
    "CognitiveProfile":        ("star_graph.personality_formation", "CognitiveProfile"),
    "CognitiveStyle":          ("star_graph.personality_formation", "CognitiveStyle"),
    "ValueSystem":             ("star_graph.personality_formation", "ValueSystem"),
    "EvolutionTrajectory":     ("star_graph.personality_formation", "EvolutionTrajectory"),

    # ── Phase 7: Memory Evolution — Cognitive Identity (7.4) ──
    "CognitiveIdentityManager": ("star_graph.cognitive_identity", "CognitiveIdentityManager"),
    "CognitiveIdentity":       ("star_graph.cognitive_identity", "CognitiveIdentity"),
    "UserIdentitySnapshot":    ("star_graph.cognitive_identity", "UserIdentitySnapshot"),

    # ── Phase 5: Cognitive Depth (2026-05-22) ──
    # memory budget & token budget
    "MemoryBudgetConfig":      ("star_graph.memory_budget", "MemoryBudgetConfig"),
    "TokenBudgetConfig":       ("star_graph.memory_budget", "TokenBudgetConfig"),
    "MemoryBudget":            ("star_graph.memory_budget", "MemoryBudget"),
    "TokenBudget":             ("star_graph.memory_budget", "TokenBudget"),
    "infer_memory_layer":      ("star_graph.memory_budget", "infer_memory_layer"),

    # quality score
    "MemoryQualityScore":      ("star_graph.quality_score", "MemoryQualityScore"),
    "QualityScorer":           ("star_graph.quality_score", "QualityScorer"),

    # stability control
    "StabilityConfig":         ("star_graph.stability_control", "StabilityConfig"),
    "StabilityScore":          ("star_graph.stability_control", "StabilityScore"),
    "StabilityController":     ("star_graph.stability_control", "StabilityController"),

    # cognitive priority
    "PriorityLevel":           ("star_graph.cognitive_priority", "PriorityLevel"),
    "CognitivePriority":       ("star_graph.cognitive_priority", "CognitivePriority"),
    "PriorityEngine":          ("star_graph.cognitive_priority", "PriorityEngine"),

    # memory layers (4-layer pyramid)
    "MemoryLayer":             ("star_graph.memory_layers", "MemoryLayer"),
    "LayerPolicy":             ("star_graph.memory_layers", "LayerPolicy"),
    "LayerManager":            ("star_graph.memory_layers", "LayerManager"),

    # typed memory
    "MemoryType":              ("star_graph.typed_memory", "MemoryType"),
    "TypeStrategy":            ("star_graph.typed_memory", "TypeStrategy"),
    "TypeManager":             ("star_graph.typed_memory", "TypeManager"),

    # abstraction chain
    "AbstractionLevel":        ("star_graph.abstraction_chain", "AbstractionLevel"),
    "AbstractionNode":         ("star_graph.abstraction_chain", "AbstractionNode"),
    "AbstractionChainConfig":  ("star_graph.abstraction_chain", "AbstractionChainConfig"),
    "AbstractionChain":        ("star_graph.abstraction_chain", "AbstractionChain"),

    # domain graph
    "Domain":                  ("star_graph.domain_graph", "Domain"),
    "DomainConfig":            ("star_graph.domain_graph", "DomainConfig"),
    "DomainManager":           ("star_graph.domain_graph", "DomainManager"),

    # context routing
    "RoutingContext":          ("star_graph.context_routing", "RoutingContext"),
    "RoutingWeights":          ("star_graph.context_routing", "RoutingWeights"),
    "ContextRouter":           ("star_graph.context_routing", "ContextRouter"),

    # hebbian learning
    "HebbianConfig":           ("star_graph.hebbian_learning", "HebbianConfig"),
    "CoActivationTracker":     ("star_graph.hebbian_learning", "CoActivationTracker"),
    "HebbianLearner":          ("star_graph.hebbian_learning", "HebbianLearner"),

    # agent state memory
    "AgentState":              ("star_graph.agent_state", "AgentState"),
    "GoalNode":                ("star_graph.agent_state", "GoalNode"),
    "ToolCallRecord":          ("star_graph.agent_state", "ToolCallRecord"),
    "Checkpoint":              ("star_graph.agent_state", "Checkpoint"),
    "AgentStateManager":       ("star_graph.agent_state", "AgentStateManager"),

    # cognitive closure
    "FeedbackRecord":          ("star_graph.cognitive_closure", "FeedbackRecord"),
    "ClosureConfig":           ("star_graph.cognitive_closure", "ClosureConfig"),
    "CognitiveClosure":        ("star_graph.cognitive_closure", "CognitiveClosure"),

    # v1.2.11: observatory & gravity well
    "Lantern":                 ("star_graph.observatory", "Lantern"),
    "Observatory":             ("star_graph.observatory", "Observatory"),
    "IlluminationResult":      ("star_graph.observatory", "IlluminationResult"),
    "compute_luminosity":      ("star_graph.observatory", "compute_luminosity"),
    "GravityWell":             ("star_graph.gravity_well", "GravityWell"),
    "GravityWellManager":      ("star_graph.gravity_well", "GravityWellManager"),
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
