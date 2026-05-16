"""Consolidation — sleep-cycle memory maintenance (lazy-loaded)."""


def __getattr__(name: str):
    _registry = {
        "SleepCycle": ("star_graph.sleep", "SleepCycle"),
        "SleepNREM": ("star_graph.sleep_nrem", "SleepNREM"),
        "SleepREM": ("star_graph.sleep_rem", "SleepREM"),
        "SleepConsolidate": ("star_graph.sleep_consolidate", "SleepConsolidate"),
        "SleepReport": ("star_graph.sleep_report", "SleepReport"),
        "PhaseMetrics": ("star_graph.sleep_report", "PhaseMetrics"),
        "MemoryEvolutionEngine": ("star_graph.evolution", "MemoryEvolutionEngine"),
        "GhostSubsystem": ("star_graph.ghost", "GhostSubsystem"),
        "CompressionLevel": ("star_graph.compression", "CompressionLevel"),
        "MultiLevelCompressor": ("star_graph.compression", "MultiLevelCompressor"),
        "SessionCompressor": ("star_graph.compression", "SessionCompressor"),
        "MicroSleep": ("star_graph.micro_sleep", "MicroSleep"),
        "SurvivalFunction": ("star_graph.survival", "SurvivalFunction"),
        "SurvivalRegistry": ("star_graph.survival", "SurvivalRegistry"),
    }
    if name in _registry:
        mod_name, attr = _registry[name]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'star_graph.consolidation' has no attribute '{name}'")


__all__ = [
    "SleepCycle", "SleepNREM", "SleepREM", "SleepConsolidate",
    "SleepReport", "PhaseMetrics",
    "MemoryEvolutionEngine",
    "GhostSubsystem",
    "CompressionLevel", "MultiLevelCompressor", "SessionCompressor",
    "MicroSleep",
    "SurvivalFunction", "SurvivalRegistry",
]
