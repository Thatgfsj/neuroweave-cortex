"""Benchmark Suite — re-exported from contrib/ (P2 architecture slimdown)."""
from .contrib.benchmark import (
    BenchmarkSuite, BenchmarkScenario, BenchmarkResult,
    ScenarioResult, Category, run_benchmark, compare_systems,
)

__all__ = [
    "BenchmarkSuite", "BenchmarkScenario", "BenchmarkResult",
    "ScenarioResult", "Category", "run_benchmark", "compare_systems",
]
