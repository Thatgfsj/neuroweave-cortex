"""Standard Benchmark Suite — evaluate and compare memory retrieval quality.

Five test categories aligned with the evaluation framework described in plan.md:
1. EXACT_FACT: Single-token precision — name→birthday, key→value
2. ASSOCIATIVE: Multi-hop path reasoning A→B→C
3. TEMPORAL: Old vs new memory interference resistance
4. NOISE: Precision under heavy ghost/noise conditions
5. COMPRESSION: Recall fidelity before vs after sleep compression

Also includes baseline comparison harness for:
- pure-embedding (cosine similarity only)
- pure-bm25 (keyword matching only)
- star-graph (our cognitive memory system)

Usage:
    from star_graph.benchmark import BenchmarkSuite, run_benchmark
    suite = BenchmarkSuite()
    suite.add_exact_fact_scenarios()
    results = suite.run(manager)
    print(results.report())
"""

from __future__ import annotations

import enum
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional, Callable


# ── Scenario categories ───────────────────────────────────────

class Category(enum.Enum):
    EXACT_FACT = "exact_fact"
    ASSOCIATIVE = "associative"
    TEMPORAL = "temporal"
    NOISE = "noise"
    COMPRESSION = "compression"


@dataclass
class BenchmarkScenario:
    """One test case: store memories, then query, expect specific answer."""
    name: str                              # unique scenario name
    category: Category
    description: str = ""
    # Memories to insert before querying
    memories: list[dict] = field(default_factory=list)
    # Query to test
    query: str = ""
    # Expected: substring that should appear in answer
    expected_answer: str = ""
    # Expected anchor IDs (for path recall)
    expected_anchor_ids: list[str] = field(default_factory=list)
    # Minimum score for has_answer
    min_relevance: float = 0.0
    # Noise memories to inject (for NOISE category)
    noise_memories: list[dict] = field(default_factory=list)
    # Tags to use for the scenario
    tags: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result for one scenario."""
    scenario: str = ""
    category: str = ""
    passed: bool = False
    metrics: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    top_items: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class CategorySummary:
    """Aggregated metrics for one category."""
    category: str = ""
    scenarios: int = 0
    passed: int = 0
    exact_match: float = 0.0       # 0..1
    has_answer: float = 0.0        # 0..1
    avg_recall_at_k: float = 0.0   # 0..1
    avg_precision: float = 0.0     # 0..1
    avg_duration_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.scenarios if self.scenarios > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Full benchmark run result."""
    name: str = ""
    total_duration_ms: float = 0.0
    scenarios: list[ScenarioResult] = field(default_factory=list)
    categories: dict[str, CategorySummary] = field(default_factory=dict)
    system_stats: dict = field(default_factory=dict)

    @property
    def total_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def total_passed(self) -> int:
        return sum(1 for s in self.scenarios if s.passed)

    @property
    def overall_exact_match(self) -> float:
        matches = [s for s in self.scenarios if s.metrics.get("exact_match", False)]
        return len(matches) / max(1, len(self.scenarios))

    @property
    def overall_has_answer(self) -> float:
        answers = [s for s in self.scenarios if s.metrics.get("has_answer", False)]
        return len(answers) / max(1, len(self.scenarios))

    def report(self) -> str:
        """Generate a plain-text benchmark report."""
        sep = "=" * 60
        lines = [
            sep,
            f"  Benchmark Report: {self.name}",
            sep,
            f"  Scenarios: {self.total_scenarios} total, {self.total_passed} passed",
            f"  Exact Match: {self.overall_exact_match:.1%}",
            f"  Has Answer:  {self.overall_has_answer:.1%}",
            sep,
        ]

        for cat_name, summary in self.categories.items():
            lines.append(
                f"  {cat_name:<20} EM={summary.exact_match:.1%} "
                f"HA={summary.has_answer:.1%} Pass={summary.pass_rate:.0%}"
            )

        lines.append(sep)

        ss = self.system_stats
        if ss:
            lines.append(f"  Anchors: {ss.get('anchors', 0)}  Edges: {ss.get('edges', 0)}")
            lines.append(f"  Exact Cache hit rate: {ss.get('exact_cache_hit_rate', 0):.1%}")
            lines.append(f"  Total duration: {self.total_duration_ms:.0f}ms")

        lines.append(sep)
        for sr in self.scenarios:
            status = "PASS" if sr.passed else "FAIL"
            em = "Y" if sr.metrics.get("exact_match") else "N"
            ha = "Y" if sr.metrics.get("has_answer") else "N"
            recall = sr.metrics.get("recall_at_k", 0)
            top = sr.top_items[0][:60] if sr.top_items else "(none)"
            err = f" ERR={sr.error}" if sr.error else ""
            lines.append(
                f"  [{status}] {sr.scenario[:45]:<45} "
                f"EM={em} HA={ha} R@k={recall:.2f}{err} -> {top}"
            )

        return "\n".join(lines)


# ── Baseline retrievers ────────────────────────────────────────

class BaselineRetriever:
    """Simple embedding-based retriever (cosine similarity only)."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._texts: list[str] = []
        self._embeddings: list[list[float]] = []

    def add(self, text: str):
        self._texts.append(text)
        self._embeddings.append(self.embedder.encode(text))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = self.embedder.encode(query)
        scores = []
        for i, emb in enumerate(self._embeddings):
            sim = _cosine_sim(q_emb, emb)
            scores.append((self._texts[i], sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class BM25Retriever:
    """Simple BM25 keyword-based retriever."""

    def __init__(self):
        self._texts: list[str] = []
        self._tokens: list[list[str]] = []

    def add(self, text: str):
        self._texts.append(text)
        self._tokens.append(_tokenize(text))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_tokens = _tokenize(query)
        N = len(self._texts)
        if N == 0:
            return []

        doc_lens = [len(t) for t in self._tokens]
        avg_dl = sum(doc_lens) / max(1, N)
        df: dict[str, int] = {}
        for tokens in self._tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        k1, b = 1.5, 0.75
        scores = []
        for i, tokens in enumerate(self._tokens):
            score = 0.0
            for qt in q_tokens:
                if qt not in df:
                    continue
                tf = tokens.count(qt)
                idf = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_lens[i] / max(1, avg_dl))
                score += idf * numerator / max(0.01, denominator)
            scores.append((self._texts[i], score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ── The benchmark suite ────────────────────────────────────────

class BenchmarkSuite:
    """Comprehensive memory evaluation framework.

    Usage:
        suite = BenchmarkSuite()
        suite.add_standard_scenarios()
        result = suite.run(memory_manager)
        print(result.report())

        # Compare baselines
        comparison = suite.compare_baselines(manager, embedder)
        print(comparison.report())
    """

    def __init__(self):
        self.scenarios: list[BenchmarkScenario] = []

    def add(self, scenario: BenchmarkScenario):
        self.scenarios.append(scenario)

    def add_standard_scenarios(self):
        """Populate with standard test scenarios across all 5 categories."""
        self.add_exact_fact_scenarios()
        self.add_associative_scenarios()
        self.add_temporal_scenarios()
        self.add_noise_scenarios()
        self.add_compression_scenarios()

    # ── Scenario builders ───────────────────────────────────

    def add_exact_fact_scenarios(self):
        """Category 1: Exact fact recall — single-token precision."""

        # Person → attribute
        self.add(BenchmarkScenario(
            name="person-birthday",
            category=Category.EXACT_FACT,
            description="Name → birthday lookup",
            memories=[
                {"text": "Alice's birthday is May 10th", "tags": ["person", "birthday"]},
                {"text": "Bob's birthday is November 3rd", "tags": ["person", "birthday"]},
                {"text": "Charlie was born on July 22nd", "tags": ["person", "birthday"]},
            ],
            query="What is Alice's birthday?",
            expected_answer="May 10th",
        ))

        # Config key → value
        self.add(BenchmarkScenario(
            name="config-port",
            category=Category.EXACT_FACT,
            description="Key → value lookup",
            memories=[
                {"text": "The Redis server uses port 6379", "tags": ["redis", "config"]},
                {"text": "MySQL runs on port 3306", "tags": ["mysql", "config"]},
                {"text": "The HTTP API listens on port 8080", "tags": ["api", "config"]},
            ],
            query="What port does Redis use?",
            expected_answer="6379",
        ))

        # Preference
        self.add(BenchmarkScenario(
            name="preference-theme",
            category=Category.EXACT_FACT,
            description="User preference recall",
            memories=[
                {"text": "User prefers dark mode for the UI", "tags": ["preference", "ui"]},
                {"text": "User likes Python for scripting tasks", "tags": ["preference", "language"]},
                {"text": "User uses VS Code as primary editor", "tags": ["preference", "editor"]},
            ],
            query="What theme does the user prefer?",
            expected_answer="dark mode",
        ))

        # Version string
        self.add(BenchmarkScenario(
            name="version-string",
            category=Category.EXACT_FACT,
            description="Version number recall",
            memories=[
                {"text": "Python version 3.11.9 is installed", "tags": ["python", "version"]},
                {"text": "Rust version 1.95.0 is the current stable", "tags": ["rust", "version"]},
            ],
            query="What version of Python is installed?",
            expected_answer="3.11.9",
        ))

        # Bug fix
        self.add(BenchmarkScenario(
            name="bug-fix-recall",
            category=Category.EXACT_FACT,
            description="Bug fix detail recall",
            memories=[
                {"text": "Fixed connection pool bug — increased size from 10 to 20", "tags": ["bug", "fix"]},
                {"text": "Fixed memory leak in rendering pipeline", "tags": ["bug", "memory"]},
            ],
            query="What was the connection pool bug fix?",
            expected_answer="10 to 20",
        ))

    def add_associative_scenarios(self):
        """Category 2: Associative transfer — multi-hop reasoning."""

        self.add(BenchmarkScenario(
            name="assoc-redis-debug",
            category=Category.ASSOCIATIVE,
            description="A→B: Redis → timeout → fix",
            memories=[
                {"text": "Redis connection timed out after 30 seconds", "tags": ["redis", "error"]},
                {"text": "The timeout was caused by exhausted connection pool", "tags": ["debug", "redis"]},
                {"text": "Fixed by increasing pool size from 10 to 20", "tags": ["fix", "redis"]},
            ],
            query="How did we fix the Redis timeout?",
            expected_answer="pool size",
        ))

        self.add(BenchmarkScenario(
            name="assoc-project-stack",
            category=Category.ASSOCIATIVE,
            description="A→B→C: project → language → tooling",
            memories=[
                {"text": "The API service is written in Python 3.11", "tags": ["api", "python"]},
                {"text": "Python 3.11 uses FastAPI for HTTP handling", "tags": ["python", "fastapi"]},
                {"text": "FastAPI endpoints are served via Uvicorn on port 8080", "tags": ["fastapi", "server"]},
            ],
            query="What port does the API service use?",
            expected_answer="8080",
        ))

        self.add(BenchmarkScenario(
            name="assoc-error-chain",
            category=Category.ASSOCIATIVE,
            description="Error → root cause → permanent fix",
            memories=[
                {"text": "Production crash: NullPointerException in auth module", "tags": ["error", "crash"]},
                {"text": "Root cause: session token was not validated before use", "tags": ["debug", "root_cause"]},
                {"text": "Permanent fix: added token validation at middleware layer", "tags": ["fix", "auth"]},
            ],
            query="What was the root cause of the auth crash?",
            expected_answer="token",
        ))

    def add_temporal_scenarios(self):
        """Category 3: Temporal confusion — old vs new interference."""

        self.add(BenchmarkScenario(
            name="temporal-port-change",
            category=Category.TEMPORAL,
            description="Old port → new port (should return new)",
            memories=[
                {"text": "Redis server port was changed from 6379 to 6380", "tags": ["redis", "config"]},
            ],
            query="What port is Redis currently on?",
            expected_answer="6380",
        ))

        self.add(BenchmarkScenario(
            name="temporal-version-upgrade",
            category=Category.TEMPORAL,
            description="Old version → new version",
            memories=[
                {"text": "Upgraded Python from 3.10 to 3.11.9", "tags": ["python", "upgrade"]},
                {"text": "Originally used Python 3.10 with asyncio issues", "tags": ["python", "history"]},
            ],
            query="What Python version are we currently using?",
            expected_answer="3.11.9",
        ))

        self.add(BenchmarkScenario(
            name="temporal-bug-timeline",
            category=Category.TEMPORAL,
            description="Before fix → after fix",
            memories=[
                {"text": "Memory leak found on Tuesday, patched on Wednesday", "tags": ["bug", "timeline"]},
                {"text": "The leak was in the image processing pipeline", "tags": ["bug", "root_cause"]},
                {"text": "After the fix, memory usage dropped from 4GB to 800MB", "tags": ["fix", "metrics"]},
            ],
            query="When was the memory leak patched?",
            expected_answer="Wednesday",
        ))

    def add_noise_scenarios(self):
        """Category 4: Noise resistance — precision under distractions."""

        self.add(BenchmarkScenario(
            name="noise-exact-fact",
            category=Category.NOISE,
            description="Exact fact retrieval with 10 distracting memories",
            memories=[
                {"text": "The database password is 's3cur3_p@ss'", "tags": ["security", "credential"]},
            ],
            query="What is the database password?",
            expected_answer="s3cur3_p@ss",
            noise_memories=[
                {"text": f"Random noise memory {i}: the quick brown fox jumps over the lazy dog number {i}", "tags": ["noise"]}
                for i in range(10)
            ],
        ))

        self.add(BenchmarkScenario(
            name="noise-associative",
            category=Category.NOISE,
            description="Associative recall with noisy similar-looking items",
            memories=[
                {"text": "The payment service API key is 'pk_live_abc123'", "tags": ["payment", "key"]},
                {"text": "Payment service uses Stripe for processing", "tags": ["payment", "stripe"]},
            ],
            query="What is the payment service API key?",
            expected_answer="pk_live_abc123",
            noise_memories=[
                {"text": f"Random key reference {i}: some-api-key-{i:04d} is used for testing", "tags": ["noise", "key"]}
                for i in range(8)
            ],
        ))

    def add_compression_scenarios(self):
        """Category 5: Compression fidelity — recall before vs after sleep."""

        self.add(BenchmarkScenario(
            name="compress-fact-survival",
            category=Category.COMPRESSION,
            description="Fact should survive sleep compression",
            memories=[
                {"text": "The project requires Python 3.11+ for match-case syntax", "tags": ["python", "requirement"]},
                {"text": "Dependencies are managed with pip and requirements.txt", "tags": ["python", "deps"]},
                {"text": "Production deployment uses Docker with multi-stage builds", "tags": ["docker", "deploy"]},
                {"text": "CI/CD pipeline runs on GitHub Actions with 3 stages", "tags": ["ci", "github"]},
                {"text": "Test coverage must stay above 80% for PR merge", "tags": ["testing", "standard"]},
            ],
            query="What Python version is required?",
            expected_answer="3.11",
        ))

    # ── Runner ──────────────────────────────────────────────

    def run(self, manager, system_name: str = "star-graph") -> BenchmarkResult:
        """Execute all scenarios against a MemoryManager.

        For each scenario:
        1. Store target memories
        2. Optionally inject noise memories
        3. Run recall query
        4. Compute metrics
        5. (For COMPRESSION) run sleep, then recall again
        """
        t_start = time.time()
        result = BenchmarkResult(name=system_name)

        for scenario in self.scenarios:
            sr = self._run_scenario(manager, scenario)
            result.scenarios.append(sr)

        # Compute category summaries
        cat_results: dict[str, list[ScenarioResult]] = {}
        for sr in result.scenarios:
            cat_results.setdefault(sr.category, []).append(sr)

        for cat_name, srs in cat_results.items():
            result.categories[cat_name] = self._summarize_category(cat_name, srs)

        result.total_duration_ms = (time.time() - t_start) * 1000
        result.system_stats = {
            "anchors": len(manager.graph.anchors),
            "edges": len(manager.graph.edges),
            "exact_cache_hit_rate": manager.exact_cache.hit_rate,
            "exact_cache_entries": manager.exact_cache.size,
        }

        return result

    def _run_scenario(self, manager, scenario: BenchmarkScenario) -> ScenarioResult:
        """Execute one scenario."""
        t0 = time.time()

        try:
            # Insert target memories
            for mem in scenario.memories:
                manager.remember(
                    mem["text"],
                    tags=mem.get("tags", []),
                    source_session=scenario.name,
                )

            # Insert noise if category is NOISE
            if scenario.category == Category.NOISE:
                for nm in scenario.noise_memories:
                    manager.remember(
                        nm["text"],
                        tags=nm.get("tags", []) + ["noise"],
                        source_session=scenario.name + "_noise",
                    )

            # Run recall
            ctx = manager.recall(query=scenario.query, max_items=10)
            items = ctx.items

            # Compute metrics
            metrics = self._compute_metrics(scenario, items)

            # For compression: run sleep then re-recall
            if scenario.category == Category.COMPRESSION:
                pre_recall = metrics.get("has_answer", False)
                manager.sleep()
                ctx2 = manager.recall(query=scenario.query, max_items=10)
                post_items = ctx2.items
                post_metrics = self._compute_metrics(scenario, post_items)
                metrics["pre_compression_recall"] = pre_recall
                metrics["post_compression_recall"] = post_metrics.get("has_answer", False)
                metrics["recall_drop"] = 1.0 if pre_recall and not post_metrics.get("has_answer") else 0.0

            top_texts = []
            for item in items[:5]:
                if item.anchor:
                    top_texts.append(item.anchor.text[:80])
                else:
                    top_texts.append((getattr(item, 'compressed_text', '') or '')[:80])

            passed = metrics.get("has_answer", False)

            return ScenarioResult(
                scenario=scenario.name,
                category=scenario.category.value,
                passed=passed,
                metrics=metrics,
                duration_ms=(time.time() - t0) * 1000,
                top_items=top_texts,
            )

        except Exception as exc:
            return ScenarioResult(
                scenario=scenario.name,
                category=scenario.category.value,
                passed=False,
                duration_ms=(time.time() - t0) * 1000,
                error=str(exc),
            )

    def _compute_metrics(self, scenario: BenchmarkScenario,
                         items: list) -> dict:
        """Compute all metrics for one recall result."""
        if not items:
            return {"exact_match": False, "has_answer": False,
                    "recall_at_k": 0.0, "precision": 0.0}

        texts = []
        for item in items:
            if item.anchor:
                texts.append(item.anchor.text.lower())
            else:
                texts.append((getattr(item, 'compressed_text', '') or '').lower())

        expected = scenario.expected_answer.lower()

        # Exact match: any result text contains the exact expected string
        exact = any(expected in t for t in texts)

        # Has answer: expected substring found in at least one result
        ha = exact  # for exact_fact tests, same thing; for associative may differ

        # Relaxed has_answer: check word overlap
        if not ha:
            expected_words = set(expected.split())
            for t in texts:
                t_words = set(t.split())
                if len(expected_words & t_words) >= len(expected_words) * 0.5:
                    ha = True
                    break

        # Recall@k: was expected answer in top-k?
        recall_k = 1.0 if ha else 0.0

        # Precision: fraction of results that are relevant
        relevant = 0
        for t in texts:
            if expected in t or _word_overlap(t, expected) > 0.5:
                relevant += 1
        precision = relevant / max(1, len(texts))

        scores = [item.relevance_score for item in items[:3]]
        top_score = scores[0] if scores else 0.0

        return {
            "exact_match": exact,
            "has_answer": ha,
            "recall_at_k": recall_k,
            "precision": round(precision, 3),
            "top_score": round(top_score, 3),
            "results_count": len(items),
        }

    def _summarize_category(self, name: str,
                           results: list[ScenarioResult]) -> CategorySummary:
        """Compute aggregate metrics for a category."""
        n = len(results)
        passed = sum(1 for r in results if r.passed)
        ems = [r.metrics.get("exact_match", False) for r in results]
        has = [r.metrics.get("has_answer", False) for r in results]
        recalls = [r.metrics.get("recall_at_k", 0.0) for r in results]
        precisions = [r.metrics.get("precision", 0.0) for r in results]
        durations = [r.duration_ms for r in results]

        return CategorySummary(
            category=name,
            scenarios=n,
            passed=passed,
            exact_match=sum(ems) / n if n else 0.0,
            has_answer=sum(has) / n if n else 0.0,
            avg_recall_at_k=statistics.mean(recalls) if recalls else 0.0,
            avg_precision=statistics.mean(precisions) if precisions else 0.0,
            avg_duration_ms=statistics.mean(durations) if durations else 0.0,
        )

    # ── Baseline comparison ────────────────────────────────

    def compare_baselines(self, manager,
                          embedder=None) -> list[BenchmarkResult]:
        """Compare Star-Graph against baseline retrievers.

        Returns [star_graph_result, embedding_result, bm25_result].
        """
        if embedder is None:
            from .embedding import get_embedder
            embedder = get_embedder()

        results = []

        # 1. Star-Graph (our system)
        results.append(self.run(manager, system_name="star-graph"))

        # 2. Pure embedding (cosine similarity)
        emb_retriever = BaselineRetriever(embedder)
        emb_result = self._run_with_baseline(emb_retriever, "pure-embedding")
        results.append(emb_result)

        # 3. Pure BM25 (keyword)
        bm25_retriever = BM25Retriever()
        bm25_result = self._run_with_baseline(bm25_retriever, "pure-bm25")
        results.append(bm25_result)

        return results

    def _run_with_baseline(self, retriever,
                          system_name: str) -> BenchmarkResult:
        """Run scenarios against a baseline retriever."""
        t_start = time.time()
        result = BenchmarkResult(name=system_name)

        for scenario in self.scenarios:
            t0 = time.time()
            try:
                # Baseline has no graph — just run search on stored texts
                metrics = {"exact_match": False, "has_answer": False,
                          "recall_at_k": 0.0, "precision": 0.0}
                top_texts: list[str] = []

                # Store and search simultaneously (stateless)
                all_texts = [m["text"] for m in scenario.memories]
                if scenario.category == Category.NOISE:
                    all_texts += [nm["text"] for nm in scenario.noise_memories]

                for text in all_texts:
                    retriever.add(text)

                search_results = retriever.search(scenario.query, top_k=10)
                top_texts = [t[:80] for t, _ in search_results[:5]]

                expected = scenario.expected_answer.lower()
                for text, _ in search_results:
                    if expected in text.lower():
                        metrics["has_answer"] = True
                        metrics["exact_match"] = True
                        metrics["recall_at_k"] = 1.0
                        break

                sr = ScenarioResult(
                    scenario=scenario.name,
                    category=scenario.category.value,
                    passed=metrics["has_answer"],
                    metrics=metrics,
                    duration_ms=(time.time() - t0) * 1000,
                    top_items=top_texts,
                )
                result.scenarios.append(sr)

            except Exception as exc:
                result.scenarios.append(ScenarioResult(
                    scenario=scenario.name,
                    category=scenario.category.value,
                    error=str(exc),
                ))

        result.total_duration_ms = (time.time() - t_start) * 1000
        return result

    def compare_report(self, results: list[BenchmarkResult]) -> str:
        """Generate a comparison report across systems."""
        sep = "=" * 60
        lines = [
            sep,
            "  System Comparison Report",
            sep,
        ]

        header = f"  {'System':<20} {'EM':>6}  {'HA':>6}  {'Pass':>6}  {'Time':>8}"
        lines.append(header)
        lines.append("  " + "-" * 54)

        for r in results:
            em_pct = f"{r.overall_exact_match:.1%}"
            ha_pct = f"{r.overall_has_answer:.1%}"
            pass_pct = f"{r.total_passed / max(1, r.total_scenarios):.1%}"
            time_str = f"{r.total_duration_ms:.0f}ms"
            lines.append(
                f"  {r.name:<20} {em_pct:>6}  {ha_pct:>6}  {pass_pct:>6}  {time_str:>8}"
            )

        lines.append(sep)

        # Category breakdown
        for cat in Category:
            lines.append(f"\n  {cat.value}:")
            for r in results:
                summary = r.categories.get(cat.value)
                if summary:
                    lines.append(
                        f"    {r.name:<20} EM={summary.exact_match:.1%} "
                        f"HA={summary.has_answer:.1%} "
                        f"Pass={summary.pass_rate:.0%}"
                    )

        return "\n".join(lines)


# ── Convenience ────────────────────────────────────────────────

def run_benchmark(manager, scenarios: str = "standard") -> BenchmarkResult:
    """Run the standard benchmark suite against a manager.

    Args:
        manager: MemoryManager instance.
        scenarios: "standard" (all), "exact_fact", "associative",
                   "temporal", "noise", "compression"
    """
    suite = BenchmarkSuite()

    if scenarios == "standard":
        suite.add_standard_scenarios()
    elif scenarios == "exact_fact":
        suite.add_exact_fact_scenarios()
    elif scenarios == "associative":
        suite.add_associative_scenarios()
    elif scenarios == "temporal":
        suite.add_temporal_scenarios()
    elif scenarios == "noise":
        suite.add_noise_scenarios()
    elif scenarios == "compression":
        suite.add_compression_scenarios()

    return suite.run(manager)


def compare_systems(manager, embedder=None) -> list[BenchmarkResult]:
    """Run baseline comparison: star-graph vs pure-embedding vs pure-bm25."""
    suite = BenchmarkSuite()
    suite.add_standard_scenarios()
    return suite.compare_baselines(manager, embedder)


# ── Helpers ────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-zA-Z_]\w{1,}', text.lower())


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-8)


def _word_overlap(text: str, expected: str) -> float:
    t_words = set(text.split())
    e_words = set(expected.split())
    if not e_words:
        return 0.0
    return len(t_words & e_words) / len(e_words)
