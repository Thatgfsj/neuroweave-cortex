"""Synthetic data generation, baselines, metrics and reporting for the benchmark.

Extracted verbatim from ``examples/memory_benchmark.py`` (issue #10) to keep the
CLI entry point small. Function bodies are unchanged.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from star_graph import StarGraph, OscillationResonanceRetriever


# ═══════════════════════════════════════════════════════════════════
# Synthetic Data Generator
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConversationTurn:
    id: str
    session_id: str
    timestamp: float
    speaker: str  # "user" or "assistant"
    text: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    emotional_valence: float = 0.0
    ground_truth_category: str = ""  # for eval


@dataclass
class TestQuery:
    query: str
    category: str  # long_context, cross_session, interference, etc.
    ground_truth_ids: list[str] = field(default_factory=list)  # expected turn IDs
    required_keywords: list[str] = field(default_factory=list)  # must appear in retrieved text
    forbidden_keywords: list[str] = field(default_factory=list)  # must NOT appear (for interference)
    description: str = ""


class SyntheticDataGenerator:
    """Generates realistic multi-session conversations with known ground truth.

    Plants specific "memory probes" — distinctive facts at known positions —
    and generates queries that target those exact facts for reliable evaluation.
    """

    # Memory probes: (session, position_hint, text, tags, importance, valence)
    # These are planted at specific positions so we know exactly where they are.
    PROBES = [
        # ── Long-context probes (early sessions, tested from late queries) ──
        ("s01", 0, "I strongly prefer Python for all backend development work. "
         "It's been my primary language for 5 years and I'm most productive in it.",
         ["python", "preference", "backend"], 0.9, 0.7),
        ("s01", 5, "My development environment is Neovim with a custom configuration "
         "for Python development including LSP, debugging, and test integration.",
         ["tools", "editor", "neovim"], 0.7, 0.5),
        ("s02", 3, "I'm building a campus network auto-login tool called CampusNet-AutoLogin. "
         "It uses Selenium to automate the captive portal login, with a Tkinter GUI.",
         ["project", "campus-net", "selenium", "tkinter"], 0.95, 0.6),

        # ── Cross-session probes (same project across multiple sessions) ──
        ("s03", 1, "Update on CampusNet-AutoLogin: I added error handling for when "
         "the network is down. It now retries 3 times with exponential backoff.",
         ["project", "campus-net", "error-handling"], 0.8, 0.4),
        ("s05", 8, "CampusNet-AutoLogin now has a system tray icon and auto-reconnect. "
         "It monitors the network and re-authenticates when the connection drops.",
         ["project", "campus-net", "system-tray", "auto-reconnect"], 0.85, 0.5),
        ("s07", 4, "Fixed a critical race condition in CampusNet-AutoLogin where "
         "the login would fail if the page loaded faster than expected.",
         ["project", "campus-net", "race-condition", "bug-fix"], 0.9, -0.3),
        ("s09", 10, "Packaged CampusNet-AutoLogin as a standalone Windows executable "
         "using PyInstaller. Now it runs on startup via a scheduled task.",
         ["project", "campus-net", "deployment", "pyinstaller"], 0.9, 0.8),

        # ── Bug history probes (set up as problem → diagnosis → fix) ──
        ("s02", 10, "BUG: Redis connection pool keeps timing out in production. "
         "Error: 'ConnectionPool exhausted, unable to acquire connection'. "
         "Happens under peak load with 200+ concurrent users.",
         ["bug", "redis", "connection-pool", "timeout"], 0.85, -0.7),
        ("s03", 15, "ROOT CAUSE for Redis timeout: the connection pool was configured "
         "for 10 connections but peak load needs 50+. Used redis-cli INFO clients "
         "to verify the connection count was saturated.",
         ["bug", "redis", "root-cause", "diagnosis"], 0.9, -0.3),
        ("s04", 20, "FIX: Increased Redis max_connections from 10 to 50, added "
         "connection timeout of 5 seconds, and implemented a circuit breaker "
         "that returns stale cache instead of failing.",
         ["bug", "redis", "fix", "circuit-breaker"], 0.9, 0.6),

        ("s06", 8, "BUG: Docker builds take 8 minutes even when I only change "
         "one line of Python. The cache is always invalidated at the pip install step.",
         ["bug", "docker", "cache", "build-performance"], 0.8, -0.5),
        ("s07", 12, "ROOT CAUSE for Docker cache: the Dockerfile does COPY . . "
         "before RUN pip install, so any source change invalidates all layers. "
         "Reordered to copy requirements.txt first.",
         ["bug", "docker", "root-cause", "dockerfile-order"], 0.85, -0.2),
        ("s08", 6, "FIX: Restructured Dockerfile — COPY requirements.txt first, "
         "RUN pip install, then COPY . . Build time went from 8 min to 45 seconds.",
         ["bug", "docker", "fix", "build-optimization"], 0.9, 0.7),

        # ── Conflicting/evolving information probes ──
        ("s01", 12, "I'm a big fan of monolithic architecture. Microservices are "
         "overengineered for most projects. Monoliths are easier to develop, test, and deploy.",
         ["architecture", "monolith", "preference"], 0.7, 0.5),
        ("s06", 14, "I've changed my mind about architecture. After dealing with "
         "a monolith that was hard to scale, I now strongly prefer microservices "
         "for any project that needs to grow beyond a single team.",
         ["architecture", "microservices", "preference-change"], 0.85, 0.3),

        ("s02", 16, "I love using ORMs like SQLAlchemy. They make database work "
         "so much cleaner and prevent SQL injection. Raw SQL is tedious and error-prone.",
         ["database", "orm", "preference"], 0.7, 0.6),
        ("s07", 18, "I've completely reversed my stance on ORMs. After debugging "
         "too many N+1 query problems, I now prefer writing raw SQL for any "
         "query more complex than a simple CRUD. ORMs hide too much.",
         ["database", "raw-sql", "preference-change"], 0.85, -0.4),

        # ── Important facts that should survive pruning ──
        ("s01", 20, "My project follows a strict code review policy: every PR "
         "must be reviewed by at least one other developer before merging. "
         "We use GitHub's branch protection rules to enforce this.",
         ["process", "code-review", "github"], 0.9, 0.5),
        ("s03", 22, "The production database backup strategy: daily full backups "
         "to S3 with 30-day retention, hourly WAL archiving for point-in-time "
         "recovery, and quarterly restore drills.",
         ["process", "backup", "database", "production"], 0.95, 0.4),
        ("s05", 24, "Security policy: all API endpoints require JWT authentication, "
         "secrets are stored in HashiCorp Vault (never in .env files), and we "
         "run OWASP ZAP scans in CI before every deploy.",
         ["process", "security", "jwt", "vault"], 0.95, 0.4),
    ]

    # Topics that evolve over time
    PREFERENCES = [
        ("Python", "programming language", 0.9),
        ("Rust", "programming language", 0.3),
        ("TypeScript", "programming language", 0.6),
        ("React", "frontend framework", 0.8),
        ("PostgreSQL", "database", 0.7),
        ("Redis", "cache", 0.6),
        ("Docker", "containerization", 0.8),
        ("GitHub Actions", "CI/CD", 0.5),
        ("Flask", "web framework", 0.4),
        ("FastAPI", "web framework", 0.7),
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.turns: list[ConversationTurn] = []
        self.queries: list[TestQuery] = []
        self._turn_counter = 0
        self._probe_index: dict[str, int] = {}  # probe description → turn index
        self._probe_turns: dict[str, list[str]] = defaultdict(list)  # category → turn ids

    def generate(self, num_sessions: int = 12, turns_per_session: int = 420) -> tuple[
        list[ConversationTurn], list[TestQuery]
    ]:
        """Generate the full dataset. ~5000 turns across 12 sessions."""
        self._num_sessions = num_sessions
        self._turns_per_full = turns_per_session
        for s in range(num_sessions):
            session_id = f"s{s+1:02d}"
            self._generate_session(session_id, turns_per_session)

        self._generate_queries()
        return self.turns, self.queries

    def _generate_session(self, session_id: str, num_turns: int) -> None:
        session_topics = self._session_topics(session_id)
        session_num = int(session_id[1:])

        # Plant probes for this session — map probe sessions to available range
        probes_for_session = []
        for (s, pos, text, tags, imp, val) in self.PROBES:
            probe_sn = int(s[1:])
            # Scale probe session number to fit available range
            mapped_sn = max(1, probe_sn * self._num_sessions // 12)
            if mapped_sn == session_num:
                # Scale position to fit turns_per_session
                scaled_pos = int(pos * num_turns / self._turns_per_full)
                probes_for_session.append((scaled_pos, text, tags, imp, val))

        for i in range(num_turns):
            self._turn_counter += 1
            is_user = i % 2 == 0

            # Check if a probe should be placed at this position
            probe = next((p for p in probes_for_session if p[0] <= i < p[0] + 2), None)
            if probe and is_user:
                _, text, tags, importance, valence = probe
                self._probe_turns["_all"].append(f"turn_{self._turn_counter:05d}")
            else:
                topic = self.rng.choice(session_topics) if session_topics else "general"
                if is_user:
                    text = self._generate_user_message(session_id, i, topic)
                else:
                    text = self._generate_assistant_message(session_id, i, topic)
                importance = self._compute_importance(topic)
                valence = self._compute_valence(topic, text)
                tags = [topic] if topic != "general" else []

            turn = ConversationTurn(
                id=f"turn_{self._turn_counter:05d}",
                session_id=session_id,
                timestamp=1000.0 * self._turn_counter + self.rng.uniform(0, 100),
                speaker="user" if is_user else "assistant",
                text=text,
                tags=tags,
                importance=importance,
                emotional_valence=valence,
                ground_truth_category=tags[0] if tags else "general",
            )
            self.turns.append(turn)

    def _session_topics(self, session_id: str) -> list[str]:
        """Assign topics per session for realistic structure."""
        si = int(session_id[1:])
        base_topics = ["python", "frontend", "database", "devops", "debugging", "architecture"]

        # Each session has 2-3 dominant topics
        primary = base_topics[si % len(base_topics)]
        secondary = base_topics[(si + 2) % len(base_topics)]
        return [primary, secondary, "general"]

    def _generate_user_message(self, session_id: str, idx: int, topic: str) -> str:
        templates = {
            "python": [
                "I'm working on a Python script that {action}. Can you help?",
                "How do I {action} using Python?",
                "Getting an error with my Python code when trying to {action}.",
                "Is there a better way to {action} in Python?",
                "Can you explain how {concept} works in Python?",
            ],
            "frontend": [
                "My React component is {action}. How do I fix this?",
                "What's the best way to {action} in React?",
                "I need to {action} on the frontend. Any suggestions?",
                "The CSS for {component} is not working as expected.",
                "Should I use {tool_a} or {tool_b} for this frontend task?",
            ],
            "database": [
                "I'm seeing slow queries when {action}. How do I optimize?",
                "What's the best indexing strategy for {scenario}?",
                "Need to {action} in PostgreSQL. What approach?",
                "My Redis cache is {action}. How to debug?",
                "Should I use {db_a} or {db_b} for this use case?",
            ],
            "devops": [
                "Docker build is failing when {action}. Error logs attached.",
                "How do I set up CI/CD for {scenario}?",
                "Kubernetes pod keeps {action}. How to troubleshoot?",
                "Need to {action} in my deployment pipeline.",
                "What's the best practice for {action} in production?",
            ],
            "debugging": [
                "Getting a weird bug: {action}. Stack trace: {trace}",
                "This error keeps appearing: {action}. I've tried everything.",
                "The {component} is {action} intermittently. Race condition?",
                "Memory leak when {action}. How to profile this?",
                "Production is {action} after the last deploy. Need urgent help.",
            ],
            "architecture": [
                "I'm designing a system that {action}. What architecture?",
                "Monolith vs microservices for {scenario}?",
                "How should I structure the {component} layer?",
                "Need to decide between {tech_a} and {tech_b} for {use_case}.",
                "What's the best pattern for {action} across services?",
            ],
            "general": [
                "I was thinking about {action}. What do you think?",
                "Can you help me understand {concept} better?",
                "I read about {concept} yesterday. Have you used it?",
                "What's your experience with {tool}?",
                "I need advice on {action} for my project.",
            ],
        }

        actions = [
            "parsing JSON from an API response",
            "handling async tasks properly",
            "managing database connections",
            "setting up authentication",
            "processing large CSV files",
            "connecting to a WebSocket",
            "implementing rate limiting",
            "logging and monitoring",
            "error handling and retries",
            "caching frequently accessed data",
            "validating user input",
            "deploying to staging",
            "running integration tests",
            "refactoring legacy code",
            "adding type hints",
            "configuring environment variables",
            "optimizing build times",
            "setting up hot reload",
            "managing state across components",
            "securing API endpoints",
            "migrating from an old library",
            "handling timezone conversions",
            "implementing search functionality",
            "adding pagination to list endpoints",
            "debugging a race condition",
        ]
        concepts = [
            "dependency injection", "async/await", "connection pooling",
            "middleware", "serialization", "authentication flow",
            "event sourcing", "CQRS", "reactive programming",
            "state machines", "message queues", "load balancing",
        ]
        tools = ["pytest", "Docker Compose", "Kubernetes", "Terraform",
                 "NGINX", "Grafana", "ELK stack", "Prometheus"]

        template = self.rng.choice(templates.get(topic, templates["general"]))
        result = template.format(
            action=self.rng.choice(actions),
            concept=self.rng.choice(concepts),
            tool=self.rng.choice(tools),
            scenario=self.rng.choice(["a multi-tenant SaaS", "an e-commerce site",
                                        "a real-time dashboard", "a data pipeline"]),
            component=self.rng.choice(["sidebar", "navbar", "modal", "table", "form"]),
            tool_a=self.rng.choice(tools), tool_b=self.rng.choice(tools),
            db_a="PostgreSQL", db_b="MongoDB",
            tech_a="gRPC", tech_b="REST",
            use_case=self.rng.choice(["user management", "payment processing",
                                       "notification system", "analytics pipeline"]),
            trace="Traceback (most recent call last):\n  File 'app.py', line 42, in handler\n    result = await process()",
        )

        return result

    def _generate_assistant_message(self, session_id: str, idx: int, topic: str) -> str:
        templates = [
            "Here's how you can {action}: first, {step1}, then {step2}.",
            "The issue is likely {cause}. Try {solution}.",
            "Good question! The key concept is {concept}. In practice, {practice}.",
            "I recommend using {tool} for this. Here's why: {reason}.",
            "The error suggests {diagnosis}. Let me show you the fix.",
            "Based on your setup, the best approach is {approach}.",
            "This is a common pattern. The standard solution is {solution}.",
            "Let me break this down: {step1}, {step2}, {step3}.",
            "The root cause is probably {cause}. Here's a step-by-step fix.",
            "I'd suggest {approach} because {reason}. Alternative: {alt}.",
        ]

        solutions = [
            "update your connection pool settings",
            "add proper error handling with try/except",
            "use async/await for I/O-bound operations",
            "implement a retry mechanism with exponential backoff",
            "add caching with Redis to reduce database load",
            "refactor to use dependency injection",
        ]

        template = self.rng.choice(templates)
        return template.format(
            action=self.rng.choice(["set up the project", "configure the server",
                                      "write the query", "structure the code"]),
            step1="install the dependencies",
            step2="configure the connection",
            step3="test with a simple query",
            cause=self.rng.choice(["a misconfigured connection pool",
                                    "an expired token", "a race condition"]),
            solution=self.rng.choice(solutions),
            concept=self.rng.choice(["connection pooling", "async I/O", "caching strategies"]),
            practice="you want to minimize the number of simultaneous connections",
            tool=self.rng.choice(["SQLAlchemy", "Redis", "Celery", "FastAPI"]),
            reason="it handles edge cases well and has good documentation",
            diagnosis="there's a resource leak in the connection handler",
            approach=self.rng.choice(["use a connection pool with proper sizing",
                                        "implement circuit breaker pattern"]),
            alt="you could also use raw TCP connections",
        )

    def _compute_importance(self, topic: str) -> float:
        base = 0.5
        if topic in ("debugging", "architecture"):
            base += 0.2
        return min(1.0, base + self.rng.uniform(-0.1, 0.1))

    def _compute_valence(self, topic: str, text: str) -> float:
        if "error" in text.lower() or "bug" in text.lower() or "failing" in text.lower():
            return -0.5 + self.rng.uniform(-0.3, 0.1)
        if "prefer" in text.lower() or "love" in text.lower() or "great" in text.lower():
            return 0.5 + self.rng.uniform(0.0, 0.3)
        return self.rng.uniform(-0.2, 0.2)

    def _generate_queries(self) -> None:
        """Generate test queries targeting planted probes across all 5 categories."""

        def _find_turns_with(*keywords: str) -> list[str]:
            ids = []
            for t in self.turns:
                text_lower = t.text.lower()
                if all(kw.lower() in text_lower for kw in keywords):
                    ids.append(t.id)
            return ids

        # ── Category 1: Long Context Recall ──
        # Probes planted in s01-s02, queried from the end
        python_pref = _find_turns_with("python", "backend", "primary language")
        editor_pref = _find_turns_with("neovim", "development environment")
        self.queries.append(TestQuery(
            query="What programming language does the user prefer for backend development?",
            category="long_context",
            ground_truth_ids=python_pref,
            required_keywords=["python", "backend"],
            description="Recall language preference from session 1",
        ))
        self.queries.append(TestQuery(
            query="What editor and development environment does the user use?",
            category="long_context",
            ground_truth_ids=editor_pref,
            required_keywords=["neovim", "lsp"],
            description="Recall editor preference from early session",
        ))

        # ── Category 2: Cross-session Recall ──
        campus_turns = _find_turns_with("campusnet", "autologin")
        self.queries.append(TestQuery(
            query="Tell me about the CampusNet-AutoLogin project — what is it and what tech does it use?",
            category="cross_session",
            ground_truth_ids=campus_turns[:2],
            required_keywords=["campus", "login", "selenium"],
            description="Cross-session recall of multi-session project",
        ))
        self.queries.append(TestQuery(
            query="How was the CampusNet-AutoLogin project deployed and packaged?",
            category="cross_session",
            ground_truth_ids=_find_turns_with("pyinstaller", "executable"),
            required_keywords=["pyinstaller", "executable", "startup"],
            description="Recall deployment details from later session",
        ))

        # ── Category 3: Semantic Compression ──
        redis_fix = _find_turns_with("redis", "max_connections", "circuit breaker")
        docker_fix = _find_turns_with("docker", "requirements.txt", "45 seconds")
        self.queries.append(TestQuery(
            query="How was the Redis connection pool timeout diagnosed and fixed?",
            category="compression",
            ground_truth_ids=redis_fix,
            required_keywords=["redis", "connection"],
            description="Recall multi-step bug fix after consolidation",
        ))
        self.queries.append(TestQuery(
            query="What was the Docker build cache problem and how was it resolved?",
            category="compression",
            ground_truth_ids=docker_fix,
            required_keywords=["docker", "cache", "requirements"],
            description="Recall Docker optimization after consolidation",
        ))

        # ── Category 4: Forgetting Resistance ──
        backup = _find_turns_with("backup", "wal", "s3")
        security = _find_turns_with("vault", "jwt", "zap")
        self.queries.append(TestQuery(
            query="What is the production database backup strategy?",
            category="forgetting",
            ground_truth_ids=backup,
            required_keywords=["backup", "database", "recovery"],
            description="Important backup policy should survive pruning",
        ))
        self.queries.append(TestQuery(
            query="What security policies and practices does the team follow?",
            category="forgetting",
            ground_truth_ids=security,
            required_keywords=["security", "authentication", "vault"],
            description="Critical security policy should survive pruning",
        ))

        # ── Category 5: Memory Interference ──
        microservice_turns = _find_turns_with("microservices", "changed my mind")
        self.queries.append(TestQuery(
            query="Does the user prefer monolithic architecture or microservices now?",
            category="interference",
            ground_truth_ids=microservice_turns,
            required_keywords=["microservices"],
            forbidden_keywords=["monolithic", "overengineered"],
            description="Architecture preference: new (microservices) should beat old (monolith)",
        ))
        raw_sql_turns = _find_turns_with("raw sql", "reversed my stance")
        self.queries.append(TestQuery(
            query="Does the user prefer using ORMs or raw SQL for database queries now?",
            category="interference",
            ground_truth_ids=raw_sql_turns,
            required_keywords=["raw sql", "n+1"],
            forbidden_keywords=["sqlalchemy", "cleaner"],
            description="Database approach: new (raw SQL) should beat old (ORM)",
        ))


# ═══════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════

class RawHistoryBaseline:
    """Full context — upper bound on recall, no compression."""

    def __init__(self, turns: list[ConversationTurn]):
        self.turns = turns
        self.texts = [t.text for t in turns]

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # Simple keyword overlap as retrieval
        query_words = set(query.lower().split())
        scored = []
        for t in self.turns:
            text_words = set(t.text.lower().split())
            overlap = len(query_words & text_words) / max(1, len(query_words))
            scored.append((overlap, t.id))
        scored.sort(key=lambda x: -x[0])
        return [tid for _, tid in scored[:top_k]]

    @property
    def token_count(self) -> int:
        return sum(len(t.split()) for t in self.texts)


class TFIDFBaseline:
    """TF-IDF vector search — no graph structure, no temporal awareness."""

    def __init__(self, turns: list[ConversationTurn]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.turns = turns
        self.texts = [t.text for t in turns]
        self.vec = TfidfVectorizer(max_features=384)
        self.matrix = self.vec.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        q_vec = self.vec.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().flatten()
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [self.turns[i].id for i, _ in ranked[:top_k]]

    @property
    def token_count(self) -> int:
        return sum(len(t.split()) for t in self.texts)



# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def recall_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    if not ground_truth_ids:
        return 1.0
    gt_set = set(ground_truth_ids)
    found = len(set(retrieved_ids[:k]) & gt_set)
    return found / len(gt_set)


def content_recall_at_k(retrieved_texts: list[str], required_keywords: list[str],
                         k: int) -> float:
    """Check if top-k retrieved texts contain required keywords (content-based recall)."""
    if not required_keywords:
        return 1.0
    if not retrieved_texts:
        return 0.0
    texts = retrieved_texts[:k]
    found = 0
    for kw in required_keywords:
        if any(kw.lower() in t.lower() for t in texts):
            found += 1
    return found / len(required_keywords)


def content_interference_score(retrieved_texts: list[str], required_keywords: list[str],
                                forbidden_keywords: list[str]) -> float:
    """Score interference resolution: required keywords present, forbidden absent."""
    if not required_keywords and not forbidden_keywords:
        return 1.0
    if not retrieved_texts:
        return 0.0

    score = 0.0
    total = 0.0
    texts_combined = " ".join(retrieved_texts[:5]).lower()

    for kw in required_keywords:
        total += 1.0
        if kw.lower() in texts_combined:
            score += 1.0

    for kw in forbidden_keywords:
        total += 1.0
        if kw.lower() not in texts_combined:
            score += 1.0

    return score / max(1.0, total)


def precision_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    if not retrieved_ids[:k]:
        return 0.0
    gt_set = set(ground_truth_ids)
    return len(set(retrieved_ids[:k]) & gt_set) / k


def mrr(retrieved_lists: list[list[str]], ground_truth_lists: list[list[str]]) -> float:
    total = 0.0
    for retrieved, gt in zip(retrieved_lists, ground_truth_lists):
        if not gt:
            continue
        gt_set = set(gt)
        for rank, rid in enumerate(retrieved, 1):
            if rid in gt_set:
                total += 1.0 / rank
                break
    return total / max(1, len(ground_truth_lists))


def temporal_recall(retrieved_ids: list[str], ground_truth: list[ConversationTurn],
                    turns: list[ConversationTurn]) -> float:
    """Weighted recall: finding recent info matters more than old."""
    if not ground_truth:
        return 1.0
    id_to_turn = {t.id: t for t in turns}
    max_ts = max(t.timestamp for t in turns) if turns else 1

    total_weight = 0.0
    found_weight = 0.0
    for gt in ground_truth:
        recency = gt.timestamp / max_ts  # normalized recency
        weight = 1.0 + recency  # recent items weighted more
        total_weight += weight
        if gt.id in retrieved_ids:
            found_weight += weight

    return found_weight / max(1.0, total_weight)


def conflict_resolution_accuracy(graph: StarGraph, conflict_queries: list[TestQuery]) -> float:
    """Check that new beliefs have higher activation than old conflicting ones."""
    if not conflict_queries:
        return 1.0

    correct = 0
    total = 0
    for q in conflict_queries:
        # Check if retrieval results favor new information
        ret = OscillationResonanceRetriever(graph)
        result = ret.retrieve(q.query)
        # For interference queries, check if the most recent relevant anchor
        # has higher activation than older ones on same topic
        anchors = []
        for c in result.constellations:
            for a in c.anchors:
                anchors.append(a)

        # Simple heuristic: if the top result is from a newer session, it resolved correctly
        if anchors:
            newest_relevant = max(
                (a for a in anchors if any(tag in str(a.tags).lower()
                 for tag in ["architecture", "editor", "sql", "orm", "microservice"])),
                key=lambda a: a.created_at, default=None
            )
            if newest_relevant and newest_relevant.created_at > (anchors[0].created_at if anchors else 0):
                correct += 1
        total += 1

    return correct / max(1, total)


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def banner(text: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def _estimate_graph_tokens(graph: StarGraph) -> int:
    """Estimate token count for the compressed graph representation."""
    total = 0
    for a in graph.anchors.values():
        total += len(a.text.split())
    for s in graph.schemas.values():
        total += len(s.template.split())
    return total


def _print_summary(results: dict) -> None:
    banner("Benchmark Results Summary")

    sleep = results["sleep_stats"]
    print(f"\n  Sleep Consolidation:")
    print(f"    Anchors:  {sleep['anchors_before']} → {sleep['anchors_after']} "
          f"({sleep['merged']} merged, {sleep['pruned']} pruned)")
    print(f"    Ghosts:   {sleep['ghosts']}")
    print(f"    Schemas:  {sleep['schemas']}")
    print(f"    Compress: {sleep['compression_ratio']:.2f}x")

    agg = results["aggregate"]
    hyb = agg.get("hybrid_fusion", {})
    print(f"\n  ┌{'─'*72}┐")
    print(f"  │ {'Method':<22} {'Tokens':>8} {'C-R@3':>7} {'C-R@5':>7} {'Interf':>7} {'Note':>10} │")
    print(f"  ├{'─'*72}┤")
    print(f"  │ {'Raw History':<22} {agg['raw_history']['tokens']:>8,} "
          f"{'N/A':>7} {'N/A':>7} {'N/A':>7} {'upper bnd':>10} │")
    print(f"  │ {'TF-IDF Vector':<22} {agg['tfidf']['tokens']:>8,} "
          f"{'N/A':>7} {'N/A':>7} {'N/A':>7} {'keyword':>10} │")
    print(f"  │ {'SG VectorSimilarity':<22} {agg['star_graph']['tokens']:>8,} "
          f"{agg['vector_only'].get('content_recall@3', 0):>7.3f} "
          f"{agg['vector_only'].get('content_recall@5', 0):>7.3f} "
          f"{'N/A':>7} {'semantic':>10} │")
    sg = agg['star_graph']
    print(f"  │ {'SG OscillationRes':<22} {sg['tokens']:>8,} "
          f"{sg['content_recall@3']:>7.3f} {sg['content_recall@5']:>7.3f} "
          f"{sg.get('interference_score', 0):>7.3f} {'phase+graph':>10} │")
    print(f"  │ {'SG HybridFusion':<22} {sg['tokens']:>8,} "
          f"{hyb.get('content_recall@3', 0):>7.3f} {hyb.get('content_recall@5', 0):>7.3f} "
          f"{hyb.get('interference_score', 0):>7.3f} {'multi-signal':>10} │")
    print(f"  └{'─'*72}┘")

    print(f"\n  Per-Category Content Recall@3:")
    print(f"  {'Category':<25} {'OscRes':>7} {'VecSim':>7} {'HybFus':>7} {'Best':>7} {'Interf':>7}")
    print(f"  {'─'*60}")
    for cat_name, cat_data in results.get("categories", {}).items():
        sg = cat_data.get("star_graph", {})
        vec = cat_data.get("vector_only", {})
        hyb_cat = cat_data.get("hybrid_fusion", {})
        osc_cr = sg.get('content_recall@3', 0)
        vec_cr = vec.get('content_recall@3', 0)
        hyb_cr = hyb_cat.get('content_recall@3', 0)
        best = max(osc_cr, vec_cr, hyb_cr)
        interf = hyb_cat.get("interference_score", sg.get("interference_score", 0))
        interf_str = f"{interf:.3f}" if interf > 0 else "N/A"
        print(f"  {cat_name:<25} {osc_cr:>7.3f} {vec_cr:>7.3f} {hyb_cr:>7.3f} {best:>7.3f} {interf_str:>7}")

    # Compression efficiency
    raw_tokens = agg["raw_history"]["tokens"]
    sg_tokens = agg["star_graph"]["tokens"]
    compression = raw_tokens / max(1, sg_tokens)
    print(f"\n  Compression: {compression:.1f}x ({raw_tokens:,} → {sg_tokens:,} tokens)")
