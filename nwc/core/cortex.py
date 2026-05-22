"""Cortex — the main cognitive runtime. Single entry point for all NWC operations.

Usage:
    from nwc import Cortex
    ctx = Cortex()
    ctx.remember("用户喜欢科幻小说")
    results = ctx.recall("用户喜欢什么")
    context = ctx.context("推荐一本书")
"""

from dataclasses import dataclass, field
from typing import Optional

from nwc.config.config import NwcConfig, get_config


@dataclass
class RecallResult:
    memory: list[dict] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    summary: str = ""


@dataclass
class ContextFrame:
    """Compressed cognitive context for LLM injection."""
    focus: str = ""
    active_goals: list[str] = field(default_factory=list)
    relevant_memories: list[dict] = field(default_factory=list)
    active_concepts: list[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    summary: str = ""

    def to_system_prompt(self) -> str:
        lines = ["# Cognitive State Summary"]
        if self.focus:
            lines.append(f"## Current Focus\n{self.focus}")
        if self.active_goals:
            lines.append("## Active Goals\n" + "\n".join(f"- {g}" for g in self.active_goals))
        if self.active_concepts:
            lines.append("## Active Concepts\n" + ", ".join(self.active_concepts))
        if self.relevant_memories:
            lines.append("## Relevant Memories")
            for m in self.relevant_memories[:5]:
                lines.append(f"- {m.get('content', m.get('text', str(m)))}")
        lines.append(f"\n## Emotional Tone\n{self.emotional_tone}")
        if self.summary:
            lines.append(f"\n## Summary\n{self.summary}")
        return "\n".join(lines)


class Cortex:
    """Cognitive runtime — wraps the full NWC stack behind a simple facade.

    Key methods:
        remember(text, tags, importance) — store a memory
        recall(query, max_items) — semantic retrieval
        context(prompt) — get compressed cognitive context for LLM injection
        reflect() — generate long-term cognitive summary
        evolve() — run memory consolidation cycle
        forget(anchor_id) — remove a memory with ghost trace
        stats() — memory system statistics
        save(path) / load(path) — persist to disk
    """

    def __init__(self, config: Optional[NwcConfig] = None):
        self.config = config or get_config()
        self._manager = None
        self._perception = None
        self._workspace = None
        self._concept_cortex = None
        self._activation_engine = None
        self._goal_system = None
        self._salience = None
        self._self_model = None
        # Phase 7
        self._importance_engine = None
        self._belief_system = None
        self._personality_engine = None
        self._identity_manager = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from star_graph import MemoryManager
        storage_path = self.config.storage.path.replace("~", str(__import__("pathlib").Path.home()))
        self._manager = MemoryManager(storage_path=storage_path)
        try:
            self._manager.load()
        except Exception:
            pass
        self._loaded = True

    def _ensure_cortex(self):
        self._ensure_loaded()
        if self._perception is None:
            from star_graph import (
                PerceptionLayer, CognitiveWorkspace, ConceptCortex as CC,
                ActivationEngine, GoalSystem, SalienceEngine, SelfModel,
            )
            graph = getattr(self._manager, 'graph', None)
            self._perception = PerceptionLayer()
            self._workspace = CognitiveWorkspace(max_items=self.config.memory.working_capacity)
            self._concept_cortex = CC()
            self._activation_engine = ActivationEngine(graph=graph)
            self._goal_system = GoalSystem()
            self._salience = SalienceEngine()
            self._self_model = SelfModel(
                workspace=self._workspace, goal_system=self._goal_system,
                concept_cortex=self._concept_cortex, salience_engine=self._salience,
            )

    # ── Core API ────────────────────────────────────────────

    def remember(self, text: str, tags: list[str] | None = None,
                 importance: float = 0.5, emotional_valence: float = 0.0) -> str:
        """Store a memory. Returns anchor ID."""
        self._ensure_loaded()
        anchor = self._manager.remember(text, tags=tags or [])
        return getattr(anchor, 'id', '') if anchor else ''

    def remember_working(self, text: str, tags: list[str] | None = None) -> dict:
        """Store in working memory buffer (short-term, high-activation).

        Returns a dict with pseudo-id, text, tags, importance, created_at.
        WorkingMemoryEntry has no .id (it's ephemeral, not persisted).
        """
        self._ensure_loaded()
        entry = self._manager.remember_working(text, tags=tags or [])
        import hashlib
        pseudo_id = f"wm_{hashlib.md5(entry.text.encode()).hexdigest()[:8]}"
        return {"id": pseudo_id, "text": entry.text[:200], "tags": entry.tags,
                "importance": entry.importance, "created_at": entry.created_at}

    def recall(self, query: str, max_items: int | None = None,
               context: dict | None = None) -> RecallResult:
        """Context-aware semantic retrieval. Returns structured result."""
        self._ensure_loaded()
        if max_items is None:
            max_items = self.config.retrieval.top_k
        result = self._manager.recall(query, max_items=max_items)
        memories = []
        entities = set()
        relations = []
        if hasattr(result, 'memories'):
            for m in result.memories:
                if getattr(m, 'anchor', None) is None:
                    continue
                item = {"id": getattr(m, 'id', '') or getattr(m.anchor, 'id', ''),
                        "content": getattr(m, 'content', '') or getattr(m.anchor, 'text', ''),
                        "score": getattr(m, 'score', 0.0), "tags": getattr(m, 'tags', []) or getattr(m.anchor, 'tags', [])}
                memories.append(item)
                if hasattr(m.anchor, 'tags'):
                    for t in m.anchor.tags:
                        entities.add(t)
        elif isinstance(result, list):
            for m in result:
                if getattr(m, 'content', None) is None and getattr(m, 'text', None) is None:
                    continue
                item = {"id": getattr(m, 'id', '') or getattr(getattr(m, 'anchor', None), 'id', ''),
                        "content": getattr(m, 'content', '') or getattr(m, 'text', ''),
                        "score": getattr(m, 'score', 0.0)}
                memories.append(item)
        return RecallResult(
            memory=memories,
            entities=list(entities),
            relations=relations,
            summary=f"Retrieved {len(memories)} memories for: {query}",
        )

    def context(self, prompt: str = "") -> ContextFrame:
        """Get compressed cognitive context for LLM injection. Uses SelfModel if Phase 6 is loaded."""
        self._ensure_loaded()
        try:
            self._ensure_cortex()
            frame = self._perception.perceive(prompt) if prompt else None
            if frame:
                self._workspace.on_perception(frame)
            state = self._self_model.construct_state()
            return ContextFrame(
                focus=state.current_focus if hasattr(state, 'current_focus') else prompt,
                active_goals=state.active_goals if hasattr(state, 'active_goals') else [],
                active_concepts=state.dominant_concepts if hasattr(state, 'dominant_concepts') else [],
                emotional_tone=state.emotional_state if hasattr(state, 'emotional_state') else "neutral",
                summary=state.summary if hasattr(state, 'summary') else "",
            )
        except Exception:
            # Fall back to basic retrieval if Phase 6 not available
            result = self.recall(prompt, max_items=5) if prompt else self.recall("recent", max_items=5)
            return ContextFrame(
                focus=prompt,
                relevant_memories=result.memory,
                summary=result.summary,
            )

    def reflect(self) -> str:
        """Generate long-term cognitive summary via sleep consolidation."""
        self._ensure_loaded()
        report = self._manager.sleep()
        return str(report) if report else "Consolidation complete."

    def evolve(self) -> str:
        """Run memory evolution cycle (decay, boost, conflict resolution)."""
        self._ensure_loaded()
        report = self._manager.evolve()
        return str(report) if report else "Evolution complete."

    def forget(self, anchor_id: str) -> bool:
        """Remove a memory, creating a ghost trace for potential revival."""
        self._ensure_loaded()
        return self._manager.forget(anchor_id)

    def fuzzy_recall(self, query: str, threshold: float = 0.2) -> RecallResult:
        """Low-confidence recall from ghost traces ('I seem to remember...')."""
        self._ensure_loaded()
        result = self._manager.fuzzy_recall(query, threshold=threshold)
        return RecallResult(
            memory=result if isinstance(result, list) else [],
            summary=f"Fuzzy recall for: {query}",
        )

    def get_profile(self) -> dict:
        """Inferred user profile from accumulated memories."""
        self._ensure_loaded()
        return self._manager.get_profile() if hasattr(self._manager, 'get_profile') else {}

    # ── Phase 7: Internal initializers ──────────────────────

    def _ensure_importance(self):
        if self._importance_engine is None:
            self._ensure_loaded()
            from star_graph.importance_engine import ImportanceEngine
            self._importance_engine = ImportanceEngine(goal_system=self._goal_system)

    def _ensure_beliefs(self):
        if self._belief_system is None:
            from star_graph.belief_system import BeliefSystem
            self._belief_system = BeliefSystem()

    def _ensure_personality(self):
        if self._personality_engine is None:
            self._ensure_beliefs()
            from star_graph.personality_formation import PersonalityFormationEngine
            self._personality_engine = PersonalityFormationEngine(belief_system=self._belief_system)

    def _ensure_identity(self):
        if self._identity_manager is None:
            from star_graph.cognitive_identity import CognitiveIdentityManager
            storage = self.config.storage.path.replace("~", str(__import__("pathlib").Path.home()))
            self._identity_manager = CognitiveIdentityManager(storage_dir=f"{storage}/identity")
            # Auto-load existing identity
            self._identity_manager.record_interaction()  # bootstrap if new

    # ── Phase 7: Importance Engine ──────────────────────────

    def evaluate_importance(self, text: str, *,
                            tags: list[str] | None = None,
                            concepts: list[str] | None = None,
                            emotion_valence: float = 0.0,
                            active_goals: list[str] | None = None) -> dict:
        """Score the importance of incoming text. Returns full importance analysis.

        Returns dict with keys: score, level, should_store, decay_rate, signal breakdown
        """
        self._ensure_importance()
        result = self._importance_engine.evaluate(
            text, emotion_valence=emotion_valence,
            active_goals=active_goals or [],
            extracted_concepts=concepts or [],
            topic_tags=tags or [],
        )
        return {
            "score": result.score,
            "level": result.level,
            "should_store": result.should_store,
            "decay_rate": result.decay_rate,
            "consolidation_priority": result.consolidation_priority,
            "signal": {
                "emotion": result.signal.emotion_weight,
                "repetition": result.signal.repetition_weight,
                "goal_relation": result.signal.goal_relation,
                "novelty": result.signal.novelty,
                "future_probability": result.signal.future_probability,
            },
        }

    # ── Phase 7: Belief System ──────────────────────────────

    def beliefs(self, category: str | None = None, min_strength: float = 0.5) -> list[dict]:
        """List beliefs about the user. Filter by category and minimum strength.

        Categories: preference, value, worldview, identity, capability
        """
        self._ensure_beliefs()
        beliefs = self._belief_system.list_all(category=category, min_strength=min_strength)
        return [{"id": b.id, "statement": b.statement, "category": b.category,
                 "strength": b.strength, "stability": b.stability} for b in beliefs]

    def add_belief(self, statement: str, category: str = "general",
                   evidence_ids: list[str] | None = None) -> str:
        """Form a new belief from evidence. Returns belief ID."""
        self._ensure_beliefs()
        belief = self._belief_system.form_belief(
            statement, category=category, evidence_ids=evidence_ids or [])
        return belief.id

    def reinforce_belief(self, belief_id: str, evidence_id: str = "") -> bool:
        self._ensure_beliefs()
        return self._belief_system.reinforce(belief_id, evidence_id)

    def challenge_belief(self, belief_id: str, contradiction_id: str = "") -> bool:
        self._ensure_beliefs()
        return self._belief_system.challenge(belief_id, contradiction_id)

    # ── Phase 7: Personality Formation ──────────────────────

    def profile(self) -> dict:
        """Get the full user cognitive profile — identity, values, patterns, evolution.

        This is the top of the 6-layer pyramid. NOT raw memories — compressed cognition.
        """
        self._ensure_personality()
        self._personality_engine.sync_from_beliefs()
        return self._personality_engine.to_dict()

    def profile_injection(self) -> str:
        """Get compressed cognitive profile as LLM prompt injection."""
        self._ensure_personality()
        self._personality_engine.sync_from_beliefs()
        return self._personality_engine.get_injection()

    def observe_pattern(self, pattern: str):
        """Record a behavioral pattern observation."""
        self._ensure_personality()
        self._personality_engine.ingest_memory({
            "content": pattern, "tags": [pattern], "concepts": [],
            "importance": 0.5,
        })

    def observe_topic(self, topic: str, emotion_valence: float = 0.0):
        """Track topic frequency for emerging/fading interest detection."""
        self._ensure_personality()
        self._personality_engine.ingest_memory({
            "content": topic, "tags": [topic], "concepts": [topic],
            "importance": 0.5,
        })

    def extract_beliefs_from_memories(self, min_occurrences: int = 3) -> list[dict]:
        """Auto-extract belief candidates by scanning stored memory patterns.

        Returns list of candidate beliefs ready for review/confirmation.
        """
        self._ensure_loaded()
        self._ensure_beliefs()
        # Get recent memories in a simple format
        memories = []
        try:
            result = self._manager.recall("", max_items=200)
            if hasattr(result, 'memories'):
                for m in result.memories:
                    memories.append({
                        "content": getattr(m, 'content', str(m)),
                        "tags": getattr(m, 'tags', []),
                        "id": getattr(m, 'id', ''),
                    })
        except Exception:
            pass
        candidates = self._belief_system.extract_patterns_from_memories(
            memories, min_occurrences=min_occurrences)
        # Auto-form beliefs from high-confidence candidates
        formed = []
        for c in candidates:
            if c["confidence"] >= 0.4:
                belief = self._belief_system.form_belief(
                    c["statement"], category=c["category"],
                    evidence_ids=c["evidence_ids"])
                formed.append({"id": belief.id, "statement": belief.statement,
                               "category": belief.category, "confidence": c["confidence"]})
        return formed

    # ── Phase 7: Cognitive Identity ─────────────────────────

    def identity(self) -> dict:
        """Get persistent cognitive identity — cross-session user understanding.

        This survives across conversations, restarts, and deployments.
        """
        self._ensure_identity()
        return self._identity_manager.get_stats()

    def identity_injection(self) -> str:
        """Get identity for LLM prompt injection."""
        self._ensure_identity()
        return self._identity_manager.get_injection()

    def identity_evolution(self) -> str:
        """Human-readable evolution summary of the AI-user relationship."""
        self._ensure_identity()
        return self._identity_manager.get_evolution_summary()

    def identity_snapshot(self):
        """Take a snapshot of current identity state."""
        self._ensure_identity()
        self._ensure_personality()
        profile = self._personality_engine.to_dict()
        self._identity_manager.update_profile(profile)
        self._identity_manager.snapshot()
        self._identity_manager.save()

    # ── Full cognitive pipeline ─────────────────────────────

    def perceive(self, text: str, *,
                 tags: list[str] | None = None,
                 intent: str = "",
                 emotion_valence: float = 0.0) -> dict:
        """Full 6-layer cognitive pipeline.

        ① Sensory:    PerceptionLayer → structured frame
        ② Evaluate:   ImportanceEngine → score, gate decision
        ③ Episodic:   store if worth it (MemoryManager)
        ④ Semantic:   ConceptCortex activation
        ⑤ Pattern:    PersonalityFormation ingestion
        ⑥ Identity:   CognitiveIdentity update

        Returns full pipeline trace with each layer's output.
        """
        self._ensure_loaded()
        self._ensure_importance()
        self._ensure_beliefs()
        self._ensure_personality()
        self._ensure_identity()
        self._ensure_cortex()

        trace = {"input": text[:200], "layers": {}}

        # ── Layer ①: Sensory / Perception ──────────────────
        perception_frame = None
        concepts = []
        entities = []
        try:
            perception_frame = self._perception.perceive(text)
            concepts = getattr(perception_frame, 'extracted_concepts', [])
            entities = getattr(perception_frame, 'extracted_entities', [])
            intent = intent or getattr(perception_frame, 'intent', 'statement')
            emotion_valence = emotion_valence or getattr(perception_frame, 'valence', 0.0)
            trace["layers"]["sensory"] = {
                "intent": intent, "valence": emotion_valence,
                "concepts": concepts[:10], "entities": entities[:10],
            }
        except Exception:
            trace["layers"]["sensory"] = {"status": "fallback"}

        all_tags = (tags or []) + concepts[:5]

        # ── Layer ②: Evaluate / Importance Gate ────────────
        imp_result = self._importance_engine.evaluate(
            text, perception_frame=perception_frame,
            emotion_valence=emotion_valence,
            active_goals=self._get_active_goals(),
            extracted_concepts=concepts,
            extracted_entities=entities,
            topic_tags=all_tags,
            intent=intent,
        )
        trace["layers"]["evaluate"] = {
            "score": round(imp_result.score, 3),
            "level": imp_result.level,
            "should_store": imp_result.should_store,
            "decay_rate": imp_result.decay_rate,
            "richness": {
                "word_count": imp_result.richness.word_count,
                "entity_count": imp_result.richness.entity_count,
                "specificity": round(imp_result.richness.specificity_score, 2),
                "actionability": round(imp_result.richness.actionability_score, 2),
                "information_density": round(imp_result.richness.information_density, 2),
            },
        }

        # ── Layer ③: Episodic / Store ──────────────────────
        anchor_id = None
        if imp_result.should_store:
            anchor_id = self.remember(text, tags=all_tags)
            trace["layers"]["episodic"] = {"stored": True, "anchor_id": anchor_id}
        else:
            trace["layers"]["episodic"] = {"stored": False, "reason": "below_importance_threshold"}

        # ── Layer ④: Semantic / Concepts ──────────────────
        activated = []
        try:
            for concept in concepts[:8]:
                cc_result = self._concept_cortex.get_or_create_concept(concept)
                activated.append(concept)
            trace["layers"]["semantic"] = {"activated_concepts": activated}
        except Exception:
            trace["layers"]["semantic"] = {"activated_concepts": []}

        # ── Layer ⑤: Pattern / Personality ────────────────
        try:
            self._personality_engine.ingest_memory({
                "content": text, "tags": all_tags, "concepts": concepts,
                "importance": imp_result.score,
                "id": anchor_id or "",
            })
            self._personality_engine.analyze_interaction(
                text, intent=intent, concepts=concepts, emotion_valence=emotion_valence)
            self._personality_engine.infer_values_from_text(text)
            trace["layers"]["pattern"] = {"patterns_tracked": True}
        except Exception:
            trace["layers"]["pattern"] = {"status": "skipped"}

        # ── Layer ⑥: Identity / Persist ────────────────────
        self._identity_manager.record_interaction()
        interactions = self._identity_manager.get_identity().total_interactions
        trace["layers"]["identity"] = {
            "total_interactions": interactions,
            "days_known": round(self._identity_manager.get_identity().days_since_first(), 1),
        }
        if interactions % 50 == 0:
            try:
                self.identity_snapshot()
                trace["layers"]["identity"]["snapshot_taken"] = True
            except Exception:
                pass

        # ── Aggregate ───────────────────────────────────────
        return {
            "anchor_id": anchor_id,
            "importance": imp_result.score,
            "level": imp_result.level,
            "stored": imp_result.should_store,
            "concepts_activated": activated,
            "interaction_count": interactions,
            "trace": trace,
        }

    def _get_active_goals(self) -> list[str]:
        """Get active goal descriptions from the goal system."""
        try:
            self._ensure_cortex()
            goals = self._goal_system.summarize_goals()
            if isinstance(goals, list):
                return [g.get("description", str(g)) if isinstance(g, dict) else str(g) for g in goals[:5]]
        except Exception:
            pass
        return []

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str | None = None):
        """Persist memory graph to disk."""
        self._ensure_loaded()
        target = path or self.config.storage.path.replace("~", str(__import__("pathlib").Path.home()))
        self._manager.save(target)

    def load(self, path: str | None = None):
        """Load memory graph from disk."""
        target = path or self.config.storage.path.replace("~", str(__import__("pathlib").Path.home()))
        self._manager.load(target)
        self._loaded = True

    def stats(self) -> dict:
        """Memory system statistics."""
        self._ensure_loaded()
        stats = self._manager.stats if hasattr(self._manager, 'stats') else None
        if stats is None:
            return {}
        if hasattr(stats, 'to_dict'):
            d = stats.to_dict()
        else:
            d = dict(stats) if isinstance(stats, dict) else {"error": "stats unavailable"}
        # Remove cognitive_health to avoid slow snapshot on first access
        d.pop("cognitive_health", None)
        return d

    # ── Async API ───────────────────────────────────────────

    async def aremember(self, text: str, tags: list[str] | None = None) -> str:
        return self.remember(text, tags)

    async def arecall(self, query: str, max_items: int | None = None) -> RecallResult:
        return self.recall(query, max_items)

    async def acontext(self, prompt: str = "") -> ContextFrame:
        return self.context(prompt)
