"""RuntimeCognitive — mixin for reflection, self-model, and compression APIs.

Extracted verbatim from ``runtime.py`` ``MemoryRuntime`` (issue #9). Bodies are
unchanged; methods still call the lazy accessors from :class:`RuntimeSubsystems`
(``autobiography``, ``compressor``, ``compiler``, ``reflection_loop``, ...).
"""

from __future__ import annotations


class RuntimeCognitive:
    """Mixin: reflections, autobiographical memory, compression, worldviews."""

    # ── Reflection (meta-cognitive insights) ───────────────

    def reflect(self, text: str, source_anchor_ids: list[str],
                reflection_type: str = "lesson_learned",
                confidence: float = 0.6) -> str:
        """Create a meta-cognitive reflection on past memories."""
        from ..graph import ReflectionNode
        if reflection_type == "failure_analysis":
            r = ReflectionNode.from_failure(text, source_anchor_ids, confidence)
        elif reflection_type == "success_pattern":
            r = ReflectionNode.from_success(text, source_anchor_ids, confidence)
        elif reflection_type == "root_cause":
            import hashlib
            rid = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
            r = ReflectionNode(id=rid, text=text, reflection_type="root_cause",
                              source_anchor_ids=source_anchor_ids,
                              confidence=confidence, strength=0.5)
        else:
            r = ReflectionNode.from_lesson(text, source_anchor_ids, confidence)
        return self.graph.add_reflection(r)

    def get_reflections(self, anchor_ids: list[str],
                        types: list[str] | None = None) -> list[dict]:
        nodes = self.graph.find_reflections(anchor_ids, types)
        return [
            {"id": r.id, "text": r.text, "type": r.reflection_type,
             "confidence": r.confidence, "strength": r.strength}
            for r in nodes
        ]

    def strengthen_reflection(self, reflection_id: str) -> bool:
        if reflection_id in self.graph.reflections:
            self.graph.reflections[reflection_id].reinforce()
            return True
        return False

    def weaken_reflection(self, reflection_id: str) -> bool:
        if reflection_id in self.graph.reflections:
            self.graph.reflections[reflection_id].weaken()
            return True
        return False

    # ── Autobiographical memory (self-model) ───────────────

    def remember_self(self, episode_summary: str = "",
                      self_belief: str = "",
                      emotional_tone: float = 0.0,
                      source_session: str = "",
                      source_anchor_ids: list[str] | None = None,
                      tags: list[str] | None = None):
        """Record a first-person narrative — the agent's own experience.

        This is NOT a fact about the user/world (use remember() for that).
        This is "I discussed X," "I believe Y about the user," "my tone was Z."
        """
        return self.autobiography.form_from_interaction(
            episode_summary=episode_summary,
            self_belief=self_belief,
            emotional_tone=emotional_tone,
            source_session=source_session,
            source_anchor_ids=source_anchor_ids,
            tags=tags,
        )

    def recall_self(self, query: str = "", *,
                    top_k: int = 10,
                    min_stability: float = 0.05):
        """Recall what the agent knows about itself — 'what do I know about myself?' """
        return self.autobiography.recall_self(
            query=query, top_k=top_k, min_stability=min_stability,
        )

    def get_self_beliefs(self, min_stability: float = 0.2) -> list[dict]:
        """Get the agent's stable self-beliefs — its current self-model."""
        return self.autobiography.get_beliefs(min_stability=min_stability)

    def update_self_belief(self, belief_substring: str, correction: str = ""):
        """Update or correct a self-belief when the agent's understanding changes."""
        return self.autobiography.contradict_belief(belief_substring, correction)

    def self_emotional_profile(self, session_id: str = "") -> dict:
        """Get the agent's emotional profile across interactions."""
        return self.autobiography.get_emotional_profile(session_id)

    # ── Compression API ──────────────────────────────────────────

    def compress_session(self, session_id: str) -> list:
        from ..compression import SessionCompressor
        anchors_list = list(self.graph.anchors.values())
        compressor = SessionCompressor()
        summaries = compressor.compress(anchors_list, session_id)
        if summaries:
            _mcomp = self.compressor
            _mcomp.add_to_graph(self.graph, summaries, edge_type="compresses")
        return summaries

    def compress_all(self) -> dict:
        from collections import defaultdict
        from ..compression import CompressionLevel

        anchors_by_session: dict[str, list] = defaultdict(list)
        for anchor in self.graph.anchors.values():
            if anchor.embedding and anchor.source_session:
                anchors_by_session[anchor.source_session].append(anchor)

        if not anchors_by_session:
            return {
                "levels": {"episodic": 0, "strategic": 0, "meta": 0},
                "total_summaries": 0,
                "total_edges": 0,
            }

        compressor = self.compressor
        results = compressor.compress_pipeline(anchors_by_session)

        total_edges = 0
        level_counts: dict[str, int] = {}
        for level, summaries in results.items():
            level_name = level.name.lower()
            level_counts[level_name] = len(summaries)
            if summaries:
                total_edges += compressor.add_to_graph(self.graph, summaries, edge_type="compresses")

        return {
            "levels": level_counts,
            "total_summaries": sum(level_counts.values()),
            "total_edges": total_edges,
        }

    def get_compressed_view(self,
                            min_level: int = 1,
                            max_summaries: int = 50) -> list[dict]:
        from ..compression import CompressionLevel
        level_tags = {
            CompressionLevel.EPISODIC: "level:episodic",
            CompressionLevel.STRATEGIC: "level:strategic",
            CompressionLevel.META: "level:meta",
        }

        summaries: list[dict] = []
        for aid, anchor in self.graph.anchors.items():
            level_tag = None
            level_val = 0
            for lvl, tag in level_tags.items():
                if tag in anchor.tags:
                    level_tag = tag
                    level_val = lvl.value
                    break

            if level_val < min_level:
                continue

            source_count = 0
            for neighbor in self.graph._adjacency.get(aid, set()):
                edge_key = self.graph._key(aid, neighbor)
                edge = self.graph.edges.get(edge_key)
                if edge and edge.edge_type == "compresses":
                    source_count += 1

            summaries.append({
                "id": aid,
                "text": anchor.text,
                "level": level_val,
                "level_name": CompressionLevel(level_val).name.lower() if level_val > 0 else "raw",
                "confidence": anchor.vector.confidence,
                "stability": anchor.vector.stability,
                "source_count": source_count,
                "tags": [t for t in anchor.tags if not t.startswith("level:")],
            })

        summaries.sort(key=lambda x: -x["confidence"])
        return summaries[:max_summaries]

    # ── Cognitive Compiler API (#45) ─────────────────────────

    def compile_worldviews(self) -> dict:
        """Run the full cognitive compilation pipeline and return results."""
        return self.compiler.compile(self.graph)

    def get_worldviews(self, min_stability: float = 0.5) -> list:
        """Get stable worldview beliefs about the user."""
        wvs = self.compiler.get_stable_worldviews(min_stability=min_stability)
        return [
            {
                "id": wv.id,
                "label": wv.label,
                "description": wv.description,
                "type": wv.worldview_type,
                "confidence": round(wv.confidence, 3),
                "stability": round(wv.stability, 3),
                "evidence_count": wv.evidence_count,
                "domain": wv.domain,
                "tags": wv.tags,
            }
            for wv in wvs
        ]

    def get_user_profile(self) -> dict | None:
        """Get the synthesized user profile from worldview consensus."""
        profile = self.compiler.profile
        if profile is None:
            return None
        return {
            "summary": profile.summary,
            "preferences": profile.preferences,
            "expertise_areas": profile.expertise_areas,
            "working_style": profile.working_style,
            "values": profile.values,
            "habits": profile.habits,
            "confidence": round(profile.confidence, 3),
            "version": profile.version,
        }

    # ── Self-Reflection API (#46) ────────────────────────────

    def get_corrections(self, topic: str = "") -> list[dict]:
        """Get correction reports. 'what did I get wrong about X?'"""
        if topic:
            return self.reflection_loop.get_corrections_for_topic(topic)
        return self.reflection_loop.get_recent_corrections()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for dedup comparison."""
        import re
        return re.sub(r'\s+', ' ', text.lower().strip())[:100]
