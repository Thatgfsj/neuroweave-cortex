"""Cognitive Closure — learn correct evolve loop for self-improving memory.

Closed-loop feedback: record recall outcomes, reflect on patterns, correct
low-quality memories, reinforce high-quality ones. Called during sleep
consolidation after other phases.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .memory_core.graph import StarGraph
    from .memory_core.anchor import Anchor


@dataclass
class FeedbackRecord:
    """Outcome trace for a single memory recall event."""

    anchor_id: str
    recall_id: str
    was_used: bool
    was_helpful: bool
    user_referenced: bool
    task_success: bool
    timestamp: float
    context: str


@dataclass
class ClosureConfig:
    """Hyperparameters governing the closure cycle."""

    reflection_interval: int = 10
    min_memories_for_reflection: int = 100
    correction_threshold: float = 0.3
    reinforcement_threshold: float = 0.7
    llm_enabled: bool = False
    llm_model: str = ""
    auto_correct: bool = True
    max_corrections_per_cycle: int = 10


class CognitiveClosure:
    """Closed-loop memory refinement: learn from feedback, correct errors, evolve."""

    def __init__(
        self,
        graph: StarGraph,
        config: ClosureConfig | None = None,
        embedder=None,
        llm_fn=None,
    ) -> None:
        self._graph = graph
        self._config = config or ClosureConfig()
        self._embedder = embedder
        self._llm_fn = llm_fn
        self._cycle_count: int = 0
        self._correction_history: dict[str, int] = defaultdict(int)
        self._last_reflection_ts: float = 0.0
        self._stats: dict[str, Any] = {
            "total_feedback_records": 0,
            "total_corrections": 0,
            "total_reinforcements": 0,
            "cycles_completed": 0,
        }

    # ── Feedback storage helpers ───────────────────────────

    def _feedback_to_anchor(self, record: FeedbackRecord) -> Anchor:
        from .memory_core.anchor import Anchor
        payload = json.dumps({
            "anchor_id": record.anchor_id,
            "recall_id": record.recall_id,
            "was_used": record.was_used,
            "was_helpful": record.was_helpful,
            "user_referenced": record.user_referenced,
            "task_success": record.task_success,
            "context": record.context,
        })
        anchor = Anchor.create(
            text=payload,
            source_session="closure",
            tags=["__feedback__", record.anchor_id],
            importance=0.3,
        )
        anchor.created_at = record.timestamp
        return anchor

    def _anchor_to_feedback(self, anchor: Anchor) -> FeedbackRecord | None:
        try:
            data = json.loads(anchor.text)
            return FeedbackRecord(
                anchor_id=data["anchor_id"],
                recall_id=data["recall_id"],
                was_used=data["was_used"],
                was_helpful=data["was_helpful"],
                user_referenced=data["user_referenced"],
                task_success=data["task_success"],
                timestamp=anchor.created_at,
                context=data.get("context", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _get_feedback_records(self, anchor_id: str = "",
                              window: int = 0) -> list[FeedbackRecord]:
        records: list[FeedbackRecord] = []
        for aid, anchor in self._graph.anchors.items():
            if "__feedback__" not in anchor.tags:
                continue
            fb = self._anchor_to_feedback(anchor)
            if fb is None:
                continue
            if anchor_id and fb.anchor_id != anchor_id:
                continue
            records.append(fb)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        if window > 0:
            records = records[:window]
        return records

    def _correction_count_for(self, anchor_id: str) -> int:
        return self._correction_history.get(anchor_id, 0)

    def _record_correction(self, anchor_id: str) -> None:
        self._correction_history[anchor_id] = self._correction_count_for(anchor_id) + 1

    # ── Public API ─────────────────────────────────────────

    def record_feedback(
        self,
        anchor_id: str,
        was_used: bool,
        was_helpful: bool,
        task_success: bool,
        context: str = "",
        user_referenced: bool = False,
    ) -> FeedbackRecord:
        recall_id = hashlib.blake2b(
            f"{anchor_id}:{time.time()}:{context}".encode(), digest_size=8
        ).hexdigest()
        record = FeedbackRecord(
            anchor_id=anchor_id,
            recall_id=recall_id,
            was_used=was_used,
            was_helpful=was_helpful,
            user_referenced=user_referenced,
            task_success=task_success,
            timestamp=time.time(),
            context=context,
        )
        fb_anchor = self._feedback_to_anchor(record)
        self._graph.add_anchor(fb_anchor)
        self._stats["total_feedback_records"] += 1

        if anchor_id in self._graph.anchors:
            anchor = self._graph.anchors[anchor_id]
            if was_helpful:
                anchor.record_success(benefit=0.05)
            elif was_used:
                anchor.record_failure(penalty=0.03)

        return record

    def get_recall_success_rate(self, anchor_id: str = "",
                                 window: int = 100) -> float:
        records = self._get_feedback_records(anchor_id=anchor_id, window=window)
        if not records:
            return 0.5
        return sum(1 for r in records if r.was_helpful) / len(records)

    # ── Reflection ─────────────────────────────────────────

    def reflect(self) -> dict:
        self._cycle_count += 1
        now = time.time()
        self._last_reflection_ts = now

        records = self._get_feedback_records(window=500)
        total = len(records)

        per_anchor: dict[str, list[FeedbackRecord]] = defaultdict(list)
        for r in records:
            per_anchor[r.anchor_id].append(r)

        anchor_stats: dict[str, dict] = {}
        for aid, recs in per_anchor.items():
            helpful = sum(1 for r in recs if r.was_helpful)
            used = sum(1 for r in recs if r.was_used)
            success = sum(1 for r in recs if r.task_success)
            n = len(recs)
            anchor_stats[aid] = {
                "total_recalls": n,
                "helpful_rate": helpful / n if n else 0.0,
                "usage_rate": used / n if n else 0.0,
                "task_success_rate": success / n if n else 0.0,
                "recent_contexts": [r.context for r in recs[:3]],
            }

        overall_success_rate = (
            sum(1 for r in records if r.was_helpful) / total
        ) if total else 0.5

        correction_candidates = [
            aid for aid, s in anchor_stats.items()
            if s["helpful_rate"] < self._config.correction_threshold
            and s["total_recalls"] >= 3
            and aid in self._graph.anchors
        ][:self._config.max_corrections_per_cycle]

        reinforcement_candidates = [
            aid for aid, s in anchor_stats.items()
            if s["helpful_rate"] > self._config.reinforcement_threshold
            and s["total_recalls"] >= 2
            and aid in self._graph.anchors
        ]

        low_quality_count = len(correction_candidates)
        high_quality_count = len(reinforcement_candidates)

        pattern_insights = self.identify_patterns() if total >= 10 else []

        health = self.compute_overall_health()

        recommendations: list[str] = []
        if low_quality_count > 0:
            recommendations.append(
                f"{low_quality_count} memories need correction (success rate < {self._config.correction_threshold})"
            )
        if high_quality_count > 0:
            recommendations.append(
                f"{high_quality_count} memories ready for reinforcement"
            )
        if overall_success_rate < 0.4:
            recommendations.append(
                "Overall recall success rate low — consider improving memory quality at encoding"
            )
        if overall_success_rate > 0.7:
            recommendations.append(
                "Memory system performing well — maintain current strategy"
            )

        report = {
            "cycle": self._cycle_count,
            "total_feedback_records": total,
            "overall_recall_success_rate": round(overall_success_rate, 3),
            "low_quality_count": low_quality_count,
            "high_quality_count": high_quality_count,
            "correction_candidates": correction_candidates,
            "reinforcement_candidates": reinforcement_candidates,
            "pattern_insights": pattern_insights,
            "health_score": round(health, 3),
            "recommendations": recommendations,
        }

        return report

    # ── Correction ─────────────────────────────────────────

    def identify_corrections(self) -> list[str]:
        records = self._get_feedback_records(window=300)
        per_anchor: dict[str, list[FeedbackRecord]] = defaultdict(list)
        for r in records:
            per_anchor[r.anchor_id].append(r)

        candidates = []
        for aid, recs in per_anchor.items():
            if aid not in self._graph.anchors:
                continue
            helpful = sum(1 for r in recs if r.was_helpful)
            n = len(recs)
            if n < 3:
                continue
            rate = helpful / n
            if rate < self._config.correction_threshold:
                candidates.append(aid)

        candidates.sort(
            key=lambda aid: sum(1 for r in per_anchor[aid] if r.was_helpful)
            / max(1, len(per_anchor[aid]))
        )
        return candidates[:self._config.max_corrections_per_cycle]

    def correct_memory(self, anchor_id: str) -> Anchor | None:
        if anchor_id not in self._graph.anchors:
            return None

        anchor = self._graph.anchors[anchor_id]
        feedback = self._get_feedback_records(anchor_id=anchor_id, window=50)
        if not feedback:
            return anchor

        helpful = sum(1 for r in feedback if r.was_helpful)
        n = len(feedback)
        rate = helpful / n

        correction_applied = False

        if rate < self._config.correction_threshold and n >= 3:
            anchor.vector.importance = max(0.01, anchor.vector.importance - 0.2)
            anchor.vector.confidence = max(0.01, anchor.vector.confidence - 0.15)
            correction_applied = True

        if feedback and feedback[0].context:
            contexts = [r.context for r in feedback if r.was_helpful]
            if contexts:
                context_words = set()
                for ctx in contexts:
                    for word in ctx.lower().split():
                        if len(word) > 3:
                            context_words.add(word)
                existing_tags_lower = {t.lower() for t in anchor.tags}
                suggested_tags = {
                    word for word in context_words
                    if word not in existing_tags_lower and len(word) > 3
                }
                if suggested_tags:
                    anchor.tags.extend(sorted(suggested_tags)[:5])
                    correction_applied = True

        text_lower = anchor.text.lower().strip()
        for other_id, other_anchor in self._graph.anchors.items():
            if other_id == anchor_id:
                continue
            if other_anchor.text.lower().strip() == text_lower:
                if other_anchor.created_at > anchor.created_at:
                    anchor.invalid_at = time.time()
                    anchor.conflict_candidate = True
                else:
                    other_anchor.invalid_at = time.time()
                    other_anchor.conflict_candidate = True
                correction_applied = True
                break

        self._record_correction(anchor_id)
        correction_count = self._correction_count_for(anchor_id)

        if correction_count >= 2 and anchor.retention_score < self._config.correction_threshold:
            anchor.transition("prune")

        if correction_applied:
            self._stats["total_corrections"] += 1

        return anchor

    def reinforce_memories(self, anchor_ids: list[str]) -> None:
        for aid in anchor_ids:
            if aid not in self._graph.anchors:
                continue
            anchor = self._graph.anchors[aid]
            anchor.vector.importance = min(1.0, anchor.vector.importance + 0.1)
            anchor.vector.confidence = min(1.0, anchor.vector.confidence + 0.08)
            anchor.vector.stability = min(1.0, anchor.vector.stability + 0.05)
            anchor.record_success(benefit=0.08)
            self._stats["total_reinforcements"] += 1

    # ── Pattern detection ──────────────────────────────────

    def identify_patterns(self) -> list[dict]:
        records = self._get_feedback_records(window=500)
        if len(records) < 10:
            return []

        insights: list[dict] = []

        tag_success: dict[str, list[int]] = defaultdict(list)
        for r in records:
            anchor = self._graph.anchors.get(r.anchor_id)
            if anchor is None:
                continue
            for tag in anchor.tags:
                tag_lower = tag.lower()
                if tag_lower == "__feedback__":
                    continue
                tag_success[tag_lower].append(1 if r.was_helpful else 0)

        ranked_tags = []
        for tag, outcomes in tag_success.items():
            if len(outcomes) < 5:
                continue
            sr = sum(outcomes) / len(outcomes)
            ranked_tags.append({"tag": tag, "success_rate": round(sr, 3), "sample_size": len(outcomes)})
        ranked_tags.sort(key=lambda x: -x["success_rate"])

        if ranked_tags:
            insights.append({
                "type": "tag_correlation",
                "description": "Tags ranked by associated recall success rate",
                "data": ranked_tags[:10],
            })

        domain_stats: dict[str, list[float]] = defaultdict(list)
        for r in records:
            anchor = self._graph.anchors.get(r.anchor_id)
            if anchor is None:
                continue
            domain = anchor.cortex_path or "default"
            domain_stats[domain].append(1.0 if r.was_helpful else 0.0)

        ranked_domains = []
        for domain, scores in domain_stats.items():
            if len(scores) < 3:
                continue
            avg = sum(scores) / len(scores)
            ranked_domains.append({"domain": domain, "avg_quality": round(avg, 3), "sample_size": len(scores)})
        ranked_domains.sort(key=lambda x: x["avg_quality"])

        if ranked_domains:
            insights.append({
                "type": "domain_quality",
                "description": "Domain-level recall quality ranking",
                "data": ranked_domains,
            })

        hour_buckets: dict[int, list[float]] = defaultdict(list)
        for r in records:
            hour = time.localtime(r.timestamp).tm_hour
            hour_buckets[hour].append(1.0 if r.was_helpful else 0.0)

        best_hours = sorted(
            [(h, sum(s) / len(s)) for h, s in hour_buckets.items() if len(s) >= 3],
            key=lambda x: -x[1],
        )[:5]

        if best_hours:
            insights.append({
                "type": "time_of_day",
                "description": "Hours with best recall quality",
                "data": [{"hour": h, "avg_quality": round(q, 3)} for h, q in best_hours],
            })

        context_task_map: dict[str, list[float]] = defaultdict(list)
        for r in records:
            if r.context and r.task_success:
                ctx_key = r.context.lower().strip()[:50]
                if len(ctx_key) > 5:
                    context_task_map[ctx_key].append(1.0 if r.was_helpful else 0.0)

        ranked_tasks = sorted(
            [(k, sum(v) / len(v)) for k, v in context_task_map.items() if len(v) >= 2],
            key=lambda x: -x[1],
        )[:8]

        if ranked_tasks:
            insights.append({
                "type": "task_memory_benefit",
                "description": "Task types that benefit most from memory retrieval",
                "data": [{"context": ctx, "avg_recall_benefit": round(q, 3)} for ctx, q in ranked_tasks],
            })

        return insights

    # ── Evolution cycle ────────────────────────────────────

    def evolution_cycle(self) -> dict:
        report = self.reflect()

        if self._config.auto_correct:
            corrected = 0
            for aid in report.get("correction_candidates", []):
                if corrected >= self._config.max_corrections_per_cycle:
                    break
                result = self.correct_memory(aid)
                if result:
                    corrected += 1

            reinforced = report.get("reinforcement_candidates", [])
            if reinforced:
                self.reinforce_memories(reinforced)

        self._stats["cycles_completed"] += 1

        report["corrections_applied"] = min(
            len(report.get("correction_candidates", [])),
            self._config.max_corrections_per_cycle,
        )
        report["reinforcements_applied"] = len(
            report.get("reinforcement_candidates", [])
        )
        report["overall_health"] = round(self.compute_overall_health(), 3)

        return report

    # ── Health ─────────────────────────────────────────────

    def compute_overall_health(self) -> float:
        success_rate = self.get_recall_success_rate(window=200)

        anchors = [a for a in self._graph.anchors.values()
                   if "__feedback__" not in a.tags]
        total_anchors = len(anchors)
        if total_anchors > 0:
            avg_quality = sum(a.retention_score for a in anchors) / total_anchors
        else:
            avg_quality = 0.5

        fb_anchor_ids = {
            aid for aid, a in self._graph.anchors.items()
            if "__feedback__" in a.tags
        }
        anchored_with_feedback: set[str] = set()
        for aid in fb_anchor_ids:
            fb = self._anchor_to_feedback(self._graph.anchors[aid])
            if fb:
                anchored_with_feedback.add(fb.anchor_id)

        total_non_fb = sum(1 for a in self._graph.anchors.values()
                          if "__feedback__" not in a.tags)
        feedback_coverage = (
            len(anchored_with_feedback) / max(1, total_non_fb)
        ) if total_non_fb > 0 else 0.0

        corrections = len([
            aid for aid, count in self._correction_history.items()
            if count > 0
        ])
        correction_rate = 1.0 - (corrections / max(1, total_anchors))
        correction_rate = max(0.0, min(1.0, correction_rate))

        health = (
            0.4 * success_rate
            + 0.3 * avg_quality
            + 0.2 * feedback_coverage
            + 0.1 * correction_rate
        )
        return max(0.0, min(1.0, health))

    # ── Snapshot ───────────────────────────────────────────

    def snapshot(self) -> dict:
        total_anchors = len(self._graph.anchors)
        fb_records = self._get_feedback_records()

        return {
            "cycle_count": self._cycle_count,
            "config": {k: v for k, v in self._config.__dict__.items()
                      if not k.startswith("_")},
            "stats": dict(self._stats),
            "total_anchors": total_anchors,
            "feedback_records": len(fb_records),
            "feedback_coverage": len({
                r.anchor_id for r in fb_records
            }) / max(1, total_anchors),
            "overall_recall_success": self.get_recall_success_rate(),
            "health_score": round(self.compute_overall_health(), 3),
            "corrections_to_date": self._stats["total_corrections"],
            "reinforcements_to_date": self._stats["total_reinforcements"],
            "last_reflection": self._last_reflection_ts,
            "correction_history_size": len(self._correction_history),
        }
