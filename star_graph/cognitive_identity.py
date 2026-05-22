"""Cognitive Identity — persistent cross-session identity for AI and user.

This is the top of the 6-layer pyramid. It maintains:
- Who the user is (user cognitive profile)
- Who the AI is becoming (self-model evolution)
- Growth trajectory over time
- Persistence to disk (~/.nwc/identity/)

The Cognitive Identity is NOT ephemeral — it persists across conversations,
sessions, and deployments. It's the "soul" of the AI-user relationship.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import time
import os


# ── Identity data types ────────────────────────────────────

@dataclass
class UserIdentitySnapshot:
    """A point-in-time snapshot of the user's cognitive identity."""
    timestamp: float = field(default_factory=time.time)
    summary: str = ""
    identity_markers: list[str] = field(default_factory=list)
    cognitive_style: dict = field(default_factory=dict)
    value_system: dict = field(default_factory=dict)
    stable_patterns: list[str] = field(default_factory=list)
    emerging_interests: list[str] = field(default_factory=list)
    interaction_count: int = 0
    days_known: float = 0.0


@dataclass
class SelfIdentity:
    """What NWC knows about itself as a cognitive system."""
    version: str = ""
    formation_date: float = 0.0
    capabilities: list[str] = field(default_factory=list)
    knowledge_domains: list[str] = field(default_factory=list)
    total_memories_processed: int = 0
    total_sleep_cycles: int = 0
    cognitive_health: float = 1.0


@dataclass
class CognitiveIdentity:
    """The complete persistent identity — user + self + trajectory."""
    user_id: str = "default"
    first_interaction: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    total_interactions: int = 0
    user_profile: dict = field(default_factory=dict)
    self_identity: SelfIdentity = field(default_factory=SelfIdentity)
    snapshots: list[UserIdentitySnapshot] = field(default_factory=list)
    growth_milestones: list[dict] = field(default_factory=list)

    def add_snapshot(self, profile: dict):
        """Record a point-in-time snapshot of user identity."""
        snap = UserIdentitySnapshot(
            timestamp=time.time(),
            summary=profile.get("summary", ""),
            identity_markers=profile.get("identity_markers", []),
            cognitive_style=profile.get("cognitive_style", {}),
            value_system=profile.get("value_system", {}),
            stable_patterns=profile.get("stable_patterns", []),
            emerging_interests=profile.get("emerging_interests", []),
            interaction_count=self.total_interactions,
            days_known=(time.time() - self.first_interaction) / 86400,
        )
        self.snapshots.append(snap)
        # Keep last 100 snapshots
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]

    def record_milestone(self, event: str, detail: dict | None = None):
        self.growth_milestones.append({
            "timestamp": time.time(),
            "event": event,
            "detail": detail or {},
            "interaction_count": self.total_interactions,
        })
        if len(self.growth_milestones) > 50:
            self.growth_milestones = self.growth_milestones[-50:]

    def days_since_first(self) -> float:
        return (time.time() - self.first_interaction) / 86400

    def evolution_summary(self) -> str:
        """Human-readable evolution summary."""
        if not self.snapshots:
            return "Identity formation just beginning."

        first = self.snapshots[0]
        latest = self.snapshots[-1]
        lines = [
            f"Known user for {self.days_since_first():.0f} days",
            f"Interactions: {self.total_interactions}",
        ]
        if first.summary != latest.summary:
            lines.append(f"Evolved from: {first.summary}")
            lines.append(f"To: {latest.summary}")
        if latest.emerging_interests:
            lines.append(f"Currently exploring: {', '.join(latest.emerging_interests[:3])}")
        return "\n".join(lines)

    def to_injection(self) -> str:
        """Compressed identity for LLM prompt injection."""
        lines = ["# Persistent Cognitive Identity"]
        lines.append(f"Relationship: {self.days_since_first():.0f} days, {self.total_interactions} interactions")

        if self.snapshots:
            latest = self.snapshots[-1]
            if latest.summary:
                lines.append(f"\n## User Essence\n{latest.summary}")
            if latest.identity_markers:
                lines.append(f"\n## Core Identity\n" + "\n".join(f"- {m}" for m in latest.identity_markers[:5]))
            if latest.emerging_interests:
                lines.append(f"\n## Current Direction\nExploring: {', '.join(latest.emerging_interests[:3])}")

        if self.growth_milestones:
            recent = self.growth_milestones[-3:]
            lines.append("\n## Recent Growth")
            for m in recent:
                lines.append(f"- {m['event']}")

        return "\n".join(lines)


# ── Identity Manager ───────────────────────────────────────

class CognitiveIdentityManager:
    """Manages persistent cognitive identity across sessions.

    Usage:
        cim = CognitiveIdentityManager()
        cim.record_interaction()
        cim.update_profile(personality_engine.get_profile().to_dict())
        cim.snapshot()
        cim.save()

        # For LLM injection:
        identity_prompt = cim.get_identity().to_injection()
    """

    def __init__(self, user_id: str = "default", storage_dir: str = ""):
        self.user_id = user_id
        self._storage_dir = Path(storage_dir or os.path.expanduser("~/.nwc/identity"))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._identity = self._load()

    # ── Persistence ────────────────────────────────────────

    @property
    def _storage_path(self) -> Path:
        return self._storage_dir / f"identity_{self.user_id}.json"

    def _load(self) -> CognitiveIdentity:
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self._deserialize(data)
            except Exception:
                pass
        return CognitiveIdentity(user_id=self.user_id)

    def save(self):
        data = self._serialize()
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _serialize(self) -> dict:
        ci = self._identity
        return {
            "user_id": ci.user_id,
            "first_interaction": ci.first_interaction,
            "last_interaction": ci.last_interaction,
            "total_interactions": ci.total_interactions,
            "user_profile": ci.user_profile,
            "self_identity": asdict(ci.self_identity),
            "snapshots": [asdict(s) for s in ci.snapshots],
            "growth_milestones": ci.growth_milestones,
        }

    def _deserialize(self, data: dict) -> CognitiveIdentity:
        ci = CognitiveIdentity(
            user_id=data.get("user_id", "default"),
            first_interaction=data.get("first_interaction", time.time()),
            last_interaction=data.get("last_interaction", time.time()),
            total_interactions=data.get("total_interactions", 0),
            user_profile=data.get("user_profile", {}),
            growth_milestones=data.get("growth_milestones", []),
        )
        ci.self_identity = SelfIdentity(**data.get("self_identity", {}))
        ci.snapshots = [
            UserIdentitySnapshot(**s) for s in data.get("snapshots", [])
        ]
        return ci

    # ── Interactions ───────────────────────────────────────

    def record_interaction(self):
        """Call on every user interaction."""
        self._identity.total_interactions += 1
        self._identity.last_interaction = time.time()
        self._identity.self_identity.total_memories_processed += 1

    def record_sleep_cycle(self):
        self._identity.self_identity.total_sleep_cycles += 1

    def update_profile(self, profile: dict):
        """Update user cognitive profile from PersonalityFormationEngine."""
        self._identity.user_profile = profile

    def snapshot(self):
        """Take a point-in-time snapshot of current identity."""
        self._identity.add_snapshot(self._identity.user_profile)

    def add_milestone(self, event: str, detail: dict | None = None):
        self._identity.record_milestone(event, detail)

    # ── Access ─────────────────────────────────────────────

    def get_identity(self) -> CognitiveIdentity:
        return self._identity

    def get_injection(self) -> str:
        """Get identity prompt injection for LLM."""
        return self._identity.to_injection()

    def get_evolution_summary(self) -> str:
        return self._identity.evolution_summary()

    def get_stats(self) -> dict:
        ci = self._identity
        return {
            "user_id": ci.user_id,
            "days_known": ci.days_since_first(),
            "total_interactions": ci.total_interactions,
            "snapshots": len(ci.snapshots),
            "milestones": len(ci.growth_milestones),
            "sleep_cycles": ci.self_identity.total_sleep_cycles,
            "latest_summary": ci.snapshots[-1].summary if ci.snapshots else "forming...",
        }
