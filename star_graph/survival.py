"""Survival Functions — configurable memory decay curves.

Provides a pluggable interface for memory decay with 4 built-in curves:
- Ebbinghaus (classic forgetting curve)
- Power-law (slower long-term decay)
- Exponential (simple exponential decay)
- Custom (arbitrary lambda)

Can be configured per-anchor or per-cortex via defaults.yaml.

Integration:
    Anchor.decay() → anchor.decay_factor → survival_function.survive(t, strength)
    GhostNode.intensity → survival_function.ghost_decay(t)
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable


# ═══════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════

class SurvivalFunction(ABC):
    """Abstract survival/decay function for memory retention.

    Each implementation defines how memory strength decays over time.
    The function returns a retention factor in [0, 1] where:
      1.0 = fully retained (just created)
      0.0 = completely forgotten

    Parameters:
      t_hours: elapsed time in hours
      strength: relative memory strength (0..1), derived from
                importance, stability, emotional salience, etc.
    """

    @abstractmethod
    def survive(self, t_hours: float, strength: float = 0.5) -> float:
        """Return retention factor after t_hours given memory strength."""
        ...

    @abstractmethod
    def half_life(self, strength: float = 0.5) -> float:
        """Return time (hours) until retention reaches 0.5."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this survival function."""
        ...

    def ghost_decay(self, t_hours: float, base_strength: float = 0.3) -> float:
        """Decay specific to ghost traces — typically faster than live memory."""
        return self.survive(t_hours, base_strength * 0.5)


# ═══════════════════════════════════════════════════════════════
# Built-in implementations
# ═══════════════════════════════════════════════════════════════

class EbbinghausSurvival(SurvivalFunction):
    """Ebbinghaus forgetting curve: R = e^(-t / S), S in hours.

    S (relative strength) is derived from memory importance: S = 1 + strength * 200
    giving half-lives ranging from ~0.7h (weak) to ~140h (strong).
    This curve drops fast initially then flattens — the classic "forgetting curve."
    """

    name = "ebbinghaus"

    def __init__(self, min_hours: float = 1.0, max_hours: float = 336.0):
        self.min_hours = min_hours   # minimum S for strength=0
        self.max_hours = max_hours  # maximum S for strength=1

    def survive(self, t_hours: float, strength: float = 0.5) -> float:
        s = self.min_hours + strength * (self.max_hours - self.min_hours)
        return math.exp(-t_hours / s)

    def half_life(self, strength: float = 0.5) -> float:
        s = self.min_hours + strength * (self.max_hours - self.min_hours)
        return s * math.log(2)

    def ghost_decay(self, t_hours: float, base_strength: float = 0.3) -> float:
        # Ghosts have a compressed S range (faster decay)
        s = self.min_hours * 0.2 + base_strength * (self.max_hours * 0.15)
        return math.exp(-t_hours / s)


class PowerLawSurvival(SurvivalFunction):
    """Power-law decay: R = (1 + t / S)^(-alpha).

    Slower long-term decay than exponential. Better fits the observation
    that very old memories decay more slowly than exponential predicts.
    """

    name = "power_law"

    def __init__(self, alpha: float = 0.5, scale: float = 50.0):
        self.alpha = alpha
        self.scale = scale

    def survive(self, t_hours: float, strength: float = 0.5) -> float:
        s = self.scale * (0.2 + strength * 1.8)
        return (1.0 + t_hours / s) ** (-self.alpha)

    def half_life(self, strength: float = 0.5) -> float:
        s = self.scale * (0.2 + strength * 1.8)
        return s * (2.0 ** (1.0 / self.alpha) - 1.0)


class ExponentialSurvival(SurvivalFunction):
    """Simple exponential decay: R = e^(-lambda * t).

    Parameters:
      lambda_per_day: decay rate per day (default 0.05 → half-life ~14 days)
    """

    name = "exponential"

    def __init__(self, lambda_per_day: float = 0.05):
        self.lambda_per_hour = lambda_per_day / 24.0

    def survive(self, t_hours: float, strength: float = 0.5) -> float:
        # Strength modulates lambda: stronger = slower decay
        effective_lambda = self.lambda_per_hour * (1.0 - strength * 0.8)
        return math.exp(-effective_lambda * t_hours)

    def half_life(self, strength: float = 0.5) -> float:
        effective_lambda = self.lambda_per_hour * (1.0 - strength * 0.8)
        return math.log(2) / max(0.0001, effective_lambda)


class CustomSurvival(SurvivalFunction):
    """User-defined survival function via a lambda/callable.

    Example:
        CustomSurvival(lambda t, s: 1.0 / (1.0 + t / (s * 100)))
    """

    name = "custom"

    def __init__(self, fn: Callable[[float, float], float],
                 half_life_fn: Callable[[float], float] | None = None):
        self._fn = fn
        self._half_life_fn = half_life_fn

    def survive(self, t_hours: float, strength: float = 0.5) -> float:
        return max(0.0, min(1.0, self._fn(t_hours, strength)))

    def half_life(self, strength: float = 0.5) -> float:
        if self._half_life_fn:
            return self._half_life_fn(strength)
        # Brute-force search for half-life
        lo, hi = 0.0, 87600.0  # up to 10 years
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.survive(mid, strength) > 0.5:
                lo = mid
            else:
                hi = mid
        return lo


# ═══════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════

class SurvivalRegistry:
    """Global registry of named survival functions.

    Usage:
        reg = SurvivalRegistry()
        reg.register("ebbinghaus", EbbinghausSurvival())
        fn = reg.get("ebbinghaus")
        retention = fn.survive(t_hours=48, strength=0.7)
    """

    def __init__(self):
        self._functions: dict[str, SurvivalFunction] = {}
        # Register defaults
        self._register_defaults()

    def _register_defaults(self):
        self.register("ebbinghaus", EbbinghausSurvival())
        self.register("power_law", PowerLawSurvival())
        self.register("exponential", ExponentialSurvival())

    def register(self, name: str, fn: SurvivalFunction):
        self._functions[name] = fn

    def get(self, name: str) -> SurvivalFunction:
        if name not in self._functions:
            raise KeyError(
                f"Unknown survival function: '{name}'. "
                f"Available: {list(self._functions.keys())}"
            )
        return self._functions[name]

    @property
    def available(self) -> list[str]:
        return list(self._functions.keys())

    @classmethod
    def from_config(cls, config=None) -> SurvivalFunction:
        """Create a survival function from config or defaults.

        Reads config.survival section for: function_name, and function-specific params.
        Falls back to Ebbinghaus if not configured.
        """
        reg = cls()
        fn_name = "ebbinghaus"

        if config:
            surv_cfg = getattr(config, 'survival', None)
            if surv_cfg:
                fn_name = getattr(surv_cfg, 'function', 'ebbinghaus')

        # If custom, read params from config
        if fn_name == "ebbinghaus":
            min_h = 1.0
            max_h = 336.0
            if config and hasattr(config, 'survival'):
                sc = config.survival
                min_h = getattr(sc, 'ebbinghaus_min_hours', 1.0)
                max_h = getattr(sc, 'ebbinghaus_max_hours', 336.0)
            return EbbinghausSurvival(min_hours=min_h, max_hours=max_h)

        elif fn_name == "power_law":
            alpha = 0.5
            scale = 50.0
            if config and hasattr(config, 'survival'):
                sc = config.survival
                alpha = getattr(sc, 'power_law_alpha', 0.5)
                scale = getattr(sc, 'power_law_scale', 50.0)
            return PowerLawSurvival(alpha=alpha, scale=scale)

        elif fn_name == "exponential":
            lam = 0.05
            if config and hasattr(config, 'survival'):
                lam = getattr(config.survival, 'exponential_lambda_per_day', 0.05)
            return ExponentialSurvival(lambda_per_day=lam)

        elif fn_name in reg._functions:
            return reg._functions[fn_name]

        raise KeyError(f"Unknown survival function: {fn_name}")


# ═══════════════════════════════════════════════════════════════
# Anchor integration helper
# ═══════════════════════════════════════════════════════════════

@dataclass
class SurvivalState:
    """Runtime state for tracking survival curve application to an anchor."""
    last_decay_at: float = field(default_factory=time.time)
    current_retention: float = 1.0
    decay_count: int = 0
    function_name: str = "ebbinghaus"

    def apply(self, fn: SurvivalFunction, anchor_strength: float) -> float:
        """Apply survival function since last decay tick.

        Returns the new retention factor.
        """
        now = time.time()
        t_hours = (now - self.last_decay_at) / 3600.0
        self.last_decay_at = now
        self.current_retention = fn.survive(t_hours, anchor_strength)
        self.decay_count += 1
        return self.current_retention


def derive_strength(anchor) -> float:
    """Compute composite strength for survival function input.

    Uses: importance (30%), stability (25%), emotional salience (20%),
          frequency (15%), confidence (10%).
    """
    v = anchor.vector
    return (
        v.importance * 0.30
        + v.stability * 0.25
        + abs(v.emotional_valence) * 0.20
        + v.frequency * 0.15
        + v.confidence * 0.10
    )
