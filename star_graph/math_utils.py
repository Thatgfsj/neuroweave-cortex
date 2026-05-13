"""Shared math utilities — single source of truth for common operations.

Centralises functions that were duplicated across 20+ files:
- cosine_sim: cosine similarity between two float vectors
"""

from __future__ import annotations

import math


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors.

    Returns a value in [-1, 1]. Safe for zero-vectors (returns 0.0).
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    denom = na * nb
    if denom < 1e-12:
        return 0.0
    return dot / denom


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default when denominator is near zero."""
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def sigmoid(x: float, k: float = 1.0) -> float:
    """Logistic sigmoid 1 / (1 + exp(-k*x))."""
    try:
        return 1.0 / (1.0 + math.exp(-k * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
