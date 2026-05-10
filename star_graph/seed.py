"""Deterministic mode — reproducible benchmarks.

Without this, cognitive + graph + sleep replay systems are intrinsically
non-deterministic. seed_everything(42) makes all operations reproducible:
  - replay order
  - pruning decisions
  - abstraction generation
  - phase initialization
  - stochastic sampling in SWR replay
"""

import random
import numpy as np


_GLOBAL_SEED: int | None = None


def seed_everything(seed: int = 42) -> None:
    """Set all RNG seeds for reproducible runs."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seed() -> int | None:
    return _GLOBAL_SEED


def is_deterministic() -> bool:
    return _GLOBAL_SEED is not None
