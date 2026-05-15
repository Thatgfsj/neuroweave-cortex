"""Tests for deterministic seed functions."""

import random

import numpy as np
import pytest

from star_graph.seed import seed_everything, get_seed, is_deterministic


class TestSeedEverything:
    def test_sets_seed(self):
        seed_everything(42)
        assert get_seed() == 42
        assert is_deterministic()

    def test_default_seed_is_42(self):
        seed_everything()
        assert get_seed() == 42

    def test_is_deterministic_false_when_not_seeded(self):
        seed_everything(None)  # can't set None, so use fresh state
        # Re-import to get fresh module state
        import star_graph.seed as sd
        sd._GLOBAL_SEED = None
        assert not sd.is_deterministic()

    def test_makes_random_reproducible(self):
        seed_everything(123)
        a = random.random()
        seed_everything(123)
        b = random.random()
        assert a == b

    def test_makes_numpy_reproducible(self):
        seed_everything(42)
        a = np.random.random()
        seed_everything(42)
        b = np.random.random()
        assert a == b


class TestGetSeed:
    def test_returns_none_initially(self):
        import star_graph.seed as sd
        sd._GLOBAL_SEED = None
        assert get_seed() is None

    def test_returns_seed_after_set(self):
        seed_everything(77)
        assert get_seed() == 77


class TestIsDeterministic:
    def test_false_when_not_seeded(self):
        import star_graph.seed as sd
        sd._GLOBAL_SEED = None
        assert not is_deterministic()

    def test_true_when_seeded(self):
        seed_everything(1)
        assert is_deterministic()
