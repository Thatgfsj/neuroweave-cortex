"""Tests for shared math utilities."""

import math
import pytest

from star_graph.math_utils import cosine_sim, safe_div, clamp, sigmoid


class TestCosineSim:
    def test_identical(self):
        assert cosine_sim([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-10)

    def test_opposite(self):
        assert cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_both_zero(self):
        assert cosine_sim([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_empty_vectors(self):
        assert cosine_sim([], []) == 0.0

    def test_partial_overlap(self):
        v = cosine_sim([1.0, 2.0], [2.0, 1.0])
        assert 0.7 < v < 0.8


class TestSafeDiv:
    def test_normal_division(self):
        assert safe_div(10.0, 2.0) == 5.0

    def test_zero_denominator_returns_default(self):
        assert safe_div(10.0, 0.0) == 0.0

    def test_custom_default(self):
        assert safe_div(10.0, 0.0, default=-1.0) == -1.0

    def test_near_zero_denominator(self):
        assert safe_div(10.0, 1e-13) == 0.0

    def test_negative_denominator(self):
        assert safe_div(10.0, -2.0) == -5.0


class TestClamp:
    def test_within_range(self):
        assert clamp(0.5) == 0.5

    def test_below_range(self):
        assert clamp(-0.5) == 0.0

    def test_above_range(self):
        assert clamp(1.5) == 1.0

    def test_custom_range(self):
        assert clamp(5.0, lo=0.0, hi=10.0) == 5.0
        assert clamp(-1.0, lo=0.0, hi=10.0) == 0.0
        assert clamp(11.0, lo=0.0, hi=10.0) == 10.0

    def test_at_boundary(self):
        assert clamp(0.0) == 0.0
        assert clamp(1.0) == 1.0


class TestSigmoid:
    def test_zero(self):
        assert sigmoid(0.0) == 0.5

    def test_large_positive(self):
        assert sigmoid(10.0) == pytest.approx(1.0, abs=1e-4)

    def test_large_negative(self):
        assert sigmoid(-10.0) == pytest.approx(0.0, abs=1e-4)

    def test_custom_k(self):
        steep = sigmoid(1.0, k=5.0)
        flat = sigmoid(1.0, k=0.5)
        assert steep > flat  # steeper curve at same x

    def test_overflow_handling(self):
        # Very large negative values should not overflow
        result = sigmoid(-1000.0)
        assert 0.0 <= result <= 1.0
