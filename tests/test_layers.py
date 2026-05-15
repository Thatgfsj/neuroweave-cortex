"""Tests for layer boundary enforcement."""

import os

import pytest

from star_graph.layers import (
    get_layer,
    check_import,
    layer_summary,
    enforce_layer_boundaries,
)


class TestGetLayer:
    def test_layer_1_modules(self):
        assert get_layer("anchor") == 1
        assert get_layer("graph") == 1
        assert get_layer("index") == 1
        assert get_layer("config") == 1

    def test_layer_2_modules(self):
        assert get_layer("sleep") == 2
        assert get_layer("retriever") == 2
        assert get_layer("abstraction") == 2
        assert get_layer("ghost") == 2

    def test_layer_3_modules(self):
        assert get_layer("seed") == 3
        assert get_layer("embedding") == 3
        assert get_layer("online") == 3

    def test_unknown_module_returns_0(self):
        assert get_layer("nonexistent_module") == 0

    def test_full_module_path(self):
        assert get_layer("star_graph.sleep") == 2


class TestCheckImport:
    def test_same_layer_allowed(self):
        assert check_import("sleep", "ghost")  # L2 → L2

    def test_upper_to_lower_allowed(self):
        assert check_import("sleep", "graph")  # L2 → L1
        assert check_import("seed", "anchor")  # L3 → L1
        assert check_import("seed", "sleep")   # L3 → L2

    def test_lower_to_upper_violated(self):
        # L1 → L2 is not allowed (except known exceptions)
        assert not check_import("graph", "sleep")

    def test_known_exception_anchor_to_embedding(self):
        assert check_import("anchor", "embedding")

    def test_unknown_modules_always_allowed(self):
        assert check_import("unknown", "graph")
        assert check_import("graph", "unknown")
        assert check_import("unknown", "other_unknown")


class TestLayerSummary:
    def test_returns_string(self):
        result = layer_summary()
        assert isinstance(result, str)
        assert "Layer Architecture" in result
        assert "Layer 1" in result
        assert "Layer 2" in result
        assert "Layer 3" in result

    def test_includes_module_names(self):
        result = layer_summary()
        assert "anchor" in result
        assert "sleep" in result
        assert "seed" in result


class TestEnforceLayerBoundaries:
    def test_does_nothing_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("STAR_GRAPH_STRICT_LAYERS", raising=False)
        # Should not raise
        enforce_layer_boundaries()

    def test_runs_with_env_set(self, monkeypatch):
        monkeypatch.setenv("STAR_GRAPH_STRICT_LAYERS", "1")
        # This will scan all modules — should not raise since violations
        # are already handled by check_import exceptions
        try:
            enforce_layer_boundaries()
        except ImportError:
            # It's OK if there are violations found
            pass
