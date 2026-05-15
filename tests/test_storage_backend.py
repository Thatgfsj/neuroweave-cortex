"""Tests for StorageBackend ABC."""

import pytest

from star_graph.storage_backend import StorageBackend
from star_graph.graph import StarGraph


class MockStorage(StorageBackend):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self):
        self._saved = None
        self._graph = StarGraph()

    def save(self, graph):
        self._saved = graph

    def load(self):
        return self._graph

    @property
    def exists(self):
        return self._saved is not None


class TestStorageBackend:
    def test_save_and_load(self):
        store = MockStorage()
        g = StarGraph()
        store.save(g)
        assert store.exists
        loaded = store.load()
        assert isinstance(loaded, StarGraph)

    def test_exists_false_initially(self):
        store = MockStorage()
        assert not store.exists

    def test_fine_grained_ops_are_noop(self):
        store = MockStorage()
        store.save_anchor("a1", {"text": "test"})
        store.delete_anchor("a1")
        store.save_edge("a1", "a2", {"weight": 0.5})
        store.delete_edge("a1", "a2")
        store.save_ghost("g1", {})
        store.delete_ghost("g1")
        store.save_schema("s1", {})
        store.close()
        # All should complete without error

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            StorageBackend()
