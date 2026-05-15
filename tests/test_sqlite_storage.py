"""Tests for SQLite storage backend."""

import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from star_graph.anchor import Anchor, AnchorVector
from star_graph.graph import StarGraph, Schema
from star_graph.sqlite_storage import (
    SQLiteStorage,
    DEFAULT_SQLITE_PATH,
    _embedding_to_blob,
    _blob_to_embedding,
)


@contextmanager
def temp_store_dir():
    """Create a temp directory that's safe to clean up on Windows."""
    td = tempfile.mkdtemp()
    try:
        yield Path(td)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def make_anchor(name: str, text: str = "", embedding: list | None = None,
                tags: list | None = None) -> Anchor:
    a = Anchor.create(text=text or f"Memory for {name}", tags=tags or [])
    a.id = name
    if embedding:
        a.embedding = embedding
    return a


class TestBlobConversion:
    def test_embedding_to_blob_roundtrip(self):
        emb = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = _embedding_to_blob(emb)
        assert isinstance(blob, bytes)
        result = _blob_to_embedding(blob)
        for a, b in zip(emb, result):
            assert a == pytest.approx(b, abs=1e-6)

    def test_embedding_none_to_blob(self):
        assert _embedding_to_blob(None) is None

    def test_blob_none_to_embedding(self):
        assert _blob_to_embedding(None) is None

    def test_empty_embedding(self):
        blob = _embedding_to_blob([])
        assert blob == b""
        result = _blob_to_embedding(blob)
        assert result == []


class TestSQLiteStorageInit:
    def test_default_path(self):
        expected = Path.home() / ".star_graph" / "memory.db"
        assert DEFAULT_SQLITE_PATH == expected

    def test_custom_path(self):
        store = SQLiteStorage(Path("/tmp/test.db"))
        assert store.path == Path("/tmp/test.db")


class TestSQLiteStorageSaveLoad:
    def test_save_and_load_roundtrip(self):
        g = StarGraph()
        a = make_anchor("a1", "Hello world", embedding=[0.1, 0.2, 0.3])
        g.add_anchor(a)

        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save(g)
            store.close()
            assert store.exists

            store2 = SQLiteStorage(tmp_path)
            loaded = store2.load()
            assert "a1" in loaded.anchors
            assert loaded.anchors["a1"].text == "Hello world"
            assert loaded.anchors["a1"].embedding == pytest.approx([0.1, 0.2, 0.3])
            store2.close()

    def test_load_empty_when_no_tables(self):
        with temp_store_dir() as td:
            path = td / "empty.db"
            store = SQLiteStorage(path)
            graph = store.load()
            assert len(graph.anchors) == 0
            store.close()

    def test_load_multiple_anchors(self):
        g = StarGraph()
        for i in range(5):
            a = make_anchor(f"a{i}", f"Anchor {i}")
            g.add_anchor(a)

        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save(g)
            store.close()

            store2 = SQLiteStorage(tmp_path)
            loaded = store2.load()
            assert len(loaded.anchors) == 5
            store2.close()

    def test_save_and_load_edges(self):
        g = StarGraph()
        a0 = make_anchor("a0")
        a1 = make_anchor("a1")
        g.add_anchor(a0)
        g.add_anchor(a1)
        g.add_edge("a0", "a1", weight=0.8, edge_type="causes")

        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save(g)
            store.close()

            store2 = SQLiteStorage(tmp_path)
            loaded = store2.load()
            assert len(loaded.edges) == 1
            edge = list(loaded.edges.values())[0]
            assert edge.weight == pytest.approx(0.88)  # 0.8 * 1.1 strong-relation boost
            assert edge.edge_type == "causes"
            store2.close()

    def test_save_and_load_schemas(self):
        g = StarGraph()
        a = make_anchor("s1")
        g.add_anchor(a)
        schema = Schema(id="sc1", template="test template", slots={},
                        instance_ids=["s1"])
        g.schemas["sc1"] = schema

        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save(g)
            store.close()

            store2 = SQLiteStorage(tmp_path)
            loaded = store2.load()
            assert "sc1" in loaded.schemas
            assert loaded.schemas["sc1"].template == "test template"
            store2.close()

    def test_exists_false_before_save(self):
        with temp_store_dir() as td:
            path = td / "test.db"
            store = SQLiteStorage(path)
            assert not store.exists
            store.close()

    def test_close_reopens(self):
        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            assert not store.exists
            store.save(StarGraph())
            store.close()
            assert store._conn is None
            assert store.exists
            store.close()


class TestSQLiteStorageFineGrained:
    def test_save_anchor(self):
        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save_anchor("a1", {
                "text": "fine-grained anchor",
                "tags": ["test"],
                "embedding": [0.1, 0.2, 0.3],
                "replay_count": 5,
            })
            store.close()

            store2 = SQLiteStorage(tmp_path)
            graph = store2.load()
            assert "a1" in graph.anchors
            assert graph.anchors["a1"].text == "fine-grained anchor"
            assert graph.anchors["a1"].tags == ["test"]
            store2.close()

    def test_delete_anchor(self):
        g = StarGraph()
        a = make_anchor("del1", "to delete")
        g.add_anchor(a)

        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save(g)
            store.delete_anchor("del1")
            store.close()

            store2 = SQLiteStorage(tmp_path)
            loaded = store2.load()
            assert "del1" not in loaded.anchors
            store2.close()

    def test_save_edge(self):
        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save_anchor("a1", {"text": "node 1"})
            store.save_anchor("a2", {"text": "node 2"})
            store.save_edge("a1", "a2", {
                "weight": 0.7,
                "edge_type": "related",
                "co_activation_count": 3,
            })
            store.close()

            store2 = SQLiteStorage(tmp_path)
            loaded = store2.load()
            assert len(loaded.edges) == 1
            edge = list(loaded.edges.values())[0]
            assert edge.weight == 0.7
            assert edge.edge_type == "related"
            assert edge.co_activation_count == 3
            store2.close()

    def test_delete_edge(self):
        g = StarGraph()
        a0 = make_anchor("a0")
        a1 = make_anchor("a1")
        g.add_anchor(a0)
        g.add_anchor(a1)
        g.add_edge("a0", "a1")

        with temp_store_dir() as td:
            tmp_path = td / "test.db"
            store = SQLiteStorage(tmp_path)
            store.save(g)
            store.delete_edge("a0", "a1")
            store.close()

            store2 = SQLiteStorage(tmp_path)
            loaded = store2.load()
            assert len(loaded.edges) == 0
            store2.close()
