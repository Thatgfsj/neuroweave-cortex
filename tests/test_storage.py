"""Tests for JSON persistence layer."""

import json
import tempfile
from pathlib import Path

import pytest

from star_graph.anchor import Anchor
from star_graph.graph import StarGraph
from star_graph.storage import JSONStorage, Storage, DEFAULT_PATH


class TestJSONStorage:
    def test_default_path_is_home_dir(self):
        expected = Path.home() / ".star_graph" / "memory.json"
        assert DEFAULT_PATH == expected

    def test_save_and_load_roundtrip(self):
        g = StarGraph()
        a = Anchor.create(text="Hello world", tags=["test"])
        a.id = "test1"
        g.add_anchor(a)
        g.add_edge("test1", "test1")  # ignored by graph

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            store = JSONStorage(tmp_path)
            store.save(g)
            assert store.exists

            loaded = store.load()
            assert "test1" in loaded.anchors
            assert loaded.anchors["test1"].text == "Hello world"
            assert loaded.anchors["test1"].tags == ["test"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nonexistent.json"
            store = JSONStorage(path)
            graph = store.load()
            assert len(graph.anchors) == 0

    def test_load_multiple_anchors_with_edges(self):
        g = StarGraph()
        for i in range(5):
            a = Anchor.create(text=f"Anchor {i}")
            a.id = f"a{i}"
            g.add_anchor(a)
        g.add_edge("a0", "a1", weight=0.8, edge_type="causes")
        g.add_edge("a1", "a2", weight=0.5, edge_type="related")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            store = JSONStorage(tmp_path)
            store.save(g)

            loaded = store.load()
            assert len(loaded.anchors) == 5
            assert len(loaded.edges) == 2
            assert ("a0", "a1") in loaded.edges or ("a1", "a0") in loaded.edges
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_and_load_with_schemas(self):
        from star_graph.graph import Schema
        g = StarGraph()
        a = Anchor.create(text="Schema test")
        a.id = "s1"
        g.add_anchor(a)
        schema = Schema(id="sc1", template="test template", slots={}, instance_ids=["s1"])
        g.schemas["sc1"] = schema

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            store = JSONStorage(tmp_path)
            store.save(g)

            loaded = store.load()
            assert "sc1" in loaded.schemas
            assert loaded.schemas["sc1"].template == "test template"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exists_property(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.json"
            store = JSONStorage(path)
            assert not store.exists
            store.save(StarGraph())
            assert store.exists

    def test_storage_alias(self):
        assert Storage is JSONStorage

    def test_can_load_v1_format_without_oscillator(self):
        v1_data = {
            "version": 1,
            "saved_at": 1700000000.0,
            "anchors": [
                {
                    "id": "old1",
                    "text": "old anchor",
                    "vector": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    "embedding": None,
                    "prediction": None,
                    "oscillator": {},
                    "created_at": 1700000000.0,
                    "last_activated_at": 1700000000.0,
                    "source_session": "old",
                    "tags": ["legacy"],
                    "schema_ref": None,
                    "replay_count": 0,
                }
            ],
            "edges": [],
            "schemas": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            json.dump(v1_data, f)
            tmp_path = f.name

        try:
            store = JSONStorage(tmp_path)
            graph = store.load()
            assert "old1" in graph.anchors
        finally:
            Path(tmp_path).unlink(missing_ok=True)
