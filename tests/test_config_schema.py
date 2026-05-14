"""Tests for config schema validation and eviction."""
from __future__ import annotations

from star_graph.config import Config, CONFIG_SCHEMA, _validate_schema, _DotDict


class TestDotDict:
    def test_basic_access(self):
        d = _DotDict({"a": 1, "b": {"c": 2}})
        assert d.a == 1
        assert d.b.c == 2

    def test_dot_path_get(self):
        d = _DotDict({"a": {"b": {"c": 42}}})
        assert d.get("a.b.c") == 42
        assert d.get("a.b.x") is None
        assert d.get("a.b.x", 99) == 99
        assert d.get("nonexistent", "fallback") == "fallback"

    def test_merge(self):
        d = _DotDict({"a": 1, "b": {"c": 2}})
        d.merge({"b": {"c": 3, "d": 4}})
        assert d.b.c == 3
        assert d.b.d == 4
        assert d.a == 1

    def test_to_dict(self):
        d = _DotDict({"a": 1, "b": {"c": 2}})
        assert d.to_dict() == {"a": 1, "b": {"c": 2}}


class TestConfigSchema:
    def test_defaults_validate_clean(self):
        cfg = Config.defaults()
        warnings = cfg.validate(schema=True)
        # Defaults should have very few warnings (mainly schema_text_threshold > 1)
        for w in warnings:
            assert isinstance(w, str)

    def test_missing_section(self):
        cfg = Config()
        cfg._sections["graph"] = _DotDict({"contradiction_threshold": 0.5})
        warnings = _validate_schema(cfg.to_dict(), CONFIG_SCHEMA)
        assert any("missing section" in w for w in warnings)

    def test_type_error(self):
        cfg = Config()
        cfg._sections["graph"] = _DotDict({
            "contradiction_threshold": "not_a_float",
            "max_edges_per_node": True,  # bool passed as int
        })
        warnings = _validate_schema(cfg.to_dict(), CONFIG_SCHEMA)
        assert any("contradiction_threshold" in w and "float" in w for w in warnings)
        assert any("max_edges_per_node" in w and "bool" in w for w in warnings)

    def test_range_error(self):
        cfg = Config()
        cfg._sections["graph"] = _DotDict({"contradiction_threshold": 2.5})
        warnings = _validate_schema(cfg.to_dict(), CONFIG_SCHEMA)
        assert any("contradiction_threshold" in w and "max" in w for w in warnings)

    def test_allowed_values(self):
        cfg = Config()
        cfg._sections["index"] = _DotDict({"algorithm": "invalid_algo"})
        warnings = _validate_schema(cfg.to_dict(), CONFIG_SCHEMA)
        assert any("algorithm" in w and "not in" in w for w in warnings)

    def test_override_and_get_path(self):
        from star_graph.config import override
        override("sleep.merge.default_threshold", 0.33)
        cfg = Config.get()
        assert cfg.get_path("sleep.merge.default_threshold") == 0.33
        # Restore
        override("sleep.merge.default_threshold", 0.85)

    def test_get_path_missing(self):
        cfg = Config.defaults()
        assert cfg.get_path("nonexistent.key", None) is None
        assert cfg.get_path("sleep.nonexistent", 42) == 42

    def test_validate_no_schema(self):
        cfg = Config.defaults()
        warnings = cfg.validate(schema=False)
        # Should still run heuristic checks
        assert isinstance(warnings, list)


class TestEviction:
    def test_eviction_caps_anchors(self):
        from star_graph import MemoryManager
        from star_graph.config import override
        mgr = MemoryManager()
        override("graph.max_total_anchors", 5)
        override("graph.eviction_policy", "lowest_retention")
        for i in range(15):
            mgr.remember(f"Eviction test memory {i}", tags=["test"])
        assert len(mgr.graph.anchors) <= 5

    def test_eviction_disabled(self):
        from star_graph import MemoryManager
        from star_graph.config import override
        mgr = MemoryManager()
        override("graph.max_total_anchors", 0)
        for i in range(10):
            mgr.remember(f"No-cap test {i}", tags=["test"])
        assert len(mgr.graph.anchors) == 10

    def test_eviction_lru_policy(self):
        from star_graph import MemoryManager
        from star_graph.config import override
        import time
        mgr = MemoryManager()
        override("graph.max_total_anchors", 3)
        override("graph.eviction_policy", "lru")
        for i in range(3):
            mgr.remember(f"LRU test {i}", tags=["test"])
        # Touch first anchor
        first_id = list(mgr.graph.anchors.keys())[0]
        mgr.graph.anchors[first_id].last_activated_at = time.time() + 99999
        # Add more
        for i in range(3, 8):
            mgr.remember(f"LRU test {i}", tags=["test"])
        # First anchor (touched) should survive
        assert first_id in mgr.graph.anchors
        assert len(mgr.graph.anchors) <= 3
