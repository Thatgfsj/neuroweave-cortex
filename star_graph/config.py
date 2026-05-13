"""Centralized YAML config with dot-notation access and programmatic override.

Usage:
    from star_graph.config import config, Config

    # Use defaults
    threshold = config.sleep.merge.default_threshold  # 0.85

    # Override programmatically
    config.sleep.merge.default_threshold = 0.75

    # Or load from a custom YAML
    cfg = Config.from_yaml("my_params.yaml")
    cfg.apply_globally()
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any


class _DotDict:
    """Recursive dot-notation dict: cfg.sleep.swr.valence_weight."""

    def __init__(self, data: dict | None = None):
        object.__setattr__(self, "_data", {})
        if data:
            for k, v in data.items():
                if isinstance(v, dict):
                    self._data[k] = _DotDict(v)
                else:
                    self._data[k] = v

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        # Fall back to class methods (e.g., merge, to_dict, validate)
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            raise AttributeError(f"No config key '{name}'. Available: {list(data.keys())}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, dict):
                self._data[name] = _DotDict(value)
            else:
                self._data[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, name: str, default: Any = None) -> Any:
        """Get a config value by dotted path. Supports 'sleep.swr.tau' traversal.

        If name contains no dots, behaves like dict.get(name, default).
        If a dotted path is given, walks nested _DotDict objects.
        Returns default if any key in the path is missing.
        """
        if '.' not in name:
            return self._data.get(name, default)

        parts = name.split(".")
        current = self
        for part in parts:
            if not isinstance(current, _DotDict) or part not in current._data:
                return default
            current = current._data[part]
        return current

    def to_dict(self) -> dict:
        result = {}
        for k, v in self._data.items():
            result[k] = v.to_dict() if isinstance(v, _DotDict) else v
        return result

    def merge(self, other: _DotDict | dict) -> None:
        """Recursively merge another config, overriding leaf values."""
        if isinstance(other, dict):
            other = _DotDict(other)
        for k, v in other._data.items():
            if k in self._data and isinstance(self._data[k], _DotDict) and isinstance(v, _DotDict):
                self._data[k].merge(v)
            elif isinstance(v, _DotDict):
                self._data[k] = _DotDict(v.to_dict())
            else:
                self._data[k] = v

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def validate(self, schema: _DotDict | None = None) -> list[str]:
        """Validate config values are in reasonable ranges. Returns list of warnings."""
        warnings = []
        data = self.to_dict()
        _check_ranges(data, "", warnings)
        return warnings


def _check_ranges(data: dict, path: str, warnings: list[str]) -> None:
    """Recursively check that numeric values are in [0, 1] or otherwise reasonable."""
    for k, v in data.items():
        full = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _check_ranges(v, full, warnings)
        elif isinstance(v, (int, float)):
            # Probabilities/weights should be in [0, 1]
            if "weight" in k or "factor" in k or "boost" in k or "penalty" in k:
                if not 0.0 <= v <= 5.0:
                    warnings.append(f"{full}={v} outside expected [0,5] for weight/factor")
            elif "threshold" in k or "prob" in k:
                if not 0.0 <= v <= 1.0:
                    warnings.append(f"{full}={v} outside expected [0,1] for threshold/probability")
            elif "fraction" in k:
                if not 0.0 <= v <= 1.0:
                    warnings.append(f"{full}={v} outside expected [0,1] for fraction")


class Config:
    """Global config singleton with YAML loading and programmatic override."""

    _instance: Config | None = None

    def __init__(self):
        self._sections: dict[str, _DotDict] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls()
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                cfg._sections[section_name] = _DotDict(section_data)
        return cfg

    @classmethod
    def defaults(cls) -> Config:
        """Load the built-in defaults.yaml."""
        defaults_path = Path(__file__).parent / "defaults.yaml"
        return cls.from_yaml(defaults_path)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name in self._sections:
            return self._sections[name]
        raise AttributeError(f"No config section '{name}'. Available: {list(self._sections.keys())}")

    def section(self, name: str) -> _DotDict | None:
        return self._sections.get(name)

    def apply_globally(self) -> None:
        """Set this config as the global singleton."""
        Config._instance = self

    @classmethod
    def get(cls) -> Config:
        """Get the global config singleton, loading defaults if needed."""
        if cls._instance is None:
            cls._instance = cls.defaults()
        return cls._instance

    def merge(self, other: Config | dict) -> None:
        """Merge another config on top of this one."""
        if isinstance(other, Config):
            for name, section in other._sections.items():
                if name in self._sections:
                    self._sections[name].merge(section)
                else:
                    self._sections[name] = section
        elif isinstance(other, dict):
            for name, section_data in other.items():
                section = _DotDict(section_data) if isinstance(section_data, dict) else section_data
                if name in self._sections and isinstance(section, _DotDict):
                    self._sections[name].merge(section)
                else:
                    self._sections[name] = section

    def get_path(self, dotted_path: str, default: Any = None) -> Any:
        """Get a config value by dotted path: cfg.get_path('exact_cache.auto_harvest', True).

        Splits on '.', looks up the section, then walks nested _DotDict keys.
        Returns default if any segment is missing.
        """
        parts = dotted_path.split(".")
        if len(parts) < 2:
            return default
        section_name = parts[0]
        section = self._sections.get(section_name)
        if section is None:
            return default
        return section.get(".".join(parts[1:]), default)

    def override(self, dotted_path: str, value: Any) -> None:
        """Set a single config value by dotted path. E.g., 'sleep.swr.tau=30.0'."""
        parts = dotted_path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Override path must have at least section.key: {dotted_path}")
        section_name = parts[0]
        if section_name not in self._sections:
            self._sections[section_name] = _DotDict({})
        current = self._sections[section_name]
        for part in parts[1:-1]:
            if part not in current:
                current._data[part] = _DotDict({})
            current = current._data[part]
        current._data[parts[-1]] = value

    def to_dict(self) -> dict:
        return {name: section.to_dict() for name, section in self._sections.items()}

    def validate(self) -> list[str]:
        warnings = []
        for name, section in self._sections.items():
            section_warnings = section.validate()
            for w in section_warnings:
                warnings.append(f"[{name}]{w}")
        return warnings

    def __repr__(self) -> str:
        return f"Config(sections={list(self._sections.keys())})"


# Module-level convenience
config: Config = Config.get()


def reload_defaults() -> Config:
    """Reload defaults, discarding any overrides."""
    cfg = Config.defaults()
    cfg.apply_globally()
    return cfg


def override(dotted_path: str, value: Any) -> None:
    """Override a single config value globally. E.g., override('sleep.merge.default_threshold', 0.75)."""
    Config.get().override(dotted_path, value)


def load_config(path: str | Path) -> Config:
    """Load a YAML config and set it as global."""
    cfg = Config.from_yaml(path)
    cfg.apply_globally()
    return cfg
