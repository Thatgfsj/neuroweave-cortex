#!/usr/bin/env python3
"""Repository statistics for NWC docs — the single source of quoted counts.

Usage:
    python scripts/repo_stats.py            # human-readable
    python scripts/repo_stats.py --json     # machine-readable

Counting rules (统计口径), so every doc quotes the same numbers:

* Python modules: files matching ``star_graph/**/*.py`` + ``nwc/**/*.py``,
  reported as total / regular modules / ``__init__.py`` files, plus a
  per-package breakdown.
* Test files: test modules under ``tests/**/*.py`` (``__init__.py`` excluded).
* Test functions: occurrences of ``def test_`` under ``tests/`` (grep-style,
  class methods included).
* Python lines: physical lines in package files, test files, and both.
* Version: literal ``__version__`` in ``star_graph/_version.py`` (parsed with
  ``ast``; the package is never imported).

Standard library only. Docs must cite this script instead of hardcoding counts
(see docs/engineering/DOC_POLICY.md §2).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

PACKAGES = ("star_graph", "nwc")
TESTS = "tests"
VERSION_FILE = "star_graph/_version.py"
TEST_DEF = re.compile(r"\bdef\s+test_")


def _tracked_py_files(root: Path, relative: str) -> list[Path]:
    """Return git-tracked ``.py`` files under *relative* (same scope as CI)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", relative],
            check=True, capture_output=True, text=True,
        ).stdout
        tracked = [line for line in out.split("\0") if line.endswith(".py")]
        if tracked:
            return sorted(root / line for line in tracked)
    except (OSError, subprocess.CalledProcessError):
        pass
    # Fallback for non-git checkouts / tarballs.
    base = root / relative
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _py_files(root: Path, relative: str) -> list[Path]:
    return _tracked_py_files(root, relative)


def _count_lines(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        data = path.read_bytes()
        total += data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    return total


def _read_version(root: Path) -> str:
    text = (root / VERSION_FILE).read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        targets = getattr(node, "targets", [])
        if any(getattr(t, "id", None) == "__version__" for t in targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise SystemExit(f"{VERSION_FILE}: no literal __version__ assignment")


def collect(root: Path) -> dict:
    """Collect all statistics for ``root`` as a JSON-serialisable dict."""
    packages = {name: _py_files(root, name) for name in PACKAGES}
    package_files = [p for name in PACKAGES for p in packages[name]]
    inits = [p for p in package_files if p.name == "__init__.py"]
    test_files = [p for p in _py_files(root, TESTS) if p.name != "__init__.py"]
    test_functions = sum(
        len(TEST_DEF.findall(p.read_text(encoding="utf-8"))) for p in test_files
    )
    package_lines = _count_lines(package_files)
    test_lines = _count_lines(test_files)
    return {
        "version": _read_version(root),
        "python_modules": {
            "total": len(package_files),
            "regular": len(package_files) - len(inits),
            "init": len(inits),
            "by_package": {name: len(files) for name, files in packages.items()},
        },
        "test_files": len(test_files),
        "test_functions": test_functions,
        "python_lines": {
            "package": package_lines,
            "tests": test_lines,
            "total": package_lines + test_lines,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report NWC repository statistics.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    stats = collect(root)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    modules = stats["python_modules"]
    lines = stats["python_lines"]
    by_pkg = ", ".join(f"{k} {v}" for k, v in modules["by_package"].items())
    print(f"NWC repository statistics (root: {root})")
    print(f"  version          : {stats['version']}")
    print(f"  python modules   : {modules['total']} ({by_pkg})")
    print(f"                     {modules['regular']} regular + {modules['init']} __init__.py")
    print(f"  test files       : {stats['test_files']}")
    print(f"  test functions   : {stats['test_functions']}  (def test_)")
    print(f"  python lines     : {lines['total']} ({lines['package']} package + {lines['tests']} tests)")
    print("\nDocs quote these numbers with this script as the source: python scripts/repo_stats.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
