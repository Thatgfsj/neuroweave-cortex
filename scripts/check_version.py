#!/usr/bin/env python3
"""Version consistency check for the four authoritative version sources.

All must match pyproject.toml [project].version:
pyproject.toml, star_graph/__init__.py, nwc/__init__.py, README.md badge.

TODO(#14): star_graph/defaults.yaml (1.3.5) and the server modules
(nwc/api/server.py, star_graph/contrib/mcp_server.py, both 1.4.0) are known
drift and deliberately NOT checked here until #14 unifies the version source.

Exit codes: 0 consistent, 1 mismatch, 2 missing/unreadable source.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SOURCES = (
    ("pyproject.toml", re.compile(r'(?m)^version\s*=\s*"([^"]+)"')),
    ("star_graph/__init__.py", re.compile(r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']')),
    ("nwc/__init__.py", re.compile(r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']')),
    ("README.md", re.compile(r"badge/version-([0-9]+\.[0-9]+(?:\.[0-9]+)*)")),
)
REFERENCE = "pyproject.toml"


def read_version(root: Path, relative: str, pattern: re.Pattern[str]) -> tuple[str | None, str]:
    path = root / relative
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    match = pattern.search(text)
    if not match:
        return None, f"no version match in {relative} (pattern: {pattern.pattern})"
    line = text[: match.start()].count("\n") + 1
    return match.group(1), f"{relative}:{line}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that the four authoritative version sources agree.",
        epilog="Run from the repository root, or pass --root.",
    )
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--verbose", action="store_true", help="print matched paths and line numbers")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    results: list[tuple[str, str | None, str]] = []
    for relative, pattern in SOURCES:
        version, where = read_version(root, relative, pattern)
        results.append((relative, version, where))

    expected = next((version for name, version, _ in results if name == REFERENCE), None)
    if expected is None:
        print("ERROR: cannot determine reference version from pyproject.toml", file=sys.stderr)
        for name, _, where in results:
            print(f"  {name}: {where}", file=sys.stderr)
        return 2

    print(f"reference: {REFERENCE} -> {expected}")
    failures = 0
    for name, version, where in results:
        source = where if args.verbose else name
        if version is None:
            print(f"  FAIL  {source}: {where}")
            failures += 1
        elif version != expected:
            print(f"  FAIL  {source}: {version} (expected {expected})")
            failures += 1
        else:
            print(f"  ok    {source}: {version}")

    if failures:
        print(
            f"\nFAIL: {failures} source(s) disagree with {REFERENCE}. "
            "Update every version source in the same commit (see #5).",
            file=sys.stderr,
        )
        return 1

    print(f"\nPASS: all {len(results)} version sources match {expected}.")
    print("note: defaults.yaml and server modules are checked by #14, not here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
