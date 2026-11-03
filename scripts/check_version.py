#!/usr/bin/env python3
"""Strict version-consistency check for NeuroWeave Cortex.

``star_graph/_version.py`` is the single source of truth. Every other place
that mentions the version must either import it or be a documentation mirror
kept in sync by ``scripts/bump_version.py``.

Checked:

1. ``star_graph/_version.py`` defines a literal ``__version__``, imports
   nothing, and no other shipped module hardcodes ``__version__ = "..."``.
2. ``pyproject.toml`` declares ``dynamic = ["version"]`` and wires
   ``[tool.setuptools.dynamic].version.attr`` to
   ``star_graph._version.__version__``; it never carries a static
   ``[project].version``.
3. ``star_graph/__init__.py`` and ``nwc/__init__.py`` re-export the value from
   ``_version`` instead of hardcoding it.
4. Documentation mirrors (README badge, README_CN header, plan.md header,
   ROADMAP 现状) show the current version.
5. No legacy hardcoded versions: ``star_graph/defaults.yaml`` has no top-level
   ``version:`` key and the HTTP/MCP server modules import ``__version__``
   instead of embedding a quoted ``x.y.z`` literal.
6. The release workflow (``.github/workflows/verify.yml``) reads the tag
   version from ``star_graph/_version.py`` — never from a static
   ``[project].version`` that no longer exists — compares the tag on ``v*``
   pushes, and still reports the version on ``workflow_dispatch``.

This module uses only the standard library and never imports ``star_graph``
(``pyproject.toml`` is parsed with ``tomllib``; sources are parsed with
``ast``), so it is safe in the no-build PR checks.

Exit codes: 0 all checks pass, 1 a check failed, 2 a critical source is
unreadable.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

SEMVER = r"\d+\.\d+\.\d+"

VERSION_MODULE = "star_graph/_version.py"
PYPROJECT = "pyproject.toml"
VERIFY_WORKFLOW = ".github/workflows/verify.yml"
EXPECTED_ATTR = "star_graph._version.__version__"

README_BADGE = re.compile(rf"badge/version-({SEMVER})")
README_CN_HEADER = re.compile(rf"(?m)^v({SEMVER})\s*\|")
PLAN_HEADER = re.compile(rf"(?m)^> 最后更新:.*\|\s*\*\*v({SEMVER})\*\*")
ROADMAP_HEADER = re.compile(rf"(?m)^##\s*现状（v({SEMVER})")
ROADMAP_CURRENT = re.compile(rf"当前版本\s*v({SEMVER})")
YAML_TOP_LEVEL_VERSION = re.compile(r"(?m)^version\s*:")
QUOTED_SEMVER = re.compile(rf"""["']{SEMVER}["']""")

# `from ._version import __version__` / `from star_graph._version import __version__`
VERSION_IMPORT = re.compile(r"from\s+(?:\.+|star_graph\.)_version\s+import\s+.*\b__version__")

DOC_MIRRORS = (
    ("README.md", "README badge", README_BADGE),
    ("README_CN.md", "README_CN header", README_CN_HEADER),
    ("plan.md", "plan.md header", PLAN_HEADER),
    ("ROADMAP.md", "ROADMAP 现状 heading", ROADMAP_HEADER),
)

SERVER_MODULES = (
    "nwc/api/server.py",
    "star_graph/contrib/mcp_server.py",
    "nwc/mcp/server.py",
    "star_graph/cortex_api/server.py",
)

PACKAGES = ("star_graph", "nwc")


class MissingSourceError(RuntimeError):
    """A critical version source is missing or unreadable."""


@dataclass(frozen=True)
class Result:
    """One check outcome."""

    name: str
    ok: bool
    detail: str


def _read(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MissingSourceError(f"cannot read {path}: {exc}") from exc


def _line_of(text: str, position: int) -> int:
    return text[:position].count("\n") + 1


def extract_source_version(root: Path) -> tuple[str, str]:
    """Return ``(__version__, "path:line")`` from the single source module.

    Raises ``MissingSourceError`` if the file is missing/unreadable and
    ``ValueError`` if it is malformed (imports, non-literal, duplicated).
    """
    text = _read(root, VERSION_MODULE)
    tree = ast.parse(text, filename=VERSION_MODULE)
    found: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(
                f"{VERSION_MODULE}:{node.lineno} imports modules; the version source must be import-free"
            )
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ValueError(f"{VERSION_MODULE}:{node.lineno} __version__ is not a string literal")
            found.append((node.value.value, node.lineno))
    if not found:
        raise ValueError(f"{VERSION_MODULE}: no module-level __version__ assignment")
    if len(found) > 1:
        raise ValueError(f"{VERSION_MODULE}: __version__ assigned {len(found)} times")
    return found[0][0], f"{VERSION_MODULE}:{found[0][1]}"


def _check_pyproject(root: Path) -> Result:
    try:
        data = tomllib.loads(_read(root, PYPROJECT))
    except MissingSourceError:
        raise
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise MissingSourceError(f"cannot parse {PYPROJECT}: {exc}") from exc

    project = data.get("project", {})
    attr = (
        data.get("tool", {})
        .get("setuptools", {})
        .get("dynamic", {})
        .get("version", {})
        .get("attr")
    )
    problems: list[str] = []
    if "version" in project:
        problems.append("[project] still pins a static version (must be dynamic only)")
    if project.get("dynamic") != ["version"]:
        problems.append(f'[project].dynamic is {project.get("dynamic")!r}, expected ["version"]')
    if attr != EXPECTED_ATTR:
        problems.append(f"[tool.setuptools.dynamic].version.attr is {attr!r}, expected {EXPECTED_ATTR!r}")
    if problems:
        return Result(PYPROJECT, False, "; ".join(problems))
    return Result(PYPROJECT, True, f"dynamic version -> {EXPECTED_ATTR}")


def _check_init(root: Path, relative: str) -> Result:
    try:
        text = _read(root, relative)
    except MissingSourceError as exc:
        return Result(relative, False, str(exc))
    literal = QUOTED_SEMVER.search(text)
    if literal:
        line = _line_of(text, literal.start())
        return Result(relative, False, f"{relative}:{line} hardcodes {literal.group(0)} (import from _version instead)")
    match = VERSION_IMPORT.search(text)
    if not match:
        return Result(relative, False, "does not import __version__ from _version")
    return Result(relative, True, f"{relative}:{_line_of(text, match.start())} imports _version.__version__")


def _check_doc(root: Path, relative: str, label: str, pattern: re.Pattern[str], expected: str) -> Result:
    try:
        text = _read(root, relative)
    except MissingSourceError as exc:
        return Result(label, False, str(exc))
    match = pattern.search(text)
    if not match:
        return Result(label, False, f"{relative}: no {label} match (pattern: {pattern.pattern})")
    found = match.group(1)
    line = _line_of(text, match.start())
    if found != expected:
        return Result(label, False, f"{relative}:{line} says {found}, expected {expected}")
    return Result(label, True, f"{relative}:{line} = {found}")


def _check_defaults_yaml(root: Path) -> Result:
    try:
        text = _read(root, "star_graph/defaults.yaml")
    except MissingSourceError as exc:
        return Result("defaults.yaml", False, str(exc))
    match = YAML_TOP_LEVEL_VERSION.search(text)
    if match:
        line = _line_of(text, match.start())
        return Result(
            "defaults.yaml",
            False,
            f"star_graph/defaults.yaml:{line} still defines a top-level version field (drop it; the version lives in _version.py)",
        )
    return Result("defaults.yaml", True, "no top-level version field")


def _check_server(root: Path, relative: str) -> Result:
    try:
        text = _read(root, relative)
    except MissingSourceError as exc:
        return Result(relative, False, str(exc))
    literal = QUOTED_SEMVER.search(text)
    if literal:
        line = _line_of(text, literal.start())
        return Result(relative, False, f"{relative}:{line} hardcodes {literal.group(0)}; import __version__ instead")
    match = VERSION_IMPORT.search(text)
    if not match:
        return Result(relative, False, "does not import __version__ from _version")
    return Result(relative, True, f"{relative}:{_line_of(text, match.start())} imports _version.__version__")


def _check_verify_workflow(root: Path) -> Result:
    """The release workflow must take the tag version from the single source.

    ``pyproject.toml`` is dynamic, so ``[project].version`` no longer exists;
    the old release job read it and raised ``KeyError`` on every ``v*`` tag.
    """
    try:
        text = _read(root, VERIFY_WORKFLOW)
    except MissingSourceError as exc:
        return Result(VERIFY_WORKFLOW, False, str(exc))
    problems: list[str] = []
    if not re.search(r"star_graph[./]_version", text):
        problems.append("does not read star_graph/_version.py")
    if "tomllib" in text or re.search(r"""\[["']version["']\]""", text):
        problems.append("parses a static [project].version from pyproject.toml (the version is dynamic)")
    if "refs/tags/" not in text or "github.ref_name" not in text:
        problems.append("does not compare the pushed tag against the source version")
    if re.search(r"if:\s*startsWith\(github\.ref", text):
        problems.append("tag step is gated to tag refs, so workflow_dispatch never reports the version")
    if problems:
        return Result(VERIFY_WORKFLOW, False, "; ".join(problems))
    return Result(VERIFY_WORKFLOW, True, "tag check reads star_graph/_version.py")


def _check_unique_source(root: Path) -> Result:
    offenders: list[str] = []
    for package in PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            if relative == VERSION_MODULE:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if re.match(r"^__version__\s*=", line):
                    offenders.append(f"{relative}:{lineno}")
    if offenders:
        return Result(
            "unique version source",
            False,
            "hardcoded __version__ outside star_graph/_version.py: " + ", ".join(offenders),
        )
    return Result("unique version source", True, "only star_graph/_version.py defines __version__")


def run_checks(root: Path) -> list[Result]:
    """Run every check and return ordered results.

    Raises ``MissingSourceError`` when the single source or pyproject.toml cannot
    be read/parsed; those are fatal (exit 2) rather than reportable mismatches.
    """
    root = Path(root)
    results: list[Result] = []

    full_version, where = extract_source_version(root)
    results.append(Result(VERSION_MODULE, True, f"{where} = {full_version}"))
    results.append(_check_unique_source(root))
    results.append(_check_pyproject(root))
    results.append(_check_init(root, "star_graph/__init__.py"))
    results.append(_check_init(root, "nwc/__init__.py"))

    for relative, label, pattern in DOC_MIRRORS:
        results.append(_check_doc(root, relative, label, pattern, full_version))

    results.append(_check_doc(root, "ROADMAP.md", "ROADMAP current version", ROADMAP_CURRENT, full_version))
    results.append(_check_defaults_yaml(root))
    for relative in SERVER_MODULES:
        results.append(_check_server(root, relative))
    results.append(_check_verify_workflow(root))
    return results


def format_results(results: list[Result], verbose: bool = False) -> str:
    lines = []
    for result in results:
        status = "ok  " if result.ok else "FAIL"
        if verbose and result.ok:
            lines.append(f"  {status}  {result.detail}")
        elif verbose:
            lines.append(f"  {status}  {result.name}: {result.detail}")
        else:
            lines.append(f"  {status}  {result.name}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that every NWC version reference matches star_graph/_version.py.",
        epilog="Run from the repository root, or pass --root.",
    )
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--verbose", action="store_true", help="print file:line details for every check")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        results = run_checks(root)
        version, _ = extract_source_version(root)
    except MissingSourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"version source: {VERSION_MODULE} -> {version}")
    print(format_results(results, verbose=args.verbose))

    failures = [result for result in results if not result.ok]
    if failures:
        print(
            f"\nFAIL: {len(failures)} check(s) disagree with {VERSION_MODULE} ({version}). "
            "Use scripts/bump_version.py --bump or update the mirrors in the same commit.",
            file=sys.stderr,
        )
        return 1

    print(f"\nPASS: {len(results)} checks agree on {version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
