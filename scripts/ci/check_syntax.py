#!/usr/bin/env python3
"""Syntax-only validation: nothing is imported, executed or installed.

* tracked (and untracked) .py files are parsed with ast.parse;
* changed .toml / .json files (vs. the PR base, or the working tree when no
  base is available) are parsed with tomllib / json;
* failures are reported as `path:line:col: message`.

Exit codes: 0 all valid, 1 invalid file(s), 2 tooling error.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

MIN_PYTHON = (3, 11)
if sys.version_info < MIN_PYTHON:
    print(
        f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
        f"found {sys.version.split()[0]}.",
        file=sys.stderr,
    )
    sys.exit(2)

TEXT_FORMATS = {".toml", ".json"}
BASE_CANDIDATES = ("origin/main", "HEAD~1")


def git(root: Path, *args: str) -> tuple[int, str]:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return result.returncode, result.stdout


def ls_files(root: Path, pattern: str = "", untracked: bool = False) -> set[str]:
    extra = ["--others", "--exclude-standard"] if untracked else []
    code, out = git(root, "ls-files", "-z", *extra, pattern)
    return {name for name in out.split("\0") if name} if code == 0 else set()


def resolve_base(root: Path, explicit: str | None) -> str | None:
    candidates = [explicit] if explicit else []
    env_base = os.environ.get("GITHUB_BASE_REF", "").strip()
    if env_base:
        candidates.append(f"origin/{env_base}")
    candidates.extend(BASE_CANDIDATES)
    for candidate in candidates:
        code, _ = git(root, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        if code == 0:
            return candidate
    return None


def changed_files(root: Path, base: str | None) -> set[str]:
    names: set[str] = set()
    for spec in ([f"{base}...HEAD"] if base else []) + ["HEAD", ""]:
        args = ["diff", "--name-only", "--diff-filter=ACMR"] + ([spec] if spec else [])
        code, out = git(root, *args)
        if code == 0:
            names.update(line.strip() for line in out.splitlines() if line.strip())
    return names | ls_files(root, untracked=True)


def validate(root: Path, files: list[str]) -> list[str]:
    failures: list[str] = []
    for relative in files:
        path = root / relative
        suffix = path.suffix.lower()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"{relative}: cannot read ({exc})")
            continue
        try:
            if suffix == ".py":
                ast.parse(text, filename=relative)
            elif suffix == ".toml":
                tomllib.loads(text)
            elif suffix == ".json":
                json.loads(text)
        except SyntaxError as exc:
            failures.append(f"{relative}:{exc.lineno}:{exc.offset}: {exc.msg}")
        except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"{relative}: invalid {suffix.lstrip('.')} ({exc})")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse tracked .py with ast (no imports) and changed .toml/.json.",
        epilog="Run from the repository root, or pass --root.",
    )
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--base", help="git base ref for changed files (default: auto)")
    parser.add_argument("--verbose", action="store_true", help="print parsed file counts")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    python_files = sorted(ls_files(root, "*.py") | ls_files(root, "*.py", untracked=True))
    base = resolve_base(root, args.base)
    data_files = sorted(
        name for name in changed_files(root, base) if Path(name).suffix.lower() in TEXT_FORMATS
    )

    failures = validate(root, python_files + data_files)
    print(f"python files parsed: {len(python_files)} (ast.parse, no imports)")
    print(f"changed base: {base or 'none (working tree only)'}")
    print(f"changed toml/json parsed: {len(data_files)}")
    if args.verbose:
        for name in data_files:
            print(f"  data: {name}")
    if failures:
        print(f"\nFAIL: {len(failures)} invalid file(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nPASS: all files parse cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
