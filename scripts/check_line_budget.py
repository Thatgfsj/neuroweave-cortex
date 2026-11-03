#!/usr/bin/env python3
"""Line budget gate: no tracked Python file may exceed 1000 lines.

Files already over the limit are pinned in scripts/ci/line_budget_allowlist.txt
(one `path  # issue #N` per line) as a ratchet: each must be split by its
tracked issue, and entries should disappear over time. A file over the limit
that is NOT allowlisted fails the check.

Exit codes: 0 within budget, 1 violation, 2 tooling error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAX_LINES = 1000
DEFAULT_ALLOWLIST = "scripts/ci/line_budget_allowlist.txt"


def git_tracked_python(root: Path) -> list[str]:
    try:
        result = subprocess.run(["git", "ls-files", "-z", "*.py"], cwd=root,
                                capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"git ls-files failed: {exc}") from exc
    return sorted(item.decode("utf-8", "replace") for item in result.stdout.split(b"\0") if item)


def count_lines(path: Path) -> int:
    try:
        data = path.read_bytes()
    except OSError:
        return 0
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def read_allowlist(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.is_file():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries[line] = raw.split("#", 1)[1].strip() if "#" in raw else ""
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when a tracked .py file exceeds the line budget (ratchet).",
        epilog="Allowlist lines look like: star_graph/memory_core/graph.py  # issue #8",
    )
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--max-lines", type=int, default=MAX_LINES,
                        help=f"per-file line limit (default {MAX_LINES})")
    parser.add_argument("--allowlist", default=DEFAULT_ALLOWLIST,
                        help=f"allowlist path (default {DEFAULT_ALLOWLIST})")
    parser.add_argument("--verbose", action="store_true", help="print allowlist notes")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        files = git_tracked_python(root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    allowlist = read_allowlist(root / args.allowlist)
    over = sorted(
        ((name, count_lines(root / name)) for name in files
         if count_lines(root / name) > args.max_lines),
        key=lambda item: -item[1],
    )
    violations = [item for item in over if item[0] not in allowlist]

    print(f"tracked python files: {len(files)}; limit: {args.max_lines} lines/file")
    for name, lines in over:
        state = "allowlisted" if name in allowlist else "VIOLATION"
        print(f"  {state:<11} {lines:>5}  {name}")
        if args.verbose and name in allowlist:
            print(f"              ratchet issue: {allowlist[name] or 'none recorded'}")
    stale = sorted(set(allowlist) - {name for name, _ in over})
    if stale:
        print("ratchet note: entries no longer over the limit, remove them:")
        for name in stale:
            print(f"  - {name}")

    print(f"ratchet: {len(over)} file(s) over budget, {len(over) - len(violations)} allowlisted, "
          f"{len(violations)} violation(s); {len(allowlist)} entry(ies) in allowlist.")
    if violations:
        print("FAIL: split these files (or add a Lead-approved allowlist entry): "
              + ", ".join(name for name, _ in violations), file=sys.stderr)
        return 1
    print("PASS: line budget respected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
