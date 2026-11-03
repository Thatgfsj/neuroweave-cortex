#!/usr/bin/env python3
"""Bump the NWC version and keep every mirror in sync.

``star_graph/_version.py`` is the single source of truth; this script is the
only supported way to change it because the documentation mirrors must move in
the same commit.

Usage::

    python scripts/bump_version.py --check          # read-only consistency check
    python scripts/bump_version.py --bump patch     # 1.4.3 -> 1.4.4
    python scripts/bump_version.py --bump minor     # 1.4.3 -> 1.5.0
    python scripts/bump_version.py --set 2.0.0      # explicit, idempotent target
    python scripts/bump_version.py --bump patch --dry-run

``--check`` reuses :mod:`check_version` (the same logic CI runs), so the two
scripts can never drift apart. ``--bump`` updates:

* ``star_graph/_version.py`` — ``__version__`` and ``__version_info__``
* ``README.md`` — version badge
* ``README_CN.md`` — header line inside the code fence
* ``plan.md`` — 顶部 "最后更新" 行
* ``ROADMAP.md`` — "现状" heading and 当前版本 lines

Historical entries (plan.md 版本历史 rows, ROADMAP bullets) are never
rewritten. The script is standard-library only and idempotent: a file that
already contains the target version is left untouched, and ``--set`` with the
current version changes nothing.

Exit codes: 0 updated/consistent, 1 check failed, 2 unreadable source.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import check_version  # noqa: E402 — sibling module, needs the sys.path bootstrap above

VERSION_MODULE = check_version.VERSION_MODULE
SEMVER = check_version.SEMVER

MIRROR_FILES = ("README.md", "README_CN.md", "plan.md", "ROADMAP.md")


def read_version(root: Path) -> str:
    """Read the current version from the single source module."""
    return check_version.extract_source_version(root)[0]


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse ``X.Y.Z``; raise ``ValueError`` on anything else."""
    if not re.fullmatch(SEMVER, version):
        raise ValueError(f"invalid version {version!r}; expected MAJOR.MINOR.PATCH")
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def bump_version(version: str, part: str) -> str:
    """Return the next version after bumping ``part`` (major/minor/patch)."""
    major, minor, patch = parse_version(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump part {part!r}; expected major, minor or patch")


def _rewrite_version_module(old: str, new: str) -> str:
    def transform(text: str) -> str:
        text = text.replace(f'__version__ = "{old}"', f'__version__ = "{new}"')
        major, minor, patch = parse_version(new)
        text = re.sub(
            r"(?m)^__version_info__\s*=\s*\([^)]*\)",
            f"__version_info__ = ({major}, {minor}, {patch})",
            text,
            count=1,
        )
        return text

    return transform


def _rewrite_readme(old: str, new: str) -> str:
    def transform(text: str) -> str:
        return text.replace(f"badge/version-{old}", f"badge/version-{new}")

    return transform


def _rewrite_readme_cn(old: str, new: str) -> str:
    def transform(text: str) -> str:
        return re.sub(rf"(?m)^v{re.escape(old)}(\s*\|)", rf"v{new}\1", text)

    return transform


def _rewrite_plan(old: str, new: str, today: _dt.date) -> str:
    def transform(text: str) -> str:
        pattern = re.compile(
            rf"(?m)^(> 最后更新: )\d{{4}}-\d{{2}}-\d{{2}}( \| \*\*v){re.escape(old)}(\*\*.*)$"
        )
        match = pattern.search(text)
        if not match:
            return text
        replacement = f"{match.group(1)}{today.isoformat()}{match.group(2)}{new}{match.group(3)}"
        return text[: match.start()] + replacement + text[match.end():]

    return transform


def _rewrite_roadmap(old: str, new: str, today: _dt.date) -> str:
    def transform(text: str) -> str:
        heading = re.compile(rf"(?m)^(##\s*现状（v){re.escape(old)}( — )\d{{4}}-\d{{2}}(）)$")
        text = heading.sub(rf"\g<1>{new}\g<2>{today:%Y-%m}\g<3>", text)
        text = text.replace(f"当前版本 v{old}", f"当前版本 v{new}")
        text = text.replace(f'star_graph.__version__ = "{old}"', f'star_graph.__version__ = "{new}"')
        return text

    return transform


def _transformers(old: str, new: str, today: _dt.date) -> dict[str, object]:
    return {
        VERSION_MODULE: _rewrite_version_module(old, new),
        "README.md": _rewrite_readme(old, new),
        "README_CN.md": _rewrite_readme_cn(old, new),
        "plan.md": _rewrite_plan(old, new, today),
        "ROADMAP.md": _rewrite_roadmap(old, new, today),
    }


def apply_version(root: Path, new: str, dry_run: bool = False) -> tuple[list[str], list[str]]:
    """Rewrite every mirror to ``new``. Returns (changed, unchanged) paths."""
    old = read_version(root)
    if old == new:
        return [], [VERSION_MODULE, *MIRROR_FILES]

    changed: list[str] = []
    unchanged: list[str] = []
    today = _dt.date.today()
    for relative, transform in _transformers(old, new, today).items():
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise check_version.MissingSourceError(f"cannot read {path}: {exc}") from exc
        updated = transform(text)
        if updated == text:
            unchanged.append(relative)
            continue
        if not dry_run:
            path.write_text(updated, encoding="utf-8")
        changed.append(relative)
    return changed, unchanged


def _print_summary(old: str, new: str, changed: list[str], unchanged: list[str], dry_run: bool) -> None:
    verb = "would update" if dry_run else "updated"
    print(f"version: {old} -> {new}")
    for relative in changed:
        print(f"  {verb} {relative}")
    for relative in unchanged:
        print(f"  unchanged {relative} (already at {new})")


def _run_check(root: Path, verbose: bool) -> int:
    argv = ["--root", str(root)]
    if verbose:
        argv.append("--verbose")
    return check_version.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check or bump the NWC version (single source: star_graph/_version.py).",
        epilog="Updates star_graph/_version.py plus README/README_CN/plan.md/ROADMAP mirrors.",
    )
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--verbose", action="store_true", help="print file:line details when checking")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without writing")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify consistency and exit (read-only)")
    mode.add_argument("--bump", choices=("major", "minor", "patch"), help="increment the given part")
    mode.add_argument("--set", metavar="X.Y.Z", help="set an explicit version (idempotent)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if args.check:
        return _run_check(root, args.verbose)

    try:
        old = read_version(root)
        new = bump_version(old, args.bump) if args.bump else args.set
        if new is None:
            raise ValueError("no target version given")
        parse_version(new)
        changed, unchanged = apply_version(root, new, dry_run=args.dry_run)
    except check_version.MissingSourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_summary(old, new, changed, unchanged, args.dry_run)
    if args.dry_run:
        print("\ndry-run: nothing written.")
        return 0

    print()
    status = _run_check(root, args.verbose)
    if status == 0:
        print(f"\nPASS: version is now {new} everywhere.")
    return status


if __name__ == "__main__":
    sys.exit(main())
