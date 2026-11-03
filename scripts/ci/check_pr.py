#!/usr/bin/env python3
"""PR guard: issue linkage + change size (pure stdlib: no build, no installs).

A. Linkage -- the visible PR body must contain `Closes|Fixes|Resolves #N`; that
   issue must exist, be created no later than the PR, and carry a type label
   (bug/feat/docs/ci/refactor/chore/governance/spec). `Refs #N` is advisory
   only: it is reported as a warning, never as proof of linkage.

   Before matching, the body is stripped of fenced code blocks, inline code,
   HTML comments and blockquote lines, so "example" text in a PR template does
   not create phantom links.

B. Size -- additions+deletions in source files must stay below 1000 lines.
   Source = CODE_EXTS (code only; docs/config/data such as .md/.json/.csv/
   .yml/.toml are listed but never counted). The `size/exception` label
   overrides the limit and raises a loud warning.

Usage:
  PR_NUMBER=12 GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/name \
      python3 scripts/ci/check_pr.py [--verbose]
  python3 scripts/ci/check_pr.py --self-test    # offline, no network

Env: GITHUB_TOKEN, GITHUB_REPOSITORY, PR_NUMBER, GITHUB_EVENT_PATH,
GITHUB_STEP_SUMMARY (optional). Requires Python >= 3.11.
TODO(#14): version drift lives in scripts/check_version.py, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info < (3, 11):
    print(f"ERROR: Python 3.11+ required; found {sys.version.split()[0]}.", file=sys.stderr)
    sys.exit(2)

API_ROOT = "https://api.github.com"
STRONG_REF_RE = re.compile(
    r"\b(?:closes|fixes|resolves)\b\s*:?\s*#\s*(\d+)", re.IGNORECASE
)
ADVISORY_REF_RE = re.compile(r"\b(?:refs?|references?)\b\s*:?\s*#\s*(\d+)", re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
TYPE_LABELS = frozenset(
    {"bug", "feat", "docs", "ci", "refactor", "chore", "governance", "spec"}
)
CODE_EXTS = frozenset(
    {
        ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
        ".sh", ".bash", ".ps1", ".sql", ".rs", ".go", ".java", ".kt",
        ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".lua",
    }
)
SIZE_LIMIT = 1000
EXCEPTION_LABEL = "size/exception"
MAX_PAGES = 30


class ApiError(RuntimeError):
    """GitHub API failure; status 0 means a transport/parse error."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"HTTP {status}: {reason}" if status else str(reason))
        self.status = status

    @property
    def is_tooling_error(self) -> bool:
        """True when retrying/permissions -- not PR content -- is the issue."""
        return self.status == 0 or self.status in (401, 403, 429) or self.status >= 500


class GitHub:
    def __init__(self, token: str, repo: str, api_root: str = API_ROOT) -> None:
        self.token, self.repo, self.api_root = token, repo, api_root.rstrip("/")

    def get(self, path: str, params: dict | None = None):
        url = f"{self.api_root}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "nwc-pr-guard",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = str(exc.reason)
            if exc.code in (403, 429):
                detail += " (rate limit or missing permission; retry later)"
            raise ApiError(exc.code, detail) from None
        except urllib.error.URLError as exc:
            raise ApiError(0, f"network error: {exc.reason}") from None
        try:
            return json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError as exc:
            raise ApiError(0, f"invalid JSON response for {path} "
                               f"(line {exc.lineno}: {exc.msg})") from None

    def paged(self, path: str, max_pages: int = MAX_PAGES) -> list:
        items: list = []
        page = 1
        while page <= max_pages:
            batch = self.get(path, {"per_page": 100, "page": page})
            if not isinstance(batch, list):
                raise ApiError(
                    0,
                    f"unexpected response for {path} page {page}: "
                    f"expected a list, got {type(batch).__name__}",
                )
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1
        print(f"::warning::{path}: pagination capped at {max_pages} pages; "
              "change size may be undercounted.", file=sys.stderr)
        return items


@dataclass
class Check:
    title: str
    ok: bool
    details: list[str] = field(default_factory=list)
    warning: bool = False


def label_names(pr: dict) -> set[str]:
    return {label.get("name", "") for label in pr.get("labels") or []}


def parse_time(value) -> datetime | None:
    """Parse an ISO timestamp to an aware UTC datetime (naive input assumed UTC)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def visible_body(body: str) -> str:
    """Strip fenced code, inline code, HTML comments and blockquote lines."""
    text = HTML_COMMENT_RE.sub("", body or "")
    kept: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        marker = FENCE_RE.match(line)
        if fence is not None:
            if marker and marker.group(1)[0] == fence:
                fence = None
            continue
        if marker:
            fence = marker.group(1)[0]
            continue
        if line.lstrip().startswith(">"):
            continue
        kept.append(INLINE_CODE_RE.sub("", line))
    return "\n".join(kept)


def _numbers(pattern: re.Pattern[str], text: str) -> list[int]:
    found: list[int] = []
    for match in pattern.finditer(text):
        number = int(match.group(1))
        if number not in found:
            found.append(number)
    return found


def scan_body(body: str) -> tuple[list[int], list[int]]:
    """Return (strong Closes/Fixes/Resolves refs, advisory Refs refs)."""
    visible = visible_body(body)
    return _numbers(STRONG_REF_RE, visible), _numbers(ADVISORY_REF_RE, visible)


def check_linkage(pr: dict, refs: list[int],
                  lookups: dict[int, tuple[dict | None, str | None]],
                  advisories: list[int] | None = None) -> Check:
    advisories = advisories or []
    if not refs:
        details = [
            "PR body has no issue reference.",
            "Add a line such as `Closes #12` (Closes|Fixes|Resolves #N).",
        ]
        if advisories:
            listed = ", ".join(f"#{number}" for number in advisories)
            details.append(
                f"WARNING: `Refs` is advisory only and does not satisfy linkage: {listed}"
            )
        return Check("Issue linkage", False, details, warning=bool(advisories))

    details: list[str] = []
    warnings = False
    ok = True
    pr_created = parse_time(pr.get("created_at"))
    if pr_created is None:
        warnings = True
        details.append(
            "WARNING: PR created_at is missing/unparseable; "
            "skipping the issue-age comparison."
        )
    for ref in refs:
        issue, error = lookups.get(ref, (None, "issue was not fetched"))
        if error:
            ok = False
            details.append(f"#{ref}: {error}")
            continue
        issue = issue or {}
        types = sorted(label_names(issue) & TYPE_LABELS)
        created = parse_time(issue.get("created_at"))
        problems: list[str] = []
        if not types:
            problems.append("no type label (bug/feat/docs/ci/refactor/chore/governance/spec)")
        if created is None:
            warnings = True
            details.append(
                f"#{ref}: WARNING: no parseable created_at; skipping the age comparison."
            )
        elif pr_created is not None and created > pr_created:
            problems.append(
                f"issue created {created.isoformat()} after PR ({pr_created.isoformat()}); "
                "open the issue first"
            )
        if issue.get("pull_request"):
            problems.append("reference points to a pull request, not an issue")
        if problems:
            ok = False
            details.extend(f"#{ref}: {problem}" for problem in problems)
        else:
            details.append(f"#{ref}: linked, type labels: {', '.join(types)}")
    return Check("Issue linkage", ok, details, warning=warnings)


def check_size(pr: dict, files: list[dict], limit: int = SIZE_LIMIT) -> Check:
    total = 0
    code: list[tuple[str, int]] = []
    gray: list[tuple[str, int]] = []
    for entry in files:
        name = str(entry.get("filename") or "")
        delta = int(entry.get("additions") or 0) + int(entry.get("deletions") or 0)
        if Path(name).suffix.lower() in CODE_EXTS:
            total += delta
            code.append((name, delta))
        else:
            gray.append((name, delta))

    details = [f"Code files: {len(code)} ({total} lines, limit {limit})."]
    if gray:
        details.append(
            f"Non-code (not counted): {len(gray)} file(s), {sum(d for _, d in gray)} lines."
        )
        details.extend(f"  gray {name} (+{delta})" for name, delta in gray[:15])
        if len(gray) > 15:
            details.append(f"  ... and {len(gray) - 15} more")
    if total < limit:
        return Check("Change size", True, details)
    if EXCEPTION_LABEL in label_names(pr):
        details.insert(
            0,
            "!!! SIZE EXCEPTION GRANTED: `size/exception` label present -- "
            f"{total} >= {limit} lines approved by the Lead. Keep it smaller next time. !!!",
        )
        return Check("Change size", True, details, warning=True)
    details.append(
        f"FAIL: {total} >= {limit} code lines. Split the PR, or ask the Lead "
        f"for the `{EXCEPTION_LABEL}` label."
    )
    return Check("Change size", False, details)


def summarize(pr: dict, checks: list[Check]) -> str:
    number = pr.get("number") or os.environ.get("PR_NUMBER", "?")
    lines = [f"## PR guard — PR #{number}", "", "| Check | Result | Notes |", "| --- | --- | --- |"]
    for check in checks:
        note = " / ".join(check.details[:2]).replace("|", "\\|")
        mark = "✅" if check.ok else "❌"
        if check.ok and check.warning:
            mark = "⚠️"
        lines.append(f"| {check.title} | {mark} | {note} |")
    lines.append("")
    for check in checks:
        lines += [f"<details><summary>{check.title}</summary>", ""]
        lines += [f"- {detail}" for detail in check.details]
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def write_summary(markdown: str, path: str | None = None) -> None:
    target = path or os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        print("(no GITHUB_STEP_SUMMARY; summary printed to stdout only)")
        return
    try:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
        print(f"summary written to {target}")
    except OSError as exc:
        print(f"WARNING: cannot write summary: {exc}", file=sys.stderr)


def print_checks(checks: list[Check]) -> None:
    for check in checks:
        mark = "PASS" if check.ok else "FAIL"
        if check.ok and check.warning:
            mark = "PASS(!)"
        print(f"[{mark}] {check.title}")
        for detail in check.details:
            print(f"        {detail}")


def load_event(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: cannot read GITHUB_EVENT_PATH: {exc}", file=sys.stderr)
        return {}


def fetch_pr(api: GitHub, number: int, event: dict) -> dict:
    payload = event.get("pull_request")
    if isinstance(payload, dict) and int(payload.get("number") or 0) == number:
        return payload
    return api.get(f"/repos/{api.repo}/pulls/{number}")


def fetch_lookups(api: GitHub, refs: list[int]) -> dict[int, tuple[dict | None, str | None]]:
    lookups: dict[int, tuple[dict | None, str | None]] = {}
    for ref in refs:
        try:
            lookups[ref] = (api.get(f"/repos/{api.repo}/issues/{ref}"), None)
        except ApiError as exc:
            if exc.is_tooling_error:
                raise
            lookups[ref] = (None, f"issue lookup failed ({exc})")
    return lookups


def self_test() -> int:
    issue_ok = {"number": 42, "created_at": "2026-10-01T00:00:00Z",
                "labels": [{"name": "ci"}]}
    pr_created = "2026-11-01T00:00:00Z"

    def pr(body: str, labels: tuple[str, ...] = (), created: str | None = pr_created) -> dict:
        payload = {"number": 7, "body": body,
                   "labels": [{"name": name} for name in labels]}
        if created is not None:
            payload["created_at"] = created
        return payload

    small = [
        {"filename": "scripts/ci/new.py", "additions": 120, "deletions": 8},
        {"filename": "docs/notes.md", "additions": 60, "deletions": 0},
    ]
    large = [
        {"filename": "star_graph/big.py", "additions": 900, "deletions": 250},
        {"filename": "README.md", "additions": 10, "deletions": 0},
    ]
    exactly_limit = [{"filename": "star_graph/big.py", "additions": 700, "deletions": 300}]
    under_limit = [{"filename": "star_graph/big.py", "additions": 699, "deletions": 300}]
    docs_only = [{"filename": "pyproject.toml", "additions": 5000, "deletions": 0},
                 {"filename": ".github/workflows/ci.yml", "additions": 5000, "deletions": 0}]
    cases = [
        ("linked + small", pr("Closes #42"), small, {42: (issue_ok, None)}, True, False),
        ("linked + oversized", pr("Fixes #42"), large, {42: (issue_ok, None)}, False, False),
        ("no issue link", pr("Update docs only"), small, {}, False, False),
        ("size/exception label", pr("Closes #42", ("ci", "size/exception")),
         large, {42: (issue_ok, None)}, True, True),
        ("issue newer than PR", pr("Closes #42"), small,
         {42: ({**issue_ok, "created_at": "2026-11-02T00:00:00Z"}, None)}, False, False),
        ("issue without type label", pr("Closes #42"), small,
         {42: ({**issue_ok, "labels": [{"name": "question"}]}, None)}, False, False),
        ("missing issue (404)", pr("Closes #999"), small,
         {999: (None, "HTTP 404: Not Found")}, False, False),
        ("fenced code is not a link", pr("Example:\n```markdown\nCloses #42\n```\n"), small, {}, False, False),
        ("HTML comment is not a link", pr("<!-- Closes #42 -->\nNo link here."), small, {}, False, False),
        ("blockquote/inline code are not links",
         pr("> Closes #42\nInline `Closes #42` example."), small, {}, False, False),
        ("Refs alone does not satisfy linkage", pr("Refs #42"), small,
         {42: (issue_ok, None)}, False, True),
        ("exactly 1000 code lines fails", pr("Closes #42"), exactly_limit,
         {42: (issue_ok, None)}, False, False),
        ("999 code lines passes", pr("Closes #42"), under_limit,
         {42: (issue_ok, None)}, True, False),
        ("reference to a pull request", pr("Closes #42"), small,
         {42: ({**issue_ok, "pull_request": {"url": "https://example.invalid"}}, None)}, False, False),
        ("missing issue created_at warns but passes", pr("Closes #42"), small,
         {42: ({k: v for k, v in issue_ok.items() if k != "created_at"}, None)}, True, True),
        ("missing PR created_at warns instead of skipping blindly",
         pr("Closes #42", created=None), small, {42: (issue_ok, None)}, True, True),
        ("docs/config do not count as code", pr("Closes #42"), docs_only, {42: (issue_ok, None)}, True, False),
    ]
    failures = 0
    for name, pull, files, lookups, want_ok, want_warning in cases:
        refs, advisories = scan_body(pull.get("body") or "")
        checks = [check_linkage(pull, refs, lookups, advisories),
                  check_size(pull, files)]
        got_ok = all(check.ok for check in checks)
        got_warning = any(check.warning for check in checks)
        passed = got_ok == want_ok and got_warning == want_warning
        if not passed:
            failures += 1
        print(f"[self-test] {name}: {'PASS' if got_ok else 'FAIL'} "
              f"(expected {'PASS' if want_ok else 'FAIL'}, "
              f"warning={'yes' if got_warning else 'no'}) -> "
              f"{'ok' if passed else 'MISMATCH'}")
        for check in checks:
            for detail in check.details:
                print(f"    {check.title}: {detail}")
    print(f"self-test: {len(cases) - failures}/{len(cases)} cases passed")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PR issue-linkage and change-size guard (no build, stdlib only).",
        epilog="PR_NUMBER=12 GITHUB_TOKEN=$TOKEN GITHUB_REPOSITORY=owner/name "
               "python3 scripts/ci/check_pr.py",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run offline mock cases: pass/oversize/no-issue/exception")
    parser.add_argument("--pr-number", type=int, help="PR number (default: $PR_NUMBER)")
    parser.add_argument("--repo", help="owner/name (default: $GITHUB_REPOSITORY)")
    parser.add_argument("--api-root", default=API_ROOT, help="GitHub API base URL")
    parser.add_argument("--max-changes", type=int, default=SIZE_LIMIT,
                        help=f"code change line limit (default {SIZE_LIMIT})")
    parser.add_argument("--summary", help="summary file (default: $GITHUB_STEP_SUMMARY)")
    parser.add_argument("--verbose", action="store_true", help="extra diagnostics")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    raw_number = args.pr_number if args.pr_number is not None else os.environ.get("PR_NUMBER", "")
    if not repo or "/" not in repo:
        print("ERROR: set GITHUB_REPOSITORY=owner/name (or pass --repo).", file=sys.stderr)
        return 2
    if not token:
        print("ERROR: set GITHUB_TOKEN (pull-requests:read, issues:read, contents:read).",
              file=sys.stderr)
        return 2
    try:
        number = int(raw_number)
    except (TypeError, ValueError):
        print(f"ERROR: invalid PR_NUMBER={raw_number!r} (use --pr-number).", file=sys.stderr)
        return 2

    api = GitHub(token, repo, args.api_root)
    try:
        pr = fetch_pr(api, number, load_event(os.environ.get("GITHUB_EVENT_PATH")))
        files = api.paged(f"/repos/{repo}/pulls/{number}/files")
        body = pr.get("body") or ""
        refs, advisories = scan_body(body)
        lookups = fetch_lookups(api, refs)
    except ApiError as exc:
        if exc.is_tooling_error:
            print(
                "ERROR: GitHub API tooling failure (not a PR rule violation): "
                f"{exc}. Retry later or check the token permissions.",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: GitHub API failure: {exc}", file=sys.stderr)
        return 2
    if args.verbose:
        print(f"repo={repo} pr=#{number} files={len(files)} title={pr.get('title', '')!r}")
        print(f"strong refs={refs} advisory refs={advisories}")

    checks = [
        check_linkage(pr, refs, lookups, advisories),
        check_size(pr, files, args.max_changes),
    ]
    print_checks(checks)
    markdown = summarize(pr, checks)
    write_summary(markdown, args.summary)
    if args.verbose:
        print(markdown)
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
