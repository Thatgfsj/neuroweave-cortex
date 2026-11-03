#!/usr/bin/env python3
"""PR guard: issue linkage + change size (pure stdlib: no build, no installs).

A. Linkage -- the body must contain `(Closes|Fixes|Resolves|Refs) #N`; that
   issue must exist, be created no later than the PR, and carry a type label
   (bug/feat/docs/ci/refactor/chore/governance/spec).
B. Size -- code-file additions+deletions must stay below 1000 lines. Non-code
   files (docs/data) are listed but never counted. The `size/exception` label
   overrides the limit and raises a loud warning.

Usage:
  PR_NUMBER=12 GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/name \
      python3 scripts/ci/check_pr.py [--verbose]
  python3 scripts/ci/check_pr.py --self-test    # offline, no network

Env: GITHUB_TOKEN, GITHUB_REPOSITORY, PR_NUMBER, GITHUB_EVENT_PATH,
GITHUB_STEP_SUMMARY (optional). TODO(#14): version drift lives in
scripts/check_version.py, not here.
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
from datetime import datetime
from pathlib import Path

API_ROOT = "https://api.github.com"
ISSUE_REF_RE = re.compile(r"(?:closes|fixes|resolves|refs)\s+#(\d+)", re.IGNORECASE)
TYPE_LABELS = frozenset(
    {"bug", "feat", "docs", "ci", "refactor", "chore", "governance", "spec"}
)
CODE_EXTS = frozenset(
    {
        ".py", ".ts", ".js", ".sh", ".yml", ".yaml", ".toml", ".sql",
        ".rs", ".go", ".java", ".c", ".h", ".cpp",
    }
)
SIZE_LIMIT = 1000
EXCEPTION_LABEL = "size/exception"


class ApiError(RuntimeError):
    """GitHub API failure; status 0 means a transport error."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"HTTP {status}: {reason}" if status else str(reason))
        self.status = status


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
            raise ApiError(exc.code, str(exc.reason)) from None
        except urllib.error.URLError as exc:
            raise ApiError(0, str(exc.reason)) from None
        return json.loads(raw) if raw.strip() else None

    def paged(self, path: str) -> list:
        items: list = []
        page = 1
        while True:
            batch = self.get(path, {"per_page": 100, "page": page}) or []
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1


@dataclass
class Check:
    title: str
    ok: bool
    details: list[str] = field(default_factory=list)
    warning: bool = False


def label_names(pr: dict) -> set[str]:
    return {label.get("name", "") for label in pr.get("labels") or []}


def parse_time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def issue_refs(body: str) -> list[int]:
    refs: list[int] = []
    for match in ISSUE_REF_RE.finditer(body or ""):
        if int(match.group(1)) not in refs:
            refs.append(int(match.group(1)))
    return refs


def check_linkage(pr: dict, body: str, refs: list[int],
                  lookups: dict[int, tuple[dict | None, str | None]]) -> Check:
    if not refs:
        return Check("Issue linkage", False, [
            "PR body has no issue reference.",
            "Add a line such as `Closes #12` (Closes|Fixes|Resolves|Refs #N).",
        ])
    details: list[str] = []
    ok = True
    pr_created = parse_time(pr.get("created_at"))
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
            problems.append("issue has no parseable created_at")
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
    return Check("Issue linkage", ok, details)


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
        lines.append(f"| {check.title} | {'✅' if check.ok else '❌'} | {note} |")
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
            lookups[ref] = (None, f"issue lookup failed ({exc})")
    return lookups


def self_test() -> int:
    issue_ok = {"number": 42, "created_at": "2026-10-01T00:00:00Z",
                "labels": [{"name": "ci"}]}
    pr_created = "2026-11-01T00:00:00Z"

    def pr(body: str, labels: tuple[str, ...] = ()) -> dict:
        return {"number": 7, "body": body, "created_at": pr_created,
                "labels": [{"name": name} for name in labels]}

    small = [
        {"filename": "scripts/ci/new.py", "additions": 120, "deletions": 8},
        {"filename": "docs/notes.md", "additions": 60, "deletions": 0},
    ]
    large = [
        {"filename": "star_graph/big.py", "additions": 900, "deletions": 250},
        {"filename": "README.md", "additions": 10, "deletions": 0},
    ]
    cases = [
        ("linked + small", pr("Closes #42"), small, {42: (issue_ok, None)}, True),
        ("linked + oversized", pr("Fixes #42"), large, {42: (issue_ok, None)}, False),
        ("no issue link", pr("Update docs only"), small, {}, False),
        ("size/exception label", pr("Closes #42", ("ci", "size/exception")),
         large, {42: (issue_ok, None)}, True),
        ("issue newer than PR", pr("Closes #42"), small,
         {42: ({**issue_ok, "created_at": "2026-11-02T00:00:00Z"}, None)}, False),
        ("issue without type label", pr("Closes #42"), small,
         {42: ({**issue_ok, "labels": [{"name": "question"}]}, None)}, False),
        ("missing issue (404)", pr("Refs #999"), small,
         {999: (None, "HTTP 404: Not Found")}, False),
    ]
    failures = 0
    for name, pull, files, lookups, want_ok in cases:
        body = pull.get("body") or ""
        checks = [check_linkage(pull, body, issue_refs(body), lookups),
                  check_size(pull, files)]
        got_ok = all(check.ok for check in checks)
        if got_ok != want_ok:
            failures += 1
        verdict = "PASS" if got_ok else "FAIL"
        print(f"[self-test] {name}: {verdict} "
              f"(expected {'PASS' if want_ok else 'FAIL'}) -> "
              f"{'ok' if got_ok == want_ok else 'MISMATCH'}")
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
    except ApiError as exc:
        print(f"ERROR: GitHub API failure: {exc}", file=sys.stderr)
        return 2
    if args.verbose:
        print(f"repo={repo} pr=#{number} files={len(files)} title={pr.get('title', '')!r}")

    body = pr.get("body") or ""
    refs = issue_refs(body)
    checks = [
        check_linkage(pr, body, refs, fetch_lookups(api, refs)),
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
