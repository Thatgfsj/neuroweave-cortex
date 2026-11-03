#!/usr/bin/env bash
# Final verification: build ONE wheel, verify its contents, install it into a
# throwaway venv outside the repository and smoke-test the installed package.
# Used by .github/workflows/verify.yml and runnable locally from any cwd.
#   PYTHON=.venv/bin/python bash scripts/smoke_test.sh
# This script is NOT part of the PR checks (PR checks never build or install).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || {
  echo "FAIL: python interpreter '$PY' not found; set PYTHON=/path/to/python" >&2; exit 2; }

"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "FAIL: Python >= 3.11 is required (found $("$PY" -V 2>&1)); set PYTHON=/path/to/python3.11+" >&2
  exit 2
}

say() { printf '\n== %s ==\n' "$*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

VERSION="$("$PY" -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])')" \
  || fail "cannot read [project].version from pyproject.toml"
[ -n "$VERSION" ] || fail "empty [project].version in pyproject.toml"

say "1/4 build wheel (old wheels removed, version $VERSION)"
rm -f dist/*.whl
mkdir -p dist
"$PY" -m pip wheel . --no-deps -w dist/ --disable-pip-version-check \
  || fail "wheel build failed"

say "2/4 verify wheel contents"
mapfile -t wheels < <(find dist -maxdepth 1 -type f -iname "nwcortex-${VERSION}-*.whl" | sort)
[ "${#wheels[@]}" -eq 1 ] \
  || fail "expected exactly one nwcortex-${VERSION}-*.whl in dist/, found ${#wheels[@]}"
WHEEL="${wheels[0]}"
[ -f "$WHEEL" ] || fail "wheel missing: $WHEEL"
"$PY" - "$WHEEL" <<'PY' || exit 1
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
missing = [pkg for pkg in ("star_graph/", "nwc/") if not any(n.startswith(pkg) for n in names)]
print(f"wheel: {wheel} ({len(names)} members)")
if missing:
    print(f"FAIL: wheel is missing packages: {', '.join(missing)}")
    sys.exit(1)
print("PASS: wheel contains star_graph and nwc")
PY

say "3/4 install wheel into a throwaway venv and import it outside the repo"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/nwc-smoke.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT
case "$(cd "$WORKDIR" && pwd -P)" in
  "$ROOT"/*) fail "temporary directory is inside the repository: $WORKDIR" ;;
esac
VENV="$WORKDIR/venv"
"$PY" -m venv "$VENV" || fail "cannot create venv at $VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install --disable-pip-version-check "$WHEEL" pytest \
  || fail "cannot install $WHEEL + pytest into $VENV"
(
  cd "$WORKDIR"
  env -u PYTHONPATH -u PYTHONHOME "$VPY" - "$ROOT" "$VERSION" <<'PY' || exit 1
import importlib.metadata as metadata
import os
import sys

import nwc
import star_graph

repo = os.path.realpath(sys.argv[1])
expected = sys.argv[2]
star_file = os.path.realpath(star_graph.__file__)
nwc_file = os.path.realpath(nwc.__file__)
print(f"star_graph: {star_file}")
print(f"nwc: {nwc_file}")
for label, path in (("star_graph", star_file), ("nwc", nwc_file)):
    if path == repo or path.startswith(repo + os.sep):
        print(f"FAIL: {label} was imported from the repository ({path}); wheel is shadowed")
        sys.exit(1)
metadata_version = metadata.version("NWcortex")
print(f"metadata: NWcortex {metadata_version}; star_graph {star_graph.__version__}; "
      f"nwc {nwc.__version__}")
if metadata_version != star_graph.__version__:
    print("FAIL: importlib.metadata.version('NWcortex') != star_graph.__version__")
    sys.exit(1)
if star_graph.__version__ != nwc.__version__ or metadata_version != expected:
    print(f"FAIL: package versions disagree (expected {expected})")
    sys.exit(1)
from star_graph.graph import StarGraph

if not callable(nwc.Cortex) or StarGraph() is None:
    print("FAIL: API smoke failed")
    sys.exit(1)
print("PASS: installed wheel imports + StarGraph() API smoke ok")
PY
)

say "4/4 README doctest + graph tests (source tree)"
test_files=()
missing=()
for candidate in tests/test_readme_doctest.py tests/test_graph.py; do
  if [ -f "$candidate" ]; then
    test_files+=("$candidate")
  else
    missing+=("$candidate")
  fi
done
if [ "${#test_files[@]}" -eq 0 ]; then
  fail "required smoke test files are missing: ${missing[*]}"
elif [ "${#missing[@]}" -gt 0 ]; then
  echo "::warning::smoke test file(s) missing: ${missing[*]} (was the file moved?)"
fi
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$VPY" -m pytest "${test_files[@]}" -q \
  || fail "pytest smoke tests failed"
echo "PASS: ${test_files[*]}"

say "RESULT"
echo "PASS: single build + wheel install + smoke verification complete"
