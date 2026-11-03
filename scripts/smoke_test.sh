#!/usr/bin/env bash
# Final verification: build a wheel and smoke-test it. Idempotent, safe to re-run.
# Used by .github/workflows/verify.yml and runnable locally from any cwd.
#   PYTHON=.venv/bin/python bash scripts/smoke_test.sh
# This script is NOT part of the PR checks (PR checks never build or install).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || {
  echo "FAIL: python interpreter '$PY' not found; set PYTHON=/path/to/python" >&2; exit 1; }

say() { printf '\n== %s ==\n' "$*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
smoke_env() { PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$@"; }

say "1/4 build wheel (pip wheel --no-deps)"
mkdir -p dist
"$PY" -m pip wheel . --no-deps -w dist/ --disable-pip-version-check \
  || fail "wheel build failed"

say "2/4 verify wheel contents"
shopt -s nullglob
wheels=(dist/*.whl)
shopt -u nullglob
[ "${#wheels[@]}" -gt 0 ] || fail "no dist/*.whl produced"
WHEEL="$(ls -t dist/*.whl | head -n 1)"
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

say "3/4 README doctest + graph tests"
test_files=()
for candidate in tests/test_readme_doctest.py tests/test_graph.py; do
  if [ -f "$candidate" ]; then
    test_files+=("$candidate")
  fi
done
if [ "${#test_files[@]}" -eq 0 ]; then
  echo "SKIP: neither tests/test_readme_doctest.py nor tests/test_graph.py exists"
elif ! "$PY" -c "import pytest" >/dev/null 2>&1; then
  fail "pytest is required for smoke tests: $PY -m pip install pytest"
else
  smoke_env "$PY" -m pytest "${test_files[@]}" -q || fail "pytest smoke tests failed"
  echo "PASS: ${test_files[*]}"
fi

say "4/4 import + API smoke"
smoke_env "$PY" - <<'PY' || exit 1
import nwc
import star_graph
from star_graph.graph import StarGraph

print(f"star_graph {star_graph.__version__} / nwc {nwc.__version__}")
assert star_graph.__version__ == nwc.__version__, "package version mismatch"
assert callable(nwc.Cortex) and StarGraph() is not None, "API smoke failed"
print("PASS: import + StarGraph() smoke ok")
PY

say "RESULT"
echo "PASS: build + smoke verification complete"
