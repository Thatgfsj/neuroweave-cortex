"""Guardrails for the single-source version layout introduced by issue #5.

``star_graph/_version.py`` is the only place allowed to define the version.
Everything else imports it, and the documentation mirrors are kept in sync by
``scripts/bump_version.py`` / verified by ``scripts/check_version.py``.
"""

from __future__ import annotations

import ast
import importlib.metadata as metadata
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

import nwc
import star_graph
from star_graph import _version

ROOT = Path(__file__).resolve().parent.parent
# Never hardcode the release number here: bump_version.py only edits the
# single source + documentation mirrors, so a literal would break the next bump.
EXPECTED = _version.__version__
SEMVER = r"\d+\.\d+\.\d+"

SERVER_MODULES = (
    "nwc/api/server.py",
    "star_graph/contrib/mcp_server.py",
    "nwc/mcp/server.py",
    "star_graph/cortex_api/server.py",
)

DOC_MIRRORS = {
    "README.md": re.compile(rf"badge/version-({SEMVER})"),
    "README_CN.md": re.compile(rf"(?m)^v({SEMVER})\s*\|"),
    "plan.md": re.compile(rf"(?m)^> 最后更新:.*\|\s*\*\*v({SEMVER})\*\*"),
    "ROADMAP.md": re.compile(rf"(?m)^##\s*现状（v({SEMVER})"),
}


def test_versions_agree():
    assert star_graph.__version__ == EXPECTED
    assert nwc.__version__ == EXPECTED
    assert _version.__version__ == EXPECTED


def test_version_info_matches_string():
    major, minor, patch = (int(part) for part in EXPECTED.split("."))
    assert _version.__version_info__ == (major, minor, patch)


def test_version_module_is_import_free():
    tree = ast.parse((ROOT / "star_graph/_version.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports == [], "star_graph/_version.py must not import anything"


def test_version_module_is_the_only_definition():
    offenders = []
    for package in ("star_graph", "nwc"):
        for path in (ROOT / package).rglob("*.py"):
            if path.name == "_version.py" or "__pycache__" in path.parts:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.startswith("__version__"):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert offenders == []


def test_pyproject_uses_dynamic_version():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in data["project"], "[project] must not pin a static version"
    assert data["project"]["dynamic"] == ["version"]
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "star_graph._version.__version__"


@pytest.mark.parametrize(
    ("relative", "pattern"),
    [
        ("star_graph/__init__.py", r"from\s+\._version\s+import\s+.*\b__version__"),
        ("nwc/__init__.py", r"from\s+star_graph\._version\s+import\s+.*\b__version__"),
    ],
)
def test_package_inits_import_single_source(relative, pattern):
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert re.search(pattern, text), f"{relative} must import __version__ from _version"
    assert not re.search(rf"""["']{SEMVER}["']""", text), f"{relative} must not hardcode a version"


@pytest.mark.parametrize(("relative", "pattern"), DOC_MIRRORS.items())
def test_documentation_mirrors_show_current_version(relative, pattern):
    match = pattern.search((ROOT / relative).read_text(encoding="utf-8"))
    assert match, f"{relative} is missing its version marker"
    assert match.group(1) == EXPECTED, f"{relative} documents {match.group(1)}, expected {EXPECTED}"


def test_roadmap_current_version_line():
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert f"当前版本 v{EXPECTED}" in text


def test_defaults_yaml_has_no_version_field():
    text = (ROOT / "star_graph/defaults.yaml").read_text(encoding="utf-8")
    assert not re.search(r"(?m)^version\s*:", text), "defaults.yaml must not duplicate the version"


@pytest.mark.parametrize("relative", SERVER_MODULES)
def test_server_modules_use_single_source(relative):
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert not re.search(rf"""["']{SEMVER}["']""", text), f"{relative} hardcodes a version"
    assert re.search(r"from\s+(?:\.+|star_graph\.)_version\s+import\s+.*\b__version__", text)


def test_installed_metadata_matches_source():
    try:
        installed = metadata.version("NWcortex")
    except metadata.PackageNotFoundError:
        pytest.skip("NWcortex is not installed in this environment")

    if installed == star_graph.__version__:
        return

    direct = metadata.distribution("NWcortex").read_text("direct_url.json")
    info = json.loads(direct) if direct else {}
    url = info.get("url", "")
    source_root = Path(unquote(urlparse(url).path)).resolve() if url.startswith("file://") else None
    editable = bool(info.get("dir_info", {}).get("editable"))

    if source_root is not None and source_root != ROOT:
        pytest.skip(f"installed NWcortex {installed} belongs to another checkout: {source_root}")
    if editable:
        pytest.skip(
            f"editable install metadata ({installed}) is stale for {ROOT}; "
            "re-run `pip install -e .` or rely on scripts/smoke_test.sh for wheel metadata"
        )
    assert installed == star_graph.__version__


def test_check_version_script_passes():
    result = subprocess.run(
        [sys.executable, "scripts/check_version.py", "--root", str(ROOT), "--verbose"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bump_version_check_passes():
    result = subprocess.run(
        [sys.executable, "scripts/bump_version.py", "--root", str(ROOT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bump_helpers_are_semver_aware():
    spec = importlib.util.spec_from_file_location("nwc_bump_version", ROOT / "scripts/bump_version.py")
    assert spec and spec.loader
    bump = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bump)

    assert bump.bump_version("1.4.3", "patch") == "1.4.4"
    assert bump.bump_version("1.4.3", "minor") == "1.5.0"
    assert bump.bump_version("1.4.3", "major") == "2.0.0"
    with pytest.raises(ValueError):
        bump.bump_version("1.4", "patch")
