# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "generate_security_constraints.py"
SPEC = importlib.util.spec_from_file_location("generate_security_constraints", SCRIPT_PATH)
assert SPEC and SPEC.loader
SECURITY_CONSTRAINTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SECURITY_CONSTRAINTS)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.uv]
constraint-dependencies = [
    "urllib3>=2.8.0",
    "Pillow>=13; python_version >= '3.11'",
    "certifi>=2026.10.19",
]
""",
        encoding="utf-8",
    )
    return tmp_path


def test_exports_floors_including_markers(workspace: Path) -> None:
    assert SECURITY_CONSTRAINTS.generate_constraints(workspace) == 0
    lines = (workspace / "constraints.txt").read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if not line.startswith("#")] == [
        "certifi>=2026.10.19",
        'Pillow>=13; python_version >= "3.11"',
        "urllib3>=2.8.0",
    ]


def test_check_accepts_current_export_without_rewriting(workspace: Path) -> None:
    SECURITY_CONSTRAINTS.generate_constraints(workspace)
    destination = workspace / "constraints.txt"
    before = destination.stat().st_mtime_ns
    assert SECURITY_CONSTRAINTS.generate_constraints(workspace, check=True) == 0
    assert destination.stat().st_mtime_ns == before


@pytest.mark.parametrize("existing", [None, "certifi>=2020\n"], ids=["missing", "stale"])
def test_check_rejects_stale_export_without_writing(workspace: Path, existing: str | None) -> None:
    destination = workspace / "constraints.txt"
    if existing is not None:
        destination.write_text(existing, encoding="utf-8")
    assert SECURITY_CONSTRAINTS.generate_constraints(workspace, check=True) == 1
    assert (destination.read_text(encoding="utf-8") if destination.exists() else None) == existing


def test_cli_fails_for_changed_floors(workspace: Path) -> None:
    scripts = workspace / "scripts"
    scripts.mkdir()
    script = scripts / SCRIPT_PATH.name
    script.write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True)
    (workspace / "pyproject.toml").write_text(
        '[tool.uv]\nconstraint-dependencies = ["certifi>=2027"]\n', encoding="utf-8"
    )
    result = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "make update-security-constraints" in result.stdout


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("requests[socks]>=2.32", "requests>=2.32"),
        ("requests[socks]>=2.32; python_version >= '3.11'", 'requests>=2.32; python_version >= "3.11"'),
        ("example @ https://example.com/example.whl", "example @ https://example.com/example.whl"),
        ("example[feature] @ https://example.com/example.whl", "example @ https://example.com/example.whl"),
    ],
    ids=["extras", "extras-with-marker", "direct-url", "direct-url-with-extras"],
)
def test_export_is_pip_compatible(workspace: Path, raw: str, expected: str) -> None:
    (workspace / "pyproject.toml").write_text(
        "[tool.uv]\nconstraint-dependencies = " + json.dumps([raw]) + "\n", encoding="utf-8"
    )
    assert SECURITY_CONSTRAINTS.generate_constraints(workspace) == 0
    lines = (workspace / "constraints.txt").read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if not line.startswith("#")] == [expected]
    assert SECURITY_CONSTRAINTS.generate_constraints(workspace, check=True) == 0


@pytest.mark.parametrize(
    ("changed_path", "should_run"),
    [
        ("pyproject.toml", True),
        ("packages/data-designer-engine/pyproject.toml", True),
        ("tests_e2e/pyproject.toml", True),
        ("uv.lock", True),
        ("constraints.txt", True),
        ("scripts/generate_security_constraints.py", True),
        ("README.md", False),
        ("packages/data-designer/src/data_designer/interface/results.py", False),
    ],
)
def test_audit_workflow_filters_unrelated_prs(changed_path: str, should_run: bool) -> None:
    workflow_path = SCRIPT_PATH.parents[1] / ".github" / "workflows" / "dependency-security.yml"
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    for trigger in ("pull_request", "push"):
        paths = workflow["on"][trigger]["paths"]
        assert any(fnmatchcase(changed_path, pattern) for pattern in paths) == should_run
    assert workflow["on"]["schedule"]
