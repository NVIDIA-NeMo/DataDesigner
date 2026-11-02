# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).parents[3] / ".github/workflows/agentic-ci-daily.yml"


@pytest.fixture
def daily_scripts() -> dict[str, str]:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return {step["id"]: step["run"] for step in workflow["jobs"]["audit"]["steps"] if "run" in step and "id" in step}


@pytest.fixture
def scope_check(daily_scripts: dict[str, str]) -> str:
    script = "FILE_CAP=3" + daily_scripts["scope_gate"].split("FILE_CAP=3", 1)[1]
    return script.split("# Docs suite:", 1)[0] + '\nprintf "%s" "$REASONS"\n'


@pytest.mark.parametrize(
    "suite,pattern,files,added,removed,accepted",
    [
        ("docs-and-references", "architecture/{i}.md", 10, 100, 100, True),
        ("docs-and-references", "architecture/{i}.md", 11, 10, 10, False),
        ("docs-and-references", "architecture/{i}.md", 1, 101, 100, False),
        ("docs-and-references", "fern/versions/latest/pages/{i}.mdx", 4, 30, 30, True),
        ("docs-and-references", "fern/versions/latest/pages/devnotes/posts/{i}.mdx", 1, 1, 1, False),
        ("docs-and-references", "packages/engine/src/{i}.py", 1, 2, 2, True),
        ("docs-and-references", "docs/{i}.md", 1, 1, 1, False),
        ("docs-and-references", ".agents/{i}.md", 1, 1, 1, False),
        ("docs-and-references", ".github/workflows/{i}.yml", 1, 1, 1, False),
        ("docs-and-references", "pyproject.toml", 1, 1, 1, False),
        ("dependencies", "packages/pkg{i}/pyproject.toml", 3, 25, 25, True),
        ("dependencies", "packages/pkg{i}/pyproject.toml", 4, 1, 1, False),
        ("dependencies", "packages/pkg{i}/pyproject.toml", 1, 26, 25, False),
        ("structure", "packages/engine/src/{i}.py", 3, 25, 25, True),
        ("structure", "packages/engine/src/{i}.py", 4, 1, 1, False),
        ("structure", "packages/engine/src/{i}.py", 1, 26, 25, False),
        ("code-quality", "packages/engine/src/{i}.py", 3, 25, 25, True),
        ("code-quality", "packages/engine/src/{i}.py", 4, 1, 1, False),
        ("code-quality", "packages/engine/src/{i}.py", 1, 26, 25, False),
    ],
)
def test_suite_scope_limits(
    scope_check: str, suite: str, pattern: str, files: int, added: int, removed: int, accepted: bool
) -> None:
    script = r"""
git() {
  case "$2" in
    --name-only) printf '%s\n' "$FILES" ;;
    --shortstat) printf ' %s files changed, %s insertions(+), %s deletions(-)\n' "$FILE_COUNT" "$ADDED" "$REMOVED" ;;
    *) return 1 ;;
  esac
}
DIFF_REF=HEAD
REASONS=""
"""
    env = os.environ | {
        "SUITE": suite,
        "FILES": "\n".join(pattern.format(i=i) for i in range(files)),
        "FILE_COUNT": str(files),
        "ADDED": str(added),
        "REMOVED": str(removed),
    }
    result = subprocess.run(["/bin/bash", "-e", "-c", script + scope_check], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (result.stdout == "") == accepted, result.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="Workflow backlog checks require jq")
@pytest.mark.parametrize("include_ready_finding", [False, True])
@pytest.mark.parametrize(
    "suite,data,expected",
    [
        ("docs-and-references", {}, 0),
        ("docs-and-references", {"task_key": "models"}, 0),
        ("docs-and-references", {"scope_complete": True}, 0),
        ("docs-and-references", {"task_key": "", "scope_complete": True}, 0),
        ("docs-and-references", {"task_key": "models", "scope_complete": False}, 0),
        ("docs-and-references", {"task_key": "models", "scope_complete": "true"}, 0),
        ("docs-and-references", {"task_key": "models", "scope_complete": True}, 1),
        ("docs-and-references", "invalid", 0),
        ("docs-and-references", None, 0),
        ("docs-and-references", {"task_key": 123, "scope_complete": True}, 0),
        ("dependencies", {}, 1),
        ("structure", {}, 1),
        ("code-quality", {}, 1),
    ],
)
def test_fix_phase_requires_completed_docs_scope(
    daily_scripts: dict[str, str],
    tmp_path: Path,
    suite: str,
    data: dict[str, str | bool | int] | str | None,
    expected: int,
    include_ready_finding: bool,
) -> None:
    state = tmp_path / ".agentic-ci-state"
    state.mkdir()
    backlog = [{"id": "finding", "data": data}]
    if include_ready_finding:
        backlog.append({"id": "ready", "data": {"task_key": "ready", "scope_complete": True}})
    (state / "runner-state.json").write_text(json.dumps({"fix_backlog": backlog}))
    output = tmp_path / "output"
    env = os.environ | {"SUITE": suite, "GITHUB_OUTPUT": str(output)}
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", daily_scripts["backlog"]], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == f"size={expected + include_ready_finding}"


@pytest.mark.skipif(shutil.which("jq") is None, reason="Workflow backlog checks require jq")
@pytest.mark.parametrize("data", [None, "invalid", {}, {"task_key": "models", "scope_complete": True}])
def test_docs_scope_reset(daily_scripts: dict[str, str], tmp_path: Path, data: object) -> None:
    state = tmp_path / ".agentic-ci-state"
    state.mkdir()
    state_file = state / "runner-state.json"
    attempts = [{"id": "finding", "attempts": [{"outcome": "open"}]}]
    state_file.write_text(json.dumps({"fix_backlog": [{"id": "finding", "data": data}], "attempted_fixes": attempts}))
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", daily_scripts["reset_docs_scope"]], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    updated = json.loads(state_file.read_text())
    assert updated["fix_backlog"] == [
        {"id": "finding", "data": (data if isinstance(data, dict) else {}) | {"scope_complete": False}}
    ]
    assert updated["attempted_fixes"] == attempts


@pytest.mark.skipif(shutil.which("jq") is None, reason="Workflow backlog checks require jq")
@pytest.mark.parametrize("contents", ['{"fix_backlog":', '{"fix_backlog": "invalid"}'])
def test_docs_scope_reset_defers_corrupt_state(daily_scripts: dict[str, str], tmp_path: Path, contents: str) -> None:
    state = tmp_path / ".agentic-ci-state"
    state.mkdir()
    state_file = state / "runner-state.json"
    state_file.write_text(contents)
    output = tmp_path / "output"
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", daily_scripts["reset_docs_scope"]],
        cwd=tmp_path,
        env=os.environ | {"GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == "ready=false"
    assert state_file.read_text() == contents
    assert not state_file.with_suffix(".json.tmp").exists()
