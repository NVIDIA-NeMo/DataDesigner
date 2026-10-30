# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "manage_agentic_ci_attempts.py"
SPEC = importlib.util.spec_from_file_location("manage_agentic_ci_attempts", SCRIPT_PATH)
assert SPEC and SPEC.loader
ATTEMPTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTEMPTS)


@pytest.fixture
def runner_state(tmp_path: Path) -> tuple[Path, Path]:
    state_path = tmp_path / "runner-state.json"
    prior_path = tmp_path / "prior.json"
    state_path.write_text(
        json.dumps(
            {
                "attempted_fixes": [
                    {
                        "id": finding_id,
                        "attempts": [
                            {
                                "outcome": "open",
                                "pr_number": 123,
                                "branch": "agentic-ci/chore/dependencies-batch",
                            }
                        ],
                    }
                    for finding_id in ("first", "second")
                ]
            }
        )
    )
    prior_path.write_text(json.dumps([{"id": "first", "n": 0}, {"id": "second", "n": 0}]))
    return state_path, prior_path


def test_groups_batched_attempts(runner_state: tuple[Path, Path]) -> None:
    state_path, prior_path = runner_state
    module = ATTEMPTS
    assert isinstance(module, ModuleType)

    assert module.new_open_attempt_groups(state_path, prior_path) == [
        {
            "pr_number": 123,
            "branch": "agentic-ci/chore/dependencies-batch",
            "finding_ids": ["first", "second"],
        }
    ]


def test_abandons_every_attempt_in_rejected_batch(runner_state: tuple[Path, Path]) -> None:
    state_path, _ = runner_state
    module = ATTEMPTS
    assert isinstance(module, ModuleType)

    module.abandon_open_attempts(state_path, ["first", "second"], "lockfile_missing")

    attempts = [entry["attempts"][-1] for entry in json.loads(state_path.read_text())["attempted_fixes"]]
    assert attempts == [
        {
            "outcome": "abandoned",
            "pr_number": 123,
            "branch": "agentic-ci/chore/dependencies-batch",
            "lockfile_missing": True,
        },
        {
            "outcome": "abandoned",
            "pr_number": 123,
            "branch": "agentic-ci/chore/dependencies-batch",
            "lockfile_missing": True,
        },
    ]


@pytest.mark.parametrize("pr_state", ["OPEN", "CLOSED", "MERGED"])
def test_reconciles_each_pr_once(
    runner_state: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, pr_state: str
) -> None:
    state_path, _ = runner_state
    query = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps({"state": pr_state})))
    monkeypatch.setattr(ATTEMPTS.subprocess, "run", query)

    ATTEMPTS.reconcile_open_attempts(state_path, "NVIDIA-NeMo/DataDesigner")

    query.assert_called_once_with(
        ["gh", "pr", "view", "123", "--repo", "NVIDIA-NeMo/DataDesigner", "--json", "state"],
        check=True,
        capture_output=True,
        text=True,
    )
    for entry in json.loads(state_path.read_text())["attempted_fixes"]:
        assert entry["attempts"] == [
            {"outcome": pr_state.lower(), "pr_number": 123, "branch": "agentic-ci/chore/dependencies-batch"}
        ]


def test_reconciliation_failure_preserves_state(
    runner_state: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path, _ = runner_state
    original = json.loads(state_path.read_text())
    original["attempted_fixes"][1]["attempts"][-1]["pr_number"] = 456
    state_path.write_text(json.dumps(original))
    query = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, '{"state": "CLOSED"}'),
            subprocess.CalledProcessError(1, "gh"),
        ]
    )
    monkeypatch.setattr(ATTEMPTS.subprocess, "run", query)

    with pytest.raises(subprocess.CalledProcessError):
        ATTEMPTS.reconcile_open_attempts(state_path, "NVIDIA-NeMo/DataDesigner")

    assert json.loads(state_path.read_text()) == original


@pytest.mark.skipif(shutil.which("jq") is None, reason="Workflow backlog checks require jq")
def test_closed_docs_pr_can_reenter_audit_without_another_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = yaml.safe_load((SCRIPT_PATH.parents[1] / ".github/workflows/agentic-ci-daily.yml").read_text())
    steps = {step["id"]: step for step in workflow["jobs"]["audit"]["steps"] if "id" in step}
    order = list(steps)
    assert (
        order.index("reset_docs_scope")
        < order.index("reconcile_docs_prs")
        < order.index("audit")
        < order.index("backlog")
    )
    assert "backlog" not in steps["reconcile_docs_prs"]["if"]
    state = tmp_path / ".agentic-ci-state"
    state.mkdir()
    state_path = state / "runner-state.json"
    state_path.write_text(
        json.dumps(
            {
                "fix_backlog": [{"id": "docs", "data": {"task_key": "models", "scope_complete": True}}],
                "attempted_fixes": [{"id": "docs", "attempts": [{"outcome": "open", "pr_number": 123}]}],
            }
        )
    )
    output = tmp_path / "output"
    env = os.environ | {"GITHUB_OUTPUT": str(output), "SUITE": "docs-and-references"}
    subprocess.run(["/bin/bash", "-e", "-c", steps["reset_docs_scope"]["run"]], cwd=tmp_path, env=env, check=True)
    with monkeypatch.context() as patch:
        patch.setattr(
            ATTEMPTS.subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, '{"state": "CLOSED"}'))
        )
        ATTEMPTS.reconcile_open_attempts(state_path, "NVIDIA-NeMo/DataDesigner")

    updated = json.loads(state_path.read_text())
    assert updated["attempted_fixes"] == [{"id": "docs", "attempts": [{"outcome": "closed", "pr_number": 123}]}]
    assert updated["fix_backlog"][0]["data"]["scope_complete"] is False
    # The finding is eligible for the audit again; a completed audit admits its retry.
    updated["fix_backlog"][0]["data"]["scope_complete"] = True
    state_path.write_text(json.dumps(updated))
    subprocess.run(["/bin/bash", "-e", "-c", steps["backlog"]["run"]], cwd=tmp_path, env=env, check=True)
    assert output.read_text().strip() == "size=1"
