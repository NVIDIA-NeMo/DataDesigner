# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

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
