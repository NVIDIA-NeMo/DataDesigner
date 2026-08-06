# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[4] / "docs" / "scripts" / "build_notebooks_cached.sh"


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("NOTEBOOK_EXECUTION_ATTEMPTS", "invalid", "must be a positive integer"),
        ("NOTEBOOK_RETRY_DELAY_SECONDS", "-1", "must be a non-negative integer"),
    ],
)
def test_rejects_invalid_retry_settings(tmp_path: Path, setting: str, value: str, message: str) -> None:
    env = {**os.environ, setting: value}

    result = subprocess.run([SCRIPT_PATH, tmp_path], check=False, capture_output=True, text=True, env=env)

    assert result.returncode != 0
    assert message in result.stdout


def test_retries_failed_notebook_execution(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script = repo_root / "docs" / "scripts" / SCRIPT_PATH.name
    script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT_PATH, script)

    source_dir = repo_root / "docs" / "notebook_source"
    source_dir.mkdir()
    (source_dir / "_README.md").write_text("README\n")
    (source_dir / "_pyproject.toml").write_text('[project]\nname = "notebooks"\n')
    (source_dir / "example.py").write_text("print('example')\n")

    attempts_file = tmp_path / "attempts"
    jupytext = tmp_path / "jupytext"
    jupytext.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
attempt=0
if [ -f "$ATTEMPTS_FILE" ]; then
    attempt=$(<"$ATTEMPTS_FILE")
fi
attempt=$((attempt + 1))
printf '%s' "$attempt" > "$ATTEMPTS_FILE"
if [ "$attempt" -lt 2 ]; then
    exit 1
fi
src="${!#}"
printf '%s\n' '{"cells": []}' > "${src%.py}.ipynb"
"""
    )
    jupytext.chmod(0o755)
    env = {
        **os.environ,
        "ATTEMPTS_FILE": str(attempts_file),
        "DOCS_JUPYTEXT": str(jupytext),
        "NOTEBOOK_CACHE_CONTEXT": "test-context",
        "NOTEBOOK_EXECUTION_ATTEMPTS": "2",
        "NOTEBOOK_RETRY_DELAY_SECONDS": "0",
    }

    result = subprocess.run([script], check=True, capture_output=True, text=True, env=env)

    assert attempts_file.read_text() == "2"
    assert "Attempt 1 failed; retrying" in result.stdout
    assert (repo_root / "docs" / "notebooks" / "example.ipynb").exists()
    assert (repo_root / ".notebook-cache" / "example.sha256").exists()
