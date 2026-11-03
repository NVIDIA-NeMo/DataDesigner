# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from coverage import CoverageData
from coverage.exceptions import DataError

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("report_ci_coverage", ROOT / "scripts/report_ci_coverage.py")
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())


@pytest.fixture
def coverage_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".coveragerc").write_text("[run]\nrelative_files = true\nsource = .\n")
    (tmp_path / "sample.py").write_text("first = 1\nsecond = 2\n")
    for package, lines in (("config", {1}), ("engine", {2}), ("interface", {1})):
        data = CoverageData(basename=str(tmp_path / f".coverage.{package}"))
        data.add_lines({"sample.py": lines})
        data.write()
    return tmp_path


def test_combines_coverage_before_enforcing_threshold(coverage_files: Path) -> None:
    assert REPORT.report_coverage() == 100
    assert (coverage_files / "coverage.xml").is_file()


@pytest.mark.parametrize("package", ["config", "engine", "interface"])
@pytest.mark.parametrize("invalid", ["missing", "empty", "corrupt"])
def test_rejects_incomplete_coverage(coverage_files: Path, package: str, invalid: str) -> None:
    path = coverage_files / f".coverage.{package}"
    path.unlink()
    if invalid == "empty":
        data = CoverageData(basename=str(path))
        data.add_lines({})
        data.write()
    elif invalid == "corrupt":
        path.write_text("not coverage data")
    with pytest.raises((ValueError, DataError)):
        REPORT.report_coverage()


def test_rejects_low_combined_coverage(coverage_files: Path) -> None:
    for path in coverage_files.glob(".coverage.*"):
        path.unlink()
        data = CoverageData(basename=str(path))
        data.add_lines({"sample.py": {1}})
        data.write()
    with pytest.raises(ValueError, match="below 90%"):
        REPORT.report_coverage()


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", None])
def test_gate_rejects_unsuccessful_dependencies(result: str | None) -> None:
    gate = WORKFLOW["jobs"]["test-summary"]
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == set(WORKFLOW["jobs"]) - {"test-summary"}
    results = {name: {"result": "success"} for name in gate["needs"]}
    results["test-engine"]["result"] = result
    completed = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", gate["steps"][0]["run"]],
        env={**os.environ, "CHECK_RESULTS": json.dumps(results)},
        capture_output=True,
    )
    assert (completed.returncode == 0) is (result == "success")
