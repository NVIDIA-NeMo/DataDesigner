# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


@pytest.fixture
def load_script() -> Callable[[str], ModuleType]:
    def load(name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def test_redacts_configured_values_and_hostname(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    redact = load_script("redact_agentic_ci_output").redact_output
    monkeypatch.setenv("AGENTIC_CI_MODEL", "test/model")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-api-key")
    monkeypatch.setenv("GH_TOKEN", "fixture-github-token")

    text = "test/model https://gateway.example.invalid GATEWAY.EXAMPLE.INVALID fixture-api-key fixture-github-token"
    assert redact(text + "\nPublic report") == " ".join(["[redacted]"] * 5) + "\nPublic report"


def test_report_splits_at_headings(load_script: Callable[[str], ModuleType]) -> None:
    triage = load_script("publish_agentic_ci_triage")
    first = "### First\n" + ("a" * 31_000)
    second = "### Second\n" + ("b" * 31_000)

    parts = triage.report_parts(f"{first}\n\n{second}", "2026-11-17")

    assert len(parts) == 2
    assert parts[0].startswith("<!-- agentic-ci-issue-triage:1/2 -->\n")
    assert parts[1].startswith("<!-- agentic-ci-issue-triage:2/2 -->\n")
    assert parts[0].endswith(first + "\n")
    assert parts[1].endswith(second + "\n")


@pytest.mark.parametrize("parts,methods", [(1, ["PATCH", "DELETE"]), (3, ["PATCH", "PATCH", "POST"])])
def test_publish_reconciles_only_bot_report_comments(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch, parts: int, methods: list[str]
) -> None:
    triage = load_script("publish_agentic_ci_triage")
    marker = "<!-- agentic-ci-issue-triage:1/1 -->"
    bot = {"login": "github-actions[bot]"}
    comments = [
        {"id": 10, "user": {"login": "contributor"}, "body": marker},
        {"id": 20, "user": bot, "body": "another bot comment"},
        {"id": 30, "user": bot, "body": marker},
        {"id": 25, "user": bot, "body": marker},
    ]
    monkeypatch.setattr(triage.subprocess, "check_output", Mock(return_value=json.dumps([comments])))
    request = Mock()
    monkeypatch.setattr(triage.subprocess, "run", request)

    triage.publish_report("example/project", 1, ["report text"] * parts)

    calls = request.call_args_list
    assert [call.args[0][3] for call in calls] == methods
    assert calls[0].args[0][4].endswith("/comments/25")
    assert calls[1].args[0][4].endswith("/comments/30")
    assert calls[0].kwargs["input"] == json.dumps({"body": "report text"})


@pytest.mark.parametrize(
    "workflow_name,job,step_name",
    [("health-probe", "probe", "Ping inference API"), ("daily", "audit", "Pre-flight checks")],
)
def test_inference_connection_errors_are_generic(workflow_name: str, job: str, step_name: str) -> None:
    workflow = yaml.safe_load((WORKFLOWS / f"agentic-ci-{workflow_name}.yml").read_text())
    step = next(step for step in workflow["jobs"][job]["steps"] if step["name"] == step_name)
    mock_commands = """
claude() { return 0; }
curl() {
  [ "$1" = "-s" ] || echo "gateway.example.invalid" >&2
  return 7
}
"""

    result = subprocess.run(
        ["bash", "-e", "-c", mock_commands + step["run"]], capture_output=True, text=True, check=False
    )

    assert result.returncode == 1
    assert "curl exit 7" in result.stdout
    assert result.stderr == ""


def test_triage_generation_has_no_tools_or_write_token() -> None:
    workflow = yaml.safe_load((WORKFLOWS / "agentic-ci-issue-triage.yml").read_text())
    triage = workflow["jobs"]["triage"]
    generate = next(step for step in triage["steps"] if step["name"] == "Generate triage report")

    assert triage["permissions"]["issues"] == "read"
    assert "GH_TOKEN" not in generate["env"]
    assert "GITHUB_TOKEN" not in generate["env"]
    assert '--tools ""' in generate["run"]
    assert workflow["jobs"]["publish"]["permissions"]["issues"] == "write"
    assert workflow["jobs"]["publish"]["if"] == "github.ref == 'refs/heads/main'"
