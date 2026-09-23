# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

WORKFLOW = Path(__file__).parents[3] / ".github/workflows/pr-linked-issue.yml"


def _step(name: str) -> tuple[str, str]:
    lines = WORKFLOW.read_text().splitlines()
    start = lines.index(f"      - name: {name}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("      - name: ")), len(lines))
    step = "\n".join(lines[start:end])
    if "        run: |\n" in step:
        run = step.split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] if line.startswith("          ") else line for line in run.splitlines())
    else:
        script = next(
            line.removeprefix("        run: ") for line in step.splitlines() if line.startswith("        run: ")
        )
    return step, script.replace("${{ github.repository }}", "NVIDIA-NeMo/DataDesigner")


def test_issue_validation_only_closes_for_definite_policy_failures() -> None:
    _, script = _step("Validate issue is open and triaged")
    with TemporaryDirectory() as directory:
        path = Path(directory)
        gh = path / "gh"
        gh.write_text(
            "#!/bin/sh\n"
            'case "$SCENARIO" in\n'
            '  triaged) printf \'%s\\n\' \'{"state":"open","labels":[{"name":"triaged"}]}\' ;;\n'
            '  untriaged) printf \'%s\\n\' \'{"state":"open","labels":[]}\' ;;\n'
            '  missing) echo "gh: Not Found (HTTP 404)" >&2; exit 1 ;;\n'
            '  unavailable) echo "gh: Service Unavailable (HTTP 503)" >&2; exit 1 ;;\n'
            "esac\n"
        )
        gh.chmod(0o755)
        for scenario, expected_exit, expected_output in (
            ("triaged", 0, "is_triaged=true"),
            ("untriaged", 0, "is_triaged=false"),
            ("missing", 0, "issue_exists=false"),
            ("unavailable", 1, ""),
        ):
            output = path / "output"
            output.write_text("")
            env = os.environ | {
                "PATH": f"{path}:{os.environ['PATH']}",
                "SCENARIO": scenario,
                "ISSUE_NUM": "123",
                "GITHUB_OUTPUT": str(output),
            }
            result = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)
            assert result.returncode == expected_exit, (scenario, result.stderr)
            assert expected_output in output.read_text()


def test_invalid_issue_closes_pull_request() -> None:
    step, script = _step("Close PR without an open, triaged issue")
    assert "if: steps.comment.outputs.status == 'fail'" in step
    with TemporaryDirectory() as directory:
        path = Path(directory)
        gh = path / "gh"
        gh.write_text('#!/bin/sh\nprintf "%s\n" "$*" > "$CALL_LOG"\n')
        gh.chmod(0o755)
        call_log = path / "call.log"
        env = os.environ | {
            "PATH": f"{path}:{os.environ['PATH']}",
            "PR_NUMBER": "42",
            "REPO": "NVIDIA-NeMo/DataDesigner",
            "CALL_LOG": str(call_log),
        }
        subprocess.run(["bash", "-e", "-c", script], env=env, check=True)
        assert call_log.read_text().strip() == "pr close 42 --repo NVIDIA-NeMo/DataDesigner"


def test_contributor_permission_lookup_fails_on_api_outage() -> None:
    _, script = _step("Check author permissions")
    with TemporaryDirectory() as directory:
        path = Path(directory)
        gh = path / "gh"
        gh.write_text(
            "#!/bin/sh\n"
            'case "$SCENARIO" in\n'
            "  collaborator) echo write ;;\n"
            '  outsider) echo "gh: Not Found (HTTP 404)" >&2; exit 1 ;;\n'
            '  unavailable) echo "gh: Service Unavailable (HTTP 503)" >&2; exit 1 ;;\n'
            "esac\n"
        )
        gh.chmod(0o755)
        for scenario, expected_exit, expected_output in (
            ("collaborator", 0, "is_collaborator=true"),
            ("outsider", 0, "is_collaborator=false"),
            ("unavailable", 1, ""),
        ):
            output = path / "output"
            output.write_text("")
            env = os.environ | {
                "PATH": f"{path}:{os.environ['PATH']}",
                "SCENARIO": scenario,
                "PR_AUTHOR": "example-user",
                "HEAD_REPO": "someone/fork",
                "HEAD_REF": "feature",
                "PR_BODY": "Fixes #123",
                "REPO": "NVIDIA-NeMo/DataDesigner",
                "GITHUB_OUTPUT": str(output),
            }
            result = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)
            assert result.returncode == expected_exit, (scenario, result.stderr)
            assert expected_output in output.read_text()
