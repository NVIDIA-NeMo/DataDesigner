# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

WORKFLOW = Path(__file__).parents[3] / ".github/workflows/pr-linked-issue.yml"


def _step(name: str) -> tuple[dict[str, Any], str]:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = next(step for step in workflow["jobs"]["check"]["steps"] if step["name"] == name)
    return step, step["run"].replace("${{ github.repository }}", "NVIDIA-NeMo/DataDesigner")


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


def test_close_step_rechecks_current_pr_before_closing() -> None:
    step, script = _step("Close new or reopened PR without an open, triaged issue")
    condition = step["if"]
    assert "steps.comment.outputs.status == 'fail'" in condition
    assert "github.run_attempt == '1'" in condition
    assert "github.triggering_actor != 'github-actions[bot]'" in condition
    assert "github.event.action == 'opened'" in condition
    assert "github.event.action == 'reopened'" in condition
    result_step, _ = _step("Set check result")
    assert "steps.close.outputs.current_valid != 'true'" in result_step["if"]
    with TemporaryDirectory() as directory:
        path = Path(directory)
        gh = path / "gh"
        gh.write_text(
            r"""#!/bin/sh
if [ "$1" = "pr" ] && [ "$2" = "view" ]; then
  printf "%s\n" "$PR_JSON"
elif [ "$1" = "api" ]; then
  if [ "$2" = "-X" ]; then
    printf "%s\n" "$*" > "$COMMENT_LOG"
    exit 0
  fi
  case "$ISSUE_SCENARIO" in
    triaged) printf '%s\n' '{"state":"open","labels":[{"name":"triaged"}]}' ;;
    untriaged) printf '%s\n' '{"state":"open","labels":[]}' ;;
    unavailable) echo "gh: Service Unavailable (HTTP 503)" >&2; exit 1 ;;
  esac
elif [ "$1" = "pr" ] && [ "$2" = "close" ]; then
  printf "%s\n" "$*" > "$CALL_LOG"
fi
"""
        )
        gh.chmod(0o755)
        call_log = path / "call.log"
        comment_log = path / "comment.log"
        output = path / "output"
        for state, labels, body, issue_scenario, expected_exit, should_close in (
            ("OPEN", [], "", "", 0, True),
            ("OPEN", [], "Fixes #123", "untriaged", 0, True),
            # The opened event was invalid, but the current body now links a triaged issue.
            ("OPEN", [], "Fixes #123", "triaged", 0, False),
            ("OPEN", ["keep-open"], "", "", 0, False),
            ("CLOSED", [], "", "", 0, False),
            ("OPEN", [], "Fixes #123", "unavailable", 1, False),
        ):
            call_log.unlink(missing_ok=True)
            comment_log.unlink(missing_ok=True)
            output.write_text("")
            env = os.environ | {
                "PATH": f"{path}:{os.environ['PATH']}",
                "PR_NUMBER": "42",
                "REPO": "NVIDIA-NeMo/DataDesigner",
                "PR_JSON": json.dumps({"state": state, "labels": [{"name": label} for label in labels], "body": body}),
                "ISSUE_SCENARIO": issue_scenario,
                "CALL_LOG": str(call_log),
                "COMMENT_LOG": str(comment_log),
                "COMMENT_ID": "77",
                "GITHUB_OUTPUT": str(output),
            }
            result = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)
            assert result.returncode == expected_exit, result.stderr
            assert call_log.exists() == should_close, (
                state,
                labels,
                body,
                issue_scenario,
                result.stdout,
                result.stderr,
            )
            if should_close:
                assert call_log.read_text().strip() == "pr close 42 --repo NVIDIA-NeMo/DataDesigner"
            if issue_scenario == "triaged":
                assert "current_valid=true" in output.read_text()
                assert comment_log.read_text().strip() == (
                    "api -X DELETE repos/NVIDIA-NeMo/DataDesigner/issues/comments/77"
                )


def test_issue_parser_uses_current_pr_body() -> None:
    _, script = _step("Parse issue reference from PR body")
    with TemporaryDirectory() as directory:
        path = Path(directory)
        gh = path / "gh"
        gh.write_text('#!/bin/sh\nprintf "%s\\n" "$CURRENT_BODY"\n')
        gh.chmod(0o755)
        output = path / "output"
        for body, expected in (("Fixes #123", "issue_num=123"), ("", "issue_num=")):
            output.write_text("")
            env = os.environ | {
                "PATH": f"{path}:{os.environ['PATH']}",
                "CURRENT_BODY": body,
                "PR_NUMBER": "42",
                "REPO": "NVIDIA-NeMo/DataDesigner",
                "GITHUB_OUTPUT": str(output),
            }
            subprocess.run(["bash", "-e", "-c", script], env=env, check=True)
            assert expected in output.read_text()


def test_failure_comment_exposes_its_id_for_reconciliation() -> None:
    _, script = _step("Build comment body and post result")
    with TemporaryDirectory() as directory:
        path = Path(directory)
        gh = path / "gh"
        gh.write_text('#!/bin/sh\nif [ "$2" = "--paginate" ]; then exit 0; fi\nprintf "%s\\n" 77\n')
        gh.chmod(0o755)
        output = path / "output"
        output.write_text("")
        env = os.environ | {
            "PATH": f"{path}:{os.environ['PATH']}",
            "IS_COLLABORATOR": "false",
            "ISSUE_NUM": "",
            "ISSUE_EXISTS": "false",
            "ISSUE_OPEN": "false",
            "IS_TRIAGED": "false",
            "PR_NUMBER": "42",
            "REPO": "NVIDIA-NeMo/DataDesigner",
            "GITHUB_OUTPUT": str(output),
        }
        subprocess.run(["bash", "-e", "-c", script], env=env, check=True)
        assert "status=fail" in output.read_text()
        assert "comment_id=77" in output.read_text()


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
