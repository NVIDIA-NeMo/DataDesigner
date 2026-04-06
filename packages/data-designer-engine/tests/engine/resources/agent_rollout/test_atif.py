# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_designer.engine.resources.agent_rollout.atif import AtifAgentRolloutFormatHandler
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError


def _make_handler() -> AtifAgentRolloutFormatHandler:
    return AtifAgentRolloutFormatHandler()


def test_parse_file_comprehensive_happy_path(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trace.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "atif-session",
                "agent": {
                    "name": "harbor-agent",
                    "version": "2.0.0",
                    "model_name": "gpt-5",
                    "extra": {
                        "cwd": "/workspace",
                        "project_path": "/workspace/project",
                        "git_branch": "main",
                    },
                },
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-04-06T12:00:00Z",
                        "source": "system",
                        "message": "You are a coding agent.",
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-04-06T12:00:01Z",
                        "source": "user",
                        "message": "Inspect the repository.",
                        "is_copied_context": True,
                    },
                    {
                        "step_id": 3,
                        "timestamp": "2026-04-06T12:00:05Z",
                        "source": "agent",
                        "message": [{"type": "text", "text": "I inspected the repository."}],
                        "reasoning_content": "Check README and tests first.",
                        "tool_calls": [
                            {
                                "tool_call_id": "call-1",
                                "function_name": "read_file",
                                "arguments": {"path": "README.md"},
                            }
                        ],
                        "observation": {
                            "results": [
                                {
                                    "source_call_id": "call-1",
                                    "content": "README contents",
                                },
                                {
                                    "subagent_trajectory_ref": [
                                        {
                                            "session_id": "subagent-1",
                                            "trajectory_path": "subagent-1.json",
                                            "extra": {"summary": "Ran test audit"},
                                        }
                                    ]
                                },
                            ]
                        },
                    },
                ],
                "notes": "standalone ATIF rollout",
                "final_metrics": {"total_cost_usd": 0.42},
            }
        ),
        encoding="utf-8",
    )

    records = _make_handler().parse_file(root_path=tmp_path, relative_path="trace.json")

    assert len(records) == 1
    record = records[0]
    assert record.trace_id == "atif-session"
    assert record.root_session_id == "atif-session"
    assert record.source_kind == "atif"
    assert record.message_count == 4
    assert record.tool_call_count == 1
    assert record.final_assistant_message == "I inspected the repository."
    assert record.cwd == "/workspace"
    assert record.project_path == "/workspace/project"
    assert record.git_branch == "main"
    assert record.started_at == "2026-04-06T12:00:00Z"
    assert record.ended_at == "2026-04-06T12:00:05Z"

    assistant_message = next(message for message in record.messages if message["role"] == "assistant")
    assert assistant_message["reasoning_content"] == "Check README and tests first."
    assert assistant_message["tool_calls"][0]["id"] == "call-1"
    assert assistant_message["tool_calls"][0]["function"]["arguments"] == '{"path": "README.md"}'

    tool_message = next(message for message in record.messages if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call-1"
    assert tool_message["content"] == [{"type": "text", "text": "README contents"}]

    assert record.source_meta["schema_version"] == "ATIF-v1.6"
    assert record.source_meta["agent_name"] == "harbor-agent"
    assert record.source_meta["version"] == "2.0.0"
    assert record.source_meta["model_name"] == "gpt-5"
    assert record.source_meta["copied_context_step_ids"] == [2]
    assert record.source_meta["subagent_trajectory_refs"] == [
        {
            "step_id": 3,
            "refs": [
                {
                    "session_id": "subagent-1",
                    "trajectory_path": "subagent-1.json",
                    "extra": {"summary": "Ran test audit"},
                }
            ],
        }
    ]
    assert record.source_meta["notes"] == "standalone ATIF rollout"
    assert record.source_meta["final_metrics"] == {"total_cost_usd": 0.42}


def test_parse_file_rejects_non_object_payload(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trace.json"
    trajectory_path.write_text('["not", "an", "object"]', encoding="utf-8")

    with pytest.raises(AgentRolloutSeedParseError, match="Expected JSON object"):
        _make_handler().parse_file(root_path=tmp_path, relative_path="trace.json")


def test_is_handled_file_accepts_json_only() -> None:
    handler = _make_handler()

    assert handler.is_handled_file("trace.json") is True
    assert handler.is_handled_file("nested/run.trajectory.json") is True
    assert handler.is_handled_file("trace.jsonl") is False
