# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_designer.engine.resources.agent_rollout.hermes_agent import HermesAgentRolloutFormatHandler
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError


def _make_handler() -> HermesAgentRolloutFormatHandler:
    return HermesAgentRolloutFormatHandler()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parse_file_cli_session_log_happy_path(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "session_20260407_092759_baeaac.json",
        {
            "session_id": "20260407_092759_baeaac",
            "model": "aws/anthropic/bedrock-claude-opus-4-6",
            "base_url": "https://inference-api.nvidia.com/v1",
            "platform": "cli",
            "session_start": "2026-04-07T09:39:07.028463",
            "last_updated": "2026-04-07T09:51:07.905570",
            "system_prompt": "You are Hermes.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Run shell commands.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ],
            "message_count": 4,
            "messages": [
                {"role": "user", "content": "Set up a uv project."},
                {
                    "role": "assistant",
                    "content": "I'll initialize the project.",
                    "finish_reason": "tool_calls",
                    "reasoning": None,
                    "tool_calls": [
                        {
                            "id": "tooluse_1",
                            "call_id": "tooluse_1",
                            "response_item_id": "fc_tooluse_1",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command":"uv init"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tooluse_1",
                    "content": '{"output":"Initialized project","exit_code":0,"error":null}',
                },
                {
                    "role": "assistant",
                    "content": "Done.",
                    "finish_reason": "stop",
                    "reasoning": None,
                    "tool_calls": [],
                },
            ],
        },
    )

    handler = _make_handler()
    records = handler.parse_file(
        root_path=tmp_path,
        relative_path="session_20260407_092759_baeaac.json",
    )

    assert len(records) == 1
    record = records[0]
    assert record.trace_id == "20260407_092759_baeaac"
    assert record.root_session_id == "20260407_092759_baeaac"
    assert record.source_kind == "hermes_agent"
    assert record.started_at == "2026-04-07T09:39:07.028463"
    assert record.ended_at == "2026-04-07T09:51:07.905570"
    assert record.message_count == 4
    assert record.tool_call_count == 1
    assert record.final_assistant_message == "Done."
    assert record.source_meta["model"] == "aws/anthropic/bedrock-claude-opus-4-6"
    assert record.source_meta["base_url"] == "https://inference-api.nvidia.com/v1"
    assert record.source_meta["available_tool_names"] == ["terminal"]
    assert record.source_meta["has_system_prompt"] is True
    assert record.source_meta["system_prompt_chars"] == len("You are Hermes.")
    assert record.source_meta["finish_reasons"] == ["tool_calls", "stop"]


def test_parse_file_gateway_transcript_uses_sessions_index(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    _write_json(
        tmp_path / "sessions.json",
        {
            "slack:thread-1": {
                "session_key": "slack:thread-1",
                "session_id": "gateway-session-1",
                "created_at": "2026-04-07T08:00:00",
                "updated_at": "2026-04-07T08:05:00",
                "display_name": "ops-thread",
                "platform": "slack",
                "chat_type": "thread",
                "input_tokens": 12,
                "output_tokens": 34,
                "total_tokens": 46,
                "last_prompt_tokens": 99,
                "estimated_cost_usd": 0.12,
                "cost_status": "reported",
                "memory_flushed": True,
                "origin": {
                    "platform": "slack",
                    "chat_id": "C123",
                    "chat_name": "deployments",
                    "chat_type": "thread",
                    "thread_id": "thread-1",
                },
            }
        },
    )
    write_jsonl(
        tmp_path / "gateway-session-1.jsonl",
        [
            {"role": "user", "content": "Check the deployment status."},
            {
                "role": "assistant",
                "content": "I'll inspect the logs.",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "tooluse_logs",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"kubectl logs deploy/app"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tooluse_logs",
                "content": '{"output":"healthy","exit_code":0,"error":null}',
            },
        ],
    )

    handler = _make_handler()
    parse_context = handler.build_parse_context(root_path=tmp_path, recursive=True)
    records = handler.parse_file(
        root_path=tmp_path,
        relative_path="gateway-session-1.jsonl",
        parse_context=parse_context,
    )

    assert len(records) == 1
    record = records[0]
    assert record.trace_id == "gateway-session-1"
    assert record.started_at == "2026-04-07T08:00:00"
    assert record.ended_at == "2026-04-07T08:05:00"
    assert record.tool_call_count == 1
    assert record.final_assistant_message == "I'll inspect the logs."
    assert record.source_meta["session_format"] == "gateway_transcript"
    assert record.source_meta["session_key"] == "slack:thread-1"
    assert record.source_meta["platform"] == "slack"
    assert record.source_meta["chat_type"] == "thread"
    assert record.source_meta["tool_names_used"] == ["terminal"]
    assert record.source_meta["origin"]["chat_id"] == "C123"


def test_parse_file_cli_session_requires_messages_list(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "session_20260407_092611_298324.json",
        {
            "session_id": "20260407_092611_298324",
            "model": "aws/anthropic/bedrock-claude-opus-4-6",
        },
    )

    handler = _make_handler()
    with pytest.raises(AgentRolloutSeedParseError, match="missing a 'messages' list"):
        handler.parse_file(
            root_path=tmp_path,
            relative_path="session_20260407_092611_298324.json",
        )


def test_is_handled_file_accepts_cli_json_and_gateway_jsonl() -> None:
    handler = _make_handler()

    assert handler.is_handled_file("session_20260407_092759_baeaac.json") is True
    assert handler.is_handled_file("gateway-session-1.jsonl") is True
    assert handler.is_handled_file("nested/gateway-session-2.jsonl") is True
    assert handler.is_handled_file("sessions.json") is False
    assert handler.is_handled_file("notes.txt") is False
