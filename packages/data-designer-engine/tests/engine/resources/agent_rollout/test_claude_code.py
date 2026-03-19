# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_designer.engine.resources.agent_rollout.claude_code import (
    ClaudeCodeAgentRolloutFormatHandler,
    ClaudeCodeParseContext,
)
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError


def _make_handler() -> ClaudeCodeAgentRolloutFormatHandler:
    return ClaudeCodeAgentRolloutFormatHandler()


def test_parse_file_comprehensive_happy_path(
    tmp_path: Path, write_jsonl: Callable[[Path, list[dict[str, Any]]], None]
) -> None:
    write_jsonl(
        tmp_path / "session.jsonl",
        [
            {"type": "user", "sessionId": "s1", "message": {"content": "Do something"}},
            {
                "type": "assistant",
                "sessionId": "s1",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "read_file",
                            "input": {"path": "/tmp/x"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "s1",
                "message": {
                    "content": {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "file contents",
                    }
                },
            },
            {
                "type": "assistant",
                "sessionId": "s1",
                "message": {"content": [{"type": "text", "text": "Done"}]},
            },
        ],
    )
    session_index = {"s1": {"projectPath": "/my/project", "summary": "A test session"}}
    ctx = ClaudeCodeParseContext(session_index=session_index)
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="session.jsonl", parse_context=ctx)

    assert len(records) == 1
    record = records[0]
    assert record.root_session_id == "s1"
    assert record.source_kind == "claude_code"
    assert record.final_assistant_message == "Done"
    assert record.tool_call_count == 1
    assert record.message_count == 4

    assistant_msg = next(m for m in record.messages if m["role"] == "assistant" and "reasoning_content" in m)
    assert assistant_msg["reasoning_content"] == "Let me think..."

    tool_msg = next(m for m in record.messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "tool-1"

    assert record.project_path == "/my/project"
    assert record.source_meta["summary"] == "A test session"


def test_parse_file_skips_empty_files(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="empty.jsonl")
    assert records == []


def test_parse_file_raises_on_malformed_assistant_record(
    tmp_path: Path, write_jsonl: Callable[[Path, list[dict[str, Any]]], None]
) -> None:
    write_jsonl(
        tmp_path / "session.jsonl",
        [{"type": "assistant", "sessionId": "s1", "message": "not a dict"}],
    )
    handler = _make_handler()
    with pytest.raises(AgentRolloutSeedParseError, match="missing a message payload"):
        handler.parse_file(root_path=tmp_path, relative_path="session.jsonl")


def test_is_handled_file_accepts_jsonl_rejects_tool_results_and_history() -> None:
    handler = _make_handler()
    assert handler.is_handled_file("project/session.jsonl") is True
    assert handler.is_handled_file("project/tool-results/calls.jsonl") is False
    assert handler.is_handled_file("project/history.jsonl") is False
    assert handler.is_handled_file("project/data.json") is False
