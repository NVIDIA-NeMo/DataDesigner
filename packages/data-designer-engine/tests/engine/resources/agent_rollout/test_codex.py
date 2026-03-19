# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_designer.engine.resources.agent_rollout.codex import CodexAgentRolloutFormatHandler


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _make_handler() -> CodexAgentRolloutFormatHandler:
    return CodexAgentRolloutFormatHandler()


def test_parse_file_comprehensive_happy_path(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "rollout-abc.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "sess-1",
                    "cwd": "/project",
                    "git_branch": "main",
                    "originator": "codex-cli",
                    "cli_version": "1.0.0",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "List files"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": "Thinking step 1"},
            },
            {
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": "Thinking step 2"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Here are the files"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "shell",
                    "arguments": '{"cmd": "ls"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "file1.txt\nfile2.txt",
                },
            },
        ],
    )
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="rollout-abc.jsonl")

    assert len(records) == 1
    record = records[0]
    assert record.root_session_id == "sess-1"
    assert record.source_kind == "codex"
    assert record.message_count == 4
    assert record.final_assistant_message == "Here are the files"
    assert record.tool_call_count == 1

    assistant_msg = next(m for m in record.messages if m["role"] == "assistant")
    assert assistant_msg["reasoning_content"] == "Thinking step 1\n\nThinking step 2"

    tool_msg = next(m for m in record.messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call-1"

    assert record.cwd == "/project"
    assert record.git_branch == "main"
    assert record.source_meta["originator"] == "codex-cli"
    assert record.source_meta["cli_version"] == "1.0.0"


def test_parse_file_records_unattached_reasoning_in_source_meta(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "rollout-abc.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "sess-1"}},
            {
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": "Orphan thought"},
            },
        ],
    )
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="rollout-abc.jsonl")

    assert records[0].source_meta["unattached_reasoning"] == ["Orphan thought"]


def test_is_handled_file_requires_rollout_prefix() -> None:
    handler = _make_handler()
    assert handler.is_handled_file("rollout-abc.jsonl") is True
    assert handler.is_handled_file("subdir/rollout-xyz.jsonl") is True
    assert handler.is_handled_file("session.jsonl") is False
    assert handler.is_handled_file("rollout-abc.json") is False
