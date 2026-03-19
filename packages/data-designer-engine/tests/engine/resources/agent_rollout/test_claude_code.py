# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data_designer.engine.resources.agent_rollout.claude_code import (
    ClaudeCodeAgentRolloutFormatHandler,
    ClaudeCodeParseContext,
    coerce_raw_blocks,
    normalize_content_block,
)
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _make_handler() -> ClaudeCodeAgentRolloutFormatHandler:
    return ClaudeCodeAgentRolloutFormatHandler()


def test_normalize_content_block_handles_text_variants() -> None:
    assert normalize_content_block({"type": "text", "text": "hi"}) == {"type": "text", "text": "hi"}
    assert normalize_content_block({"type": "input_text", "text": "x"}) == {"type": "text", "text": "x"}
    assert normalize_content_block({"type": "output_text", "text": "y"}) == {"type": "text", "text": "y"}


def test_normalize_content_block_wraps_plain_strings() -> None:
    assert normalize_content_block("raw text") == {"type": "text", "text": "raw text"}


def test_coerce_raw_blocks_handles_none_dict_list_scalar() -> None:
    assert coerce_raw_blocks(None) == []
    assert coerce_raw_blocks({"type": "text", "text": "a"}) == [{"type": "text", "text": "a"}]
    assert coerce_raw_blocks([{"type": "text", "text": "a"}]) == [{"type": "text", "text": "a"}]
    assert coerce_raw_blocks(42) == [{"type": "text", "text": "42"}]


def test_parse_file_produces_normalized_record_from_minimal_trace(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "session.jsonl",
        [
            {"type": "user", "sessionId": "s1", "message": {"content": "Hello"}},
            {
                "type": "assistant",
                "sessionId": "s1",
                "message": {"content": [{"type": "text", "text": "Hi there"}]},
            },
        ],
    )
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="session.jsonl")

    assert len(records) == 1
    record = records[0]
    assert record.root_session_id == "s1"
    assert record.message_count == 2
    assert record.final_assistant_message == "Hi there"
    assert record.source_kind == "claude_code"


def test_parse_file_extracts_thinking_blocks_as_reasoning_content(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "session.jsonl",
        [
            {"type": "user", "sessionId": "s1", "message": {"content": "Explain"}},
            {
                "type": "assistant",
                "sessionId": "s1",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Here is my answer"},
                    ]
                },
            },
        ],
    )
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="session.jsonl")

    assistant_msg = next(m for m in records[0].messages if m["role"] == "assistant")
    assert assistant_msg["reasoning_content"] == "Let me think..."


def test_parse_file_normalizes_tool_use_and_tool_result_pairs(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "session.jsonl",
        [
            {"type": "user", "sessionId": "s1", "message": {"content": "Do something"}},
            {
                "type": "assistant",
                "sessionId": "s1",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "read_file",
                            "input": {"path": "/tmp/x"},
                        },
                    ]
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
        ],
    )
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="session.jsonl")

    messages = records[0].messages
    assert records[0].tool_call_count == 1
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "tool-1"


def test_parse_file_collects_session_index_metadata(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "session.jsonl",
        [
            {"type": "user", "sessionId": "s1", "message": {"content": "Hi"}},
            {
                "type": "assistant",
                "sessionId": "s1",
                "message": {"content": [{"type": "text", "text": "Hello"}]},
            },
        ],
    )
    session_index = {"s1": {"projectPath": "/my/project", "summary": "A test session"}}
    ctx = ClaudeCodeParseContext(session_index=session_index)
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="session.jsonl", parse_context=ctx)

    assert records[0].project_path == "/my/project"
    assert records[0].source_meta["summary"] == "A test session"


def test_parse_file_skips_empty_files(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="empty.jsonl")
    assert records == []


def test_parse_file_raises_on_malformed_assistant_record(tmp_path: Path) -> None:
    _write_jsonl(
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
