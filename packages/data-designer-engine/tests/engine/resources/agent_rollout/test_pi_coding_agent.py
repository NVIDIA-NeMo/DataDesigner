# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from data_designer.engine.resources.agent_rollout.pi_coding_agent import PiCodingAgentRolloutFormatHandler
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError


def _make_handler() -> PiCodingAgentRolloutFormatHandler:
    return PiCodingAgentRolloutFormatHandler()


def _make_session_header(
    *,
    session_id: str = "abc-123",
    cwd: str = "/home/user/project",
    version: int = 3,
    timestamp: str = "2026-04-07T10:00:00.000Z",
) -> dict[str, Any]:
    return {
        "type": "session",
        "version": version,
        "id": session_id,
        "timestamp": timestamp,
        "cwd": cwd,
    }


def _make_entry(
    *,
    entry_type: str = "message",
    entry_id: str,
    parent_id: str | None,
    timestamp: str = "2026-04-07T10:00:01.000Z",
    message: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": entry_type,
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": timestamp,
    }
    if message is not None:
        entry["message"] = message
    entry.update(extra)
    return entry


def test_parse_file_happy_path(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Mirrors the structure of real Pi sessions: structural entries before messages,
    block-list user content, parallel tool calls, and a final text-only assistant."""
    write_jsonl(
        tmp_path / "subdir" / "20260407_abc123.jsonl",
        [
            _make_session_header(),
            # Real sessions start with model_change + thinking_level_change
            _make_entry(
                entry_type="model_change",
                entry_id="m1",
                parent_id=None,
                provider="tcri",
                modelId="tcri/donatello-70b",
            ),
            _make_entry(
                entry_type="thinking_level_change",
                entry_id="t1",
                parent_id="m1",
                thinkingLevel="high",
            ),
            # User message with block-list content (real Pi format)
            _make_entry(
                entry_id="e1",
                parent_id="t1",
                message={
                    "role": "user",
                    "content": [{"type": "text", "text": "Analyze the project structure."}],
                    "timestamp": 1712484001000,
                },
            ),
            # Assistant with thinking + text + tool call
            _make_entry(
                entry_id="e2",
                parent_id="e1",
                message={
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me explore the repo."},
                        {"type": "text", "text": "I'll look at the directory layout."},
                        {
                            "type": "toolCall",
                            "id": "tooluse_abc123",
                            "name": "bash",
                            "arguments": {"command": "find . -type f | head -20"},
                        },
                    ],
                    "api": "foot-clan",
                    "provider": "tcri",
                    "model": "tcri/donatello-70b",
                    "stopReason": "toolUse",
                    "timestamp": 1712484002000,
                },
            ),
            _make_entry(
                entry_id="e3",
                parent_id="e2",
                message={
                    "role": "toolResult",
                    "toolCallId": "tooluse_abc123",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "./src/main.py\n./README.md"}],
                    "isError": False,
                    "timestamp": 1712484003000,
                },
            ),
            # Parallel tool calls: assistant with 2 ToolCall blocks
            _make_entry(
                entry_id="e4",
                parent_id="e3",
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tooluse_read1",
                            "name": "read",
                            "arguments": {"path": "src/main.py"},
                        },
                        {
                            "type": "toolCall",
                            "id": "tooluse_read2",
                            "name": "read",
                            "arguments": {"path": "README.md"},
                        },
                    ],
                    "model": "tcri/donatello-70b",
                    "stopReason": "toolUse",
                    "timestamp": 1712484004000,
                },
            ),
            # Two consecutive toolResult messages
            _make_entry(
                entry_id="e5",
                parent_id="e4",
                message={
                    "role": "toolResult",
                    "toolCallId": "tooluse_read1",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "import sys\nprint('hello')"}],
                    "isError": False,
                    "timestamp": 1712484005000,
                },
            ),
            _make_entry(
                entry_id="e6",
                parent_id="e5",
                message={
                    "role": "toolResult",
                    "toolCallId": "tooluse_read2",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "# My Project\nA sample project."}],
                    "isError": False,
                    "timestamp": 1712484006000,
                },
            ),
            # Final assistant response (text only, stop)
            _make_entry(
                entry_id="e7",
                parent_id="e6",
                timestamp="2026-04-07T10:01:00.000Z",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The project has a simple Python entry point and a README."}],
                    "model": "tcri/donatello-70b",
                    "stopReason": "stop",
                    "timestamp": 1712484007000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(
        root_path=tmp_path,
        relative_path="subdir/20260407_abc123.jsonl",
    )

    assert len(records) == 1
    record = records[0]
    assert record.trace_id == "abc-123"
    assert record.root_session_id == "abc-123"
    assert record.source_kind == "pi_coding_agent"
    assert record.cwd == "/home/user/project"
    assert record.project_path == "/home/user/project"
    assert record.started_at == "2026-04-07T10:00:00.000Z"
    assert record.ended_at == "2026-04-07T10:01:00.000Z"
    assert record.final_assistant_message == "The project has a simple Python entry point and a README."

    # 7 normalized messages: user, assistant(+tc), tool, assistant(+2tc), tool, tool, assistant
    assert record.message_count == 7
    assert record.tool_call_count == 3

    # Message structure
    assert record.messages[0]["role"] == "user"
    assert record.messages[0]["content"][0]["text"] == "Analyze the project structure."

    assert record.messages[1]["role"] == "assistant"
    assert record.messages[1]["reasoning_content"] == "Let me explore the repo."
    assert record.messages[1]["tool_calls"][0]["function"]["name"] == "bash"

    assert record.messages[2]["role"] == "tool"
    assert record.messages[2]["tool_call_id"] == "tooluse_abc123"

    # Parallel tool calls: one assistant message → two tool results
    assert record.messages[3]["role"] == "assistant"
    assert len(record.messages[3]["tool_calls"]) == 2
    assert record.messages[3]["tool_calls"][0]["id"] == "tooluse_read1"
    assert record.messages[3]["tool_calls"][1]["id"] == "tooluse_read2"
    assert record.messages[4]["role"] == "tool"
    assert record.messages[4]["tool_call_id"] == "tooluse_read1"
    assert record.messages[5]["role"] == "tool"
    assert record.messages[5]["tool_call_id"] == "tooluse_read2"

    assert record.messages[6]["role"] == "assistant"
    assert "tool_calls" not in record.messages[6]  # no tool calls on final message

    # Tool call IDs link 1:1 with tool results
    declared_tc_ids = {tc["id"] for m in record.messages for tc in m.get("tool_calls", [])}
    tool_result_ids = {m["tool_call_id"] for m in record.messages if m.get("role") == "tool"}
    assert declared_tc_ids == tool_result_ids

    # Source meta
    assert record.source_meta["record_count"] == 10  # header + 2 structural + 7 message entries
    assert record.source_meta["session_version"] == 3
    assert record.source_meta["stop_reasons"] == ["toolUse", "stop"]
    assert record.source_meta["models_used"] == ["tcri/donatello-70b"]
    assert "model_change" in record.source_meta["entry_types"]
    assert "thinking_level_change" in record.source_meta["entry_types"]
    assert "has_branches" not in record.source_meta


def test_parse_file_assistant_tool_calls_without_text(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Real sessions frequently have assistants with only ToolCall blocks and no text."""
    write_jsonl(
        tmp_path / "tools_only.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={
                    "role": "user",
                    "content": [{"type": "text", "text": "Read all config files."}],
                    "timestamp": 1000,
                },
            ),
            _make_entry(
                entry_id="e2",
                parent_id="e1",
                message={
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "tc_a", "name": "read", "arguments": {"path": "config.yaml"}},
                        {"type": "toolCall", "id": "tc_b", "name": "read", "arguments": {"path": "settings.json"}},
                        {"type": "toolCall", "id": "tc_c", "name": "read", "arguments": {"path": ".env"}},
                    ],
                    "model": "tcri/donatello-70b",
                    "stopReason": "toolUse",
                    "timestamp": 2000,
                },
            ),
            _make_entry(
                entry_id="e3",
                parent_id="e2",
                message={
                    "role": "toolResult",
                    "toolCallId": "tc_a",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "port: 8080"}],
                    "isError": False,
                    "timestamp": 3000,
                },
            ),
            _make_entry(
                entry_id="e4",
                parent_id="e3",
                message={
                    "role": "toolResult",
                    "toolCallId": "tc_b",
                    "toolName": "read",
                    "content": [{"type": "text", "text": '{"debug": true}'}],
                    "isError": False,
                    "timestamp": 4000,
                },
            ),
            _make_entry(
                entry_id="e5",
                parent_id="e4",
                message={
                    "role": "toolResult",
                    "toolCallId": "tc_c",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "SECRET=hunter2"}],
                    "isError": False,
                    "timestamp": 5000,
                },
            ),
            _make_entry(
                entry_id="e6",
                parent_id="e5",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Found 3 config files with settings."}],
                    "stopReason": "stop",
                    "timestamp": 6000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="tools_only.jsonl")
    record = records[0]

    assert record.message_count == 6  # user, assistant(3tc), tool, tool, tool, assistant
    assert record.tool_call_count == 3

    # The tools-only assistant still produces a valid message with empty content
    assistant_msg = record.messages[1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["tool_calls"]) == 3
    tool_names = [tc["function"]["name"] for tc in assistant_msg["tool_calls"]]
    assert tool_names == ["read", "read", "read"]

    # All three tool results are present and linked
    for i, expected_id in enumerate(["tc_a", "tc_b", "tc_c"]):
        assert record.messages[2 + i]["role"] == "tool"
        assert record.messages[2 + i]["tool_call_id"] == expected_id

    assert record.final_assistant_message == "Found 3 config files with settings."


def test_parse_file_resolves_active_branch(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """When a session has branches, the active path follows the last entry back to root."""
    write_jsonl(
        tmp_path / "branched.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={"role": "user", "content": "Hello", "timestamp": 1000},
            ),
            # Branch A (abandoned): e2a follows e1
            _make_entry(
                entry_id="e2a",
                parent_id="e1",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Branch A"}],
                    "stopReason": "stop",
                    "timestamp": 2000,
                },
            ),
            # Branch B (active): e2b also follows e1, and is the last entry
            _make_entry(
                entry_id="e2b",
                parent_id="e1",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Branch B"}],
                    "stopReason": "stop",
                    "timestamp": 3000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="branched.jsonl")

    assert len(records) == 1
    record = records[0]
    # Active path is e1 → e2b (not e2a)
    assert record.message_count == 2
    assert record.messages[0]["role"] == "user"
    assert record.messages[1]["content"][0]["text"] == "Branch B"
    assert record.source_meta["has_branches"] is True


def test_parse_file_normalizes_bash_execution(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    write_jsonl(
        tmp_path / "bash_session.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={
                    "role": "bashExecution",
                    "command": "ls -la",
                    "output": "total 42\ndrwxr-xr-x 5 user user 160 Apr 7 10:00 .",
                    "exitCode": 0,
                    "cancelled": False,
                    "truncated": False,
                    "timestamp": 1000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="bash_session.jsonl")
    record = records[0]

    # bashExecution normalizes to tool-call pair
    assert record.message_count == 2
    assert record.messages[0]["role"] == "assistant"
    assert record.messages[0]["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(record.messages[0]["tool_calls"][0]["function"]["arguments"])["command"] == "ls -la"
    assert record.messages[1]["role"] == "tool"
    assert record.messages[1]["tool_call_id"] == "bash_e1"
    assert "[exit code: 0]" in record.messages[1]["content"][0]["text"]
    assert record.source_meta["bash_execution_count"] == 1


def test_parse_file_skips_excluded_bash_execution(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    write_jsonl(
        tmp_path / "excluded_bash.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={
                    "role": "bashExecution",
                    "command": "echo hello",
                    "output": "hello",
                    "exitCode": 0,
                    "cancelled": False,
                    "truncated": False,
                    "excludeFromContext": True,
                    "timestamp": 1000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="excluded_bash.jsonl")
    assert records[0].message_count == 0


def test_parse_file_requires_session_header(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    write_jsonl(
        tmp_path / "no_header.jsonl",
        [
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={"role": "user", "content": "Hello", "timestamp": 1000},
            ),
        ],
    )

    handler = _make_handler()
    with pytest.raises(AgentRolloutSeedParseError, match="missing a session header"):
        handler.parse_file(root_path=tmp_path, relative_path="no_header.jsonl")


def test_parse_file_empty_file_returns_empty_list(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    write_jsonl(tmp_path / "empty.jsonl", [])

    handler = _make_handler()
    assert handler.parse_file(root_path=tmp_path, relative_path="empty.jsonl") == []


def test_is_handled_file_accepts_jsonl_only() -> None:
    handler = _make_handler()

    assert handler.is_handled_file("20260407_abc123.jsonl") is True
    assert handler.is_handled_file("subdir/20260407_abc123.jsonl") is True
    assert handler.is_handled_file("config.json") is False
    assert handler.is_handled_file("notes.txt") is False


def test_parse_file_includes_custom_display_messages(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    write_jsonl(
        tmp_path / "custom_msg.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={
                    "role": "custom",
                    "customType": "my-extension",
                    "content": "Extension context injected.",
                    "display": True,
                    "timestamp": 1000,
                },
            ),
            _make_entry(
                entry_id="e2",
                parent_id="e1",
                message={
                    "role": "custom",
                    "customType": "my-extension",
                    "content": "Hidden context.",
                    "display": False,
                    "timestamp": 2000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="custom_msg.jsonl")
    record = records[0]

    # Only display=True custom message is included
    assert record.message_count == 1
    assert record.messages[0]["role"] == "system"


def test_parse_file_tracks_model_changes(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    write_jsonl(
        tmp_path / "model_change.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={"role": "user", "content": "Hello", "timestamp": 1000},
            ),
            _make_entry(
                entry_type="model_change",
                entry_id="e2",
                parent_id="e1",
                provider="dimension-x",
                modelId="krang/raphael-405b",
            ),
            _make_entry(
                entry_id="e3",
                parent_id="e2",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi!"}],
                    "model": "krang/raphael-405b",
                    "stopReason": "stop",
                    "timestamp": 2000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="model_change.jsonl")
    assert records[0].source_meta["models_used"] == ["krang/raphael-405b"]


def test_parse_file_includes_compaction_summary_message(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """compactionSummary role inside a message entry is normalized as a system message."""
    write_jsonl(
        tmp_path / "compaction_msg.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={
                    "role": "compactionSummary",
                    "summary": "User discussed project setup and ran several commands.",
                    "tokensBefore": 50000,
                    "timestamp": 1000,
                },
            ),
            _make_entry(
                entry_id="e2",
                parent_id="e1",
                message={"role": "user", "content": "Continue please.", "timestamp": 2000},
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="compaction_msg.jsonl")
    record = records[0]

    assert record.message_count == 2
    assert record.messages[0]["role"] == "system"
    assert "project setup" in record.messages[0]["content"][0]["text"]
    assert record.messages[1]["role"] == "user"


def test_parse_file_handles_compaction_entry(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Entry-level compaction (type=compaction) is normalized as a system message."""
    write_jsonl(
        tmp_path / "compaction_entry.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_type="compaction",
                entry_id="e1",
                parent_id=None,
                summary="Earlier discussion about API design and error handling.",
                firstKeptEntryId="e0",
                tokensBefore=40000,
            ),
            _make_entry(
                entry_id="e2",
                parent_id="e1",
                message={"role": "user", "content": "Now let's add tests.", "timestamp": 2000},
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="compaction_entry.jsonl")
    record = records[0]

    assert record.message_count == 2
    assert record.messages[0]["role"] == "system"
    assert "API design" in record.messages[0]["content"][0]["text"]
    assert record.messages[1]["role"] == "user"


def test_parse_file_handles_branch_summary_entry(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Entry-level branch_summary is normalized as a system message on the active path."""
    write_jsonl(
        tmp_path / "branch_summary.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={"role": "user", "content": "Try approach A", "timestamp": 1000},
            ),
            _make_entry(
                entry_id="e2",
                parent_id="e1",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Approach A failed."}],
                    "stopReason": "stop",
                    "timestamp": 2000,
                },
            ),
            # User switched branches; branch summary injected for abandoned branch
            _make_entry(
                entry_type="branch_summary",
                entry_id="e3",
                parent_id="e1",
                fromId="e2",
                summary="Explored approach A which failed due to compatibility issues.",
            ),
            _make_entry(
                entry_id="e4",
                parent_id="e3",
                message={"role": "user", "content": "Try approach B instead.", "timestamp": 3000},
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="branch_summary.jsonl")
    record = records[0]

    # Active path: e1 → e3 (branch_summary) → e4
    assert record.message_count == 3
    assert record.messages[0]["role"] == "user"
    assert record.messages[0]["content"][0]["text"] == "Try approach A"
    assert record.messages[1]["role"] == "system"
    assert "approach A" in record.messages[1]["content"][0]["text"]
    assert record.messages[2]["role"] == "user"
    assert record.messages[2]["content"][0]["text"] == "Try approach B instead."


def test_parse_file_handles_custom_message_entry(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Entry-level custom_message (display=true) is included; display=false is skipped."""
    write_jsonl(
        tmp_path / "custom_entry.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_type="custom_message",
                entry_id="e1",
                parent_id=None,
                customType="lint-hook",
                content="Lint found 3 warnings in src/app.ts.",
                display=True,
            ),
            _make_entry(
                entry_type="custom_message",
                entry_id="e2",
                parent_id="e1",
                customType="internal-state",
                content="Cache refreshed.",
                display=False,
            ),
            _make_entry(
                entry_id="e3",
                parent_id="e2",
                message={"role": "user", "content": "Fix those warnings.", "timestamp": 2000},
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="custom_entry.jsonl")
    record = records[0]

    assert record.message_count == 2
    assert record.messages[0]["role"] == "system"
    assert "3 warnings" in record.messages[0]["content"][0]["text"]
    assert record.messages[1]["role"] == "user"


def test_parse_file_handles_branch_summary_message_role(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """branchSummary role inside a message entry is normalized as a system message."""
    write_jsonl(
        tmp_path / "branch_msg.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={
                    "role": "branchSummary",
                    "summary": "Abandoned branch tried regex-based parsing.",
                    "fromId": "prev1",
                    "timestamp": 1000,
                },
            ),
            _make_entry(
                entry_id="e2",
                parent_id="e1",
                message={"role": "user", "content": "Use an AST parser instead.", "timestamp": 2000},
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="branch_msg.jsonl")
    record = records[0]

    assert record.message_count == 2
    assert record.messages[0]["role"] == "system"
    assert "regex-based parsing" in record.messages[0]["content"][0]["text"]
    assert record.messages[1]["role"] == "user"


def test_parse_file_includes_parent_session_in_source_meta(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Forked sessions include the parent session path in source_meta."""
    header = _make_session_header()
    header["parentSession"] = "/home/user/.pi/agent/sessions/--project--/original.jsonl"
    write_jsonl(
        tmp_path / "forked.jsonl",
        [
            header,
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={"role": "user", "content": "Continue from where we left off.", "timestamp": 1000},
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="forked.jsonl")
    assert records[0].source_meta["parent_session"] == "/home/user/.pi/agent/sessions/--project--/original.jsonl"


def test_parse_file_assistant_bare_string_content(
    tmp_path: Path,
    write_jsonl: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Assistant messages with bare string content (not a block list) are normalized."""
    write_jsonl(
        tmp_path / "bare_string.jsonl",
        [
            _make_session_header(),
            _make_entry(
                entry_id="e1",
                parent_id=None,
                message={
                    "role": "assistant",
                    "content": "Plain text response.",
                    "stopReason": "stop",
                    "timestamp": 1000,
                },
            ),
        ],
    )

    handler = _make_handler()
    records = handler.parse_file(root_path=tmp_path, relative_path="bare_string.jsonl")
    record = records[0]

    assert record.message_count == 1
    assert record.final_assistant_message == "Plain text response."
