# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed_source import TraceSeedFormat, TraceSeedSource
from data_designer.engine.resources.seed_reader import SeedReaderError, TraceSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver


def test_trace_seed_reader_normalizes_claude_code_directory(tmp_path: Path) -> None:
    session_dir = tmp_path / "project-a"
    subagents_dir = session_dir / "subagents"
    subagents_dir.mkdir(parents=True)

    _write_jsonl(
        session_dir / "session-1.jsonl",
        [
            {
                "type": "system",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "version": "2.1.7",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "user",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {"role": "user", "content": "Inspect the repo"},
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Need to inspect"},
                        {"type": "tool_use", "id": "toolu_1", "name": "ReadFile", "input": {"path": "README.md"}},
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:03Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README contents"}],
                },
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:04Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Repo inspected"}]},
            },
        ],
    )
    _write_jsonl(
        subagents_dir / "agent-a.jsonl",
        [
            {
                "type": "user",
                "sessionId": "session-1",
                "agentId": "agent-a",
                "isSidechain": True,
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:01:00Z",
                "message": {"role": "user", "content": "Check tests"},
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "agentId": "agent-a",
                "isSidechain": True,
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:01:01Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Tests checked"}]},
            },
        ],
    )
    (session_dir / "sessions-index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "sessionId": "session-1",
                        "projectPath": "/repo-from-index",
                        "summary": "Investigate repository",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reader = TraceSeedReader()
    reader.attach(
        TraceSeedSource(path=str(tmp_path), format=TraceSeedFormat.CLAUDE_CODE_DIR),
        PlaintextResolver(),
    )

    assert set(reader.get_column_names()) >= {"trace_id", "messages", "final_assistant_message", "source_meta"}

    df = lazy.pd.read_json(reader.get_dataset_uri(), lines=True).sort_values("trace_id").reset_index(drop=True)
    assert list(df["trace_id"]) == ["session-1", "session-1:agent-a"]

    root_row = df.iloc[0]
    assert root_row["project_path"] == "/repo-from-index"
    assert bool(root_row["is_sidechain"]) is False
    assert root_row["tool_call_count"] == 1
    assert root_row["final_assistant_message"] == "Repo inspected"
    assert [message["role"] for message in root_row["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert root_row["messages"][1]["reasoning_content"] == "Need to inspect"

    sidechain_row = df.iloc[1]
    assert sidechain_row["agent_id"] == "agent-a"
    assert bool(sidechain_row["is_sidechain"]) is True
    assert sidechain_row["final_assistant_message"] == "Tests checked"


def test_trace_seed_reader_normalizes_codex_directory(tmp_path: Path) -> None:
    codex_dir = tmp_path / "sessions" / "2026" / "03" / "10"
    codex_dir.mkdir(parents=True)
    _write_jsonl(
        codex_dir / "rollout-2026-03-10T00-00-00-session.jsonl",
        [
            {
                "timestamp": "2026-03-10T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "codex-session",
                    "timestamp": "2026-03-10T00:00:00Z",
                    "cwd": "/workspace",
                    "cli_version": "0.108.0",
                    "originator": "codex_cli_rs",
                    "model_provider": "openai",
                    "source": "api",
                },
            },
            {
                "timestamp": "2026-03-10T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Follow repo rules"}],
                },
            },
            {
                "timestamp": "2026-03-10T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "List files"}],
                },
            },
            {
                "timestamp": "2026-03-10T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Need to run ls"}],
                },
            },
            {
                "timestamp": "2026-03-10T00:00:04Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                    "call_id": "call_1",
                },
            },
            {
                "timestamp": "2026-03-10T00:00:05Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "README.md\nsrc",
                },
            },
            {
                "timestamp": "2026-03-10T00:00:06Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Listed files"}],
                },
            },
        ],
    )

    reader = TraceSeedReader()
    reader.attach(
        TraceSeedSource(path=str(tmp_path), format=TraceSeedFormat.CODEX_DIR),
        PlaintextResolver(),
    )

    df = lazy.pd.read_json(reader.get_dataset_uri(), lines=True)
    assert len(df) == 1
    row = df.iloc[0]

    assert row["trace_id"] == "codex-session"
    assert row["source_kind"] == "codex"
    assert row["cwd"] == "/workspace"
    assert row["tool_call_count"] == 1
    assert row["final_assistant_message"] == "Listed files"
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert row["messages"][2]["reasoning_content"] == "Need to run ls"
    assert row["messages"][2]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert row["messages"][3]["tool_call_id"] == "call_1"
    assert row["source_meta"]["cli_version"] == "0.108.0"


def test_trace_seed_reader_normalizes_chat_completion_jsonl_directory(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "chat.jsonl",
        [
            {
                "trace_id": "row-1",
                "session_id": "sess-1",
                "split": "train",
                "messages": [
                    {"role": "developer", "content": "Use tools if needed"},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            },
            {"prompt": "Question?", "completion": "Answer."},
        ],
    )

    reader = TraceSeedReader()
    reader.attach(
        TraceSeedSource(path=str(tmp_path), format=TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR),
        PlaintextResolver(),
    )

    df = lazy.pd.read_json(reader.get_dataset_uri(), lines=True).sort_values("trace_id").reset_index(drop=True)

    assert list(df["trace_id"]) == ["chat:2", "row-1"]
    assert df.iloc[0]["final_assistant_message"] == "Answer."
    assert [message["role"] for message in df.iloc[1]["messages"]] == ["system", "user", "assistant"]
    assert df.iloc[1]["source_meta"]["split"] == "train"
    assert df.iloc[1]["source_meta"]["file_line"] == 1


def test_trace_seed_reader_errors_on_invalid_chat_completion_rows(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "invalid.jsonl", [{"unexpected": "shape"}])

    reader = TraceSeedReader()
    reader.attach(
        TraceSeedSource(path=str(tmp_path), format=TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR),
        PlaintextResolver(),
    )

    with pytest.raises(SeedReaderError, match="invalid.jsonl line 1"):
        reader.get_dataset_uri()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(f"{json.dumps(row)}\n")
