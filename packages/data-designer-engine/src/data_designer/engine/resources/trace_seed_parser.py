# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_designer.config.seed_source import TraceSeedFormat, TraceSeedSource
from data_designer.engine.models.utils import ChatMessage


class TraceSeedParseError(ValueError): ...


SOURCE_KIND_BY_FORMAT: dict[TraceSeedFormat, str] = {
    TraceSeedFormat.CLAUDE_CODE_DIR: "claude_code",
    TraceSeedFormat.CODEX_DIR: "codex",
    TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR: "chat_completion_jsonl",
}


@dataclass
class NormalizedTraceRecord:
    trace_id: str
    source_kind: str
    source_path: str
    root_session_id: str
    agent_id: str | None
    is_sidechain: bool
    cwd: str | None
    project_path: str | None
    git_branch: str | None
    started_at: str | None
    ended_at: str | None
    messages: list[dict[str, Any]]
    message_count: int
    tool_call_count: int
    final_assistant_message: str | None
    source_meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "root_session_id": self.root_session_id,
            "agent_id": self.agent_id,
            "is_sidechain": self.is_sidechain,
            "cwd": self.cwd,
            "project_path": self.project_path,
            "git_branch": self.git_branch,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "messages": self.messages,
            "message_count": self.message_count,
            "tool_call_count": self.tool_call_count,
            "final_assistant_message": self.final_assistant_message,
            "source_meta": self.source_meta,
        }


def normalize_trace_source(source: TraceSeedSource) -> list[dict[str, Any]]:
    root_path = Path(source.path)
    if source.format == TraceSeedFormat.CLAUDE_CODE_DIR:
        records = _normalize_claude_directory(root_path)
    elif source.format == TraceSeedFormat.CODEX_DIR:
        records = _normalize_codex_directory(root_path)
    elif source.format == TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR:
        records = _normalize_chat_completion_directory(root_path)
    else:
        raise TraceSeedParseError(f"Unsupported trace format: {source.format}")
    return [record.to_dict() for record in records]


def _normalize_claude_directory(root_path: Path) -> list[NormalizedTraceRecord]:
    source_kind = SOURCE_KIND_BY_FORMAT[TraceSeedFormat.CLAUDE_CODE_DIR]
    session_index = _load_claude_session_index(root_path)
    trace_files = sorted(
        path for path in root_path.rglob("*.jsonl") if "tool-results" not in path.parts and path.name != "history.jsonl"
    )
    if not trace_files:
        raise TraceSeedParseError(f"No Claude Code session JSONL files found under {root_path}")

    normalized_records: list[NormalizedTraceRecord] = []
    for file_path in trace_files:
        rows = _load_jsonl_rows(file_path)
        if not rows:
            raise TraceSeedParseError(f"Claude Code trace file {file_path} is empty")

        messages: list[dict[str, Any]] = []
        timestamps: list[str] = []
        versions: set[str] = set()
        raw_types: set[str] = set()
        session_id: str | None = None
        agent_id: str | None = None
        cwd: str | None = None
        git_branch: str | None = None
        is_sidechain = False

        for _, raw_record in rows:
            raw_types.add(str(raw_record.get("type", "unknown")))
            if timestamp := _coerce_optional_str(raw_record.get("timestamp")):
                timestamps.append(timestamp)
            session_id = session_id or _coerce_optional_str(raw_record.get("sessionId"))
            agent_id = agent_id or _coerce_optional_str(raw_record.get("agentId"))
            cwd = cwd or _coerce_optional_str(raw_record.get("cwd"))
            git_branch = git_branch or _coerce_optional_str(raw_record.get("gitBranch"))
            version = _coerce_optional_str(raw_record.get("version"))
            if version:
                versions.add(version)
            is_sidechain = is_sidechain or bool(raw_record.get("isSidechain"))

            record_type = raw_record.get("type")
            if record_type == "assistant":
                messages.extend(_normalize_claude_assistant_messages(raw_record))
            elif record_type == "user":
                messages.extend(_normalize_claude_user_messages(raw_record))

        session_key = session_id or file_path.stem
        index_entry = session_index.get(session_key, {})
        project_path = _coerce_optional_str(index_entry.get("projectPath")) or cwd
        trace_id = f"{session_key}:{agent_id}" if agent_id else session_key
        source_meta = {
            "record_count": len(rows),
            "record_types": sorted(raw_types),
        }
        if versions:
            source_meta["claude_versions"] = sorted(versions)
        if summary := _coerce_optional_str(index_entry.get("summary")):
            source_meta["summary"] = summary
        if first_prompt := _coerce_optional_str(index_entry.get("firstPrompt")):
            source_meta["first_prompt"] = first_prompt

        normalized_records.append(
            _build_trace_record(
                trace_id=trace_id,
                source_kind=source_kind,
                source_path=str(file_path),
                root_session_id=session_key,
                agent_id=agent_id,
                is_sidechain=is_sidechain,
                cwd=cwd,
                project_path=project_path,
                git_branch=git_branch,
                started_at=min(timestamps) if timestamps else None,
                ended_at=max(timestamps) if timestamps else None,
                messages=messages,
                source_meta=source_meta,
            )
        )

    return normalized_records


def _normalize_codex_directory(root_path: Path) -> list[NormalizedTraceRecord]:
    source_kind = SOURCE_KIND_BY_FORMAT[TraceSeedFormat.CODEX_DIR]
    trace_files = sorted(root_path.rglob("rollout-*.jsonl"))
    if not trace_files:
        raise TraceSeedParseError(f"No Codex rollout JSONL files found under {root_path}")

    normalized_records: list[NormalizedTraceRecord] = []
    for file_path in trace_files:
        rows = _load_jsonl_rows(file_path)
        if not rows:
            raise TraceSeedParseError(f"Codex rollout file {file_path} is empty")

        messages: list[dict[str, Any]] = []
        timestamps: list[str] = []
        pending_reasoning: list[str] = []
        raw_types: set[str] = set()
        response_item_types: set[str] = set()
        session_meta: dict[str, Any] = {}

        for _, raw_record in rows:
            record_type = _coerce_optional_str(raw_record.get("type")) or "unknown"
            raw_types.add(record_type)
            if timestamp := _coerce_optional_str(raw_record.get("timestamp")):
                timestamps.append(timestamp)

            payload = raw_record.get("payload")
            if not isinstance(payload, dict):
                continue

            if record_type == "session_meta":
                session_meta = payload
                if session_timestamp := _coerce_optional_str(payload.get("timestamp")):
                    timestamps.append(session_timestamp)
                continue

            if record_type == "event_msg":
                event_type = _coerce_optional_str(payload.get("type"))
                if event_type == "agent_reasoning" and (reasoning_text := _coerce_optional_str(payload.get("text"))):
                    pending_reasoning.append(reasoning_text)
                continue

            if record_type != "response_item":
                continue

            item_type = _coerce_optional_str(payload.get("type")) or "unknown"
            response_item_types.add(item_type)

            if item_type == "message":
                role = _normalize_message_role(payload.get("role"), context=f"Codex message in {file_path}")
                reasoning_content = _consume_pending_reasoning(pending_reasoning) if role == "assistant" else None
                messages.append(
                    _build_message(
                        role=role,
                        content=payload.get("content"),
                        reasoning_content=reasoning_content,
                    )
                )
            elif item_type == "function_call":
                messages.append(
                    _build_message(
                        role="assistant",
                        content="",
                        reasoning_content=_consume_pending_reasoning(pending_reasoning),
                        tool_calls=[
                            {
                                "id": _require_string(payload.get("call_id"), f"Codex tool call id in {file_path}"),
                                "type": "function",
                                "function": {
                                    "name": _require_string(payload.get("name"), f"Codex tool name in {file_path}"),
                                    "arguments": _stringify_json_value(payload.get("arguments")),
                                },
                            }
                        ],
                    )
                )
            elif item_type == "function_call_output":
                messages.append(
                    _build_message(
                        role="tool",
                        content=payload.get("output"),
                        tool_call_id=_require_string(payload.get("call_id"), f"Codex tool output id in {file_path}"),
                    )
                )
            elif item_type == "reasoning":
                pending_reasoning.extend(_extract_codex_reasoning_summaries(payload))

        session_id = _coerce_optional_str(session_meta.get("id")) or file_path.stem
        source_meta = {
            "record_count": len(rows),
            "record_types": sorted(raw_types),
            "response_item_types": sorted(response_item_types),
        }
        for field_name in ("originator", "cli_version", "model_provider", "source"):
            value = _coerce_optional_str(session_meta.get(field_name))
            if value:
                source_meta[field_name] = value
        if pending_reasoning:
            source_meta["unattached_reasoning"] = pending_reasoning

        normalized_records.append(
            _build_trace_record(
                trace_id=session_id,
                source_kind=source_kind,
                source_path=str(file_path),
                root_session_id=session_id,
                agent_id=None,
                is_sidechain=False,
                cwd=_coerce_optional_str(session_meta.get("cwd")),
                project_path=_coerce_optional_str(session_meta.get("cwd")),
                git_branch=_coerce_optional_str(session_meta.get("git_branch")),
                started_at=_coerce_optional_str(session_meta.get("timestamp"))
                or (min(timestamps) if timestamps else None),
                ended_at=max(timestamps) if timestamps else None,
                messages=messages,
                source_meta=source_meta,
            )
        )

    return normalized_records


def _normalize_chat_completion_directory(root_path: Path) -> list[NormalizedTraceRecord]:
    source_kind = SOURCE_KIND_BY_FORMAT[TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR]
    trace_files = sorted(root_path.rglob("*.jsonl"))
    if not trace_files:
        raise TraceSeedParseError(f"No chat-completion JSONL files found under {root_path}")

    normalized_records: list[NormalizedTraceRecord] = []
    for file_path in trace_files:
        rows = _load_jsonl_rows(file_path)
        if not rows:
            raise TraceSeedParseError(f"Chat-completion JSONL file {file_path} is empty")

        for line_number, raw_record in rows:
            if "messages" in raw_record:
                raw_messages = raw_record.get("messages")
                if not isinstance(raw_messages, list):
                    raise TraceSeedParseError(
                        f"Expected 'messages' to be a list in {file_path} line {line_number}, got {type(raw_messages).__name__}"
                    )
                messages = [
                    _normalize_chat_completion_message(message, file_path=file_path, line_number=line_number)
                    for message in raw_messages
                ]
            elif "prompt" in raw_record and "completion" in raw_record:
                messages = [
                    _build_message(role="user", content=raw_record.get("prompt")),
                    _build_message(role="assistant", content=raw_record.get("completion")),
                ]
            else:
                raise TraceSeedParseError(
                    f"Unsupported chat-completion JSONL row in {file_path} line {line_number}; expected "
                    "'messages' or 'prompt'/'completion'"
                )

            trace_id = (
                _coerce_optional_str(raw_record.get("trace_id"))
                or _coerce_optional_str(raw_record.get("id"))
                or f"{file_path.stem}:{line_number}"
            )
            root_session_id = _coerce_optional_str(raw_record.get("session_id")) or trace_id
            source_meta = {
                key: value
                for key, value in raw_record.items()
                if key not in {"messages", "prompt", "completion", "trace_id", "id", "session_id"}
            }
            source_meta["file_line"] = line_number

            normalized_records.append(
                _build_trace_record(
                    trace_id=trace_id,
                    source_kind=source_kind,
                    source_path=str(file_path),
                    root_session_id=root_session_id,
                    agent_id=None,
                    is_sidechain=False,
                    cwd=None,
                    project_path=None,
                    git_branch=None,
                    started_at=None,
                    ended_at=None,
                    messages=messages,
                    source_meta=source_meta,
                )
            )

    return normalized_records


def _normalize_claude_assistant_messages(raw_record: dict[str, Any]) -> list[dict[str, Any]]:
    message_payload = raw_record.get("message")
    if not isinstance(message_payload, dict):
        raise TraceSeedParseError(f"Claude assistant record is missing a message payload: {raw_record}")

    content_blocks = _coerce_raw_blocks(message_payload.get("content"))
    assistant_content: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []

    for block in content_blocks:
        block_type = _coerce_optional_str(block.get("type"))
        if block_type == "text":
            assistant_content.append(_normalize_content_block(block))
        elif block_type == "thinking":
            reasoning_text = _coerce_optional_str(block.get("thinking")) or _coerce_optional_str(block.get("text"))
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
        elif block_type == "tool_use":
            tool_calls.append(_normalize_claude_tool_call(block))
        elif block_type == "tool_result":
            tool_messages.append(
                _build_message(
                    role="tool",
                    content=block.get("content"),
                    tool_call_id=_require_string(block.get("tool_use_id"), "Claude tool_result tool_use_id"),
                )
            )
        else:
            assistant_content.append(_normalize_content_block(block))

    normalized_messages: list[dict[str, Any]] = []
    if assistant_content or reasoning_parts or tool_calls:
        normalized_messages.append(
            _build_message(
                role="assistant",
                content=assistant_content,
                reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
                tool_calls=tool_calls,
            )
        )
    normalized_messages.extend(tool_messages)
    return normalized_messages


def _normalize_claude_user_messages(raw_record: dict[str, Any]) -> list[dict[str, Any]]:
    message_payload = raw_record.get("message")
    if not isinstance(message_payload, dict):
        raise TraceSeedParseError(f"Claude user record is missing a message payload: {raw_record}")

    content = message_payload.get("content")
    if isinstance(content, dict) and _coerce_optional_str(content.get("type")) == "tool_result":
        return [
            _build_message(
                role="tool",
                content=content.get("content"),
                tool_call_id=_require_string(content.get("tool_use_id"), "Claude tool_result tool_use_id"),
            )
        ]
    if isinstance(content, list):
        user_content: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        for block in _coerce_raw_blocks(content):
            if _coerce_optional_str(block.get("type")) == "tool_result":
                tool_messages.append(
                    _build_message(
                        role="tool",
                        content=block.get("content"),
                        tool_call_id=_require_string(block.get("tool_use_id"), "Claude tool_result tool_use_id"),
                    )
                )
            else:
                user_content.append(_normalize_content_block(block))

        normalized_messages: list[dict[str, Any]] = []
        if user_content:
            normalized_messages.append(_build_message(role="user", content=user_content))
        normalized_messages.extend(tool_messages)
        return normalized_messages

    return [_build_message(role="user", content=content)]


def _normalize_chat_completion_message(
    raw_message: Any,
    *,
    file_path: Path,
    line_number: int,
) -> dict[str, Any]:
    if not isinstance(raw_message, dict):
        raise TraceSeedParseError(
            f"Expected chat-completion message object in {file_path} line {line_number}, got {type(raw_message).__name__}"
        )

    role = _normalize_message_role(raw_message.get("role"), context=f"{file_path} line {line_number}")
    tool_call_id = _coerce_optional_str(raw_message.get("tool_call_id"))
    if role == "tool" and tool_call_id is None:
        raise TraceSeedParseError(f"Tool message is missing tool_call_id in {file_path} line {line_number}")

    raw_tool_calls = raw_message.get("tool_calls")
    tool_calls = None
    if raw_tool_calls is not None:
        if not isinstance(raw_tool_calls, list):
            raise TraceSeedParseError(f"'tool_calls' must be a list in {file_path} line {line_number}")
        tool_calls = [_normalize_tool_call(raw_tool_call) for raw_tool_call in raw_tool_calls]

    reasoning_content = _coerce_optional_str(raw_message.get("reasoning_content"))
    return _build_message(
        role=role,
        content=raw_message.get("content"),
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def _normalize_claude_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _require_string(block.get("id"), "Claude tool_use id"),
        "type": "function",
        "function": {
            "name": _require_string(block.get("name"), "Claude tool_use name"),
            "arguments": _stringify_json_value(block.get("input")),
        },
    }


def _normalize_tool_call(raw_tool_call: Any) -> dict[str, Any]:
    if not isinstance(raw_tool_call, dict):
        raise TraceSeedParseError(f"Tool call must be a dict, got {type(raw_tool_call).__name__}")

    if "function" in raw_tool_call:
        function_payload = raw_tool_call.get("function")
        if not isinstance(function_payload, dict):
            raise TraceSeedParseError("Tool call 'function' payload must be a dict")
        return {
            "id": _coerce_optional_str(raw_tool_call.get("id")) or "",
            "type": _coerce_optional_str(raw_tool_call.get("type")) or "function",
            "function": {
                "name": _require_string(function_payload.get("name"), "tool call function name"),
                "arguments": _stringify_json_value(function_payload.get("arguments")),
            },
        }

    if "name" in raw_tool_call:
        return {
            "id": _coerce_optional_str(raw_tool_call.get("id")) or "",
            "type": "function",
            "function": {
                "name": _require_string(raw_tool_call.get("name"), "tool call name"),
                "arguments": _stringify_json_value(raw_tool_call.get("arguments") or raw_tool_call.get("input")),
            },
        }

    raise TraceSeedParseError(f"Unsupported tool call structure: {raw_tool_call}")


def _build_trace_record(
    *,
    trace_id: str,
    source_kind: str,
    source_path: str,
    root_session_id: str,
    agent_id: str | None,
    is_sidechain: bool,
    cwd: str | None,
    project_path: str | None,
    git_branch: str | None,
    started_at: str | None,
    ended_at: str | None,
    messages: list[dict[str, Any]],
    source_meta: dict[str, Any],
) -> NormalizedTraceRecord:
    final_assistant_message = None
    for message in reversed(messages):
        if message.get("role") == "assistant":
            text = _extract_text_from_content(message.get("content"))
            if text:
                final_assistant_message = text
                break

    tool_call_count = sum(len(message.get("tool_calls", [])) for message in messages)
    return NormalizedTraceRecord(
        trace_id=trace_id,
        source_kind=source_kind,
        source_path=source_path,
        root_session_id=root_session_id,
        agent_id=agent_id,
        is_sidechain=is_sidechain,
        cwd=cwd,
        project_path=project_path,
        git_branch=git_branch,
        started_at=started_at,
        ended_at=ended_at,
        messages=messages,
        message_count=len(messages),
        tool_call_count=tool_call_count,
        final_assistant_message=final_assistant_message,
        source_meta=source_meta,
    )


def _build_message(
    *,
    role: str,
    content: Any,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    normalized_role = _normalize_message_role(role, context="trace message")
    return ChatMessage(
        role=normalized_role,
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or [],
        tool_call_id=tool_call_id,
    ).to_dict()


def _normalize_message_role(raw_role: Any, *, context: str) -> str:
    role = _coerce_optional_str(raw_role)
    if role == "developer":
        return "system"
    if role in {"user", "assistant", "system", "tool"}:
        return role
    raise TraceSeedParseError(f"Unsupported message role {raw_role!r} in {context}")


def _normalize_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        block_type = _coerce_optional_str(block.get("type"))
        if block_type in {"text", "input_text", "output_text"} and "text" in block:
            return {"type": "text", "text": _stringify_text_value(block.get("text"))}
        if block_type == "thinking":
            return {"type": "text", "text": _stringify_text_value(block.get("thinking") or block.get("text"))}
        if block_type is not None:
            return block
    return {"type": "text", "text": _stringify_text_value(block)}


def _coerce_content_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, list):
        return [_normalize_content_block(block) for block in content]
    return [_normalize_content_block(content)]


def _coerce_raw_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, list):
        return [
            block if isinstance(block, dict) else {"type": "text", "text": _stringify_text_value(block)}
            for block in content
        ]
    if isinstance(content, dict):
        return [content]
    return [{"type": "text", "text": _stringify_text_value(content)}]


def _extract_text_from_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = _coerce_optional_str(block.get("type"))
                if block_type in {"text", "input_text", "output_text"} and "text" in block:
                    text_parts.append(_stringify_text_value(block.get("text")))
                elif "text" in block:
                    text_parts.append(_stringify_text_value(block.get("text")))
            else:
                text_parts.append(_stringify_text_value(block))
        text = "\n\n".join(part for part in text_parts if part)
        return text or None
    return _coerce_optional_str(content)


def _extract_codex_reasoning_summaries(payload: dict[str, Any]) -> list[str]:
    summaries = payload.get("summary")
    if not isinstance(summaries, list):
        return []
    reasoning_parts: list[str] = []
    for summary in summaries:
        if isinstance(summary, dict) and (text := _coerce_optional_str(summary.get("text"))):
            reasoning_parts.append(text)
    return reasoning_parts


def _consume_pending_reasoning(pending_reasoning: list[str]) -> str | None:
    if not pending_reasoning:
        return None
    joined = "\n\n".join(pending_reasoning)
    pending_reasoning.clear()
    return joined


def _load_claude_session_index(root_path: Path) -> dict[str, dict[str, Any]]:
    entries_by_session_id: dict[str, dict[str, Any]] = {}
    for index_path in sorted(root_path.rglob("sessions-index.json")):
        with index_path.open(encoding="utf-8") as file:
            index_payload = json.load(file)
        entries = index_payload.get("entries", [])
        if not isinstance(entries, list):
            raise TraceSeedParseError(f"Claude sessions index at {index_path} is missing an 'entries' list")
        for entry in entries:
            if isinstance(entry, dict) and (session_id := _coerce_optional_str(entry.get("sessionId"))):
                entries_by_session_id[session_id] = entry
    return entries_by_session_id


def _load_jsonl_rows(file_path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with file_path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                parsed_line = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise TraceSeedParseError(f"Invalid JSON in {file_path} line {line_number}: {error.msg}") from error
            if not isinstance(parsed_line, dict):
                raise TraceSeedParseError(
                    f"Expected JSON object in {file_path} line {line_number}, got {type(parsed_line).__name__}"
                )
            rows.append((line_number, parsed_line))
    return rows


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TraceSeedParseError(f"Expected non-empty string for {context}, got {value!r}")
    return value


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _stringify_json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, sort_keys=True)


def _stringify_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
