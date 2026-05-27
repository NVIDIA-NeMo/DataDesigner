# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from data_designer.config.seed_source import AgentRolloutFormat
from data_designer.engine.resources.agent_rollout.base import AgentRolloutFormatHandler, AgentRolloutParseContext
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError, NormalizedAgentRolloutRecord
from data_designer.engine.resources.agent_rollout.utils import (
    build_message,
    coerce_optional_str,
    load_jsonl_rows,
    normalize_message_content,
    require_string,
    stringify_json_value,
)

logger = logging.getLogger(__name__)


class PiCodingAgentRolloutFormatHandler(AgentRolloutFormatHandler):
    """Normalize Pi Coding Agent session artifacts into rollout seed rows."""

    format: ClassVar[AgentRolloutFormat] = AgentRolloutFormat.PI_CODING_AGENT

    def is_handled_file(self, relative_path: str) -> bool:
        """Return whether a file path should be parsed as a Pi session.

        Args:
            relative_path: File path relative to the configured rollout root.

        Returns:
            ``True`` for ``.jsonl`` session files.
        """
        return Path(relative_path).suffix == ".jsonl"

    def parse_file(
        self,
        *,
        root_path: Path,
        relative_path: str,
        parse_context: AgentRolloutParseContext | None = None,
    ) -> list[NormalizedAgentRolloutRecord]:
        """Parse one Pi session JSONL file into a normalized rollout record.

        Args:
            root_path: Root directory configured on the seed source.
            relative_path: Path to the Pi session file relative to ``root_path``.
            parse_context: Unused for Pi sessions.

        Returns:
            A single normalized record, or an empty list for empty files.
        """
        del parse_context
        file_path = root_path / relative_path
        rows = list(load_jsonl_rows(file_path))
        if not rows:
            logger.warning("Skipping empty Pi Coding Agent session file %s", file_path)
            return []

        record = parse_pi_session(file_path=file_path, rows=rows)
        return [record]


def parse_pi_session(
    *,
    file_path: Path,
    rows: list[tuple[int, dict[str, Any]]],
) -> NormalizedAgentRolloutRecord:
    """Parse a Pi Coding Agent JSONL session into a normalized record.

    Pi sessions are tree-structured via ``id``/``parentId`` fields. This
    function resolves the active conversation path by walking from the last
    entry back to the root.

    Args:
        file_path: Absolute path to the Pi session file.
        rows: Parsed JSONL rows as ``(line_number, payload)`` pairs.

    Returns:
        A normalized rollout record for the session.

    Raises:
        AgentRolloutSeedParseError: If the session header is missing or the
            payload is malformed.
    """
    _, first_row = rows[0]
    if first_row.get("type") != "session":
        raise AgentRolloutSeedParseError(f"Pi session at {file_path} is missing a session header as the first entry")

    session_header = first_row
    session_id = require_string(session_header.get("id"), f"Pi session header id in {file_path}")
    cwd = coerce_optional_str(session_header.get("cwd"))
    session_version = session_header.get("version")

    entries = [entry for _, entry in rows[1:]]
    active_entries = _resolve_active_path(entries)

    messages: list[dict[str, Any]] = []
    entry_types: set[str] = set()
    models_used: list[str] = []
    seen_models: set[str] = set()
    stop_reasons: list[str] = []
    seen_stop_reasons: set[str] = set()
    bash_execution_count = 0

    for entry in active_entries:
        entry_type = coerce_optional_str(entry.get("type")) or "unknown"
        entry_types.add(entry_type)

        if entry_type == "model_change":
            model_id = coerce_optional_str(entry.get("modelId"))
            if model_id and model_id not in seen_models:
                seen_models.add(model_id)
                models_used.append(model_id)
            continue

        # Entry-level compaction (context summary replacing earlier messages) and
        # branch_summary (LLM summary of an abandoned branch injected via /tree).
        if entry_type in ("compaction", "branch_summary"):
            summary = coerce_optional_str(entry.get("summary"))
            if summary:
                messages.append(build_message(role="system", content=summary))
            continue

        # Entry-level custom message: extension-injected content that participates
        # in LLM context.  Distinct from the ``custom`` entry type which is
        # state-only and never enters context.
        if entry_type == "custom_message":
            if entry.get("display"):
                messages.append(build_message(role="system", content=normalize_message_content(entry.get("content"))))
            continue

        if entry_type != "message":
            continue

        raw_message = entry.get("message")
        if not isinstance(raw_message, dict):
            continue

        entry_id = coerce_optional_str(entry.get("id")) or ""
        role = coerce_optional_str(raw_message.get("role"))

        if role == "assistant":
            stop_reason = coerce_optional_str(raw_message.get("stopReason"))
            if stop_reason and stop_reason not in seen_stop_reasons:
                seen_stop_reasons.add(stop_reason)
                stop_reasons.append(stop_reason)

            model = coerce_optional_str(raw_message.get("model"))
            if model and model not in seen_models:
                seen_models.add(model)
                models_used.append(model)

        if role == "bashExecution":
            bash_execution_count += 1

        normalized = _normalize_pi_message(raw_message, file_path=file_path, entry_id=entry_id)
        messages.extend(normalized)

    timestamps: list[str] = []
    session_timestamp = coerce_optional_str(session_header.get("timestamp"))
    if session_timestamp:
        timestamps.append(session_timestamp)
    for entry in entries:
        ts = coerce_optional_str(entry.get("timestamp"))
        if ts:
            timestamps.append(ts)

    has_branches = _detect_branches(entries)

    source_meta: dict[str, Any] = {
        "record_count": len(rows),
        "entry_types": sorted(entry_types),
        "stop_reasons": stop_reasons,
    }
    if session_version is not None:
        source_meta["session_version"] = session_version
    if models_used:
        source_meta["models_used"] = models_used
    if has_branches:
        source_meta["has_branches"] = True
    if bash_execution_count:
        source_meta["bash_execution_count"] = bash_execution_count
    parent_session = coerce_optional_str(session_header.get("parentSession"))
    if parent_session:
        source_meta["parent_session"] = parent_session

    return NormalizedAgentRolloutRecord(
        trace_id=session_id,
        source_kind=AgentRolloutFormat.PI_CODING_AGENT.value,
        source_path=str(file_path),
        root_session_id=session_id,
        agent_id=None,
        is_sidechain=False,
        cwd=cwd,
        project_path=cwd,
        git_branch=None,
        started_at=min(timestamps) if timestamps else None,
        ended_at=max(timestamps) if timestamps else None,
        messages=messages,
        source_meta=source_meta,
    )


def _resolve_active_path(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk from the last entry back to the root via ``parentId`` to get the active path.

    Args:
        entries: All session entries (excluding the session header).

    Returns:
        Entries on the active conversation path in chronological order.
    """
    if not entries:
        return []

    entries_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry_id = entry.get("id")
        if isinstance(entry_id, str) and entry_id:
            entries_by_id[entry_id] = entry

    path: list[dict[str, Any]] = []
    current: dict[str, Any] | None = entries[-1]
    visited: set[str] = set()

    while current is not None:
        current_id = current.get("id")
        if isinstance(current_id, str) and current_id in visited:
            break
        if isinstance(current_id, str):
            visited.add(current_id)
        path.append(current)
        parent_id = current.get("parentId")
        if parent_id is None:
            break
        current = entries_by_id.get(parent_id)

    path.reverse()
    return path


def _detect_branches(entries: list[dict[str, Any]]) -> bool:
    """Return whether any parent ID is referenced by more than one child.

    Args:
        entries: All session entries (excluding the session header).

    Returns:
        ``True`` if the session tree contains at least one branch point.
    """
    seen_parents: set[str] = set()
    for entry in entries:
        parent_id = entry.get("parentId")
        if not isinstance(parent_id, str) or not parent_id:
            continue
        if parent_id in seen_parents:
            return True
        seen_parents.add(parent_id)
    return False


def _normalize_pi_message(
    raw_message: dict[str, Any],
    *,
    file_path: Path,
    entry_id: str,
) -> list[dict[str, Any]]:
    """Normalize one Pi ``AgentMessage`` into chat-schema messages.

    Args:
        raw_message: The ``message`` payload from a Pi ``message`` entry.
        file_path: File being parsed, used for error reporting.
        entry_id: Entry ID used to synthesize tool-call IDs for bash executions.

    Returns:
        One or more normalized messages. ``bashExecution`` messages produce an
        assistant tool-call message followed by a tool-result message.
    """
    role = coerce_optional_str(raw_message.get("role"))

    if role == "user":
        return [build_message(role="user", content=normalize_message_content(raw_message.get("content")))]

    if role == "assistant":
        return [_normalize_pi_assistant_message(raw_message)]

    if role == "toolResult":
        return [
            build_message(
                role="tool",
                content=normalize_message_content(raw_message.get("content")),
                tool_call_id=require_string(
                    raw_message.get("toolCallId"),
                    f"Pi toolResult toolCallId in {file_path}",
                ),
            )
        ]

    if role == "bashExecution":
        return _normalize_pi_bash_execution(raw_message, entry_id=entry_id)

    if role == "custom":
        if raw_message.get("display"):
            return [build_message(role="system", content=normalize_message_content(raw_message.get("content")))]
        return []

    if role in ("compactionSummary", "branchSummary"):
        summary = coerce_optional_str(raw_message.get("summary"))
        if summary:
            return [build_message(role="system", content=summary)]
        return []

    return []


def _normalize_pi_assistant_message(raw_message: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Pi assistant message with structured content blocks.

    Pi assistant messages contain a list of typed content blocks:
    ``TextContent``, ``ThinkingContent``, and ``ToolCall``.

    Args:
        raw_message: Raw Pi assistant message payload.

    Returns:
        A single normalized assistant message.
    """
    content_blocks = raw_message.get("content")
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                if isinstance(block, str) and block:
                    text_parts.append(block)
                continue

            block_type = block.get("type")
            if block_type == "text":
                text = coerce_optional_str(block.get("text"))
                if text:
                    text_parts.append(text)
            elif block_type == "thinking":
                thinking = coerce_optional_str(block.get("thinking"))
                if thinking:
                    thinking_parts.append(thinking)
            elif block_type == "toolCall":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": stringify_json_value(block.get("arguments")),
                        },
                    }
                )
    elif isinstance(content_blocks, str):
        text_parts.append(content_blocks)

    content = "\n\n".join(text_parts) if text_parts else ""
    reasoning_content = "\n\n".join(thinking_parts) if thinking_parts else None

    return build_message(
        role="assistant",
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )


def _normalize_pi_bash_execution(
    raw_message: dict[str, Any],
    *,
    entry_id: str,
) -> list[dict[str, Any]]:
    """Normalize a Pi ``bashExecution`` message as a tool-call pair.

    Args:
        raw_message: Raw Pi bash execution message payload.
        entry_id: Entry ID used to synthesize a tool-call ID.

    Returns:
        An assistant message with a synthetic ``bash`` tool call followed by
        a tool-result message containing the execution output.
    """
    if raw_message.get("excludeFromContext"):
        return []

    synthetic_id = f"bash_{entry_id}" if entry_id else "bash_unknown"
    command = coerce_optional_str(raw_message.get("command")) or ""
    output = coerce_optional_str(raw_message.get("output")) or ""
    exit_code = raw_message.get("exitCode")

    result_parts = [output] if output else []
    if exit_code is not None:
        result_parts.append(f"[exit code: {exit_code}]")
    result_content = "\n".join(result_parts) if result_parts else ""

    return [
        build_message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": synthetic_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": stringify_json_value({"command": command}),
                    },
                }
            ],
        ),
        build_message(
            role="tool",
            content=result_content,
            tool_call_id=synthetic_id,
        ),
    ]
