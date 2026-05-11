# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError


def build_message(
    *,
    role: str,
    content: Any,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    normalized_role = normalize_message_role(role, context="agent rollout message")
    return ChatMessage(
        role=normalized_role,
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or [],
        tool_call_id=tool_call_id,
    ).to_dict()


def normalize_message_role(raw_role: Any, *, context: str) -> Literal["user", "assistant", "system", "tool"]:
    role = coerce_optional_str(raw_role)
    if role == "developer":
        return "system"
    if role in {"user", "assistant", "system", "tool"}:
        return role
    raise AgentRolloutSeedParseError(f"Unsupported message role {raw_role!r} in {context}")


def load_jsonl_rows(file_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with file_path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                parsed_line = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise AgentRolloutSeedParseError(
                    f"Invalid JSON in {file_path} line {line_number}: {error.msg}"
                ) from error
            if not isinstance(parsed_line, dict):
                raise AgentRolloutSeedParseError(
                    f"Expected JSON object in {file_path} line {line_number}, got {type(parsed_line).__name__}"
                )
            yield (line_number, parsed_line)


def load_json_object(file_path: Path) -> dict[str, Any]:
    with file_path.open(encoding="utf-8") as file:
        try:
            parsed_payload = json.load(file)
        except json.JSONDecodeError as error:
            raise AgentRolloutSeedParseError(f"Invalid JSON in {file_path}: {error.msg}") from error

    if not isinstance(parsed_payload, dict):
        raise AgentRolloutSeedParseError(f"Expected JSON object in {file_path}, got {type(parsed_payload).__name__}")
    return parsed_payload


def require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or value == "":
        raise AgentRolloutSeedParseError(f"Expected non-empty string for {context}, got {value!r}")
    return value


def coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def stringify_json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, sort_keys=True)


def normalize_message_content(content: Any) -> Any:
    """Coerce raw message content into the normalized content shape.

    Returns ``""`` for ``None``, passes through ``str`` and ``list``
    unchanged, and falls back to :func:`stringify_json_value` for
    everything else.
    """
    if content is None:
        return ""
    if isinstance(content, (str, list)):
        return content
    return stringify_json_value(content)


def stringify_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def min_max_timestamps(timestamps: list[str]) -> tuple[str | None, str | None]:
    """Return the chronologically earliest and latest timestamps.

    Values are parsed as ISO 8601 before comparison so that mixed UTC offsets
    and precisions order correctly (e.g. ``2025-01-01T00:30:00+01:00`` is
    earlier than ``2025-01-01T00:00:00Z``). Naive timestamps are treated as
    UTC. Unparseable values are skipped. The winning entries are returned in
    their original string form.
    """
    parsed: list[tuple[datetime, str]] = []
    for original in timestamps:
        instant = parse_iso8601(original)
        if instant is not None:
            parsed.append((instant, original))
    if not parsed:
        return None, None
    earliest = min(parsed, key=lambda pair: pair[0])[1]
    latest = max(parsed, key=lambda pair: pair[0])[1]
    return earliest, latest


def parse_iso8601(value: str) -> datetime | None:
    """Parse an ISO 8601 timestamp, treating naive values as UTC.

    Returns ``None`` for strings that cannot be parsed so callers can silently
    skip malformed entries.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
