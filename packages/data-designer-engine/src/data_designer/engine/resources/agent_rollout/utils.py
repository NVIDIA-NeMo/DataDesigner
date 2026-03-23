# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
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


def stringify_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
