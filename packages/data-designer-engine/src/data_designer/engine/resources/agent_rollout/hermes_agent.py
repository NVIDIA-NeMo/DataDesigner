# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from data_designer.config.seed_source import AgentRolloutFormat
from data_designer.engine.resources.agent_rollout.base import AgentRolloutFormatHandler, AgentRolloutParseContext
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError, NormalizedAgentRolloutRecord
from data_designer.engine.resources.agent_rollout.utils import (
    build_message,
    coerce_optional_str,
    load_json_object,
    load_jsonl_rows,
    normalize_message_role,
    require_string,
    stringify_json_value,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HermesAgentParseContext(AgentRolloutParseContext):
    session_index: dict[str, dict[str, Any]]


class HermesAgentRolloutFormatHandler(AgentRolloutFormatHandler):
    format: ClassVar[AgentRolloutFormat] = AgentRolloutFormat.HERMES_AGENT

    def build_parse_context(self, *, root_path: Path, recursive: bool) -> HermesAgentParseContext:
        return HermesAgentParseContext(session_index=load_hermes_session_index(root_path, recursive=recursive))

    def is_handled_file(self, relative_path: str) -> bool:
        path = Path(relative_path)
        if path.name == "sessions.json":
            return False
        if path.suffix == ".json":
            return path.name.startswith("session_")
        return path.suffix == ".jsonl"

    def parse_file(
        self,
        *,
        root_path: Path,
        relative_path: str,
        parse_context: AgentRolloutParseContext | None = None,
    ) -> list[NormalizedAgentRolloutRecord]:
        file_path = root_path / relative_path
        session_index: dict[str, dict[str, Any]] = {}
        if isinstance(parse_context, HermesAgentParseContext):
            session_index = parse_context.session_index

        if file_path.suffix == ".json":
            record = parse_hermes_cli_session_log(file_path)
            return [record]

        records = parse_hermes_gateway_transcript(file_path=file_path, session_index=session_index)
        return records


def parse_hermes_cli_session_log(file_path: Path) -> NormalizedAgentRolloutRecord:
    payload = load_json_object(file_path)
    raw_messages = _require_message_list(payload.get("messages"), file_path=file_path, context="Hermes CLI session")
    messages = normalize_hermes_messages(raw_messages, file_path=file_path)

    session_id = coerce_optional_str(payload.get("session_id")) or file_path.stem.removeprefix("session_")
    available_tool_names = extract_hermes_tool_names(payload.get("tools"))
    source_meta = _build_hermes_cli_source_meta(
        payload=payload, raw_messages=raw_messages, tool_names=available_tool_names
    )

    return NormalizedAgentRolloutRecord(
        trace_id=session_id,
        source_kind=AgentRolloutFormat.HERMES_AGENT.value,
        source_path=str(file_path),
        root_session_id=session_id,
        agent_id=None,
        is_sidechain=False,
        cwd=None,
        project_path=None,
        git_branch=None,
        started_at=coerce_optional_str(payload.get("session_start")),
        ended_at=coerce_optional_str(payload.get("last_updated")),
        messages=messages,
        source_meta=source_meta,
    )


def parse_hermes_gateway_transcript(
    *,
    file_path: Path,
    session_index: dict[str, dict[str, Any]],
) -> list[NormalizedAgentRolloutRecord]:
    rows = list(load_jsonl_rows(file_path))
    if not rows:
        logger.warning("Skipping empty Hermes Agent transcript file %s", file_path)
        return []

    raw_messages = [row for _, row in rows]
    messages = normalize_hermes_messages(raw_messages, file_path=file_path)
    session_id = file_path.stem
    session_meta = session_index.get(session_id, {})
    source_meta = _build_hermes_gateway_source_meta(session_meta=session_meta, raw_messages=raw_messages)

    return [
        NormalizedAgentRolloutRecord(
            trace_id=session_id,
            source_kind=AgentRolloutFormat.HERMES_AGENT.value,
            source_path=str(file_path),
            root_session_id=session_id,
            agent_id=None,
            is_sidechain=False,
            cwd=None,
            project_path=None,
            git_branch=None,
            started_at=coerce_optional_str(session_meta.get("created_at")),
            ended_at=coerce_optional_str(session_meta.get("updated_at")),
            messages=messages,
            source_meta=source_meta,
        )
    ]


def normalize_hermes_messages(raw_messages: list[dict[str, Any]], *, file_path: Path) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    for message_index, raw_message in enumerate(raw_messages, start=1):
        if not isinstance(raw_message, dict):
            raise AgentRolloutSeedParseError(
                f"Expected Hermes message object at index {message_index} in {file_path}, "
                f"got {type(raw_message).__name__}"
            )

        role = normalize_message_role(
            raw_message.get("role"),
            context=f"Hermes message #{message_index} in {file_path}",
        )
        if role == "tool":
            normalized_messages.append(
                build_message(
                    role="tool",
                    content=_normalize_message_content(raw_message.get("content")),
                    tool_call_id=require_string(
                        raw_message.get("tool_call_id"),
                        f"Hermes tool message tool_call_id #{message_index} in {file_path}",
                    ),
                )
            )
            continue

        content = _normalize_message_content(raw_message.get("content"))
        reasoning_content = coerce_optional_str(raw_message.get("reasoning"))
        tool_calls = normalize_hermes_tool_calls(
            raw_message.get("tool_calls"),
            file_path=file_path,
            message_index=message_index,
        )
        normalized_messages.append(
            build_message(
                role=role,
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
            )
        )
    return normalized_messages


def normalize_hermes_tool_calls(
    raw_tool_calls: Any,
    *,
    file_path: Path,
    message_index: int,
) -> list[dict[str, Any]]:
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise AgentRolloutSeedParseError(
            f"Expected Hermes tool_calls list for message #{message_index} in {file_path}, "
            f"got {type(raw_tool_calls).__name__}"
        )

    normalized_tool_calls: list[dict[str, Any]] = []
    for tool_call_index, raw_tool_call in enumerate(raw_tool_calls, start=1):
        if not isinstance(raw_tool_call, dict):
            raise AgentRolloutSeedParseError(
                f"Expected Hermes tool call object at message #{message_index} call #{tool_call_index} "
                f"in {file_path}, got {type(raw_tool_call).__name__}"
            )

        raw_function = raw_tool_call.get("function")
        function_payload = raw_function if isinstance(raw_function, dict) else {}
        call_context = f"Hermes tool call #{tool_call_index} on message #{message_index} in {file_path}"
        tool_call_id = require_string(raw_tool_call.get("id") or raw_tool_call.get("call_id"), f"{call_context} id")
        function_name = require_string(
            function_payload.get("name") or raw_tool_call.get("name"), f"{call_context} name"
        )
        arguments = function_payload.get("arguments", raw_tool_call.get("arguments"))
        normalized_tool_calls.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": stringify_json_value(arguments),
                },
            }
        )
    return normalized_tool_calls


def extract_hermes_tool_names(raw_tools: Any) -> list[str]:
    if not isinstance(raw_tools, list):
        return []

    tool_names: list[str] = []
    seen_names: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        function_payload = raw_tool.get("function")
        if not isinstance(function_payload, dict):
            continue
        tool_name = coerce_optional_str(function_payload.get("name"))
        if not tool_name or tool_name in seen_names:
            continue
        seen_names.add(tool_name)
        tool_names.append(tool_name)
    return tool_names


def load_hermes_session_index(root_path: Path, *, recursive: bool = True) -> dict[str, dict[str, Any]]:
    entries_by_session_id: dict[str, dict[str, Any]] = {}
    glob_method = root_path.rglob if recursive else root_path.glob
    for index_path in sorted(glob_method("sessions.json")):
        try:
            index_payload = load_json_object(index_path)
            for session_key, entry in index_payload.items():
                if not isinstance(entry, dict):
                    continue
                session_id = coerce_optional_str(entry.get("session_id"))
                if not session_id:
                    continue
                entry_with_key = dict(entry)
                entry_with_key.setdefault("session_key", session_key)
                entries_by_session_id[session_id] = entry_with_key
        except (AgentRolloutSeedParseError, OSError) as error:
            logger.warning("Skipping malformed Hermes sessions index %s: %s", index_path, error)
    return entries_by_session_id


def _require_message_list(raw_messages: Any, *, file_path: Path, context: str) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
        raise AgentRolloutSeedParseError(f"{context} at {file_path} is missing a 'messages' list")
    return raw_messages


def _normalize_message_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, (str, list)):
        return content
    return stringify_json_value(content)


def _extract_finish_reasons(raw_messages: list[dict[str, Any]]) -> list[str]:
    finish_reasons: list[str] = []
    seen_reasons: set[str] = set()
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        if raw_message.get("role") != "assistant":
            continue
        finish_reason = coerce_optional_str(raw_message.get("finish_reason"))
        if not finish_reason or finish_reason in seen_reasons:
            continue
        seen_reasons.add(finish_reason)
        finish_reasons.append(finish_reason)
    return finish_reasons


def _extract_used_tool_names(raw_messages: list[dict[str, Any]]) -> list[str]:
    tool_names: list[str] = []
    seen_names: set[str] = set()
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        raw_tool_calls = raw_message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            continue
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            function_payload = raw_tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            tool_name = coerce_optional_str(function_payload.get("name"))
            if not tool_name or tool_name in seen_names:
                continue
            seen_names.add(tool_name)
            tool_names.append(tool_name)
    return tool_names


def _build_hermes_cli_source_meta(
    *,
    payload: dict[str, Any],
    raw_messages: list[dict[str, Any]],
    tool_names: list[str],
) -> dict[str, Any]:
    source_meta: dict[str, Any] = {
        "record_count": len(raw_messages),
        "session_format": "cli_session_log",
        "available_tool_count": len(tool_names),
        "available_tool_names": tool_names,
        "finish_reasons": _extract_finish_reasons(raw_messages),
    }
    for field_name in ("model", "base_url", "platform"):
        field_value = coerce_optional_str(payload.get(field_name))
        if field_value:
            source_meta[field_name] = field_value
    system_prompt = payload.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt:
        source_meta["has_system_prompt"] = True
        source_meta["system_prompt_chars"] = len(system_prompt)
    return source_meta


def _build_hermes_gateway_source_meta(
    *,
    session_meta: dict[str, Any],
    raw_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    source_meta: dict[str, Any] = {
        "record_count": len(raw_messages),
        "session_format": "gateway_transcript",
        "finish_reasons": _extract_finish_reasons(raw_messages),
        "tool_names_used": _extract_used_tool_names(raw_messages),
    }
    for field_name in (
        "session_key",
        "display_name",
        "platform",
        "chat_type",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "last_prompt_tokens",
        "estimated_cost_usd",
        "cost_status",
        "memory_flushed",
    ):
        if field_name in session_meta:
            source_meta[field_name] = session_meta[field_name]
    origin = session_meta.get("origin")
    if isinstance(origin, dict) and origin:
        source_meta["origin"] = origin
    return source_meta
