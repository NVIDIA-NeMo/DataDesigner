# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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
    load_jsonl_rows,
    require_string,
    stringify_json_value,
    stringify_text_value,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaudeCodeParseContext(AgentRolloutParseContext):
    session_index: dict[str, dict[str, Any]]


class ClaudeCodeAgentRolloutFormatHandler(AgentRolloutFormatHandler):
    format: ClassVar[AgentRolloutFormat] = AgentRolloutFormat.CLAUDE_CODE

    def build_parse_context(self, *, root_path: Path, recursive: bool) -> ClaudeCodeParseContext:
        session_index = load_claude_session_index(root_path, recursive=recursive)
        return ClaudeCodeParseContext(session_index=session_index)

    def is_handled_file(self, relative_path: str) -> bool:
        path = Path(relative_path)
        return path.suffix == ".jsonl" and "tool-results" not in path.parts and path.name != "history.jsonl"

    def parse_file(
        self,
        *,
        root_path: Path,
        relative_path: str,
        parse_context: AgentRolloutParseContext | None = None,
    ) -> list[NormalizedAgentRolloutRecord]:
        file_path = root_path / relative_path
        session_index: dict[str, dict[str, Any]] = {}
        if isinstance(parse_context, ClaudeCodeParseContext):
            session_index = parse_context.session_index

        rows = load_jsonl_rows(file_path)
        if not rows:
            logger.warning("Skipping empty Claude Code trace file %s", file_path)
            return []

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
            if timestamp := coerce_optional_str(raw_record.get("timestamp")):
                timestamps.append(timestamp)
            session_id = session_id or coerce_optional_str(raw_record.get("sessionId"))
            agent_id = agent_id or coerce_optional_str(raw_record.get("agentId"))
            cwd = cwd or coerce_optional_str(raw_record.get("cwd"))
            git_branch = git_branch or coerce_optional_str(raw_record.get("gitBranch"))
            version = coerce_optional_str(raw_record.get("version"))
            if version:
                versions.add(version)
            is_sidechain = is_sidechain or bool(raw_record.get("isSidechain"))

            record_type = raw_record.get("type")
            if record_type == "assistant":
                messages.extend(normalize_claude_assistant_messages(raw_record))
            elif record_type == "user":
                messages.extend(normalize_claude_user_messages(raw_record))

        session_key = session_id or file_path.stem
        index_entry = session_index.get(session_key, {})
        project_path = coerce_optional_str(index_entry.get("projectPath")) or cwd
        trace_id = f"{session_key}:{agent_id}" if agent_id else session_key
        source_meta: dict[str, Any] = {
            "record_count": len(rows),
            "record_types": sorted(raw_types),
        }
        if versions:
            source_meta["claude_versions"] = sorted(versions)
        if summary := coerce_optional_str(index_entry.get("summary")):
            source_meta["summary"] = summary
        if first_prompt := coerce_optional_str(index_entry.get("firstPrompt")):
            source_meta["first_prompt"] = first_prompt

        return [
            NormalizedAgentRolloutRecord(
                trace_id=trace_id,
                source_kind=self.format.value,
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
        ]


def normalize_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        block_type = coerce_optional_str(block.get("type"))
        if block_type in {"text", "input_text", "output_text"} and "text" in block:
            return {"type": "text", "text": stringify_text_value(block.get("text"))}
        if block_type == "thinking":
            return {"type": "text", "text": stringify_text_value(block.get("thinking") or block.get("text"))}
        if block_type is not None:
            return block
    return {"type": "text", "text": stringify_text_value(block)}


def coerce_raw_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, list):
        return [
            block if isinstance(block, dict) else {"type": "text", "text": stringify_text_value(block)}
            for block in content
        ]
    if isinstance(content, dict):
        return [content]
    return [{"type": "text", "text": stringify_text_value(content)}]


def normalize_claude_assistant_messages(raw_record: dict[str, Any]) -> list[dict[str, Any]]:
    message_payload = raw_record.get("message")
    if not isinstance(message_payload, dict):
        raise AgentRolloutSeedParseError(f"Claude assistant record is missing a message payload: {raw_record}")

    content_blocks = coerce_raw_blocks(message_payload.get("content"))
    assistant_content: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []

    for block in content_blocks:
        block_type = coerce_optional_str(block.get("type"))
        if block_type == "text":
            assistant_content.append(normalize_content_block(block))
        elif block_type == "thinking":
            reasoning_text = coerce_optional_str(block.get("thinking")) or coerce_optional_str(block.get("text"))
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
        elif block_type == "tool_use":
            tool_calls.append(normalize_claude_tool_call(block))
        elif block_type == "tool_result":
            tool_messages.append(
                build_message(
                    role="tool",
                    content=block.get("content"),
                    tool_call_id=require_string(block.get("tool_use_id"), "Claude tool_result tool_use_id"),
                )
            )
        else:
            assistant_content.append(normalize_content_block(block))

    normalized_messages: list[dict[str, Any]] = []
    if assistant_content or reasoning_parts or tool_calls:
        normalized_messages.append(
            build_message(
                role="assistant",
                content=assistant_content,
                reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
                tool_calls=tool_calls,
            )
        )
    normalized_messages.extend(tool_messages)
    return normalized_messages


def normalize_claude_user_messages(raw_record: dict[str, Any]) -> list[dict[str, Any]]:
    message_payload = raw_record.get("message")
    if not isinstance(message_payload, dict):
        raise AgentRolloutSeedParseError(f"Claude user record is missing a message payload: {raw_record}")

    content = message_payload.get("content")
    if isinstance(content, dict) and coerce_optional_str(content.get("type")) == "tool_result":
        return [
            build_message(
                role="tool",
                content=content.get("content"),
                tool_call_id=require_string(content.get("tool_use_id"), "Claude tool_result tool_use_id"),
            )
        ]
    if isinstance(content, list):
        user_content: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        for block in coerce_raw_blocks(content):
            if coerce_optional_str(block.get("type")) == "tool_result":
                tool_messages.append(
                    build_message(
                        role="tool",
                        content=block.get("content"),
                        tool_call_id=require_string(block.get("tool_use_id"), "Claude tool_result tool_use_id"),
                    )
                )
            else:
                user_content.append(normalize_content_block(block))

        normalized_messages: list[dict[str, Any]] = []
        if user_content:
            normalized_messages.append(build_message(role="user", content=user_content))
        normalized_messages.extend(tool_messages)
        return normalized_messages

    return [build_message(role="user", content=content)]


def normalize_claude_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": require_string(block.get("id"), "Claude tool_use id"),
        "type": "function",
        "function": {
            "name": require_string(block.get("name"), "Claude tool_use name"),
            "arguments": stringify_json_value(block.get("input")),
        },
    }


def load_claude_session_index(root_path: Path, *, recursive: bool = True) -> dict[str, dict[str, Any]]:
    entries_by_session_id: dict[str, dict[str, Any]] = {}
    glob_method = root_path.rglob if recursive else root_path.glob
    for index_path in sorted(glob_method("sessions-index.json")):
        try:
            with index_path.open(encoding="utf-8") as file:
                index_payload = json.load(file)
            entries = index_payload.get("entries", [])
            if not isinstance(entries, list):
                raise AgentRolloutSeedParseError(f"Claude sessions index at {index_path} is missing an 'entries' list")
            for entry in entries:
                if isinstance(entry, dict) and (session_id := coerce_optional_str(entry.get("sessionId"))):
                    entries_by_session_id[session_id] = entry
        except (AgentRolloutSeedParseError, json.JSONDecodeError) as error:
            logger.warning("Skipping malformed Claude sessions index %s: %s", index_path, error)
    return entries_by_session_id
