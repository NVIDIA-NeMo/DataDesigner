# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from data_designer.config.seed_source import AgentRolloutFormat, AgentRolloutSeedSource, CodexReasoningSource
from data_designer.engine.resources.agent_rollout.base import AgentRolloutFormatHandler, AgentRolloutParseContext
from data_designer.engine.resources.agent_rollout.types import NormalizedAgentRolloutRecord
from data_designer.engine.resources.agent_rollout.utils import (
    build_message,
    coerce_optional_str,
    load_jsonl_rows,
    min_max_timestamps,
    require_string,
    stringify_json_value,
    stringify_text_value,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodexParseContext(AgentRolloutParseContext):
    reasoning_source: CodexReasoningSource


class CodexAgentRolloutFormatHandler(AgentRolloutFormatHandler):
    format: ClassVar[AgentRolloutFormat] = AgentRolloutFormat.CODEX

    def build_parse_context(self, *, root_path: Path, source: AgentRolloutSeedSource) -> CodexParseContext:
        return CodexParseContext(reasoning_source=source.codex_reasoning_source)

    def is_handled_file(self, relative_path: str) -> bool:
        path = Path(relative_path)
        return path.suffix == ".jsonl" and path.name.startswith("rollout-")

    def parse_file(
        self,
        *,
        root_path: Path,
        relative_path: str,
        parse_context: AgentRolloutParseContext | None = None,
    ) -> list[NormalizedAgentRolloutRecord]:
        reasoning_source = CodexReasoningSource.SUMMARY
        if isinstance(parse_context, CodexParseContext):
            reasoning_source = parse_context.reasoning_source
        file_path = root_path / relative_path

        rows = list(load_jsonl_rows(file_path))
        if not rows:
            logger.warning("Skipping empty Codex rollout file %s", file_path)
            return []

        messages: list[dict[str, Any]] = []
        timestamps: list[str] = []
        pending_reasoning: list[str] = []
        raw_types: set[str] = set()
        response_item_types: set[str] = set()
        session_meta: dict[str, Any] = {}

        for _, raw_record in rows:
            record_type = coerce_optional_str(raw_record.get("type")) or "unknown"
            raw_types.add(record_type)
            if timestamp := coerce_optional_str(raw_record.get("timestamp")):
                timestamps.append(timestamp)

            payload = raw_record.get("payload")
            if not isinstance(payload, dict):
                continue

            if record_type == "session_meta":
                session_meta = payload
                if session_timestamp := coerce_optional_str(payload.get("timestamp")):
                    timestamps.append(session_timestamp)
                continue

            if record_type == "event_msg":
                event_type = coerce_optional_str(payload.get("type"))
                if event_type == "agent_reasoning" and (reasoning_text := coerce_optional_str(payload.get("text"))):
                    pending_reasoning.append(reasoning_text)
                continue

            if record_type != "response_item":
                continue

            item_type = coerce_optional_str(payload.get("type")) or "unknown"
            response_item_types.add(item_type)

            if item_type == "message":
                role = require_string(payload.get("role"), f"Codex message role in {file_path}")
                reasoning_content = (
                    "\n\n".join(pending_reasoning) if role == "assistant" and pending_reasoning else None
                )
                pending_reasoning.clear()
                messages.append(
                    build_message(
                        role=role,
                        content=payload.get("content"),
                        reasoning_content=reasoning_content,
                    )
                )
            elif item_type in {"function_call", "custom_tool_call", "apply_patch_call"}:
                reasoning_content = "\n\n".join(pending_reasoning) if pending_reasoning else None
                pending_reasoning.clear()
                messages.append(
                    build_message(
                        role="assistant",
                        content="",
                        reasoning_content=reasoning_content,
                        tool_calls=[
                            {
                                "id": require_string(payload.get("call_id"), f"Codex tool call id in {file_path}"),
                                "type": "function",
                                "function": {
                                    "name": require_string(payload.get("name"), f"Codex tool name in {file_path}"),
                                    "arguments": stringify_json_value(payload.get("arguments"))
                                    if "arguments" in payload
                                    else stringify_text_value(payload.get("input")),
                                },
                            }
                        ],
                    )
                )
            elif item_type in {"function_call_output", "custom_tool_call_output", "apply_patch_call_output"}:
                messages.append(
                    build_message(
                        role="tool",
                        content=payload.get("output"),
                        tool_call_id=require_string(
                            payload.get("call_id"),
                            f"Codex tool output id in {file_path}",
                        ),
                    )
                )
            elif item_type == "reasoning":
                pending_reasoning.extend(extract_codex_reasoning(payload, reasoning_source))
            else:
                logger.warning("Skipping unrecognized Codex response_item type %r in %s", item_type, file_path)

        session_id = coerce_optional_str(session_meta.get("id")) or file_path.stem
        source_meta: dict[str, Any] = {
            "record_count": len(rows),
            "record_types": sorted(raw_types),
            "response_item_types": sorted(response_item_types),
        }
        for field_name in ("originator", "cli_version", "model_provider", "source"):
            value = coerce_optional_str(session_meta.get(field_name))
            if value:
                source_meta[field_name] = value
        if pending_reasoning:
            source_meta["unattached_reasoning"] = list(pending_reasoning)

        earliest, latest = min_max_timestamps(timestamps)
        return [
            NormalizedAgentRolloutRecord(
                trace_id=session_id,
                source_kind=self.format.value,
                source_path=str(file_path),
                root_session_id=session_id,
                agent_id=None,
                is_sidechain=False,
                cwd=coerce_optional_str(session_meta.get("cwd")),
                project_path=coerce_optional_str(session_meta.get("cwd")),
                git_branch=coerce_optional_str(session_meta.get("git_branch")),
                started_at=coerce_optional_str(session_meta.get("timestamp")) or earliest,
                ended_at=latest,
                messages=messages,
                source_meta=source_meta,
            )
        ]


def extract_codex_reasoning(payload: dict[str, Any], reasoning_source: CodexReasoningSource) -> list[str]:
    summary = extract_codex_reasoning_parts(payload.get("summary"))
    if reasoning_source == CodexReasoningSource.CONTENT_FIRST:
        return extract_codex_reasoning_parts(payload.get("content")) or summary
    return summary


def extract_codex_reasoning_parts(parts: Any) -> list[str]:
    if not isinstance(parts, list):
        return []
    reasoning_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and (text := coerce_optional_str(part.get("text"))):
            reasoning_parts.append(text)
    return reasoning_parts
