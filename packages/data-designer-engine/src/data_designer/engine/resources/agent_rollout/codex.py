# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from data_designer.config.seed_source import AgentRolloutFormat
from data_designer.engine.resources.agent_rollout.base import AgentRolloutFormatHandler, AgentRolloutParseContext
from data_designer.engine.resources.agent_rollout.types import NormalizedAgentRolloutRecord
from data_designer.engine.resources.agent_rollout.utils import (
    build_message,
    coerce_optional_str,
    load_jsonl_rows,
    require_string,
    stringify_json_value,
    stringify_text_value,
)

logger = logging.getLogger(__name__)


class CodexAgentRolloutFormatHandler(AgentRolloutFormatHandler):
    format: ClassVar[AgentRolloutFormat] = AgentRolloutFormat.CODEX

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
        del parse_context
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
                pending_reasoning.extend(extract_codex_reasoning_summaries(payload))
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
                started_at=coerce_optional_str(session_meta.get("timestamp"))
                or (min(timestamps) if timestamps else None),
                ended_at=max(timestamps) if timestamps else None,
                messages=messages,
                source_meta=source_meta,
            )
        ]


def extract_codex_reasoning_summaries(payload: dict[str, Any]) -> list[str]:
    summaries = payload.get("summary")
    if not isinstance(summaries, list):
        return []
    reasoning_parts: list[str] = []
    for summary in summaries:
        if isinstance(summary, dict) and (text := coerce_optional_str(summary.get("text"))):
            reasoning_parts.append(text)
    return reasoning_parts
