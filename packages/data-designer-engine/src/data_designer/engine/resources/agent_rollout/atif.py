# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""ATIF trajectory ingestion for standalone rollout JSON files."""

import json
from pathlib import Path
from typing import Any, ClassVar

from data_designer.config.seed_source import AgentRolloutFormat
from data_designer.engine.resources.agent_rollout.base import AgentRolloutFormatHandler, AgentRolloutParseContext
from data_designer.engine.resources.agent_rollout.types import AgentRolloutSeedParseError, NormalizedAgentRolloutRecord
from data_designer.engine.resources.agent_rollout.utils import (
    build_message,
    coerce_optional_str,
    require_string,
    stringify_json_value,
)


class AtifAgentRolloutFormatHandler(AgentRolloutFormatHandler):
    """Normalize standalone ATIF trajectories into rollout seed rows.

    This handler intentionally supports the narrow ingestion contract used by
    DataDesigner's built-in rollout readers: one standalone JSON file becomes
    one normalized rollout record. Multi-file continuations are preserved as
    metadata only and are not stitched together here.
    """

    format: ClassVar[AgentRolloutFormat] = AgentRolloutFormat.ATIF

    def is_handled_file(self, relative_path: str) -> bool:
        """Return whether a relative path should be parsed as ATIF.

        Args:
            relative_path: File path relative to the configured rollout root.

        Returns:
            True when the file uses the standalone `.json` ATIF packaging
            expected by this handler.
        """
        return Path(relative_path).suffix == ".json"

    def parse_file(
        self,
        *,
        root_path: Path,
        relative_path: str,
        parse_context: AgentRolloutParseContext | None = None,
    ) -> list[NormalizedAgentRolloutRecord]:
        """Parse one ATIF trajectory file into the shared rollout row shape.

        Args:
            root_path: Root directory configured on the seed source.
            relative_path: Path to the ATIF file relative to ``root_path``.
            parse_context: Unused for ATIF because standalone files do not need
                attachment-scoped indexing or preprocessing.

        Returns:
            A single normalized rollout record for the trajectory file.

        Raises:
            AgentRolloutSeedParseError: If the file is not valid JSON or does
                not satisfy the ATIF fields needed by the normalized schema.
        """
        file_path = root_path / relative_path
        payload = load_atif_payload(file_path)

        schema_version = require_string(payload.get("schema_version"), f"ATIF schema_version in {file_path}")
        session_id = require_string(payload.get("session_id"), f"ATIF session_id in {file_path}")
        agent = payload.get("agent")
        if not isinstance(agent, dict):
            raise AgentRolloutSeedParseError(f"ATIF trajectory in {file_path} is missing an agent object")

        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            raise AgentRolloutSeedParseError(f"ATIF trajectory in {file_path} is missing a non-empty steps list")

        messages: list[dict[str, Any]] = []
        timestamps: list[str] = []
        copied_context_step_ids: list[int] = []
        subagent_refs: list[dict[str, Any]] = []

        for step_index, raw_step in enumerate(steps, start=1):
            if not isinstance(raw_step, dict):
                raise AgentRolloutSeedParseError(
                    f"Expected ATIF step object at {file_path} steps[{step_index - 1}], got {type(raw_step).__name__}"
                )

            step_id = raw_step.get("step_id")
            if not isinstance(step_id, int):
                raise AgentRolloutSeedParseError(
                    f"Expected integer ATIF step_id at {file_path} steps[{step_index - 1}], got {step_id!r}"
                )

            if timestamp := coerce_optional_str(raw_step.get("timestamp")):
                timestamps.append(timestamp)

            if raw_step.get("is_copied_context") is True:
                copied_context_step_ids.append(step_id)

            if "message" not in raw_step:
                raise AgentRolloutSeedParseError(f"ATIF step {step_id} in {file_path} is missing message content")

            messages.append(
                build_message(
                    role=normalize_atif_role(raw_step.get("source"), file_path=file_path, step_id=step_id),
                    content=raw_step.get("message"),
                    reasoning_content=coerce_optional_str(raw_step.get("reasoning_content")),
                    tool_calls=normalize_atif_tool_calls(
                        raw_step.get("tool_calls"),
                        file_path=file_path,
                        step_id=step_id,
                    ),
                )
            )

            observation = raw_step.get("observation")
            if observation is not None:
                messages.extend(
                    normalize_atif_observation_messages(
                        observation,
                        file_path=file_path,
                        step_id=step_id,
                        subagent_refs=subagent_refs,
                    )
                )

        source_meta = build_atif_source_meta(
            payload=payload,
            file_path=file_path,
            schema_version=schema_version,
            agent=agent,
            steps=steps,
            copied_context_step_ids=copied_context_step_ids,
            subagent_refs=subagent_refs,
        )

        agent_extra = agent.get("extra") if isinstance(agent.get("extra"), dict) else {}
        cwd = coerce_optional_str(agent_extra.get("cwd"))
        project_path = coerce_optional_str(agent_extra.get("project_path")) or cwd
        git_branch = coerce_optional_str(agent_extra.get("git_branch"))

        return [
            NormalizedAgentRolloutRecord(
                trace_id=session_id,
                source_kind=self.format.value,
                source_path=str(file_path),
                root_session_id=session_id,
                agent_id=None,
                is_sidechain=False,
                cwd=cwd,
                project_path=project_path,
                git_branch=git_branch,
                started_at=min(timestamps) if timestamps else None,
                ended_at=max(timestamps) if timestamps else None,
                messages=messages,
                source_meta=source_meta,
            )
        ]


def load_atif_payload(file_path: Path) -> dict[str, Any]:
    """Load an ATIF trajectory from disk.

    Args:
        file_path: Absolute path to the ATIF JSON file.

    Returns:
        The decoded top-level JSON object.

    Raises:
        AgentRolloutSeedParseError: If the file cannot be parsed as JSON or if
            the payload is not a JSON object.
    """
    try:
        with file_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise AgentRolloutSeedParseError(f"Invalid JSON in {file_path}: {error.msg}") from error

    if not isinstance(payload, dict):
        raise AgentRolloutSeedParseError(f"Expected JSON object in {file_path}, got {type(payload).__name__}")
    return payload


def normalize_atif_role(raw_source: Any, *, file_path: Path, step_id: int) -> str:
    """Map an ATIF step source onto the normalized chat role set.

    Args:
        raw_source: Raw ``source`` value from an ATIF step.
        file_path: File being parsed, used for error reporting.
        step_id: ATIF step identifier, used for error reporting.

    Returns:
        The corresponding normalized role understood by DataDesigner.

    Raises:
        AgentRolloutSeedParseError: If the ATIF source is missing or is not one
            of the supported values.
    """
    source = require_string(raw_source, f"ATIF source in {file_path} step {step_id}")
    if source == "agent":
        return "assistant"
    if source in {"system", "user"}:
        return source
    raise AgentRolloutSeedParseError(f"Unsupported ATIF source {source!r} in {file_path} step {step_id}")


def normalize_atif_tool_calls(raw_tool_calls: Any, *, file_path: Path, step_id: int) -> list[dict[str, Any]]:
    """Convert ATIF tool calls into the shared normalized tool-call shape.

    Args:
        raw_tool_calls: Raw ``tool_calls`` payload from an ATIF step.
        file_path: File being parsed, used for error reporting.
        step_id: ATIF step identifier, used for error reporting.

    Returns:
        A list of tool-call dictionaries compatible with the shared chat
        message schema.

    Raises:
        AgentRolloutSeedParseError: If the tool-call payload is malformed.
    """
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise AgentRolloutSeedParseError(f"ATIF tool_calls in {file_path} step {step_id} must be a list")

    tool_calls: list[dict[str, Any]] = []
    for tool_index, raw_tool_call in enumerate(raw_tool_calls):
        if not isinstance(raw_tool_call, dict):
            raise AgentRolloutSeedParseError(
                f"Expected ATIF tool_call object in {file_path} step {step_id}, got {type(raw_tool_call).__name__}"
            )
        tool_calls.append(
            {
                "id": require_string(
                    raw_tool_call.get("tool_call_id"),
                    f"ATIF tool_call_id in {file_path} step {step_id} tool_calls[{tool_index}]",
                ),
                "type": "function",
                "function": {
                    "name": require_string(
                        raw_tool_call.get("function_name"),
                        f"ATIF function_name in {file_path} step {step_id} tool_calls[{tool_index}]",
                    ),
                    "arguments": stringify_json_value(raw_tool_call.get("arguments")),
                },
            }
        )
    return tool_calls


def build_atif_source_meta(
    *,
    payload: dict[str, Any],
    file_path: Path,
    schema_version: str,
    agent: dict[str, Any],
    steps: list[Any],
    copied_context_step_ids: list[int],
    subagent_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect ATIF-specific metadata that does not fit shared rollout columns.

    Args:
        payload: Parsed top-level ATIF trajectory object.
        file_path: File being parsed, used for error reporting.
        schema_version: Validated ATIF schema version string.
        agent: Parsed ATIF agent object.
        steps: Parsed ATIF steps list.
        copied_context_step_ids: Step IDs marked as copied context.
        subagent_refs: Delegated subagent references gathered from observations.

    Returns:
        A metadata dictionary stored under ``source_meta`` on the normalized
        rollout row.
    """
    source_meta: dict[str, Any] = {
        "schema_version": schema_version,
        "step_count": len(steps),
        "agent_name": require_string(agent.get("name"), f"ATIF agent.name in {file_path}"),
    }
    for field_name in ("version", "model_name"):
        value = coerce_optional_str(agent.get(field_name))
        if value:
            source_meta[field_name] = value
    if copied_context_step_ids:
        source_meta["copied_context_step_ids"] = copied_context_step_ids
    if subagent_refs:
        source_meta["subagent_trajectory_refs"] = subagent_refs
    if notes := coerce_optional_str(payload.get("notes")):
        source_meta["notes"] = notes
    if continued_trajectory_ref := coerce_optional_str(payload.get("continued_trajectory_ref")):
        source_meta["continued_trajectory_ref"] = continued_trajectory_ref
    if final_metrics := payload.get("final_metrics"):
        if isinstance(final_metrics, dict):
            source_meta["final_metrics"] = final_metrics
    if extra := payload.get("extra"):
        if isinstance(extra, dict):
            source_meta["extra"] = extra
    return source_meta


def normalize_atif_observation_messages(
    observation: Any,
    *,
    file_path: Path,
    step_id: int,
    subagent_refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize ATIF observations into tool messages and sidecar metadata.

    Args:
        observation: Raw ATIF observation payload for a step.
        file_path: File being parsed, used for error reporting.
        step_id: ATIF step identifier, used for error reporting.
        subagent_refs: Mutable accumulator used to collect delegated subagent
            references for later storage in ``source_meta``.

    Returns:
        A list of normalized tool messages derived from observation results.

    Raises:
        AgentRolloutSeedParseError: If the observation payload is malformed.
    """
    if not isinstance(observation, dict):
        raise AgentRolloutSeedParseError(f"ATIF observation in {file_path} step {step_id} must be an object")

    results = observation.get("results")
    if not isinstance(results, list):
        raise AgentRolloutSeedParseError(f"ATIF observation.results in {file_path} step {step_id} must be a list")

    messages: list[dict[str, Any]] = []
    for result_index, raw_result in enumerate(results):
        if not isinstance(raw_result, dict):
            raise AgentRolloutSeedParseError(
                f"Expected ATIF observation result object in {file_path} step {step_id}, got {type(raw_result).__name__}"
            )
        if "content" in raw_result and raw_result.get("content") is not None:
            messages.append(
                build_message(
                    role="tool",
                    content=raw_result.get("content"),
                    tool_call_id=coerce_optional_str(raw_result.get("source_call_id")),
                )
            )

        raw_refs = raw_result.get("subagent_trajectory_ref")
        if raw_refs is None:
            continue
        if not isinstance(raw_refs, list):
            raise AgentRolloutSeedParseError(
                f"ATIF subagent_trajectory_ref in {file_path} step {step_id} result {result_index} must be a list"
            )
        refs: list[dict[str, Any]] = []
        for ref_index, raw_ref in enumerate(raw_refs):
            if not isinstance(raw_ref, dict):
                raise AgentRolloutSeedParseError(
                    f"Expected subagent trajectory ref object in {file_path} step {step_id}, got {type(raw_ref).__name__}"
                )
            normalized_ref: dict[str, Any] = {
                "session_id": require_string(
                    raw_ref.get("session_id"),
                    f"ATIF subagent session_id in {file_path} step {step_id} result {result_index} ref {ref_index}",
                )
            }
            if trajectory_path := coerce_optional_str(raw_ref.get("trajectory_path")):
                normalized_ref["trajectory_path"] = trajectory_path
            if extra := raw_ref.get("extra"):
                if isinstance(extra, dict):
                    normalized_ref["extra"] = extra
            refs.append(normalized_ref)
        if refs:
            subagent_refs.append({"step_id": step_id, "refs": refs})
    return messages
