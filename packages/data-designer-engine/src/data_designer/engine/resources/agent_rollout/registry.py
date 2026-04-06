# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.seed_source import AgentRolloutFormat
from data_designer.engine.resources.agent_rollout.atif import AtifAgentRolloutFormatHandler
from data_designer.engine.resources.agent_rollout.base import AgentRolloutFormatHandler
from data_designer.engine.resources.agent_rollout.claude_code import ClaudeCodeAgentRolloutFormatHandler
from data_designer.engine.resources.agent_rollout.codex import CodexAgentRolloutFormatHandler

BUILTIN_AGENT_ROLLOUT_FORMAT_HANDLERS: dict[AgentRolloutFormat, AgentRolloutFormatHandler] = {
    handler.format: handler
    for handler in (
        AtifAgentRolloutFormatHandler(),
        ClaudeCodeAgentRolloutFormatHandler(),
        CodexAgentRolloutFormatHandler(),
    )
}


def get_format_handler(rollout_format: AgentRolloutFormat) -> AgentRolloutFormatHandler:
    handler = BUILTIN_AGENT_ROLLOUT_FORMAT_HANDLERS.get(rollout_format)
    if handler is None:
        raise KeyError(f"No AgentRollout format handler found for format {rollout_format.value!r}")
    return handler
