# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentCommandDef:
    name: str
    attr: str
    help: str
    command_pattern: str
    returns: str


AGENT_COMMANDS: tuple[AgentCommandDef, ...] = (
    AgentCommandDef(
        name="context",
        attr="context_command",
        help="Prints output from all agent subcommands to bootstrap context.",
        command_pattern="data-designer agent context",
        returns="agent_context",
    ),
    AgentCommandDef(
        name="types",
        attr="types_command",
        help="Type names, descriptions, and source files for one or all families.",
        command_pattern="data-designer agent types [family]",
        returns="agent_types",
    ),
    AgentCommandDef(
        name="state.model-aliases",
        attr="state_model_aliases_command",
        help="Model aliases and usability status.",
        command_pattern="data-designer agent state model-aliases",
        returns="agent_state_model_aliases",
    ),
    AgentCommandDef(
        name="state.persona-datasets",
        attr="state_persona_datasets_command",
        help="Persona locales and install status.",
        command_pattern="data-designer agent state persona-datasets",
        returns="agent_state_persona_datasets",
    ),
)
