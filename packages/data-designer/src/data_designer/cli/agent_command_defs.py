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
        help="Bootstrap payload with types, state, and builder.",
        command_pattern="data-designer agent context",
        returns="agent_context",
    ),
    AgentCommandDef(
        name="types",
        attr="types_command",
        help="Type names and import paths for one or all families.",
        command_pattern="data-designer agent types [family]",
        returns="agent_types",
    ),
    AgentCommandDef(
        name="schema",
        attr="schema_command",
        help="Schema for a type or entire family.",
        command_pattern="data-designer agent schema <family> <type> | --all",
        returns="agent_schema",
    ),
    AgentCommandDef(
        name="builder",
        attr="builder_command",
        help="ConfigBuilder method surface with signatures.",
        command_pattern="data-designer agent builder",
        returns="agent_builder",
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
