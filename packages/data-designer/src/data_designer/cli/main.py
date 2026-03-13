# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.lazy_group import create_lazy_typer_group
from data_designer.cli.runtime import ensure_cli_default_model_settings

_CMD = "data_designer.cli.commands"

# Initialize Typer app with custom configuration
app = typer.Typer(
    name="data-designer",
    help="Data Designer CLI - Configure model providers and models for synthetic data generation",
    cls=create_lazy_typer_group(
        {
            "preview": {
                "module": f"{_CMD}.preview",
                "attr": "preview_command",
                "help": "Generate a preview dataset for fast iteration",
                "rich_help_panel": "Generation",
            },
            "create": {
                "module": f"{_CMD}.create",
                "attr": "create_command",
                "help": "Create a full dataset and save results to disk",
                "rich_help_panel": "Generation",
            },
            "validate": {
                "module": f"{_CMD}.validate",
                "attr": "validate_command",
                "help": "Validate a Data Designer configuration",
                "rich_help_panel": "Generation",
            },
        }
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# Create config subcommand group
config_app = typer.Typer(
    name="config",
    help="Manage configuration files",
    cls=create_lazy_typer_group(
        {
            "providers": {
                "module": f"{_CMD}.providers",
                "attr": "providers_command",
                "help": "Configure model providers interactively",
            },
            "models": {
                "module": f"{_CMD}.models",
                "attr": "models_command",
                "help": "Configure models interactively",
            },
            "mcp": {
                "module": f"{_CMD}.mcp",
                "attr": "mcp_command",
                "help": "Configure MCP providers interactively",
            },
            "tools": {
                "module": f"{_CMD}.tools",
                "attr": "tools_command",
                "help": "Configure tool configs interactively",
            },
            "list": {
                "module": f"{_CMD}.list",
                "attr": "list_command",
                "help": "List current configurations",
            },
            "reset": {
                "module": f"{_CMD}.reset",
                "attr": "reset_command",
                "help": "Reset configuration files",
            },
        }
    ),
    no_args_is_help=True,
)

# Create download command group
download_app = typer.Typer(
    name="download",
    help="Download assets for Data Designer",
    cls=create_lazy_typer_group(
        {
            "personas": {
                "module": f"{_CMD}.download",
                "attr": "personas_command",
                "help": "Download Nemotron-Persona datasets",
            },
        }
    ),
    no_args_is_help=True,
)

_AGENT_CMD = f"{_CMD}.agent"

agent_app = typer.Typer(
    name="agent",
    help="Agent-only interface for dynamic Data Designer introspection",
    cls=create_lazy_typer_group(
        {
            "context": {
                "module": _AGENT_CMD,
                "attr": "context_command",
                "help": "Bootstrap payload with types, state, and builder.",
            },
            "types": {
                "module": _AGENT_CMD,
                "attr": "types_command",
                "help": "Type names and import paths for one or all families.",
            },
            "schema": {"module": _AGENT_CMD, "attr": "schema_command", "help": "Schema for a type or entire family."},
            "builder": {
                "module": _AGENT_CMD,
                "attr": "builder_command",
                "help": "ConfigBuilder method surface with signatures.",
            },
        }
    ),
    no_args_is_help=True,
)

agent_state_app = typer.Typer(
    name="state",
    help="Return current local state relevant to agents",
    cls=create_lazy_typer_group(
        {
            "model-aliases": {
                "module": _AGENT_CMD,
                "attr": "state_model_aliases_command",
                "help": "Model aliases and usability status.",
            },
            "persona-datasets": {
                "module": _AGENT_CMD,
                "attr": "state_persona_datasets_command",
                "help": "Persona locales and install status.",
            },
        }
    ),
    no_args_is_help=True,
)

agent_app.add_typer(agent_state_app, name="state")

# Add setup command groups
app.add_typer(config_app, name="config", rich_help_panel="Setup")
app.add_typer(download_app, name="download", rich_help_panel="Setup")
app.add_typer(agent_app, name="agent", rich_help_panel="Agent")


def main() -> None:
    """Main entry point for the CLI."""
    ensure_cli_default_model_settings()
    app()


if __name__ == "__main__":
    main()
