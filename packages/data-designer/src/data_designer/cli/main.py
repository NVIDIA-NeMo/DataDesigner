# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.lazy_group import create_lazy_typer_group

_CMD = "data_designer.cli.commands"

# Initialize Typer app with custom configuration
app = typer.Typer(
    name="data-designer",
    help="Data Designer CLI for humans and agents.",
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

# Create list command group
list_app = typer.Typer(
    name="list",
    help="List available types, model aliases, and persona datasets.",
    cls=create_lazy_typer_group(
        {
            "model-aliases": {
                "module": f"{_CMD}.agent_helpers.list",
                "attr": "model_aliases_command",
                "help": "List configured model aliases and backing models",
            },
            "persona-datasets": {
                "module": f"{_CMD}.agent_helpers.list",
                "attr": "persona_datasets_command",
                "help": "List Nemotron-Persona datasets and install status",
            },
            "columns": {
                "module": f"{_CMD}.agent_helpers.list",
                "attr": "column_types_command",
                "help": "List column type names and config classes",
            },
            "samplers": {
                "module": f"{_CMD}.agent_helpers.list",
                "attr": "sampler_types_command",
                "help": "List sampler type names and params classes",
            },
            "validators": {
                "module": f"{_CMD}.agent_helpers.list",
                "attr": "validator_types_command",
                "help": "List validator type names and params classes",
            },
            "processors": {
                "module": f"{_CMD}.agent_helpers.list",
                "attr": "processor_types_command",
                "help": "List processor type names and config classes",
            },
        }
    ),
    no_args_is_help=True,
)

# Create inspect command group
inspect_app = typer.Typer(
    name="inspect",
    help="Inspect detailed schemas for configuration objects and the Python API.",
    cls=create_lazy_typer_group(
        {
            "column": {
                "module": f"{_CMD}.agent_helpers.inspect",
                "attr": "columns_command",
                "help": "Show schema for a column config type",
            },
            "sampler": {
                "module": f"{_CMD}.agent_helpers.inspect",
                "attr": "samplers_command",
                "help": "Show schema for a sampler params type",
            },
            "validator": {
                "module": f"{_CMD}.agent_helpers.inspect",
                "attr": "validators_command",
                "help": "Show schema for a validator params type",
            },
            "processor": {
                "module": f"{_CMD}.agent_helpers.inspect",
                "attr": "processors_command",
                "help": "Show schema for a processor config type",
            },
            "sampler-constraints": {
                "module": f"{_CMD}.agent_helpers.inspect",
                "attr": "constraints_command",
                "help": "Show constraint schemas for sampler columns",
            },
            "config-builder": {
                "module": f"{_CMD}.agent_helpers.inspect",
                "attr": "config_builder_command",
                "help": "Show DataDesignerConfigBuilder method signatures and docstrings",
            },
        }
    ),
    no_args_is_help=True,
)

# Add setup command groups
app.add_typer(config_app, name="config", rich_help_panel="Setup Commands")
app.add_typer(download_app, name="download", rich_help_panel="Setup Commands")

# Add agent command groups
title_agent_helpers = "Agent-Helper Commands"
app.add_typer(list_app, name="list", rich_help_panel=title_agent_helpers)
app.add_typer(inspect_app, name="inspect", rich_help_panel=title_agent_helpers)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
