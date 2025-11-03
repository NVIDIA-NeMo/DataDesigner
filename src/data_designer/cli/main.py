# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import typer

# Initialize Typer app with custom configuration
app = typer.Typer(
    name="data-designer",
    help="Data Designer CLI - Configure model providers and models for synthetic data generation",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Import and register command groups
# We import here to avoid circular dependencies
from data_designer.cli.commands import list as list_cmd
from data_designer.cli.commands import models, providers

# Create config subcommand group
config_app = typer.Typer(
    name="config",
    help="Manage configuration files",
    no_args_is_help=True,
)
config_app.command(name="providers", help="Configure model providers interactively")(providers.providers_command)
config_app.command(name="models", help="Configure models interactively")(models.models_command)
config_app.command(name="list", help="List current configurations")(list_cmd.list_command)

app.add_typer(config_app, name="config")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
