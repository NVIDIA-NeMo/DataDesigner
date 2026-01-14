# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.commands import download, models, providers, reset
from data_designer.cli.commands import list as list_cmd
from data_designer.config.default_model_settings import resolve_seed_default_model_settings
from data_designer.config.utils.misc import can_run_data_designer_locally

# Resolve default model settings on import to ensure they are available when the library is used.
if can_run_data_designer_locally():
    resolve_seed_default_model_settings()

# Initialize Typer app with custom configuration
app = typer.Typer(
    name="data-designer",
    help="Data Designer CLI - Configure model providers and models for synthetic data generation",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Create config subcommand group
config_app = typer.Typer(
    name="config",
    help="Manage configuration files",
    no_args_is_help=True,
)
config_app.command(name="providers", help="Configure model providers interactively")(providers.providers_command)
config_app.command(name="models", help="Configure models interactively")(models.models_command)
config_app.command(name="list", help="List current configurations")(list_cmd.list_command)
config_app.command(name="reset", help="Reset configuration files")(reset.reset_command)

# Create download command group
download_app = typer.Typer(
    name="download",
    help="Download assets for Data Designer",
    no_args_is_help=True,
)
download_app.command(name="personas", help="Download Nemotron-Persona datasets")(download.personas_command)

# Add command groups to main app
app.add_typer(config_app, name="config")
app.add_typer(download_app, name="download")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
