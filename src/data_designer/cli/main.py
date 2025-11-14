# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import typer

from data_designer.cli.commands import list as list_cmd
from data_designer.cli.commands import models, providers, reset
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

app.add_typer(config_app, name="config")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
