# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.list_controller import ListController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME

list_app = typer.Typer(
    name="list",
    help=(
        "Enumerate valid names and classes for configuration fields."
        " Use these names as arguments to `inspect` commands for detailed schemas."
    ),
    no_args_is_help=True,
)


@list_app.command(name="model-aliases")
def model_aliases_command() -> None:
    """List all configured model aliases with their backing model identifiers. Required to set model_alias on LLM column configs."""
    ListController(DATA_DESIGNER_HOME).list_model_aliases()


@list_app.command(name="persona-datasets")
def persona_datasets_command() -> None:
    """List available Nemotron-Persona datasets and whether each is installed locally."""
    ListController(DATA_DESIGNER_HOME).list_persona_datasets()


@list_app.command(name="columns")
def column_types_command() -> None:
    """List all column type names and their config classes. Pass a name to `inspect column <name>` for the full schema."""
    ListController(DATA_DESIGNER_HOME).list_column_types()


@list_app.command(name="samplers")
def sampler_types_command() -> None:
    """List all sampler type names and their params classes. Pass a name to `inspect sampler <name>` for the full schema."""
    ListController(DATA_DESIGNER_HOME).list_sampler_types()


@list_app.command(name="validators")
def validator_types_command() -> None:
    """List all validator type names and their params classes. Pass a name to `inspect validator <name>` for the full schema."""
    ListController(DATA_DESIGNER_HOME).list_validator_types()


@list_app.command(name="processors")
def processor_types_command() -> None:
    """List all processor type names and their config classes. Pass a name to `inspect processor <name>` for the full schema."""
    ListController(DATA_DESIGNER_HOME).list_processor_types()
