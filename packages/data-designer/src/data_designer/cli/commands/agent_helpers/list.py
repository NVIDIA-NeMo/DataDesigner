# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.list_controller import ListController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME

list_app = typer.Typer(
    name="list",
    help="List valid values for configuration fields.",
    no_args_is_help=True,
)


@list_app.command(name="model-aliases")
def model_aliases_command() -> None:
    """List configured model aliases and their backing models."""
    ListController(DATA_DESIGNER_HOME).list_model_aliases()


@list_app.command(name="persona-datasets")
def persona_datasets_command() -> None:
    """List available persona datasets and their install status."""
    ListController(DATA_DESIGNER_HOME).list_persona_datasets()


@list_app.command(name="columns")
def column_types_command() -> None:
    """List available column types and their config classes."""
    ListController(DATA_DESIGNER_HOME).list_column_types()


@list_app.command(name="samplers")
def sampler_types_command() -> None:
    """List available sampler types and their params classes."""
    ListController(DATA_DESIGNER_HOME).list_sampler_types()


@list_app.command(name="validators")
def validator_types_command() -> None:
    """List available validator types and their params classes."""
    ListController(DATA_DESIGNER_HOME).list_validator_types()


@list_app.command(name="processors")
def processor_types_command() -> None:
    """List available processor types and their config classes."""
    ListController(DATA_DESIGNER_HOME).list_processor_types()
