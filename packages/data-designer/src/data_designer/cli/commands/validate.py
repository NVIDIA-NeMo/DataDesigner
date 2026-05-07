# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import typer

from data_designer.cli.commands.generation_args import resolve_generation_config_target
from data_designer.cli.controllers.generation_controller import GenerationController


def validate_command(
    workflow_args: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[CONFIG_SOURCE] [-- WORKFLOW_ARGS]",
            help=(
                "Path or URL to a config file (.yaml/.yml/.json), or a local Python module (.py)"
                " that defines a load_config_builder() function. Extra arguments after '--' are forwarded to Python"
                " workflows."
            ),
        ),
    ] = None,
    recipe: Annotated[
        str | None,
        typer.Option(
            "--recipe",
            help="Name of an installed Data Designer recipe to validate instead of a config source.",
        ),
    ] = None,
) -> None:
    """Validate a Data Designer configuration.

    Checks that the configuration is well-formed and all references (models,
    columns, seed datasets, etc.) can be resolved by the engine.

    Examples:
        # Validate a YAML config
        data-designer validate my_config.yaml

        # Validate a remote config URL
        data-designer validate https://example.com/my_config.yaml

        # Validate a Python module
        data-designer validate my_config.py
    """
    target = resolve_generation_config_target(workflow_args, recipe)
    controller = GenerationController()
    controller.run_validate(
        config_source=target.config_source, recipe=target.recipe, workflow_args=target.workflow_args
    )
