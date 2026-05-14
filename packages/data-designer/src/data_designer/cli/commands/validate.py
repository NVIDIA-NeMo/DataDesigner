# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.generation_controller import GenerationController


def validate_command(
    config_source: str = typer.Argument(
        help=(
            "Path or URL to a config file (.yaml/.yml/.json), or a local Python module (.py)"
            " that defines a load_config_builder() function."
        ),
    ),
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
    controller = GenerationController()
    controller.run_validate(config_source=config_source)
