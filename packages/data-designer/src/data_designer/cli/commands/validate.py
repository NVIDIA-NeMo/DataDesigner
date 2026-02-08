# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.ui import console, print_error, print_header, print_success
from data_designer.cli.utils.config_loader import ConfigLoadError, load_config_builder


def validate_command(
    config_source: str = typer.Argument(
        help=(
            "Path to a config file (.yaml/.yml/.json) or Python module (.py)"
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

        # Validate a Python module
        data-designer validate my_config.py
    """
    from data_designer.config.errors import InvalidConfigError
    from data_designer.interface import DataDesigner

    try:
        config_builder = load_config_builder(config_source)
    except ConfigLoadError as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    print_header("Data Designer Validate")
    console.print(f"  Config: [bold]{config_source}[/bold]")
    console.print()

    try:
        data_designer = DataDesigner()
        data_designer.validate(config_builder)
    except InvalidConfigError as e:
        print_error(f"Configuration is invalid: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"Validation failed: {e}")
        raise typer.Exit(code=1)

    print_success("Configuration is valid")
