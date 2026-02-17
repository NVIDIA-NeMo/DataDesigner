# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.introspection_controller import OutputFormat
from data_designer.cli.controllers.list_assets_controller import ListAssetsController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def list_assets_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """List installed and available Nemotron-Persona datasets."""
    ListAssetsController(DATA_DESIGNER_HOME).list_assets(output_format.value)
