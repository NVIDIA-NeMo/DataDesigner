# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import typer

from data_designer.cli.controllers.download_controller import DownloadController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME

# Create download command group
download_app = typer.Typer(
    name="download",
    help="Download assets for Data Designer",
    no_args_is_help=True,
)


@download_app.command(name="personas")
def personas_command(
    locales: list[str] = typer.Option(
        None,
        "--locale",
        "-l",
        help="Locales to download (en_US, en_IN, hi_Deva_IN, hi_Latn_IN, ja_JP). Can be specified multiple times.",
    ),
    all_locales: bool = typer.Option(
        False,
        "--all",
        help="Download all available locales",
    ),
) -> None:
    """Download persona datasets for synthetic data generation.

    Persona datasets contain diverse character profiles that can be used
    to generate contextually rich synthetic data.

    Examples:
        # Interactive selection
        data-designer download personas

        # Download specific locales
        data-designer download personas --locale en_US --locale ja_JP

        # Download all available locales
        data-designer download personas --all
    """
    controller = DownloadController(DATA_DESIGNER_HOME)
    controller.run_personas(locales=locales, all_locales=all_locales)
