# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.download_controller import DownloadController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be downloaded without actually downloading",
    ),
    list_available: bool = typer.Option(
        False,
        "--list",
        help="List available persona datasets and their sizes",
    ),
) -> None:
    """Download Nemotron-Personas datasets for synthetic data generation.

    Examples:
        # List available datasets
        data-designer download personas --list

        # Interactive selection
        data-designer download personas

        # Download specific locales
        data-designer download personas --locale en_US --locale ja_JP

        # Download all available locales
        data-designer download personas --all

        # Preview what would be downloaded
        data-designer download personas --all --dry-run
    """
    controller = DownloadController(DATA_DESIGNER_HOME)

    if list_available:
        controller.list_personas()
    else:
        controller.run_personas(locales=locales, all_locales=all_locales, dry_run=dry_run)
