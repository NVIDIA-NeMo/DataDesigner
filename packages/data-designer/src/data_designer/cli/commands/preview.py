# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer

from data_designer.cli.controllers.generation_controller import GenerationController
from data_designer.config.utils.constants import DEFAULT_DISPLAY_WIDTH, DEFAULT_NUM_RECORDS


def preview_command(
    config_source: str = typer.Argument(
        help=(
            "Path or URL to a config file (.yaml/.yml/.json), or a local Python module (.py)"
            " that defines a load_config_builder() function."
        ),
    ),
    num_records: int = typer.Option(
        DEFAULT_NUM_RECORDS,
        "--num-records",
        "-n",
        help="Number of records to generate in the preview.",
        min=1,
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Display all records at once instead of browsing interactively. Ignored when --save-results is used.",
    ),
    save_results: bool = typer.Option(
        False,
        "--save-results",
        help="Save results to disk without displaying sample records in the terminal.",
    ),
    artifact_path: str | None = typer.Option(
        None,
        "--artifact-path",
        "-o",
        help="Directory for saved results (used with --save-results). Defaults to ./artifacts.",
    ),
    theme: str = typer.Option(
        "dark",
        "--theme",
        click_type=click.Choice(["dark", "light"], case_sensitive=False),
        help="Color theme for HTML output (dark or light). Only applies when --save-results is used.",
    ),
    display_width: int = typer.Option(
        DEFAULT_DISPLAY_WIDTH,
        "--display-width",
        help="Maximum width of the rendered record output in characters.",
        min=40,
    ),
) -> None:
    """Generate a preview dataset for fast iteration on your configuration."""
    controller = GenerationController()
    controller.run_preview(
        config_source=config_source,
        num_records=num_records,
        non_interactive=non_interactive,
        save_results=save_results,
        artifact_path=artifact_path,
        theme=theme,
        display_width=display_width,
    )
