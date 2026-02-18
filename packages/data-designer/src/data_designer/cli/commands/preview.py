# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.generation_controller import GenerationController
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS


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
        help="Display all records at once instead of browsing interactively.",
    ),
    save_report: bool = typer.Option(
        False,
        "--save-report",
        help="Save the analysis report as an HTML file in the artifact path.",
    ),
    artifact_path: str | None = typer.Option(
        None,
        "--artifact-path",
        "-o",
        help="Path where the report will be saved. Defaults to ./artifacts.",
    ),
) -> None:
    """Generate a preview dataset for fast iteration on your configuration.

    Preview results are displayed in the terminal. Use this to quickly validate
    your configuration before running a full dataset creation.

    By default, records are displayed one at a time in interactive mode. Use
    --non-interactive to display all records at once (also used automatically
    when output is piped).

    Examples:
        # Preview from a YAML config
        data-designer preview my_config.yaml

        # Preview from a Python module
        data-designer preview my_config.py

        # Preview with custom number of records
        data-designer preview my_config.yaml --num-records 5

        # Preview from a remote config URL
        data-designer preview https://example.com/my_config.yaml

        # Display all records without interactive browsing
        data-designer preview my_config.yaml --non-interactive
    """
    controller = GenerationController()
    controller.run_preview(
        config_source=config_source,
        num_records=num_records,
        non_interactive=non_interactive,
        save_report=save_report,
        artifact_path=artifact_path,
    )
