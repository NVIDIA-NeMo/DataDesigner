# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer

from data_designer.cli.controllers.generation_controller import GenerationController
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface.results import SUPPORTED_EXPORT_FORMATS


def create_command(
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
        help="Number of records to generate.",
        min=1,
    ),
    dataset_name: str = typer.Option(
        "dataset",
        "--dataset-name",
        "-d",
        help="Name for the generated dataset folder.",
    ),
    artifact_path: str | None = typer.Option(
        None,
        "--artifact-path",
        "-o",
        help="Path where generated artifacts will be stored. Defaults to ./artifacts.",
    ),
    resume: ResumeMode = typer.Option(
        ResumeMode.NEVER,
        "--resume",
        "-r",
        help=(
            "Resume an interrupted generation run. "
            "'never' (default): always start fresh. "
            "'always': resume from the last checkpoint; raise if config changed. "
            "'if_possible': resume if config matches, otherwise start fresh silently."
        ),
        case_sensitive=False,
    ),
    output_format: str | None = typer.Option(
        None,
        "--output-format",
        "-f",
        click_type=click.Choice(list(SUPPORTED_EXPORT_FORMATS)),
        help=(
            "Export the dataset to a single file after generation. "
            "Supported formats: jsonl, csv, parquet. "
            "The file is written to <artifact-path>/<dataset-name>/<dataset-name>.<format>."
        ),
    ),
    tui: bool | None = typer.Option(
        None,
        "--tui/--no-tui",
        help=(
            "Force the terminal progress TUI on or off for this run. "
            "When omitted, uses the configured RunConfig setting."
        ),
    ),
) -> None:
    """Create a full dataset and save results to disk.

    This runs the complete generation pipeline: building the dataset, profiling
    the data, and storing all artifacts to the specified output path.

    Examples:
        # Create dataset from a YAML config
        data-designer create my_config.yaml

        # Create with custom settings
        data-designer create my_config.yaml --num-records 1000 --dataset-name my_dataset

        # Resume an interrupted run
        data-designer create my_config.yaml --resume always

        # Resume if config unchanged, otherwise start fresh
        data-designer create my_config.yaml --resume if_possible

        # Create from a Python module with custom output path
        data-designer create my_config.py --artifact-path /path/to/output
    """
    controller = GenerationController()
    controller.run_create(
        config_source=config_source,
        num_records=num_records,
        dataset_name=dataset_name,
        artifact_path=artifact_path,
        resume=resume,
        output_format=output_format,
        tui=tui,
    )
