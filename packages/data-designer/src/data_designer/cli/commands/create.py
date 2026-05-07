# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import click
import typer

from data_designer.cli.commands.generation_args import resolve_generation_config_target
from data_designer.cli.controllers.generation_controller import GenerationController
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.interface.results import SUPPORTED_EXPORT_FORMATS


def create_command(
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
            help="Name of an installed Data Designer recipe to run instead of a config source.",
        ),
    ] = None,
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
) -> None:
    """Create a full dataset and save results to disk.

    This runs the complete generation pipeline: building the dataset, profiling
    the data, and storing all artifacts to the specified output path.

    Examples:
        # Create dataset from a YAML config
        data-designer create my_config.yaml

        # Create with custom settings
        data-designer create my_config.yaml --num-records 1000 --dataset-name my_dataset

        # Create from a remote config URL
        data-designer create https://example.com/my_config.json --dataset-name my_dataset

        # Create from a Python module with custom output path
        data-designer create my_config.py --artifact-path /path/to/output
    """
    target = resolve_generation_config_target(workflow_args, recipe)
    controller = GenerationController()
    controller.run_create(
        config_source=target.config_source,
        recipe=target.recipe,
        workflow_args=target.workflow_args,
        num_records=num_records,
        dataset_name=dataset_name,
        artifact_path=artifact_path,
        output_format=output_format,
    )
