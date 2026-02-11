# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.generation_controller import GenerationController
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS


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
    controller = GenerationController()
    controller.run_create(
        config_source=config_source,
        num_records=num_records,
        dataset_name=dataset_name,
        artifact_path=artifact_path,
    )
