# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import typer

from data_designer.cli.ui import console, print_error, print_header, print_success
from data_designer.cli.utils.config_loader import ConfigLoadError, load_config_builder
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS


def create_command(
    config_source: str = typer.Argument(
        help=(
            "Path to a config file (.yaml/.yml/.json) or Python module (.py)"
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
    artifact_path: str = typer.Option(
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

        # Create from a Python module with custom output path
        data-designer create my_config.py --artifact-path /path/to/output
    """
    from data_designer.interface import DataDesigner

    try:
        config_builder = load_config_builder(config_source)
    except ConfigLoadError as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    resolved_artifact_path = Path(artifact_path) if artifact_path else Path.cwd() / "artifacts"

    print_header("Data Designer Create")
    console.print(f"  Config: [bold]{config_source}[/bold]")
    console.print(f"  Records: [bold]{num_records}[/bold]")
    console.print(f"  Dataset name: [bold]{dataset_name}[/bold]")
    console.print(f"  Artifact path: [bold]{resolved_artifact_path}[/bold]")
    console.print()

    try:
        data_designer = DataDesigner(artifact_path=resolved_artifact_path)
        results = data_designer.create(
            config_builder,
            num_records=num_records,
            dataset_name=dataset_name,
        )
    except Exception as e:
        print_error(f"Dataset creation failed: {e}")
        raise typer.Exit(code=1)

    dataset = results.load_dataset()

    analysis = results.load_analysis()
    if analysis is not None:
        console.print()
        analysis.to_report()

    console.print()
    print_success(f"Dataset created — {len(dataset)} record(s) generated")
    console.print(f"  Artifacts saved to: [bold]{results.artifact_storage.base_dataset_path}[/bold]")
    console.print()
