# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.controllers.review_controller import ReviewController


def review_command(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to the dataset file (.parquet, .csv, .json, .jsonl)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    port: int = typer.Option(
        8501,
        "--port",
        "-p",
        help="Port for Streamlit server (default: 8501)",
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        help="Host address to bind (default: localhost)",
    ),
    reviewer: str = typer.Option(
        "default",
        "--reviewer",
        "-r",
        help="Reviewer name/ID for tracking (default: 'default')",
    ),
) -> None:
    """Launch interactive dataset review UI.

    Opens a Streamlit web interface for reviewing dataset records. You can navigate through
    records, provide thumbs up/down ratings, add comments, and track your progress.

    Reviews are saved as {dataset_name}_reviews.csv in the same directory as your dataset.

    Examples:

        # Basic usage
        data-designer review --dataset output/dataset.parquet

        # Custom port and reviewer
        data-designer review -d data.csv -p 8080 -r john_doe

        # View reviews on network
        data-designer review -d data.parquet --host 0.0.0.0
    """
    controller = ReviewController(dataset_path=dataset, port=port, host=host, reviewer=reviewer)
    controller.run()
