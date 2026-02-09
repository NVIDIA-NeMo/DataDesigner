# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import typer

from data_designer.cli.ui import console, print_error, print_header, print_success, wait_for_navigation_key
from data_designer.cli.utils.config_loader import ConfigLoadError, load_config_builder
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS

if TYPE_CHECKING:
    from data_designer.config.preview_results import PreviewResults


def _display_record_with_header(results: PreviewResults, index: int, total: int) -> None:
    """Display a single record with a record number header."""
    console.print(f"  [bold]Record {index + 1} of {total}[/bold]")
    results.display_sample_record(index=index)


def _browse_records_interactively(results: PreviewResults, total: int) -> None:
    """Interactively browse records with single-keypress navigation.

    Shows the first record immediately, then waits for navigation keys.
    Controls: n/enter=next, p=previous, q/Escape/Ctrl+C=quit.
    Navigation wraps around at both ends.
    """
    current_index = 0
    _display_record_with_header(results, current_index, total)

    while True:
        console.print()
        action = wait_for_navigation_key()

        if action == "q":
            console.print("  [dim]Done browsing.[/dim]")
            break
        elif action == "p":
            current_index = (current_index - 1) % total
        else:
            current_index = (current_index + 1) % total

        _display_record_with_header(results, current_index, total)


def _display_all_records(results: PreviewResults, total: int) -> None:
    """Display all records without interactive prompts."""
    for i in range(total):
        _display_record_with_header(results, i, total)


def preview_command(
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
        help="Number of records to generate in the preview.",
        min=1,
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Display all records at once instead of browsing interactively.",
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

        # Display all records without interactive browsing
        data-designer preview my_config.yaml --non-interactive
    """
    from data_designer.interface import DataDesigner

    try:
        config_builder = load_config_builder(config_source)
    except ConfigLoadError as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    print_header("Data Designer Preview")
    console.print(f"  Config: [bold]{config_source}[/bold]")
    console.print(f"  Records: [bold]{num_records}[/bold]")
    console.print()

    try:
        data_designer = DataDesigner()
        results = data_designer.preview(config_builder, num_records=num_records)
    except Exception as e:
        print_error(f"Preview generation failed: {e}")
        raise typer.Exit(code=1)

    if results.dataset is None or len(results.dataset) == 0:
        print_error("No records were generated.")
        raise typer.Exit(code=1)

    total = len(results.dataset)
    use_interactive = not non_interactive and sys.stdin.isatty() and total > 1

    if use_interactive:
        _browse_records_interactively(results, total)
    else:
        _display_all_records(results, total)

    if results.analysis is not None:
        console.print()
        results.analysis.to_report()

    console.print()
    print_success(f"Preview complete — {total} record(s) generated")
