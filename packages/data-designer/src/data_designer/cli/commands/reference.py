# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.introspection_controller import IntrospectionController, OutputFormat

reference_app = typer.Typer(
    name="reference",
    help="Reference documentation for Data Designer (overview, interface, code structure, builder, imports).",
    no_args_is_help=True,
)


@reference_app.command(name="overview")
def overview_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show compact API cheatsheet with type counts, builder summary, and quick start commands."""
    IntrospectionController(output_format=output_format.value).show_overview()


@reference_app.command(name="code-structure")
def code_structure_command(
    depth: int = typer.Option(
        2,
        "--depth",
        "-d",
        help="Max tree depth (default: 2). Must be >= 0.",
        min=0,
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show the data_designer package structure and install paths."""
    IntrospectionController(output_format=output_format.value).show_code_structure(depth=depth)


@reference_app.command(name="builder")
def builder_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show DataDesignerConfigBuilder method signatures and documentation."""
    IntrospectionController(output_format=output_format.value).show_builder()


@reference_app.command(name="interface")
def interface_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show DataDesigner class methods, result types, and RunConfig fields."""
    IntrospectionController(output_format=output_format.value).show_interface()


@reference_app.command(name="imports")
def imports_command(
    category: str | None = typer.Argument(None, help="Filter by category (e.g., 'columns'), or omit for all."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show categorized import reference for data_designer.config and data_designer.interface."""
    IntrospectionController(output_format=output_format.value).show_imports(category)
