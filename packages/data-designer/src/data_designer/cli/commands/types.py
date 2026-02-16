# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.introspection_controller import IntrospectionController, OutputFormat

types_app = typer.Typer(
    name="types",
    help="Explore Data Designer configuration types (columns, samplers, validators, etc.).",
    no_args_is_help=True,
)


def _print_usage_hint(command_name: str) -> None:
    """Print a usage hint after the type list (text format only)."""
    typer.echo("")
    typer.echo(f"Tip: Run `data-designer types {command_name} <TYPE_NAME>` for full schema details.")
    typer.echo(f"     Run `data-designer types {command_name} all` to see every type expanded.")


@types_app.command(name="columns")
def columns_command(
    type_name: str | None = typer.Argument(
        None, help="Column type to display (e.g., 'llm-text'), or 'all' for everything."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show column configuration types and their fields."""
    ctrl = IntrospectionController(output_format=output_format.value)
    ctrl.show_columns(type_name)
    if type_name is None and output_format == OutputFormat.TEXT:
        _print_usage_hint("columns")


@types_app.command(name="samplers")
def samplers_command(
    type_name: str | None = typer.Argument(
        None, help="Sampler type to display (e.g., 'category'), or 'all' for everything."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show sampler types and their parameter fields."""
    ctrl = IntrospectionController(output_format=output_format.value)
    ctrl.show_samplers(type_name)
    if type_name is None and output_format == OutputFormat.TEXT:
        _print_usage_hint("samplers")


@types_app.command(name="validators")
def validators_command(
    type_name: str | None = typer.Argument(
        None, help="Validator type to display (e.g., 'code'), or 'all' for everything."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show validator types and their parameter fields."""
    ctrl = IntrospectionController(output_format=output_format.value)
    ctrl.show_validators(type_name)
    if type_name is None and output_format == OutputFormat.TEXT:
        _print_usage_hint("validators")


@types_app.command(name="processors")
def processors_command(
    type_name: str | None = typer.Argument(
        None, help="Processor type to display (e.g., 'drop_columns'), or 'all' for everything."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show processor types and their configuration fields."""
    ctrl = IntrospectionController(output_format=output_format.value)
    ctrl.show_processors(type_name)
    if type_name is None and output_format == OutputFormat.TEXT:
        _print_usage_hint("processors")


@types_app.command(name="models")
def models_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show model configuration types (ModelConfig, inference params, distributions)."""
    IntrospectionController(output_format=output_format.value).show_models()


@types_app.command(name="constraints")
def constraints_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show constraint types (ScalarInequality, ColumnInequality, operators)."""
    IntrospectionController(output_format=output_format.value).show_constraints()


@types_app.command(name="seeds")
def seeds_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show seed dataset types (SeedConfig, sources, sampling strategies)."""
    IntrospectionController(output_format=output_format.value).show_seeds()


@types_app.command(name="mcp")
def mcp_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show MCP provider types (MCPProvider, LocalStdioMCPProvider, ToolConfig)."""
    IntrospectionController(output_format=output_format.value).show_mcp()
