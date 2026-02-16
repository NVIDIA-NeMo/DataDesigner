# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

import typer

from data_designer.cli.controllers.agent_context_controller import AgentContextController


class OutputFormat(str, Enum):
    """Supported output formats for introspect commands."""

    TEXT = "text"
    JSON = "json"


agent_context_app = typer.Typer(
    name="introspect",
    help="Introspect Data Designer's API for agent consumption.",
    no_args_is_help=True,
)


def _make_controller(output_format: OutputFormat) -> AgentContextController:
    return AgentContextController(output_format=output_format.value)


@agent_context_app.command(name="columns")
def columns_command(
    type_name: str | None = typer.Argument(
        None, help="Column type to display (e.g., 'llm-text'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show column configuration types and their fields."""
    _make_controller(output_format).show_columns(type_name, list_mode)


@agent_context_app.command(name="samplers")
def samplers_command(
    type_name: str | None = typer.Argument(
        None, help="Sampler type to display (e.g., 'category'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show sampler types and their parameter fields."""
    _make_controller(output_format).show_samplers(type_name, list_mode)


@agent_context_app.command(name="validators")
def validators_command(
    type_name: str | None = typer.Argument(
        None, help="Validator type to display (e.g., 'code'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show validator types and their parameter fields."""
    _make_controller(output_format).show_validators(type_name, list_mode)


@agent_context_app.command(name="processors")
def processors_command(
    type_name: str | None = typer.Argument(
        None, help="Processor type to display (e.g., 'drop_columns'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show processor types and their configuration fields."""
    _make_controller(output_format).show_processors(type_name, list_mode)


@agent_context_app.command(name="models")
def models_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show model configuration types (ModelConfig, inference params, distributions)."""
    _make_controller(output_format).show_models()


@agent_context_app.command(name="builder")
def builder_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show DataDesignerConfigBuilder method signatures and documentation."""
    _make_controller(output_format).show_builder()


@agent_context_app.command(name="constraints")
def constraints_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show constraint types (ScalarInequality, ColumnInequality, operators)."""
    _make_controller(output_format).show_constraints()


@agent_context_app.command(name="seeds")
def seeds_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show seed dataset types (SeedConfig, sources, sampling strategies)."""
    _make_controller(output_format).show_seeds()


@agent_context_app.command(name="mcp")
def mcp_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show MCP provider types (MCPProvider, LocalStdioMCPProvider, ToolConfig)."""
    _make_controller(output_format).show_mcp()


@agent_context_app.command(name="interface")
def interface_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show DataDesigner class methods, result types, and RunConfig fields."""
    _make_controller(output_format).show_interface()


@agent_context_app.command(name="imports")
def imports_command(
    category: str | None = typer.Argument(None, help="Filter by category (e.g., 'columns'), or omit for all."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show categorized import reference for data_designer.config and data_designer.interface."""
    _make_controller(output_format).show_imports(category)


@agent_context_app.command(name="code-structure")
def code_structure_command(
    depth: int = typer.Option(2, "--depth", "-d", help="Max tree depth (default: 2)."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show the data_designer package structure and install paths."""
    _make_controller(output_format).show_code_structure(depth=depth)


@agent_context_app.command(name="overview")
def overview_command(
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", "-f", help="Output format: 'text' or 'json'."
    ),
) -> None:
    """Show compact API cheatsheet with type counts, builder summary, and quick start commands."""
    _make_controller(output_format).show_overview()
