# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.agent_context_controller import AgentContextController

agent_context_app = typer.Typer(
    name="agent-context",
    help="Introspect Data Designer's API for agent consumption.",
    no_args_is_help=True,
)


def _make_controller(output_format: str) -> AgentContextController:
    return AgentContextController(output_format=output_format)


@agent_context_app.command(name="columns")
def columns_command(
    type_name: str | None = typer.Argument(
        None, help="Column type to display (e.g., 'llm-text'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show column configuration types and their fields."""
    _make_controller(output_format).show_columns(type_name, list_mode)


@agent_context_app.command(name="samplers")
def samplers_command(
    type_name: str | None = typer.Argument(
        None, help="Sampler type to display (e.g., 'category'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show sampler types and their parameter fields."""
    _make_controller(output_format).show_samplers(type_name, list_mode)


@agent_context_app.command(name="validators")
def validators_command(
    type_name: str | None = typer.Argument(
        None, help="Validator type to display (e.g., 'code'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show validator types and their parameter fields."""
    _make_controller(output_format).show_validators(type_name, list_mode)


@agent_context_app.command(name="processors")
def processors_command(
    type_name: str | None = typer.Argument(
        None, help="Processor type to display (e.g., 'drop_columns'), or 'all' for everything."
    ),
    list_mode: bool = typer.Option(False, "--list", "-l", help="Show summary table of available types."),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show processor types and their configuration fields."""
    _make_controller(output_format).show_processors(type_name, list_mode)


@agent_context_app.command(name="models")
def models_command(
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show model configuration types (ModelConfig, inference params, distributions)."""
    _make_controller(output_format).show_models()


@agent_context_app.command(name="builder")
def builder_command(
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show DataDesignerConfigBuilder method signatures and documentation."""
    _make_controller(output_format).show_builder()


@agent_context_app.command(name="constraints")
def constraints_command(
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show constraint types (ScalarInequality, ColumnInequality, operators)."""
    _make_controller(output_format).show_constraints()


@agent_context_app.command(name="seeds")
def seeds_command(
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show seed dataset types (SeedConfig, sources, sampling strategies)."""
    _make_controller(output_format).show_seeds()


@agent_context_app.command(name="mcp")
def mcp_command(
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show MCP provider types (MCPProvider, LocalStdioMCPProvider, ToolConfig)."""
    _make_controller(output_format).show_mcp()


@agent_context_app.command(name="overview")
def overview_command(
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' (default) or 'json'."),
) -> None:
    """Show compact API cheatsheet with type counts, builder summary, and quick start commands."""
    _make_controller(output_format).show_overview()
