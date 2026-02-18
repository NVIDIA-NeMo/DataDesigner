# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.introspection_controller import IntrospectionController

inspect_app = typer.Typer(
    name="inspect",
    help="Show schemas and method signatures for configuration types. Run `list` to discover valid type names.",
    no_args_is_help=True,
)


@inspect_app.command(name="column")
def columns_command(
    type_name: str = typer.Argument(help="Type name (e.g. 'llm-text', 'sampler'), or 'all'."),
) -> None:
    """Show schema for a column config type. Run `list columns` for valid names."""
    IntrospectionController().show_columns(type_name)


@inspect_app.command(name="sampler")
def samplers_command(
    type_name: str = typer.Argument(help="Type name (e.g. 'category', 'uniform'), or 'all'."),
) -> None:
    """Show schema for a sampler params type. Run `list samplers` for valid names."""
    IntrospectionController().show_samplers(type_name)


@inspect_app.command(name="validator")
def validators_command(
    type_name: str = typer.Argument(help="Type name (e.g. 'code', 'python'), or 'all'."),
) -> None:
    """Show schema for a validator params type. Run `list validators` for valid names."""
    IntrospectionController().show_validators(type_name)


@inspect_app.command(name="processor")
def processors_command(
    type_name: str = typer.Argument(help="Type name (e.g. 'drop_columns'), or 'all'."),
) -> None:
    """Show schema for a processor config type. Run `list processors` for valid names."""
    IntrospectionController().show_processors(type_name)


@inspect_app.command(name="sampler-constraints")
def constraints_command() -> None:
    """Show constraint schemas for sampler columns."""
    IntrospectionController().show_sampler_constraints()


@inspect_app.command(name="config-builder")
def config_builder_command() -> None:
    """Show DataDesignerConfigBuilder method signatures and docstrings."""
    IntrospectionController().show_builder()
