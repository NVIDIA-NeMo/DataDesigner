# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.introspection_controller import IntrospectionController

inspect_app = typer.Typer(
    name="inspect",
    help="Inspect configuration types and Python API (schemas, method signatures).",
    no_args_is_help=True,
)


@inspect_app.command(name="column")
def columns_command(
    type_name: str = typer.Argument(help="Column type to display (e.g., 'llm-text'), or 'all' for everything."),
) -> None:
    """Show schema for a column config type (use `list columns` for valid names)."""
    IntrospectionController().show_columns(type_name)


@inspect_app.command(name="sampler")
def samplers_command(
    type_name: str = typer.Argument(help="Sampler type to display (e.g., 'category'), or 'all' for everything."),
) -> None:
    """Show schema for a sampler params type (use `list samplers` for valid names)."""
    IntrospectionController().show_samplers(type_name)


@inspect_app.command(name="validator")
def validators_command(
    type_name: str = typer.Argument(help="Validator type to display (e.g., 'code'), or 'all' for everything."),
) -> None:
    """Show schema for a validator params type (use `list validators` for valid names)."""
    IntrospectionController().show_validators(type_name)


@inspect_app.command(name="processor")
def processors_command(
    type_name: str = typer.Argument(help="Processor type to display (e.g., 'drop_columns'), or 'all' for everything."),
) -> None:
    """Show schema for a processor config type (use `list processors` for valid names)."""
    IntrospectionController().show_processors(type_name)


@inspect_app.command(name="sampler-constraints")
def constraints_command() -> None:
    """Show sampler constraint schemas (scalar inequality, column inequality, operators)."""
    IntrospectionController().show_sampler_constraints()


@inspect_app.command(name="builder")
def builder_command() -> None:
    """Show config builder method signatures and docstrings."""
    IntrospectionController().show_builder()
