# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.introspection_controller import IntrospectionController

inspect_app = typer.Typer(
    name="inspect",
    help=(
        "Return detailed schemas (fields, types, defaults, constraints) for configuration types,"
        " or method signatures for the Python API. Use `list` first to discover valid type names."
    ),
    no_args_is_help=True,
)


@inspect_app.command(name="column")
def columns_command(
    type_name: str = typer.Argument(
        help="Column type name (e.g., 'llm-text', 'sampler'). Pass 'all' to dump every column type."
    ),
) -> None:
    """Return the full schema for a column config type, including field names, types, defaults, and descriptions. Run `list columns` to discover valid type names."""
    IntrospectionController().show_columns(type_name)


@inspect_app.command(name="sampler")
def samplers_command(
    type_name: str = typer.Argument(
        help="Sampler type name (e.g., 'category', 'uniform'). Pass 'all' to dump every sampler type."
    ),
) -> None:
    """Return the full params schema for a sampler type, including field names, types, defaults, and descriptions. Run `list samplers` to discover valid type names."""
    IntrospectionController().show_samplers(type_name)


@inspect_app.command(name="validator")
def validators_command(
    type_name: str = typer.Argument(
        help="Validator type name (e.g., 'code', 'python'). Pass 'all' to dump every validator type."
    ),
) -> None:
    """Return the full params schema for a validator type, including field names, types, defaults, and descriptions. Run `list validators` to discover valid type names."""
    IntrospectionController().show_validators(type_name)


@inspect_app.command(name="processor")
def processors_command(
    type_name: str = typer.Argument(
        help="Processor type name (e.g., 'drop_columns'). Pass 'all' to dump every processor type."
    ),
) -> None:
    """Return the full config schema for a processor type, including field names, types, defaults, and descriptions. Run `list processors` to discover valid type names."""
    IntrospectionController().show_processors(type_name)


@inspect_app.command(name="sampler-constraints")
def constraints_command() -> None:
    """Return schemas for sampler constraint types: ScalarInequalityConstraint, ColumnInequalityConstraint, and the InequalityOperator enum. Use when adding value constraints to sampler columns."""
    IntrospectionController().show_sampler_constraints()


@inspect_app.command(name="config-builder")
def config_builder_command() -> None:
    """Return DataDesignerConfigBuilder method signatures and docstrings. Use to understand available builder methods and their parameters."""
    IntrospectionController().show_builder()
