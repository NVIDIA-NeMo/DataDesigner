# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from data_designer.cli.utils.agent_introspection import (
    AgentContextError,
    get_builder_api,
    get_context,
    get_model_aliases_state,
    get_persona_datasets_state,
    get_schema,
    get_types,
)
from data_designer.cli.utils.agent_text_formatter import (
    format_builder_text,
    format_context_text,
    format_model_aliases_text,
    format_persona_datasets_text,
    format_schema_text,
    format_types_text,
)
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def context_command() -> None:
    """Return a bootstrap payload with types, local state, and builder summary."""
    _run(lambda: get_context(DATA_DESIGNER_HOME), format_context_text)


def types_command(
    family: str | None = typer.Argument(None, help="Optional schema family name."),
) -> None:
    """Return available type names and import paths for one family or all families."""
    _run(lambda: get_types(family), format_types_text)


def schema_command(
    family: str = typer.Argument(..., help="Schema family name."),
    type_name: str | None = typer.Argument(None, help="Type name within the selected family."),
    all_types: bool = typer.Option(False, "--all", help="Return every schema in the selected family."),
) -> None:
    """Return schema for a specific type or every type in a family."""
    _run(lambda: get_schema(family, type_name, all_types=all_types), format_schema_text)


def builder_command() -> None:
    """Return the DataDesignerConfigBuilder method surface with signatures and docstrings."""
    _run(get_builder_api, format_builder_text)


def state_model_aliases_command() -> None:
    """Return configured model aliases and whether each one is currently usable."""
    _run(lambda: get_model_aliases_state(DATA_DESIGNER_HOME), format_model_aliases_text)


def state_persona_datasets_command() -> None:
    """Return built-in persona locales and whether each dataset is installed locally."""
    _run(lambda: get_persona_datasets_state(DATA_DESIGNER_HOME), format_persona_datasets_text)


def _run(get_data: Callable[[], Any], format_text: Callable[[Any], str]) -> None:
    try:
        data = get_data()
        typer.echo(format_text(data))
    except AgentContextError as exc:
        typer.echo(f"Error [{exc.code}]: {exc.message}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Error [internal_error]: {exc}", err=True)
        raise typer.Exit(code=1)
