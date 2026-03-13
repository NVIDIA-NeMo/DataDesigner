# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from data_designer.cli.utils.agent_introspection import (
    AgentIntrospectionError,
    get_builder_api,
    get_context,
    get_library_version,
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

COMPACT_OPTION = typer.Option(False, "--compact", help="Emit compact JSON without indentation.")
JSON_OPTION = typer.Option(False, "--json", help="Emit structured JSON instead of text.")


def context_command(use_json: bool = JSON_OPTION, compact: bool = COMPACT_OPTION) -> None:
    """Return a bootstrap payload with types, local state, and builder summary."""
    _run(
        lambda: get_context(DATA_DESIGNER_HOME),
        format_context_text,
        kind="agent_context",
        use_json=use_json,
        compact=compact,
    )


def types_command(
    family: str | None = typer.Argument(None, help="Optional schema family name."),
    use_json: bool = JSON_OPTION,
    compact: bool = COMPACT_OPTION,
) -> None:
    """Return available type names and import paths for one family or all families."""
    _run(lambda: get_types(family), format_types_text, kind="agent_types", use_json=use_json, compact=compact)


def schema_command(
    family: str = typer.Argument(..., help="Schema family name."),
    type_name: str | None = typer.Argument(None, help="Type name within the selected family."),
    all_types: bool = typer.Option(False, "--all", help="Return every schema in the selected family."),
    use_json: bool = JSON_OPTION,
    compact: bool = COMPACT_OPTION,
) -> None:
    """Return schema for a specific type or every type in a family."""
    _run(
        lambda: get_schema(family, type_name, all_types=all_types),
        format_schema_text,
        kind="agent_schema",
        use_json=use_json,
        compact=compact,
    )


def builder_command(use_json: bool = JSON_OPTION, compact: bool = COMPACT_OPTION) -> None:
    """Return the DataDesignerConfigBuilder method surface with signatures and docstrings."""
    _run(get_builder_api, format_builder_text, kind="agent_builder", use_json=use_json, compact=compact)


def state_model_aliases_command(use_json: bool = JSON_OPTION, compact: bool = COMPACT_OPTION) -> None:
    """Return configured model aliases and whether each one is currently usable."""
    _run(
        lambda: get_model_aliases_state(DATA_DESIGNER_HOME),
        format_model_aliases_text,
        kind="agent_state_model_aliases",
        use_json=use_json,
        compact=compact,
    )


def state_persona_datasets_command(use_json: bool = JSON_OPTION, compact: bool = COMPACT_OPTION) -> None:
    """Return built-in persona locales and whether each dataset is installed locally."""
    _run(
        lambda: get_persona_datasets_state(DATA_DESIGNER_HOME),
        format_persona_datasets_text,
        kind="agent_state_persona_datasets",
        use_json=use_json,
        compact=compact,
    )


def _run(
    get_data: Callable[[], Any],
    format_text: Callable[[Any], str],
    *,
    kind: str,
    use_json: bool,
    compact: bool,
) -> None:
    try:
        data = get_data()
        if use_json:
            _emit({"kind": kind, "library_version": get_library_version(), "data": data}, compact=compact)
        else:
            typer.echo(format_text(data))
    except AgentIntrospectionError as exc:
        _emit({"error": {"code": exc.code, "message": exc.message, "details": exc.details}}, compact=compact, err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        _emit(
            {
                "error": {
                    "code": "internal_error",
                    "message": str(exc),
                    "details": {"exception_type": type(exc).__name__},
                }
            },
            compact=compact,
            err=True,
        )
        raise typer.Exit(code=1)


def _emit(payload: Any, *, compact: bool, err: bool = False) -> None:
    typer.echo(
        json.dumps(payload, indent=None if compact else 2, separators=(",", ":") if compact else None),
        err=err,
    )
