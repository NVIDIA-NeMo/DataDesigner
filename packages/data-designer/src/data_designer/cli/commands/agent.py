# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from data_designer.cli.controllers.agent_controller import AgentController
from data_designer.cli.services.agent_introspection import AgentIntrospectionError
from data_designer.config.utils.constants import DATA_DESIGNER_HOME

COMPACT_OPTION = typer.Option(False, "--compact", help="Emit compact JSON without indentation.")


def context_command(compact: bool = COMPACT_OPTION) -> None:
    """Return the self-describing agent bootstrap payload as JSON."""
    _run_command("agent_context", lambda controller: controller.get_context(), compact=compact)


def types_command(
    family: str | None = typer.Argument(None, help="Optional schema family name."),
    compact: bool = COMPACT_OPTION,
) -> None:
    """Return available types for one family or all families as JSON."""
    _run_command("agent_types", lambda controller: controller.get_types(family), compact=compact)


def schema_command(
    family: str = typer.Argument(..., help="Schema family name."),
    type_name: str | None = typer.Argument(None, help="Type name within the selected family."),
    all_types: bool = typer.Option(False, "--all", help="Return every schema in the selected family."),
    compact: bool = COMPACT_OPTION,
) -> None:
    """Return JSON schema for one family member or all family members."""
    _run_command(
        "agent_schema",
        lambda controller: controller.get_schema(family, type_name, all_types=all_types),
        compact=compact,
    )


def builder_command(compact: bool = COMPACT_OPTION) -> None:
    """Return the DataDesignerConfigBuilder API surface as JSON."""
    _run_command("agent_builder", lambda controller: controller.get_builder(), compact=compact)


def state_model_aliases_command(compact: bool = COMPACT_OPTION) -> None:
    """Return configured model alias state as JSON."""
    _run_command("agent_state_model_aliases", lambda controller: controller.get_model_aliases_state(), compact=compact)


def state_persona_datasets_command(compact: bool = COMPACT_OPTION) -> None:
    """Return persona dataset install state as JSON."""
    _run_command(
        "agent_state_persona_datasets",
        lambda controller: controller.get_persona_datasets_state(),
        compact=compact,
    )


def _run_command(kind: str, get_data: Callable[[AgentController], Any], *, compact: bool) -> None:
    controller = AgentController(DATA_DESIGNER_HOME)

    try:
        _emit_json_response(
            kind=kind,
            library_version=controller.get_library_version(),
            data=get_data(controller),
            compact=compact,
        )
    except AgentIntrospectionError as exc:
        _write_error_response(code=exc.code, message=exc.message, details=exc.details, compact=compact)
        raise typer.Exit(code=1)
    except Exception as exc:
        _write_error_response(
            code="internal_error",
            message=str(exc),
            details={"exception_type": type(exc).__name__},
            compact=compact,
        )
        raise typer.Exit(code=1)


def _emit_json_response(*, kind: str, library_version: str, data: Any, compact: bool) -> None:
    typer.echo(
        json.dumps(
            {
                "kind": kind,
                "library_version": library_version,
                "data": data,
            },
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
    )


def _write_error_response(*, code: str, message: str, details: dict[str, Any] | None = None, compact: bool) -> None:
    typer.echo(
        json.dumps(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "details": details or {},
                },
            },
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        ),
        err=True,
    )
