# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import typer

from data_designer.cli.services.introspection.discovery import (
    discover_column_configs,
    discover_constraint_types,
    discover_processor_configs,
    discover_sampler_types,
    discover_validator_types,
)
from data_designer.cli.services.introspection.formatters import (
    format_method_info_text,
    format_type_list_text,
)
from data_designer.cli.services.introspection.method_inspector import inspect_class_methods
from data_designer.cli.services.introspection.pydantic_inspector import format_model_text
from data_designer.config.config_builder import DataDesignerConfigBuilder


@dataclass(frozen=True)
class _TypedCommandSpec:
    """Configuration for typed introspection commands."""

    discover_items: Callable[[], dict[str, type]]
    type_key: str
    type_label: str
    class_label: str
    header_title: str
    case_insensitive: bool = False
    related_inspect_tip: str | None = None


_CONFIG_IMPORT = "import data_designer.config as dd"


class IntrospectionController:
    """Controller for introspect CLI commands.

    Orchestrates discovery, inspection, formatting, and output for all
    introspect subcommands.
    """

    _TYPED_COMMAND_SPECS: dict[str, _TypedCommandSpec] = {
        "columns": _TypedCommandSpec(
            discover_items=discover_column_configs,
            type_key="column_type",
            type_label="column_type",
            class_label="config_class",
            header_title="Data Designer Column Types Reference",
            case_insensitive=True,
            related_inspect_tip=(
                "Tip: Use 'data-designer inspect sampler <TYPE>' for sampler params,"
                " 'inspect validator <TYPE>' for validator params."
            ),
        ),
        "samplers": _TypedCommandSpec(
            discover_items=discover_sampler_types,
            type_key="sampler_type",
            type_label="sampler_type",
            class_label="params_class",
            header_title="Data Designer Sampler Types Reference",
            case_insensitive=True,
        ),
        "validators": _TypedCommandSpec(
            discover_items=discover_validator_types,
            type_key="validator_type",
            type_label="validator_type",
            class_label="params_class",
            header_title="Data Designer Validator Types Reference",
            case_insensitive=True,
        ),
        "processors": _TypedCommandSpec(
            discover_items=discover_processor_configs,
            type_key="processor_type",
            type_label="processor_type",
            class_label="config_class",
            header_title="Data Designer Processor Types Reference",
            case_insensitive=True,
        ),
    }

    def _emit_import_hint(self, import_stmt: str, access: str | None = None) -> None:
        """Print a one-line import hint."""
        line = f"# {import_stmt}"
        if access:
            line += f"  \u2192  {access}"
        typer.echo(line)
        typer.echo("")

    def show_columns(self, type_name: str | None) -> None:
        """Show column configuration types."""
        self._show_typed_command(command_name="columns", type_name=type_name)

    def show_samplers(self, type_name: str | None) -> None:
        """Show sampler types and their param classes."""
        self._show_typed_command(command_name="samplers", type_name=type_name)

    def show_validators(self, type_name: str | None) -> None:
        """Show validator types and their param classes."""
        self._show_typed_command(command_name="validators", type_name=type_name)

    def show_processors(self, type_name: str | None) -> None:
        """Show processor types and their config classes."""
        self._show_typed_command(command_name="processors", type_name=type_name)

    def show_builder(self) -> None:
        """Show DataDesignerConfigBuilder method signatures and docs."""
        self._emit_import_hint(_CONFIG_IMPORT, "dd.DataDesignerConfigBuilder")
        methods = inspect_class_methods(DataDesignerConfigBuilder)
        typer.echo(format_method_info_text(methods, class_name="DataDesignerConfigBuilder"))

    def show_sampler_constraints(self) -> None:
        """Show sampler constraint types."""
        self._emit_import_hint(_CONFIG_IMPORT, "dd.<ClassName>")
        items = discover_constraint_types()
        self._show_all_schemas(items, "Data Designer Constraint Types Reference")

    def _show_typed_command(self, command_name: str, type_name: str | None) -> None:
        """Resolve a typed-command spec and render it."""
        spec = self._TYPED_COMMAND_SPECS[command_name]
        self._show_typed_items(
            items=spec.discover_items(),
            type_name=type_name,
            type_key=spec.type_key,
            type_label=spec.type_label,
            class_label=spec.class_label,
            header_title=spec.header_title,
            case_insensitive=spec.case_insensitive,
            related_inspect_tip=spec.related_inspect_tip,
        )

    def _show_typed_items(
        self,
        items: dict[str, type],
        type_name: str | None,
        type_key: str,
        type_label: str,
        class_label: str,
        header_title: str,
        case_insensitive: bool = False,
        related_inspect_tip: str | None = None,
    ) -> None:
        """Shared logic for type-based commands (columns, samplers, validators, processors)."""
        if type_name is None:
            self._emit_import_hint(_CONFIG_IMPORT, "dd.<ClassName>")
            typer.echo(format_type_list_text(items, type_label, class_label))
            return

        if type_name.lower() == "all":
            self._show_all_typed(items, type_key, header_title)
            return

        canonical_value: str | None = None
        cls: type | None = None
        if case_insensitive:
            matched = {k.lower(): (k, v) for k, v in items.items()}.get(type_name.lower())
            if matched is not None:
                canonical_value, cls = matched
        else:
            if type_name in items:
                canonical_value = type_name
                cls = items[type_name]

        if canonical_value is None or cls is None:
            available = ", ".join(sorted(items.keys()))
            typer.echo(f"Error: Unknown {type_key} '{type_name}'", err=True)
            typer.echo(f"Available types: {available}", err=True)
            raise typer.Exit(code=1)

        self._emit_import_hint(_CONFIG_IMPORT, f"dd.{cls.__name__}")
        typer.echo(format_model_text(cls, type_key=type_key, type_value=canonical_value))
        if related_inspect_tip:
            typer.echo("")
            typer.echo(related_inspect_tip)

    def _show_all_typed(
        self,
        items: dict[str, type],
        type_key: str,
        header_title: str,
    ) -> None:
        """Show all types for a typed command."""
        self._emit_import_hint(_CONFIG_IMPORT, "dd.<ClassName>")
        sorted_types = sorted(items.keys())

        seen_schemas: set[str] = set()
        lines = [f"# {header_title}", f"# {len(sorted_types)} types discovered from data_designer.config", ""]
        for type_value in sorted_types:
            cls = items[type_value]
            lines.append(format_model_text(cls, type_key=type_key, type_value=type_value, seen_schemas=seen_schemas))
            lines.append("")
        typer.echo("\n".join(lines))

    def _show_all_schemas(self, items: dict[str, type], header_title: str) -> None:
        """Show all schemas for simple discovery commands (e.g. constraints)."""
        seen_schemas: set[str] = set()
        lines = [f"# {header_title}", f"# {len(items)} types", ""]
        for name in sorted(items.keys()):
            cls = items[name]
            if hasattr(cls, "model_fields"):
                lines.append(format_model_text(cls, seen_schemas=seen_schemas))
            else:
                lines.append(f"{cls.__name__}:")
                if cls.__doc__:
                    lines.append(f"  description: {cls.__doc__.strip().split(chr(10))[0]}")
                if hasattr(cls, "__members__"):
                    members = [str(m.value) for m in cls]
                    lines.append(f"  values: [{', '.join(members)}]")
            lines.append("")
        typer.echo("\n".join(lines))
