# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from enum import Enum

import typer

from data_designer.cli.services.introspection.discovery import (
    discover_column_configs,
    discover_constraint_types,
    discover_importable_names,
    discover_interface_classes,
    discover_mcp_types,
    discover_model_configs,
    discover_namespace_tree,
    discover_processor_configs,
    discover_sampler_types,
    discover_seed_types,
    discover_validator_types,
)
from data_designer.cli.services.introspection.formatters import (
    format_imports_json,
    format_imports_text,
    format_interface_json,
    format_interface_text,
    format_method_info_json,
    format_method_info_text,
    format_model_schema_json,
    format_model_schema_text,
    format_namespace_json,
    format_namespace_text,
    format_overview_text,
    format_type_list_text,
)
from data_designer.cli.services.introspection.method_inspector import inspect_class_methods
from data_designer.cli.services.introspection.pydantic_inspector import build_model_schema
from data_designer.config.base import ConfigBase
from data_designer.config.config_builder import DataDesignerConfigBuilder


class OutputFormat(str, Enum):
    """Supported output formats for introspect commands."""

    TEXT = "text"
    JSON = "json"


class IntrospectionController:
    """Controller for introspect CLI commands.

    Orchestrates discovery, inspection, formatting, and output for all
    introspect subcommands.
    """

    def __init__(self, output_format: str = "text") -> None:
        self._format = output_format

    def show_columns(self, type_name: str | None) -> None:
        """Show column configuration types."""
        items = discover_column_configs()
        self._show_typed_items(
            items=items,
            type_name=type_name,
            type_key="column_type",
            type_label="column_type",
            class_label="config_class",
            header_title="Data Designer Column Types Reference",
        )

    def show_samplers(self, type_name: str | None) -> None:
        """Show sampler types and their param classes."""
        items = discover_sampler_types()
        self._show_typed_items(
            items=items,
            type_name=type_name,
            type_key="sampler_type",
            type_label="sampler_type",
            class_label="params_class",
            header_title="Data Designer Sampler Types Reference",
            case_insensitive=True,
            uppercase_value=True,
        )

    def show_validators(self, type_name: str | None) -> None:
        """Show validator types and their param classes."""
        items = discover_validator_types()
        self._show_typed_items(
            items=items,
            type_name=type_name,
            type_key="validator_type",
            type_label="validator_type",
            class_label="params_class",
            header_title="Data Designer Validator Types Reference",
            case_insensitive=True,
            uppercase_value=True,
        )

    def show_processors(self, type_name: str | None) -> None:
        """Show processor types and their config classes."""
        items = discover_processor_configs()
        self._show_typed_items(
            items=items,
            type_name=type_name,
            type_key="processor_type",
            type_label="processor_type",
            class_label="config_class",
            header_title="Data Designer Processor Types Reference",
            case_insensitive=True,
        )

    def show_models(self) -> None:
        """Show model configuration types."""
        items = discover_model_configs()
        self._show_all_schemas(items, "Data Designer Model Configuration Reference")

    def show_builder(self) -> None:
        """Show DataDesignerConfigBuilder method signatures and docs."""
        methods = inspect_class_methods(DataDesignerConfigBuilder)
        if self._format == "json":
            typer.echo(json.dumps(format_method_info_json(methods), indent=2))
        else:
            typer.echo(format_method_info_text(methods, class_name="DataDesignerConfigBuilder"))

    def show_constraints(self) -> None:
        """Show constraint types."""
        items = discover_constraint_types()
        self._show_all_schemas(items, "Data Designer Constraint Types Reference")

    def show_seeds(self) -> None:
        """Show seed dataset types."""
        items = discover_seed_types()
        self._show_all_schemas(items, "Data Designer Seed Dataset Types Reference")

    def show_mcp(self) -> None:
        """Show MCP provider types."""
        items = discover_mcp_types()
        self._show_all_schemas(items, "Data Designer MCP Types Reference")

    def show_interface(self) -> None:
        """Show DataDesigner, result types, and RunConfig."""
        classes = discover_interface_classes()

        classes_with_methods: list[tuple[str, list]] = []
        pydantic_schemas = []
        for name, cls in classes.items():
            if isinstance(cls, type) and issubclass(cls, ConfigBase):
                pydantic_schemas.append(build_model_schema(cls))
            else:
                methods = inspect_class_methods(cls)
                classes_with_methods.append((name, methods))

        if self._format == "json":
            typer.echo(json.dumps(format_interface_json(classes_with_methods, pydantic_schemas), indent=2))
        else:
            typer.echo(format_interface_text(classes_with_methods, pydantic_schemas))

    def show_imports(self, category: str | None = None) -> None:
        """Show categorized import reference for data_designer.config and data_designer.interface."""
        categories = discover_importable_names()

        if category is not None:
            matched_key = self._match_category(category, list(categories.keys()))
            if matched_key is None:
                available = ", ".join(sorted(categories.keys()))
                typer.echo(f"Error: No category matching '{category}'.", err=True)
                typer.echo(f"Available categories: {available}", err=True)
                raise typer.Exit(code=1)
            categories = {matched_key: categories[matched_key]}

        if self._format == "json":
            typer.echo(json.dumps(format_imports_json(categories), indent=2))
        else:
            typer.echo(format_imports_text(categories))

    def show_code_structure(self, depth: int = 2) -> None:
        """Show the data_designer package structure and install paths."""
        if depth < 0:
            typer.echo("Error: --depth must be >= 0.", err=True)
            raise typer.Exit(code=1)
        data = discover_namespace_tree(max_depth=depth)
        if self._format == "json":
            typer.echo(json.dumps(format_namespace_json(data), indent=2))
        else:
            typer.echo(format_namespace_text(data))

    def show_overview(self) -> None:
        """Show compact API overview cheatsheet."""
        type_counts = {
            "Column types": len(discover_column_configs()),
            "Sampler types": len(discover_sampler_types()),
            "Validator types": len(discover_validator_types()),
            "Processor types": len(discover_processor_configs()),
            "Model configs": len(discover_model_configs()),
            "Constraint types": len(discover_constraint_types()),
            "Seed types": len(discover_seed_types()),
            "MCP types": len(discover_mcp_types()),
        }

        builder_methods = inspect_class_methods(DataDesignerConfigBuilder)

        if self._format == "json":
            typer.echo(
                json.dumps(
                    {"type_counts": type_counts, "builder_methods": format_method_info_json(builder_methods)}, indent=2
                )
            )
        else:
            typer.echo(format_overview_text(type_counts, builder_methods))

    @staticmethod
    def _match_category(query: str, keys: list[str]) -> str | None:
        """Match a user query to a category key using progressive fuzzy matching.

        Tries: exact match, first-word stem match, any-word stem match, substring match.
        """
        normalized = query.lower().rstrip("s")

        # Exact match (case-insensitive)
        for key in keys:
            if key.lower() == query.lower():
                return key

        # First-word stem match
        for key in keys:
            first_word = key.lower().split()[0].rstrip("s")
            if first_word == normalized:
                return key

        # Any-word stem match
        for key in keys:
            words = key.lower().split()
            for word in words:
                if word.rstrip("s") == normalized:
                    return key

        # Substring match (earliest position wins)
        best_key: str | None = None
        best_pos = float("inf")
        for key in keys:
            pos = key.lower().find(query.lower())
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_key = key

        return best_key

    def _show_typed_items(
        self,
        items: dict[str, type],
        type_name: str | None,
        type_key: str,
        type_label: str,
        class_label: str,
        header_title: str,
        case_insensitive: bool = False,
        uppercase_value: bool = False,
    ) -> None:
        """Shared logic for type-based commands (columns, samplers, validators, processors)."""
        if type_name is None:
            if self._format == "json":
                typer.echo(json.dumps({k: v.__name__ for k, v in sorted(items.items())}, indent=2))
            else:
                typer.echo(format_type_list_text(items, type_label, class_label))
            return

        if type_name == "all":
            self._show_all_typed(items, type_key, header_title, uppercase_value)
            return

        lookup = type_name.lower() if case_insensitive else type_name
        if lookup not in items:
            available = ", ".join(sorted(items.keys()))
            typer.echo(f"Error: Unknown {type_key} '{type_name}'", err=True)
            typer.echo(f"Available types: {available}", err=True)
            raise typer.Exit(code=1)

        cls = items[lookup]
        display_value = lookup.upper() if uppercase_value else lookup
        schema = build_model_schema(cls, type_key=type_key, type_value=display_value)

        if self._format == "json":
            typer.echo(json.dumps(format_model_schema_json(schema), indent=2))
        else:
            typer.echo(format_model_schema_text(schema))

    def _show_all_typed(
        self,
        items: dict[str, type],
        type_key: str,
        header_title: str,
        uppercase_value: bool = False,
    ) -> None:
        """Show all types for a typed command."""
        sorted_types = sorted(items.keys())

        if self._format == "json":
            all_schemas = []
            for type_value in sorted_types:
                cls = items[type_value]
                display_value = type_value.upper() if uppercase_value else type_value
                schema = build_model_schema(cls, type_key=type_key, type_value=display_value)
                all_schemas.append(format_model_schema_json(schema))
            typer.echo(json.dumps(all_schemas, indent=2))
        else:
            seen_schemas: set[str] = set()
            lines = [f"# {header_title}", f"# {len(sorted_types)} types discovered from data_designer.config", ""]
            for type_value in sorted_types:
                cls = items[type_value]
                display_value = type_value.upper() if uppercase_value else type_value
                schema = build_model_schema(cls, type_key=type_key, type_value=display_value)
                lines.append(format_model_schema_text(schema, seen_schemas=seen_schemas))
                lines.append("")
            typer.echo("\n".join(lines))

    def _show_all_schemas(self, items: dict[str, type], header_title: str) -> None:
        """Show all schemas for simple discovery commands (models, constraints, seeds, mcp)."""
        if self._format == "json":
            all_schemas = []
            for name in sorted(items.keys()):
                cls = items[name]
                if hasattr(cls, "model_fields"):
                    schema = build_model_schema(cls)
                    all_schemas.append(format_model_schema_json(schema))
                else:
                    entry: dict = {
                        "class_name": cls.__name__,
                        "description": (cls.__doc__ or "").strip().split("\n")[0],
                    }
                    if hasattr(cls, "__members__"):
                        entry["values"] = [str(member.value) for member in cls]
                    all_schemas.append(entry)
            typer.echo(json.dumps(all_schemas, indent=2))
        else:
            seen_schemas: set[str] = set()
            lines = [f"# {header_title}", f"# {len(items)} types", ""]
            for name in sorted(items.keys()):
                cls = items[name]
                if hasattr(cls, "model_fields"):
                    schema = build_model_schema(cls)
                    lines.append(format_model_schema_text(schema, seen_schemas=seen_schemas))
                else:
                    lines.append(f"{cls.__name__}:")
                    if cls.__doc__:
                        lines.append(f"  description: {cls.__doc__.strip().split(chr(10))[0]}")
                    if hasattr(cls, "__members__"):
                        members = [m.name for m in cls]
                        lines.append(f"  values: [{', '.join(members)}]")
                lines.append("")
            typer.echo("\n".join(lines))
