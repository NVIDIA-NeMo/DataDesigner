# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.cli.utils.agent_introspection import get_library_version
from data_designer.cli.utils.pydantic_schema import PydanticModelView


def format_context_text(data: dict[str, Any]) -> str:
    """Format the full context payload as sectioned text with tables."""
    sections = [
        f"Data Designer v{get_library_version()}",
        "",
        "import data_designer.config as dd",
        "",
        'A "family" is a group of related config types that share a discriminator field.',
        "Use dd.<ClassName> to reference any type below.",
        "",
        "## Families",
        "",
        format_types_text({"families": data["families"], "items": data["types"]}),
        "",
        "## Model Aliases",
        "",
        format_model_aliases_text(data["state"]["model_aliases"]),
        "",
        "## Persona Datasets",
        "",
        format_persona_datasets_text(data["state"]["persona_datasets"]),
        "",
        "## Builder",
        "",
        format_builder_text(data["builder"]),
        "",
        "## Commands",
        "",
        _format_table(data["operations"], ["command_pattern", "description"]),
    ]
    return "\n".join(sections)


def format_types_text(data: dict[str, Any]) -> str:
    """Format type listings for one family or all families."""
    if "families" in data:
        lines: list[str] = [f"{f['family']}: {f['count']} types" for f in data["families"]]
        lines.append("")
        for family_name, items in data["items"].items():
            lines.append(_format_table(items, ["type_name", "class_name"], title=family_name))
            lines.append("")
        return "\n".join(lines).rstrip()
    return _format_table(
        data["items"],
        ["type_name", "class_name"],
        title=data.get("family"),
    )


def format_schema_text(data: dict[str, Any]) -> str:
    """Format schema data as human-readable field summaries."""
    if "items" in data:
        header = f"# {data['family']} schemas ({len(data['items'])} types)"
        schemas = "\n\n".join(_format_model_schema(item["schema_view"], item["import_path"]) for item in data["items"])
        return f"{header}\n\n{schemas}"
    return _format_model_schema(data["schema_view"], data["import_path"])


def format_builder_text(data: dict[str, Any]) -> str:
    """Format builder methods with signatures."""
    path = data["import_path"]
    hint = f"dd.{path.removeprefix('data_designer.config.')}" if path.startswith("data_designer.config.") else path
    lines: list[str] = [
        f"{data['class_name']}:",
        f"  usage: {hint}",
        "  methods:",
        "",
    ]
    for method in data["methods"]:
        lines.append(f"    {method['signature']}")
        if method.get("summary"):
            lines.append(f"      {method['summary']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_model_aliases_text(state: dict[str, Any]) -> str:
    """Format model aliases as a text table with provider summary."""
    lines: list[str] = [f"default_provider: {state.get('default_provider') or '(none)'}", ""]
    lines.append(
        _format_table(
            state.get("items", []),
            ["model_alias", "model", "generation_type", "effective_provider", "usable", "reason"],
            column_labels={"effective_provider": "provider"},
            title="model aliases",
        )
    )
    return "\n".join(lines)


def format_persona_datasets_text(state: dict[str, Any]) -> str:
    """Format persona datasets as a text table."""
    return _format_table(state.get("items", []), ["locale", "size", "installed"], title="persona datasets")


def _format_table(
    items: list[dict[str, Any]],
    columns: list[str],
    *,
    title: str | None = None,
    column_labels: dict[str, str] | None = None,
) -> str:
    labels = {col: (column_labels or {}).get(col, col) for col in columns}

    if not items:
        header = f"# {title}" if title else "# table"
        return f"{header}\n(no items)"

    col_widths = {col: max(len(labels[col]), max(len(_cell(row.get(col))) for row in items)) for col in columns}

    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("  ".join(f"{labels[col]:<{col_widths[col]}}" for col in columns))
    lines.append("  ".join("-" * col_widths[col] for col in columns))
    for row in items:
        lines.append("  ".join(f"{_cell(row.get(col)):<{col_widths[col]}}" for col in columns))

    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Convert a cell value to a string, rendering None as empty."""
    if value is None:
        return ""
    return str(value)


def _format_model_schema(pydantic_model: PydanticModelView, import_path: str, *, depth: int = 0) -> str:
    indent = "      " * depth
    lines: list[str] = [f"{indent}{pydantic_model.class_name}:"]
    if pydantic_model.summary:
        lines.append(f"{indent}  {pydantic_model.summary}")
    if depth == 0:
        lines.append(f"{indent}  import_path: {import_path}")
    lines.append("")

    for field in pydantic_model.fields:
        if field.required:
            lines.append(f"{indent}  {field.name}: {field.type_text}  [required]")
        else:
            lines.append(f"{indent}  {field.name}: {field.type_text} = {field.default_text}")

        if field.description:
            lines.append(f"{indent}      {field.description}")

        if field.enum_values:
            lines.append(f"{indent}      values: {', '.join(field.enum_values)}")

        for nested_model in field.nested_models:
            lines.append(_format_model_schema(nested_model, import_path, depth=depth + 1))

    return "\n".join(lines)
