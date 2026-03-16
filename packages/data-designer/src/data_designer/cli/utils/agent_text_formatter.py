# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def format_context_text(data: dict[str, Any]) -> str:
    """Format the full context payload as sectioned text with tables."""
    sections = [
        f"Data Designer v{data['library_version']}",
        f"config_package_path: {data['config_package_path']}",
        "",
        "Standard import for accessing config objects: import data_designer.config as dd",
        "",
        'A "family" is a group of related config types that share a discriminator field.',
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
        "## Commands",
        "",
        _format_table(data["operations"], ["command_pattern", "description"]),
    ]
    return "\n".join(sections)


def format_types_text(data: dict[str, Any]) -> str:
    """Format type listings for one family or all families."""
    columns = ["type_name", "description"]
    if "families" in data:
        lines: list[str] = [f"{f['family']}: {f['count']} types" for f in data["families"]]
        lines.append("")
        for family_info in data["families"]:
            lines.append(_format_family_header(family_info))
            lines.append(_format_table(data["items"][family_info["family"]], columns))
            lines.append("")
        return "\n".join(lines).rstrip()

    lines = [_format_family_header(data)]
    lines.append(_format_table(data["items"], columns))
    return "\n".join(lines)


def format_model_aliases_text(state: dict[str, Any]) -> str:
    """Format model aliases as a text table with provider summary."""
    lines: list[str] = [f"default_provider: {state.get('default_provider') or '(none)'}", ""]
    lines.append(
        _format_table(
            state.get("items", []),
            ["model_alias", "model", "generation_type", "effective_provider", "usable", "reason"],
            column_labels={"effective_provider": "provider"},
        )
    )
    return "\n".join(lines)


def format_persona_datasets_text(state: dict[str, Any]) -> str:
    """Format persona datasets as a text table."""
    return _format_table(state.get("items", []), ["locale", "size", "installed"])


def _format_family_header(info: dict[str, Any]) -> str:
    """Format a family header block with name and config_file."""
    name = info.get("family", "")
    lines = [f"### {name}"]
    if info.get("file"):
        lines.append(f"config_file: {info['file']}")
    lines.append("")
    return "\n".join(lines)


def _format_table(
    items: list[dict[str, Any]],
    columns: list[str],
    *,
    column_labels: dict[str, str] | None = None,
) -> str:
    labels = {col: (column_labels or {}).get(col, col) for col in columns}

    if not items:
        return "(no items)"

    col_widths = {col: max(len(labels[col]), max(len(_cell(row.get(col))) for row in items)) for col in columns}

    lines: list[str] = []
    lines.append("  ".join(f"{labels[col]:<{col_widths[col]}}" for col in columns))
    lines.append("  ".join("-" * col_widths[col] for col in columns))
    for row in items:
        lines.append("  ".join(f"{_cell(row.get(col)):<{col_widths[col]}}" for col in columns))

    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
