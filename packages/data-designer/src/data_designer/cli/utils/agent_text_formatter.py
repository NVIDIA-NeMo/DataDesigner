# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def format_context_text(data: dict[str, Any]) -> str:
    """Format the full context payload as sectioned text with tables."""
    sections = [
        f"Data Designer v{data['library_version']}",
        "",
        "## Config Module",
        "",
        "The config module contains all user-facing configuration types. Do not search other modules in the library.",
        f"config_root: {data['config_module_path']}",
        "",
        f"builder: {{config_root}}/{_strip_config_prefix(data['config_builder_file'])}",
        f"base: {{config_root}}/{_strip_config_prefix(data['base_config_file'])} (read for inherited fields shared by columns and processors)",
        "All config types are accessible via: import data_designer.config as dd",
        "",
        "## Types",
        "",
        format_types_text({"families": data["families"], "items": data["types"]}),
        "",
        "## Model Aliases",
        "",
        _format_model_aliases_context(data["state"]["model_aliases"]),
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
    columns = ["type", "description"]
    preamble = ""
    if "config_module_path" in data:
        preamble = f"config_root: {data['config_module_path']}\n\n"

    if "families" in data:
        lines: list[str] = []
        for family_info in data["families"]:
            lines.append(_format_family_header(family_info))
            lines.append(_format_table(data["items"][family_info["family"]], columns))
            lines.append("")
        return preamble + "\n".join(lines).rstrip()

    lines = [_format_family_header(data)]
    lines.append(_format_table(data["items"], columns))
    return preamble + "\n".join(lines)


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
    """Format a family header block with name and config file(s)."""
    name = info.get("family", "")
    lines = [f"### {name}"]
    for path in info.get("files", []):
        lines.append(f"file: {{config_root}}/{_strip_config_prefix(path)}")
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


def _format_model_aliases_context(state: dict[str, Any]) -> str:
    """Format model aliases for the context command, showing only usable aliases."""
    usable = [i for i in state.get("items", []) if i.get("usable")]
    if not usable:
        return (
            "No usable model aliases. Tell the user the issue and that they need to configure models"
            " -- for example, using `data-designer config models` and `data-designer config providers`."
        )
    lines: list[str] = [f"default_provider: {state.get('default_provider') or '(none)'}", ""]
    lines.append(
        _format_table(
            usable,
            ["model_alias", "model", "generation_type", "effective_provider"],
            column_labels={"effective_provider": "provider"},
        )
    )
    return "\n".join(lines)


def _strip_config_prefix(path: str) -> str:
    """Strip the ``data_designer/config/`` prefix so paths are relative to root."""
    prefix = "data_designer/config/"
    return path[len(prefix) :] if path.startswith(prefix) else path


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
