# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.method_inspector import MethodInfo, ParamInfo
from data_designer.cli.services.introspection.pydantic_inspector import FieldDetail, ModelSchema


def _format_field_text(field: FieldDetail, indent: int = 4) -> list[str]:
    """Format a single field as YAML-style text lines, recursing into nested schemas."""
    pad = " " * indent
    lines: list[str] = []
    lines.append(f"{pad}{field.name}:")
    lines.append(f"{pad}  type: {field.type_str}")
    if field.description:
        lines.append(f"{pad}  description: {field.description}")
    if field.enum_values:
        lines.append(f"{pad}  values: [{', '.join(field.enum_values)}]")
    if field.nested_schema:
        lines.append(f"{pad}  schema ({field.nested_schema.class_name}):")
        for nested_field in field.nested_schema.fields:
            lines.extend(_format_field_text(nested_field, indent=indent + 4))
    return lines


def format_model_schema_text(schema: ModelSchema, indent: int = 0) -> str:
    """Format a ModelSchema as YAML-style text for backward compatibility with the existing skill scripts."""
    lines: list[str] = []
    pad = " " * indent
    lines.append(f"{pad}{schema.class_name}:")
    if schema.type_key and schema.type_value:
        lines.append(f"{pad}  {schema.type_key}: {schema.type_value}")
    lines.append(f"{pad}  description: {schema.description}")
    lines.append(f"{pad}  fields:")
    for field in schema.fields:
        lines.extend(_format_field_text(field, indent=indent + 4))
    return "\n".join(lines)


def _format_field_json(field: FieldDetail) -> dict:
    """Convert a FieldDetail to a JSON-serializable dict, recursing into nested schemas."""
    result: dict = {
        "name": field.name,
        "type": field.type_str,
    }
    if field.description:
        result["description"] = field.description
    if field.enum_values:
        result["values"] = field.enum_values
    if field.nested_schema:
        result["schema"] = format_model_schema_json(field.nested_schema)
    return result


def format_model_schema_json(schema: ModelSchema) -> dict:
    """Convert a ModelSchema to a JSON-serializable dict."""
    result: dict = {
        "class_name": schema.class_name,
        "description": schema.description,
    }
    if schema.type_key and schema.type_value:
        result[schema.type_key] = schema.type_value
    result["fields"] = [_format_field_json(f) for f in schema.fields]
    return result


def _format_param_text(param: ParamInfo, indent: int) -> str:
    """Format a single method parameter as a text line."""
    pad = " " * indent
    parts = [f"{pad}{param.name}: {param.type_str}"]
    if param.default is not None:
        parts[0] += f" = {param.default}"
    if param.description:
        parts[0] += f" \u2014 {param.description}"
    return parts[0]


def format_method_info_text(methods: list[MethodInfo], class_name: str | None = None) -> str:
    """Format a list of MethodInfo as readable text with signatures and parameter details."""
    lines: list[str] = []
    if class_name:
        lines.append(f"{class_name} Methods:")
        lines.append("")

    for method in methods:
        lines.append(f"  {method.signature}")
        if method.description:
            lines.append(f"    {method.description}")
        if method.parameters:
            lines.append("    Parameters:")
            for param in method.parameters:
                lines.append(_format_param_text(param, indent=6))
        lines.append("")

    return "\n".join(lines).rstrip()


def _param_to_json(param: ParamInfo) -> dict:
    """Convert a ParamInfo to a JSON-serializable dict."""
    result: dict = {
        "name": param.name,
        "type": param.type_str,
    }
    if param.default is not None:
        result["default"] = param.default
    if param.description:
        result["description"] = param.description
    return result


def format_method_info_json(methods: list[MethodInfo]) -> list[dict]:
    """Convert a list of MethodInfo to a JSON-serializable list of dicts."""
    result: list[dict] = []
    for method in methods:
        entry: dict = {
            "name": method.name,
            "signature": method.signature,
            "return_type": method.return_type,
        }
        if method.description:
            entry["description"] = method.description
        if method.parameters:
            entry["parameters"] = [_param_to_json(p) for p in method.parameters]
        result.append(entry)
    return result


def format_type_list_text(items: dict[str, type], type_label: str, class_label: str) -> str:
    """Format a summary table of type->class mappings, matching the existing print_list_table style."""
    sorted_items = sorted(items.items())
    if not sorted_items:
        return f"{type_label}  {class_label}\n(no items)"

    type_width = max(len(type_value) for type_value, _ in sorted_items)
    type_width = max(type_width, len(type_label))

    lines: list[str] = []
    lines.append(f"{type_label:<{type_width}}  {class_label}")
    lines.append(f"{'-' * type_width}  {'-' * max(len(class_label), 25)}")

    for type_value, cls in sorted_items:
        lines.append(f"{type_value:<{type_width}}  {cls.__name__}")

    return "\n".join(lines)


def format_overview_text(type_counts: dict[str, int], builder_methods: list[MethodInfo]) -> str:
    """Format a compact API overview cheatsheet."""
    lines: list[str] = []
    lines.append("Data Designer API Overview")
    lines.append("=" * 26)
    lines.append("")

    lines.append("Type Counts:")
    label_width = max(len(label) for label in type_counts) + 1 if type_counts else 10
    for label, count in type_counts.items():
        lines.append(f"  {label + ':':<{label_width}} {count:>3}")
    lines.append("")

    if builder_methods:
        lines.append("Builder Methods (DataDesignerConfigBuilder):")
        sig_width = max(len(_short_sig(m)) for m in builder_methods)
        for method in builder_methods:
            short = _short_sig(method)
            desc = method.description
            lines.append(f"  {short:<{sig_width}}  \u2014 {desc}")
        lines.append("")

    lines.append("Quick Start Commands:")
    lines.append("  data-designer agent-context columns --list")
    lines.append("  data-designer agent-context columns all")
    lines.append("  data-designer agent-context columns llm-text")
    lines.append("  data-designer agent-context samplers category")
    lines.append("  data-designer agent-context builder")

    return "\n".join(lines)


def _short_sig(method: MethodInfo) -> str:
    """Create a compact signature like 'add_column(...)' for overview display."""
    return f"{method.name}(...)"
