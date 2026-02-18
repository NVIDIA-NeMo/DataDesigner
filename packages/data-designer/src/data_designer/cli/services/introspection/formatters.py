# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.method_inspector import MethodInfo, ParamInfo


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
