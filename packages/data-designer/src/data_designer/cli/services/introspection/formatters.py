# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.method_inspector import MethodInfo, ParamInfo

_MIN_CLASS_COL_WIDTH = 25


def _format_param_text(param: ParamInfo, indent: int) -> str:
    """Format a single method parameter as a text line."""
    pad = " " * indent
    line = f"{pad}{param.name}: {param.type_str}"
    if param.default is not None:
        line += f" = {param.default}"
    if param.description:
        line += f" \u2014 {param.description}"
    return line


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


def format_type_list_text(items: dict[str, type], type_label: str, class_label: str) -> str:
    """Format a summary table of type->class mappings, matching the existing print_list_table style."""
    sorted_items = sorted(items.items())
    if not sorted_items:
        return f"{type_label}  {class_label}\n(no items)"

    type_width = max(len(type_value) for type_value, _ in sorted_items)
    type_width = max(type_width, len(type_label))

    lines: list[str] = []
    lines.append(f"{type_label:<{type_width}}  {class_label}")
    lines.append(f"{'-' * type_width}  {'-' * max(len(class_label), _MIN_CLASS_COL_WIDTH)}")

    for type_value, cls in sorted_items:
        lines.append(f"{type_value:<{type_width}}  {cls.__name__}")

    return "\n".join(lines)
