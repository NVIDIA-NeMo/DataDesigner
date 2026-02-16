# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.cli.services.introspection.method_inspector import MethodInfo, ParamInfo
from data_designer.cli.services.introspection.pydantic_inspector import FieldDetail, ModelSchema

_AGENT_GUIDANCE_FOOTER = (
    "Use `data-designer types <subcommand>` to explore configuration types.\n"
    "Use `data-designer reference <subcommand>` for builder, imports, and overview.\n"
    "Only read source files directly if these commands don't cover your need."
)


def _format_field_text(field: FieldDetail, indent: int = 4, seen_schemas: set[str] | None = None) -> list[str]:
    """Format a single field as YAML-style text lines, recursing into nested schemas.

    When ``seen_schemas`` is provided, nested schemas that have already been rendered
    are replaced with a short back-reference to reduce output duplication.
    """
    pad = " " * indent
    lines: list[str] = []
    header = f"{pad}{field.name}: {field.type_str}"
    if field.default_factory:
        header += f" = {field.default_factory}()"
    elif field.has_literal_default():
        header += f" = {field.default_json!r}"
    elif field.default:
        header += f" = {field.default}"
    if field.required:
        header += "  [required]"
    lines.append(header)
    if field.description:
        lines.append(f"{pad}  description: {field.description}")
    if field.enum_values:
        lines.append(f"{pad}  values: [{', '.join(field.enum_values)}]")
    if field.constraints:
        constraint_parts = [f"{k}={v}" for k, v in field.constraints.items()]
        lines.append(f"{pad}  constraints: {', '.join(constraint_parts)}")
    if field.nested_schema:
        schema_name = field.nested_schema.class_name
        if seen_schemas is not None and schema_name in seen_schemas:
            lines.append(f"{pad}  schema: (see {schema_name} above)")
        else:
            if seen_schemas is not None:
                seen_schemas.add(schema_name)
            lines.append(f"{pad}  schema ({schema_name}):")
            for nested_field in field.nested_schema.fields:
                lines.extend(_format_field_text(nested_field, indent=indent + 4, seen_schemas=seen_schemas))
    return lines


def format_model_schema_text(schema: ModelSchema, indent: int = 0, seen_schemas: set[str] | None = None) -> str:
    """Format a ModelSchema as YAML-style text for backward compatibility with the existing skill scripts.

    When ``seen_schemas`` is provided, nested schemas that have already been rendered
    across prior calls are replaced with a short back-reference.
    """
    lines: list[str] = []
    pad = " " * indent
    lines.append(f"{pad}{schema.class_name}:")
    if schema.type_key and schema.type_value:
        lines.append(f"{pad}  {schema.type_key}: {schema.type_value}")
    lines.append(f"{pad}  description: {schema.description}")
    lines.append(f"{pad}  fields:")
    for field in schema.fields:
        lines.extend(_format_field_text(field, indent=indent + 4, seen_schemas=seen_schemas))
    return "\n".join(lines)


def _format_field_json(field: FieldDetail) -> dict:
    """Convert a FieldDetail to a JSON-serializable dict, recursing into nested schemas.

    Emits machine-typed defaults: "default" (native JSON value, including null) when
    the field has a literal default, and "default_factory" (string) when it uses a factory.
    """
    result: dict = {
        "name": field.name,
        "type": field.type_str,
        "required": field.required,
    }
    if field.default_factory:
        result["default_factory"] = field.default_factory
    elif field.has_literal_default():
        result["default"] = field.default_json
    elif field.default is not None:
        result["default"] = field.default
    if field.description:
        result["description"] = field.description
    if field.enum_values:
        result["values"] = field.enum_values
    if field.constraints:
        result["constraints"] = field.constraints
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
    lines.append("  data-designer types columns")
    lines.append("  data-designer types columns all")
    lines.append("  data-designer types columns llm-text")
    lines.append("  data-designer types samplers category")
    lines.append("  data-designer reference builder")
    lines.append("  data-designer reference interface")
    lines.append("  data-designer reference imports")

    return "\n".join(lines)


def _short_sig(method: MethodInfo) -> str:
    """Create a compact signature like 'add_column(...)' for overview display."""
    return f"{method.name}(...)"


# ---------------------------------------------------------------------------
# Namespace / code-structure formatters
# ---------------------------------------------------------------------------


def _render_tree_lines(node: dict[str, Any], prefix: str = "", is_last: bool = True) -> list[str]:
    """Recursively render a namespace tree node into box-drawing lines."""
    connector = "└── " if is_last else "├── "
    suffix = "/" if node["is_package"] else ".py"
    lines: list[str] = [f"{prefix}{connector}{node['name']}{suffix}"]

    children = node.get("children", [])
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        lines.extend(_render_tree_lines(child, child_prefix, is_last=(i == len(children) - 1)))
    return lines


def format_namespace_text(data: dict[str, Any]) -> str:
    """Format a namespace tree as a text tree diagram with box-drawing characters."""
    lines: list[str] = []
    lines.append("data_designer code structure")
    lines.append("=" * 28)
    lines.append("")

    paths = data.get("paths", [])
    if paths:
        lines.append("Install path:")
        for p in paths:
            lines.append(f"  {p}")
        lines.append("")

    tree = data["tree"]
    lines.append(f"{tree['name']}/")
    children = tree.get("children", [])
    for i, child in enumerate(children):
        lines.extend(_render_tree_lines(child, prefix="", is_last=(i == len(children) - 1)))

    import_errors = data.get("import_errors", [])
    if import_errors:
        lines.append("")
        lines.append("Warnings (submodules that could not be imported):")
        for err in import_errors:
            lines.append(f"  {err.get('module', '?')}: {err.get('message', '')}")
        lines.append("")

    lines.append("")
    lines.append(_AGENT_GUIDANCE_FOOTER)
    return "\n".join(lines)


def format_namespace_json(data: dict[str, Any]) -> dict[str, Any]:
    """Return the namespace tree dict as-is for JSON output.

    When discovery collected import_errors, they are included under "import_errors".
    """
    return data


# ---------------------------------------------------------------------------
# Interface formatters
# ---------------------------------------------------------------------------


def format_interface_text(
    classes_with_methods: list[tuple[str, list[MethodInfo]]],
    pydantic_schemas: list[ModelSchema],
) -> str:
    """Format interface classes as readable text for agent consumption."""
    lines: list[str] = []
    lines.append("Data Designer Interface Reference")
    lines.append("=" * 34)
    lines.append("")

    for class_name, methods in classes_with_methods:
        lines.append(format_method_info_text(methods, class_name=class_name))
        lines.append("")

    for schema in pydantic_schemas:
        lines.append(format_model_schema_text(schema))
        lines.append("")

    return "\n".join(lines).rstrip()


def format_interface_json(
    classes_with_methods: list[tuple[str, list[MethodInfo]]],
    pydantic_schemas: list[ModelSchema],
) -> dict[str, Any]:
    """Convert interface classes to a JSON-serializable dict."""
    methods_dict: dict[str, list[dict]] = {}
    for class_name, methods in classes_with_methods:
        methods_dict[class_name] = format_method_info_json(methods)

    schemas_list: list[dict] = [format_model_schema_json(s) for s in pydantic_schemas]

    return {"methods": methods_dict, "schemas": schemas_list}


# ---------------------------------------------------------------------------
# Imports formatters
# ---------------------------------------------------------------------------


_CONFIG_MODULE = "data_designer.config"
_INTERFACE_MODULE = "data_designer.interface"
_CONFIG_ALIAS = "dd"

_RECOMMENDED_IMPORTS = [
    f"import {_CONFIG_MODULE} as {_CONFIG_ALIAS}",
    f"from {_INTERFACE_MODULE} import DataDesigner",
]


def format_imports_text(categories: dict[str, list[dict[str, str]]]) -> str:
    """Format categorized import names as readable text with access patterns."""
    lines: list[str] = []
    lines.append("Data Designer Import Reference")
    lines.append("=" * 30)
    lines.append("")

    lines.append("Recommended imports:")
    for imp in _RECOMMENDED_IMPORTS:
        lines.append(f"  {imp}")
    lines.append("")

    for category, entries in sorted(categories.items()):
        count = len(entries)
        noun = "name" if count == 1 else "names"
        lines.append(f"{category} ({count} {noun}):")

        is_config = any(e["module"] == _CONFIG_MODULE for e in entries)
        if is_config:
            for entry in sorted(entries, key=lambda e: e["name"]):
                lines.append(f"  {_CONFIG_ALIAS}.{entry['name']}")
        else:
            sorted_names = sorted(e["name"] for e in entries)
            if len(sorted_names) <= 3:
                names_str = ", ".join(sorted_names)
                lines.append(f"  from {entries[0]['module']} import {names_str}")
            else:
                module = entries[0]["module"]
                lines.append(f"  from {module} import (")
                for name in sorted_names:
                    lines.append(f"      {name},")
                lines.append("  )")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_imports_json(categories: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Return a structured JSON with recommended imports, alias, and categorized names."""
    structured: dict[str, Any] = {
        "recommended_imports": _RECOMMENDED_IMPORTS,
        "config_alias": _CONFIG_ALIAS,
        "categories": {},
    }
    for category, entries in sorted(categories.items()):
        module = entries[0]["module"] if entries else _CONFIG_MODULE
        structured["categories"][category] = {
            "module": module,
            "access_pattern": f"{_CONFIG_ALIAS}.<name>" if module == _CONFIG_MODULE else f"from {module} import <name>",
            "names": sorted(e["name"] for e in entries),
        }
    return structured
