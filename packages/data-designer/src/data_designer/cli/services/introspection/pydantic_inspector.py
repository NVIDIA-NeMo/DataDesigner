# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import types
import typing
from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def _is_basemodel_subclass(cls: Any) -> bool:
    """Return True if cls is a concrete BaseModel subclass (not BaseModel itself)."""
    return isinstance(cls, type) and issubclass(cls, BaseModel) and cls is not BaseModel


def _is_enum_subclass(cls: Any) -> bool:
    """Return True if cls is an Enum subclass (not Enum itself)."""
    return isinstance(cls, type) and issubclass(cls, Enum) and cls is not Enum


def _extract_enum_class(annotation: Any) -> type | None:
    """Unwrap a type annotation to find an Enum class, if present.

    Handles X, X | None, Annotated[X, ...].
    Returns the Enum class or None.
    """
    if annotation is None:
        return None

    # Unwrap Annotated[X, ...]
    if get_origin(annotation) is typing.Annotated:
        annotation = get_args(annotation)[0]

    if _is_enum_subclass(annotation):
        return annotation

    origin = get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            if _is_enum_subclass(arg):
                return arg

    return None


def _extract_nested_basemodel(annotation: Any) -> type | None:
    """Unwrap a type annotation to find a single nested BaseModel subclass.

    Handles: X, list[X], X | None, list[X] | None, dict[K, V], Annotated[X, ...].
    Returns None for unions of 2+ BaseModel subclasses (discriminated unions),
    primitives, enums, or BaseModel itself.
    """
    if annotation is None:
        return None

    # Unwrap Annotated[X, ...]
    if get_origin(annotation) is typing.Annotated:
        annotation = get_args(annotation)[0]

    if _is_basemodel_subclass(annotation):
        return annotation

    origin = get_origin(annotation)

    # list[X] -> check X
    if origin is list:
        args = get_args(annotation)
        if args and _is_basemodel_subclass(args[0]):
            return args[0]
        return None

    # dict[K, V] -> check V
    if origin is dict:
        args = get_args(annotation)
        if len(args) >= 2 and _is_basemodel_subclass(args[1]):
            return args[1]
        return None

    # Union: X | None, list[X] | None, or discriminated unions
    if origin is typing.Union or origin is types.UnionType:
        non_none_args = [a for a in get_args(annotation) if a is not type(None)]
        basemodel_classes: list[type] = []
        for arg in non_none_args:
            result = _extract_nested_basemodel(arg)
            if result is not None:
                basemodel_classes.append(result)
            elif _is_basemodel_subclass(arg):
                basemodel_classes.append(arg)
        if len(basemodel_classes) == 1:
            return basemodel_classes[0]
        return None

    return None


def format_type(annotation: Any) -> str:
    """Format a type annotation for readable display.

    Strips module prefixes and simplifies complex types.
    """
    type_str = str(annotation)

    # Remove module prefixes
    type_str = re.sub(r"data_designer\.config\.\w+\.", "", type_str)
    type_str = re.sub(r"pydantic\.main\.", "", type_str)
    type_str = re.sub(r"typing\.", "", type_str)

    # Clean up enum members used inside Literal or other contexts: <EnumName.MEMBER: 'value'> -> 'value'
    type_str = re.sub(r"<\w+\.\w+: '([^']+)'>", r"'\1'", type_str)

    # Clean up enum types BEFORE other replacements: <enum 'EnumName'> -> EnumName
    type_str = re.sub(r"<enum '(\w+)'>", r"\1", type_str)

    # Clean up class types: <class 'str'> -> str
    type_str = re.sub(r"<class '(\w+)'>", r"\1", type_str)

    type_str = type_str.replace("NoneType", "None")

    if "Literal[" in type_str:
        match = re.search(r"Literal\[([^\]]+)\]", type_str)
        if match:
            type_str = f"Literal[{match.group(1)}]"

    # Clean up Annotated types with Discriminator (too verbose)
    if "Annotated[" in type_str and "Discriminator" in type_str:
        start = type_str.index("Annotated[") + len("Annotated[")
        depth = 0
        for i, ch in enumerate(type_str[start:], start):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "," and depth == 0:
                type_str = type_str[start:i].strip()
                break

    return type_str


def get_brief_description(cls: type) -> str:
    """Extract first line from class docstring."""
    if cls.__doc__:
        doc = cls.__doc__.strip()
        first_line = doc.split("\n")[0].strip()
        return first_line
    return "No description available."


def _extract_constraints(field_info: Any) -> dict[str, Any] | None:
    """Extract numeric/string constraints from a Pydantic FieldInfo's metadata."""
    constraint_keys = {"ge", "le", "gt", "lt", "min_length", "max_length"}
    constraints: dict[str, Any] = {}
    for meta in getattr(field_info, "metadata", []):
        for key in constraint_keys:
            val = getattr(meta, key, None)
            if val is not None:
                constraints[key] = val
    return constraints or None


def _default_to_json(value: Any) -> Any:
    """Convert a Pydantic default value to a JSON-serializable value.

    Returns the value unchanged if it is already JSON-serializable (bool, int, float,
    str, None, list, dict with JSON-serializable values). Enum members are converted
    to their .value. Other types are returned as a string representation for stability.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_default_to_json(item) for item in value]
    if isinstance(value, dict):
        return {k: _default_to_json(v) for k, v in value.items()}
    return repr(value)


def _format_field(
    field_name: str,
    field_info: Any,
    indent: int,
    seen_schemas: set[str] | None,
    seen_types: set[type],
    max_depth: int,
    depth: int,
) -> list[str]:
    """Format a single Pydantic field as YAML-style text lines, recursing into nested schemas."""
    pad = " " * indent
    lines: list[str] = []

    type_str = format_type(field_info.annotation)
    description: str = field_info.description or ""
    required: bool = field_info.is_required()

    header = f"{pad}{field_name}: {type_str}"
    if not required:
        if field_info.default_factory is not None:
            factory_name = getattr(field_info.default_factory, "__name__", repr(field_info.default_factory))
            header += f" = {factory_name}()"
        elif field_info.default is not PydanticUndefined:
            header += f" = {_default_to_json(field_info.default)!r}"
    if required:
        header += "  [required]"
    lines.append(header)

    if description:
        lines.append(f"{pad}  description: {description}")

    enum_cls = _extract_enum_class(field_info.annotation)
    if enum_cls is not None:
        enum_values = [str(member.value) for member in enum_cls]
        lines.append(f"{pad}  values: [{', '.join(enum_values)}]")

    constraints = _extract_constraints(field_info)
    if constraints:
        constraint_parts = [f"{k}={v}" for k, v in constraints.items()]
        lines.append(f"{pad}  constraints: {', '.join(constraint_parts)}")

    nested_cls = _extract_nested_basemodel(field_info.annotation)
    if nested_cls is not None and nested_cls not in seen_types and depth < max_depth:
        schema_key = f"{nested_cls.__module__}.{nested_cls.__qualname__}"
        schema_name = nested_cls.__name__
        if seen_schemas is not None and schema_key in seen_schemas:
            lines.append(f"{pad}  schema: (see {schema_name} above)")
        else:
            if seen_schemas is not None:
                seen_schemas.add(schema_key)
            lines.append(f"{pad}  schema ({schema_name}):")
            next_seen = seen_types | {nested_cls}
            nested_model_fields: dict[str, Any] = getattr(nested_cls, "model_fields", {})
            for nested_name, nested_info in nested_model_fields.items():
                lines.extend(
                    _format_field(
                        field_name=nested_name,
                        field_info=nested_info,
                        indent=indent + 4,
                        seen_schemas=seen_schemas,
                        seen_types=next_seen,
                        max_depth=max_depth,
                        depth=depth + 1,
                    )
                )

    return lines


def format_model_text(
    cls: type,
    type_key: str | None = None,
    type_value: str | None = None,
    indent: int = 0,
    seen_schemas: set[str] | None = None,
    max_depth: int = 3,
) -> str:
    """Format a Pydantic model as YAML-style text for agent context.

    Args:
        cls: The Pydantic model class to format.
        type_key: Optional discriminator key name (e.g., "column_type").
        type_value: Optional discriminator value (e.g., "llm-text").
        indent: Base indentation level.
        seen_schemas: Set of schema refs already rendered (mutated for cross-model dedup).
        max_depth: Maximum recursion depth for nested models.
    """
    return _format_model_text(
        cls,
        type_key=type_key,
        type_value=type_value,
        indent=indent,
        seen_schemas=seen_schemas,
        seen_types=set(),
        max_depth=max_depth,
        depth=0,
    )


def _format_model_text(
    cls: type,
    type_key: str | None,
    type_value: str | None,
    indent: int,
    seen_schemas: set[str] | None,
    seen_types: set[type],
    max_depth: int,
    depth: int,
) -> str:
    """Recursive implementation of format_model_text."""
    pad = " " * indent
    lines: list[str] = []
    lines.append(f"{pad}{cls.__name__}:")
    if type_key and type_value:
        lines.append(f"{pad}  {type_key}: {type_value}")
    lines.append(f"{pad}  description: {get_brief_description(cls)}")
    lines.append(f"{pad}  fields:")

    model_fields: dict[str, Any] = getattr(cls, "model_fields", {})
    for field_name, field_info in model_fields.items():
        lines.extend(
            _format_field(
                field_name=field_name,
                field_info=field_info,
                indent=indent + 4,
                seen_schemas=seen_schemas,
                seen_types=seen_types,
                max_depth=max_depth,
                depth=depth,
            )
        )

    return "\n".join(lines)
