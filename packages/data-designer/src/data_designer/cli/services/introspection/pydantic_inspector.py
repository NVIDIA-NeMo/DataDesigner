# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import types
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

_UNDEFINED: Any = object()


@dataclass
class FieldDetail:
    """Structured representation of a single Pydantic model field."""

    name: str
    type_str: str
    description: str
    required: bool = True
    default: str | None = None
    default_json: Any = _UNDEFINED
    default_factory: str | None = None
    enum_values: list[str] | None = None
    constraints: dict[str, Any] | None = None
    nested_schema: ModelSchema | None = None

    def has_literal_default(self) -> bool:
        """True if this field has a literal default value (including None)."""
        return self.default_json is not _UNDEFINED


@dataclass
class ModelSchema:
    """Structured representation of a Pydantic model's schema."""

    class_name: str
    description: str
    schema_ref: str | None = None
    type_key: str | None = None
    type_value: str | None = None
    fields: list[FieldDetail] = field(default_factory=list)


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


def extract_nested_basemodel(annotation: Any) -> type | None:
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
            result = extract_nested_basemodel(arg)
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
        match = re.search(r"Annotated\[([^,]+(?:\s*\|\s*[^,]+)*),", type_str)
        if match:
            type_str = match.group(1).strip()

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


def get_field_info(cls: type) -> list[FieldDetail]:
    """Extract field information from a Pydantic model.

    Args:
        cls: The Pydantic model class to inspect.

    Returns:
        List of FieldDetail objects with name, type_str, description, required,
        default, enum_values, constraints, and nested_schema (initially None,
        populated by build_model_schema).
    """
    fields: list[FieldDetail] = []
    model_fields: dict[str, Any] = getattr(cls, "model_fields", {})
    if model_fields:
        for field_name, field_info in model_fields.items():
            type_str = format_type(field_info.annotation)
            description = field_info.description or ""

            required = field_info.is_required()

            default_json: Any = _UNDEFINED
            default_factory_name: str | None = None
            default_display: str | None = None
            if not required:
                if field_info.default_factory is not None:
                    default_factory_name = getattr(
                        field_info.default_factory, "__name__", repr(field_info.default_factory)
                    )
                elif field_info.default is not PydanticUndefined:
                    default_json = _default_to_json(field_info.default)
                    if default_json is not _UNDEFINED:
                        default_display = repr(default_json)

            enum_cls = _extract_enum_class(field_info.annotation)
            enum_values: list[str] | None = None
            if enum_cls is not None:
                enum_values = [str(member.value) for member in enum_cls]

            constraints = _extract_constraints(field_info)

            if default_display is None and default_factory_name is not None:
                default_display = f"{default_factory_name}()"

            fields.append(
                FieldDetail(
                    name=field_name,
                    type_str=type_str,
                    description=description,
                    required=required,
                    default=default_display,
                    default_json=default_json,
                    default_factory=default_factory_name,
                    enum_values=enum_values,
                    constraints=constraints,
                    nested_schema=None,
                )
            )
    return fields


def build_model_schema(
    cls: type,
    type_key: str | None = None,
    type_value: str | None = None,
    seen: set[type] | None = None,
    max_depth: int = 3,
    current_depth: int = 0,
) -> ModelSchema:
    """Build a structured ModelSchema from a Pydantic model class.

    Args:
        cls: The Pydantic model class to inspect.
        type_key: Optional key name for the type discriminator (e.g., "column_type").
        type_value: Optional value for the type discriminator (e.g., "llm-text").
        seen: Set of already-expanded class names to prevent cycles.
        max_depth: Maximum recursion depth for nested models.
        current_depth: Current recursion depth.

    Returns:
        A ModelSchema with recursively expanded nested schemas.
    """
    if seen is None:
        seen = set()

    class_name = cls.__name__
    description = get_brief_description(cls)
    schema_ref = f"{cls.__module__}.{cls.__qualname__}"
    fields = get_field_info(cls)

    model_fields_raw: dict[str, Any] = getattr(cls, "model_fields", {})
    for field_detail in fields:
        raw_field_info = model_fields_raw.get(field_detail.name)
        if raw_field_info is None:
            continue

        nested_cls = extract_nested_basemodel(raw_field_info.annotation)
        if nested_cls is not None and nested_cls not in seen and current_depth < max_depth:
            next_seen = seen | {nested_cls}
            field_detail.nested_schema = build_model_schema(
                nested_cls,
                seen=next_seen,
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )

    return ModelSchema(
        class_name=class_name,
        description=description,
        schema_ref=schema_ref,
        type_key=type_key,
        type_value=type_value,
        fields=fields,
    )
