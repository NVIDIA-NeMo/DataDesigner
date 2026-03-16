# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


@dataclass(frozen=True, slots=True)
class PydanticFieldView:
    name: str
    type_text: str
    required: bool
    default_text: str | None
    description: str | None
    enum_values: tuple[str, ...] = ()
    nested_models: tuple[PydanticModelView, ...] = ()


@dataclass(frozen=True, slots=True)
class PydanticModelView:
    class_name: str
    summary: str | None
    fields: tuple[PydanticFieldView, ...]
    required_field_names: tuple[str, ...]


def describe_pydantic_model(
    pydantic_model: type[BaseModel],
    *,
    hidden_fields: set[str] | frozenset[str] | None = None,
    max_depth: int = 1,
) -> PydanticModelView:
    """Describe a Pydantic model as structured metadata for downstream renderers."""
    return _describe_model(
        pydantic_model,
        hidden_fields=frozenset(hidden_fields or ()),
        depth=0,
        max_depth=max_depth,
    )


def _describe_model(
    pydantic_model: type[BaseModel],
    *,
    hidden_fields: frozenset[str],
    depth: int,
    max_depth: int,
) -> PydanticModelView:
    summary = _extract_first_paragraph(pydantic_model.__doc__) if pydantic_model.__doc__ else None
    fields: list[PydanticFieldView] = []
    required_field_names: list[str] = []
    expanded_models: set[type[BaseModel]] = set()

    for name, info in pydantic_model.model_fields.items():
        if name in hidden_fields:
            continue

        annotation = info.annotation
        required = info.is_required()
        if required:
            required_field_names.append(name)

        enum_values: list[str] = []
        nested_models: list[PydanticModelView] = []
        for leaf in _find_expandable_leaves(annotation):
            if issubclass(leaf, Enum):
                enum_values.extend(str(member.value) for member in leaf)
            elif depth < max_depth and leaf not in expanded_models:
                expanded_models.add(leaf)
                nested_models.append(
                    _describe_model(leaf, hidden_fields=frozenset(), depth=depth + 1, max_depth=max_depth)
                )

        fields.append(
            PydanticFieldView(
                name=name,
                type_text=_format_type(annotation),
                required=required,
                default_text=None if required else _format_default(info),
                description=info.description,
                enum_values=tuple(enum_values),
                nested_models=tuple(nested_models),
            )
        )

    return PydanticModelView(
        class_name=pydantic_model.__name__,
        summary=summary,
        fields=tuple(fields),
        required_field_names=tuple(required_field_names),
    )


# --- Field helpers ---


def _format_type(annotation: Any) -> str:
    """Convert a type annotation to a readable string."""
    if hasattr(annotation, "__metadata__"):
        return _format_type(get_args(annotation)[0])
    if annotation is types.NoneType:
        return "None"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        values = ", ".join(repr(arg.value) if isinstance(arg, Enum) else repr(arg) for arg in args)
        return f"Literal[{values}]"
    if origin is None:
        return annotation.__name__ if hasattr(annotation, "__name__") else strip_module_paths(str(annotation))
    if not args:
        return strip_module_paths(str(annotation))
    if origin is Union or origin is types.UnionType:
        return " | ".join(_format_type(arg) for arg in args)
    origin_name = origin.__name__ if hasattr(origin, "__name__") else str(origin)
    return f"{origin_name}[{', '.join(_format_type(arg) for arg in args)}]"


def _format_default(info: FieldInfo) -> str:
    if info.default_factory is not None:
        return f"{getattr(info.default_factory, '__name__', repr(info.default_factory))}()"
    default = info.default.value if isinstance(info.default, Enum) else info.default
    return repr(default)


def _find_expandable_leaves(annotation: Any) -> list[type]:
    """Return Enum and nested Pydantic model leaves for supported annotation shapes."""
    origin = get_origin(annotation)

    if origin is None and isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return [annotation]
        if issubclass(annotation, BaseModel) and annotation is not BaseModel:
            return [annotation]
        return []

    args = get_args(annotation)
    if not args:
        return []

    if origin is Union or origin is types.UnionType:
        non_none = [arg for arg in args if arg is not types.NoneType]
        return _find_expandable_leaves(non_none[0]) if len(non_none) == 1 else []

    if origin is list:
        return _find_expandable_leaves(args[0])

    return []


# --- Docstring helpers ---


def _extract_first_paragraph(docstring: str) -> str | None:
    lines: list[str] = []
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped.lower() in _SECTION_HEADERS:
            break
        if not stripped and lines:
            break
        if stripped:
            lines.append(stripped)
    return " ".join(lines) if lines else None


def strip_module_paths(text: str) -> str:
    """Remove module-path prefixes from a type string (e.g. 'foo.bar.Baz' -> 'Baz')."""
    return _MODULE_PATH_RE.sub(lambda match: match.group().rsplit(".", 1)[-1], text)


_MODULE_PATH_RE = re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+")

_SECTION_HEADERS = frozenset(
    {
        "args:",
        "arguments:",
        "attributes:",
        "example:",
        "examples:",
        "keyword args:",
        "keyword arguments:",
        "note:",
        "notes:",
        "raises:",
        "references:",
        "returns:",
        "see also:",
        "todo:",
        "warns:",
        "yields:",
    }
)
