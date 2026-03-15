# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# IMPORTANT: This module must NOT import from any data_designer submodules (i.e., data_designer.*).
# It supports the foundational base.py abstractions and should only depend on pydantic and Python builtins.

from __future__ import annotations

import re
import types
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo


# --- Public API ---


def generate_schema_text(cls: type[BaseModel], *, base_cls: type[BaseModel]) -> str:
    """Return an agent-friendly text summary of a model's fields, types, and defaults.

    Renders field names, annotations, defaults, descriptions, inline enum values,
    nested model expansion (1 level), and an instantiation example.
    """
    return _render_model(cls, base_cls=base_cls, depth=0)


# --- Rendering ---


def _render_model(cls: type[BaseModel], *, base_cls: type[BaseModel], depth: int) -> str:
    """Render all visible fields of a model class as indented text."""
    indent = "      " * depth
    lines: list[str] = [f"{indent}{cls.__name__}:"]
    docstring = _get_docstring_summary(cls.__doc__)
    if docstring:
        lines.append(f"{indent}  {docstring}")
    lines.append("")

    required_field_names: list[str] = []
    for name, field_info in cls.model_fields.items():
        if _is_discriminator_field(field_info) or field_info.repr is False:
            continue
        if field_info.is_required():
            required_field_names.append(name)
        lines.extend(_render_field(name, field_info, indent=indent, base_cls=base_cls, depth=depth))

    if depth == 0 and required_field_names:
        params = ", ".join(f"{n}=..." for n in required_field_names)
        lines.append(f"\n{indent}  Example: dd.{cls.__name__}({params})")

    return "\n".join(lines)


def _render_field(name: str, field_info: FieldInfo, *, indent: str, base_cls: type[BaseModel], depth: int) -> list[str]:
    """Render a single field: declaration line, description, and expandable type details."""
    lines: list[str] = []
    annotation = _format_annotation(field_info.annotation)

    if field_info.is_required():
        lines.append(f"{indent}  {name}: {annotation}  [required]")
    else:
        lines.append(f"{indent}  {name}: {annotation} = {_format_default(field_info)}")

    if field_info.description:
        lines.append(f"{indent}      {field_info.description}")

    for exp_type in _find_expandable_types(field_info.annotation, base_cls):
        if issubclass(exp_type, Enum):
            values = ", ".join(str(m.value) for m in exp_type)
            lines.append(f"{indent}      values: {values}")
        elif issubclass(exp_type, base_cls) and depth < 1:
            lines.append(_render_model(exp_type, base_cls=base_cls, depth=depth + 1))

    return lines


def _format_default(field_info: FieldInfo) -> str:
    """Format a field's default value for display."""
    if field_info.default_factory is not None:
        factory_name = getattr(field_info.default_factory, "__name__", repr(field_info.default_factory))
        return f"{factory_name}()"
    default = field_info.default
    if isinstance(default, Enum):
        default = default.value
    return repr(default)


# --- Private helpers ---


def _format_annotation(annotation: Any) -> str:
    """Convert a type annotation to a readable string, stripping module paths."""
    if get_origin(annotation) is Literal:
        args = get_args(annotation)
        if args:
            values = ", ".join(repr(a.value) if isinstance(a, Enum) else repr(a) for a in args)
            return f"Literal[{values}]"
    raw = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
    return _MODULE_PATH_RE.sub(lambda m: m.group().rsplit(".", 1)[-1], raw)


def _unwrap_annotation(annotation: Any) -> Any:
    """Strip Annotated[T, ...] wrappers, returning the inner type."""
    if hasattr(annotation, "__metadata__"):
        return get_args(annotation)[0]
    return annotation


def _is_discriminator_field(field_info: FieldInfo) -> bool:
    """Return True for single-value Literal fields whose default matches the Literal arg."""
    annotation = _unwrap_annotation(field_info.annotation)
    if get_origin(annotation) is not Literal:
        return False
    args = get_args(annotation)
    if len(args) != 1:
        return False
    arg = args[0].value if isinstance(args[0], Enum) else args[0]
    default = field_info.default
    if isinstance(default, Enum):
        default = default.value
    return arg == default


def _find_expandable_types(annotation: Any, base_cls: type[BaseModel]) -> list[type]:
    """Walk the annotation tree and return Enum / base_cls leaf types.

    Recurses into generics (list[X] -> X) and Optional[X],
    but stops at multi-member unions (e.g. discriminated unions).
    """
    annotation = _unwrap_annotation(annotation)

    if isinstance(annotation, type):
        if issubclass(annotation, (Enum, base_cls)):
            return [annotation]
        return []

    origin = get_origin(annotation)
    args = get_args(annotation)
    if not args:
        return []

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _find_expandable_types(non_none[0], base_cls)
        return []

    if origin in (list, set, frozenset, tuple):
        return [t for arg in args for t in _find_expandable_types(arg, base_cls)]

    if origin is dict and len(args) >= 2:
        return _find_expandable_types(args[1], base_cls)

    return []


def _get_docstring_summary(docstring: str | None) -> str | None:
    """Extract the first paragraph of a docstring, before any Google-style section header."""
    if not docstring:
        return None
    lines: list[str] = []
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped.lower() in _GOOGLE_SECTION_HEADERS:
            break
        if not stripped and lines:
            break
        if stripped:
            lines.append(stripped)
    return " ".join(lines) if lines else None


# --- Private constants ---

_MODULE_PATH_RE = re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+")

_GOOGLE_SECTION_HEADERS = frozenset(
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
