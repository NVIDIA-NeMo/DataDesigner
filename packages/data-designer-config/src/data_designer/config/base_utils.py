# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# IMPORTANT: This module must NOT import from any data_designer submodules (i.e., data_designer.*).
# It supports the foundational base.py abstractions and should only depend on pydantic and Python builtins.

from __future__ import annotations

import inspect
import re
import sys
import types
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo


# --- Public API ---


def generate_schema_text(cls: type[BaseModel], *, base_cls: type[BaseModel]) -> str:
    """Return an agent-friendly text summary of a model's fields, types, and defaults."""
    return _render_model(cls, base_cls=base_cls, depth=0)


# --- Rendering ---


def _render_model(cls: type[BaseModel], *, base_cls: type[BaseModel], depth: int) -> str:
    """Render all visible fields of a model class as indented text."""
    indent = "      " * depth
    lines: list[str] = [f"{indent}{cls.__name__}:"]

    if cls.__doc__:
        summary = _extract_first_paragraph(cls.__doc__)
        if summary:
            lines.append(f"{indent}  {summary}")
    lines.append("")

    required: list[str] = []

    for name, info in cls.model_fields.items():
        if _is_discriminator(info) or info.repr is False:
            continue

        # Pydantic's info.annotation is the primary source: it has full generic args
        # on Python 3.11+ and strips Annotated metadata for clean display.
        # On Python 3.10, generic args can be lost (e.g. list[Score] → bare list);
        # _recover_annotation falls back to the MRO only for those fields.
        ann = info.annotation
        ann_expand = _recover_annotation(cls, name, ann) if _has_degraded_generics(ann) else ann

        if info.is_required():
            required.append(name)
            lines.append(f"{indent}  {name}: {_format_type(ann)}  [required]")
        else:
            lines.append(f"{indent}  {name}: {_format_type(ann)} = {_format_default(info)}")

        if info.description:
            lines.append(f"{indent}      {info.description}")

        for leaf in _find_expandable_leaves(ann_expand, base_cls):
            if issubclass(leaf, Enum):
                lines.append(f"{indent}      values: {', '.join(str(m.value) for m in leaf)}")
            elif issubclass(leaf, base_cls) and depth < 1:
                lines.append(_render_model(leaf, base_cls=base_cls, depth=depth + 1))

    if depth == 0 and required:
        params = ", ".join(f"{n}=..." for n in required)
        lines.append(f"\n{indent}  Example: dd.{cls.__name__}({params})")

    return "\n".join(lines)


# --- Helpers ---


def _has_degraded_generics(annotation: Any) -> bool:
    """Return True when Pydantic has lost generic args (e.g. list[X] → bare list on Python 3.10)."""
    return isinstance(annotation, type) and annotation in _CONTAINER_TYPES


def _recover_annotation(cls: type, name: str, fallback: Any) -> Any:
    """Recover a single field's annotation from the MRO (Python 3.10 fallback).

    Walks the MRO from child to parent and returns the first resolved annotation
    for the given field name. Falls back to the original annotation if recovery fails.
    """
    for klass in cls.__mro__:
        if name not in vars(klass).get("__annotations__", {}):
            continue
        module = sys.modules.get(klass.__module__)
        globalns = getattr(module, "__dict__", {}) if module else {}
        localns = {klass.__name__: klass, **vars(klass)}
        try:
            hints = inspect.get_annotations(klass, globals=globalns, locals=localns, eval_str=True)
            if name in hints:
                return hints[name]
        except NameError:
            continue
    return fallback


def _is_discriminator(info: FieldInfo) -> bool:
    """Return True for single-value Literal fields whose default matches the Literal arg."""
    ann = _unwrap_annotated(info.annotation)
    if get_origin(ann) is not Literal:
        return False
    args = get_args(ann)
    if len(args) != 1:
        return False
    arg = args[0].value if isinstance(args[0], Enum) else args[0]
    default = info.default.value if isinstance(info.default, Enum) else info.default
    return arg == default


def _format_type(annotation: Any) -> str:
    """Convert a type annotation to a readable string.

    Recursively walks the annotation tree so Annotated metadata and module
    paths are stripped at every nesting level.
    """
    if hasattr(annotation, "__metadata__"):
        return _format_type(get_args(annotation)[0])
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin is Literal:
        args = get_args(annotation)
        if args:
            values = ", ".join(repr(a.value) if isinstance(a, Enum) else repr(a) for a in args)
            return f"Literal[{values}]"
    if origin is None and hasattr(annotation, "__name__"):
        return annotation.__name__
    args = get_args(annotation)
    if origin is not None and args:
        if origin is Union or origin is types.UnionType:
            return " | ".join(_format_type(a) for a in args)
        origin_name = origin.__name__ if hasattr(origin, "__name__") else str(origin)
        return f"{origin_name}[{', '.join(_format_type(a) for a in args)}]"
    return _MODULE_PATH_RE.sub(lambda m: m.group().rsplit(".", 1)[-1], str(annotation))


def _format_default(info: FieldInfo) -> str:
    """Format a field's default value for display."""
    if info.default_factory is not None:
        return f"{getattr(info.default_factory, '__name__', repr(info.default_factory))}()"
    default = info.default.value if isinstance(info.default, Enum) else info.default
    return repr(default)


def _unwrap_annotated(annotation: Any) -> Any:
    """Strip Annotated[T, ...] wrappers, returning the inner type."""
    return get_args(annotation)[0] if hasattr(annotation, "__metadata__") else annotation


def _find_expandable_leaves(annotation: Any, base_cls: type[BaseModel]) -> list[type]:
    """Walk the annotation tree and return Enum / base_cls leaf types.

    Recurses into list[X] and Optional[X], but stops at multi-member unions.
    Only handles patterns that appear in config fields today; dict/set/tuple
    with expandable inner types would need additions here.
    """
    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)

    if origin is None and isinstance(annotation, type):
        return [annotation] if issubclass(annotation, (Enum, base_cls)) else []
    args = get_args(annotation)
    if not args:
        return []

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        return _find_expandable_leaves(non_none[0], base_cls) if len(non_none) == 1 else []

    if origin is list:
        return _find_expandable_leaves(args[0], base_cls)

    return []


def _extract_first_paragraph(docstring: str) -> str | None:
    """Extract the first paragraph of a docstring, before any blank line or section header."""
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


# --- Constants ---

_MODULE_PATH_RE = re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+")

_CONTAINER_TYPES = frozenset({list, dict, set, tuple, frozenset})

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
