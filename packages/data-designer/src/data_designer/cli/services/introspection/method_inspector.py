# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field


@dataclass
class ParamInfo:
    name: str
    type_str: str
    default: str | None
    description: str


@dataclass
class MethodInfo:
    name: str
    signature: str
    description: str
    return_type: str
    parameters: list[ParamInfo] = field(default_factory=list)


_DEFAULT_INIT_DOCSTRING = "Initialize self. See help(type(self)) for accurate signature."


def _parse_google_docstring_args(docstring: str | None) -> dict[str, str]:
    """Parse Args section from a Google-style docstring.

    Returns:
        Dict mapping parameter names to their descriptions.
    """
    if not docstring:
        return {}

    lines = docstring.split("\n")
    result: dict[str, str] = {}
    in_args_section = False
    current_param: str | None = None
    current_desc_lines: list[str] = []
    args_indent: int | None = None

    section_pattern = re.compile(r"^(\s*)(Args|Returns|Raises|Yields|Note|Notes|Example|Examples|Attributes)\s*:")

    for line in lines:
        if re.match(r"^\s*Args\s*:\s*$", line):
            in_args_section = True
            args_indent = len(line) - len(line.lstrip())
            continue

        if not in_args_section:
            continue

        if not line.strip():
            if current_param is not None:
                current_desc_lines.append("")
            continue

        match = section_pattern.match(line)
        if match and match.group(2) != "Args":
            section_indent = len(line) - len(line.lstrip())
            if args_indent is not None and section_indent <= args_indent:
                break

        line_indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        param_match = re.match(r"^(\*{0,2}\w+)\s*(?:\(.+?\))?\s*:\s*(.*)$", stripped)
        if param_match and args_indent is not None and line_indent > args_indent:
            if current_param is not None:
                result[current_param] = _join_desc_lines(current_desc_lines)
            current_param = param_match.group(1)
            current_desc_lines = [param_match.group(2).strip()]
        elif current_param is not None:
            if args_indent is not None and line_indent <= args_indent:
                break
            current_desc_lines.append(stripped)

    if current_param is not None:
        result[current_param] = _join_desc_lines(current_desc_lines)

    return result


def _join_desc_lines(lines: list[str]) -> str:
    """Join description lines, collapsing whitespace and stripping trailing blanks."""
    return " ".join(part for part in lines if part)


def _format_annotation(annotation: type | str) -> str:
    """Format a type annotation to a readable string."""
    if annotation is inspect.Parameter.empty:
        return "Any"

    if isinstance(annotation, str):
        return annotation

    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return str(annotation).replace("typing.", "").replace("typing_extensions.", "")


def _format_signature(method_name: str, sig: inspect.Signature) -> str:
    """Format a method signature as a readable string, skipping 'self'."""
    params: list[str] = []
    seen_keyword_only = False
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())

    for param in sig.parameters.values():
        if param.name == "self":
            continue

        if param.kind == inspect.Parameter.KEYWORD_ONLY and not seen_keyword_only and not has_var_positional:
            seen_keyword_only = True
            params.append("*")

        type_str = _format_annotation(param.annotation)
        default_str = ""
        if param.default is not inspect.Parameter.empty:
            default_str = (
                f" = {param.default!r}" if not isinstance(param.default, type) else f" = {param.default.__name__}"
            )

        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            params.append(f"*{param.name}: {type_str}")
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            params.append(f"**{param.name}")
        else:
            params.append(f"{param.name}: {type_str}{default_str}")

    return_type = _format_return_type(sig)
    params_str = ", ".join(params)

    return f"{method_name}({params_str}) -> {return_type}"


def _format_return_type(sig: inspect.Signature) -> str:
    """Extract and format the return type from a signature."""
    if sig.return_annotation is inspect.Parameter.empty:
        return "None"

    formatted = _format_annotation(sig.return_annotation)
    if formatted == "Self":
        return "Self"

    return formatted


def _get_first_docstring_line(docstring: str | None) -> str:
    """Extract the first non-empty line from a docstring as the description."""
    if not docstring:
        return ""
    for line in docstring.strip().split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _build_param_info(sig: inspect.Signature, docstring_args: dict[str, str]) -> list[ParamInfo]:
    """Build ParamInfo list from a signature and parsed docstring args."""
    params: list[ParamInfo] = []
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            name = f"**{param.name}"
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            name = f"*{param.name}"
        else:
            name = param.name

        type_str = _format_annotation(param.annotation)
        default: str | None = None
        if param.default is not inspect.Parameter.empty:
            default = repr(param.default) if not isinstance(param.default, type) else param.default.__name__

        raw_name = param.name
        description = docstring_args.get(raw_name, "")
        if not description:
            description = docstring_args.get(f"**{raw_name}", "")
        if not description:
            description = docstring_args.get(f"*{raw_name}", "")

        params.append(ParamInfo(name=name, type_str=type_str, default=default, description=description))

    return params


def _is_dunder(name: str) -> bool:
    """Check if a method name is a dunder method (excluding __init__)."""
    return name.startswith("__") and name.endswith("__") and name != "__init__"


def _is_private(name: str) -> bool:
    """Check if a method name is private (starts with underscore, not dunder)."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_default_init_docstring(docstring: str | None) -> bool:
    """Check if a docstring is the unhelpful default __init__ docstring."""
    if not docstring:
        return False
    normalized = " ".join(docstring.strip().split())
    return normalized == _DEFAULT_INIT_DOCSTRING


def inspect_class_methods(cls: type, include_private: bool = False) -> list[MethodInfo]:
    """Introspect public methods of a class using inspect.signature() and docstring parsing.

    Detects regular methods, classmethods, and handles __init__ docstring fallback
    to the class docstring when the default is unhelpful.

    Args:
        cls: The class to introspect.
        include_private: If True, include methods starting with underscore.

    Returns:
        List of MethodInfo objects for each method.
    """
    methods: list[MethodInfo] = []

    # inspect.isfunction finds regular methods; inspect.ismethod finds classmethods
    seen: set[str] = set()
    candidates: list[tuple[str, object]] = []
    candidates.extend(inspect.getmembers(cls, predicate=inspect.isfunction))
    candidates.extend(inspect.getmembers(cls, predicate=inspect.ismethod))

    for name, method in candidates:
        if name in seen:
            continue
        seen.add(name)

        if _is_dunder(name):
            continue
        if _is_private(name) and not include_private:
            continue

        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            continue

        docstring = inspect.getdoc(method)

        if name == "__init__" and _is_default_init_docstring(docstring):
            docstring = inspect.getdoc(cls) or ""

        docstring_args = _parse_google_docstring_args(docstring)

        signature_str = _format_signature(name, sig)
        description = _get_first_docstring_line(docstring)
        return_type = _format_return_type(sig)
        parameters = _build_param_info(sig, docstring_args)

        methods.append(
            MethodInfo(
                name=name,
                signature=signature_str,
                description=description,
                return_type=return_type,
                parameters=parameters,
            )
        )

    return methods
