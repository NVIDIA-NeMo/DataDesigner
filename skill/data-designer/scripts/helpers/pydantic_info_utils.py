#!/usr/bin/env python3
"""Shared utilities for printing LLM-optimized context from Pydantic models.

This module provides common functionality for the four info scripts:
get_column_info.py, get_sampler_info.py, get_validator_info.py, and
get_processor_info.py.
"""

import re
import sys
import types
import typing
from collections.abc import Callable
from enum import Enum
from typing import get_args, get_origin

from pydantic import BaseModel


def _is_basemodel_subclass(cls) -> bool:
    """Return True if cls is a concrete BaseModel subclass (not BaseModel itself)."""
    return isinstance(cls, type) and issubclass(cls, BaseModel) and cls is not BaseModel


def _is_enum_subclass(cls) -> bool:
    """Return True if cls is an Enum subclass (not Enum itself)."""
    return isinstance(cls, type) and issubclass(cls, Enum) and cls is not Enum


def _extract_enum_class(annotation) -> type | None:
    """Unwrap a type annotation to find an Enum class, if present.

    Handles X, X | None, Annotated[X, ...].
    Returns the Enum class or None.
    """
    if annotation is None:
        return None

    # Unwrap Annotated[X, ...]
    if get_origin(annotation) is typing.Annotated:
        annotation = get_args(annotation)[0]

    # Direct enum class
    if _is_enum_subclass(annotation):
        return annotation

    # Union: X | None or typing.Union[X, None]
    origin = get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            if _is_enum_subclass(arg):
                return arg

    return None


def extract_nested_basemodel(annotation) -> type | None:
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

    # Direct BaseModel subclass
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
        basemodel_classes = []
        for arg in non_none_args:
            # Recurse to handle list[X] | None etc.
            result = extract_nested_basemodel(arg)
            if result is not None:
                basemodel_classes.append(result)
            elif _is_basemodel_subclass(arg):
                basemodel_classes.append(arg)
        # Only return if exactly one BaseModel subclass found
        if len(basemodel_classes) == 1:
            return basemodel_classes[0]
        return None

    return None


def format_type(annotation) -> str:
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

    # Simplify common patterns
    type_str = type_str.replace("NoneType", "None")

    # Clean up Literal types for readability
    if "Literal[" in type_str:
        # Extract just the value from Literal['value']
        match = re.search(r"Literal\[([^\]]+)\]", type_str)
        if match:
            type_str = f"Literal[{match.group(1)}]"

    # Clean up Annotated types with Discriminator (too verbose)
    if "Annotated[" in type_str and "Discriminator" in type_str:
        # Extract just the union type, drop the Discriminator metadata
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


def get_field_info(
    cls: type, default_descriptions: dict[str, str]
) -> list[tuple[str, str, str, type | None, type | None]]:
    """Extract field information from a Pydantic model.

    Args:
        cls: The Pydantic model class to inspect.
        default_descriptions: Fallback descriptions for fields without docstrings.

    Returns:
        List of (field_name, type_str, description, nested_basemodel_cls, enum_cls) tuples.
    """
    fields = []
    model_fields: dict = getattr(cls, "model_fields", {})
    if model_fields:
        for field_name, field_info in model_fields.items():
            type_str = format_type(field_info.annotation)
            # Use field's description if available, otherwise fall back to defaults
            description = field_info.description or default_descriptions.get(field_name, "")
            nested_cls = extract_nested_basemodel(field_info.annotation)
            enum_cls = _extract_enum_class(field_info.annotation)
            fields.append((field_name, type_str, description, nested_cls, enum_cls))
    return fields


def _print_fields(
    fields: list[tuple[str, str, str, type | None, type | None]],
    default_descriptions: dict[str, str],
    indent: int = 4,
    seen: set | None = None,
    max_depth: int = 3,
    current_depth: int = 0,
) -> None:
    """Print fields with optional nested BaseModel expansion and enum values.

    Args:
        fields: List of (field_name, type_str, description, nested_cls, enum_cls) tuples.
        default_descriptions: Fallback descriptions for nested model fields.
        indent: Current indentation level (number of spaces).
        seen: Set of already-expanded class names to prevent cycles.
        max_depth: Maximum recursion depth for nested models.
        current_depth: Current recursion depth.
    """
    if seen is None:
        seen = set()

    pad = " " * indent

    for field_name, type_str, desc, nested_cls, enum_cls in fields:
        print(f"{pad}{field_name}:")
        print(f"{pad}  type: {type_str}")
        if desc:
            print(f"{pad}  description: {desc}")

        # Expand enum values
        if enum_cls is not None:
            values = [member.name for member in enum_cls]  # type: ignore[var-annotated]
            print(f"{pad}  values: [{', '.join(values)}]")

        # Expand nested BaseModel (with cycle and depth protection)
        if nested_cls is not None and nested_cls.__name__ not in seen and current_depth < max_depth:
            seen.add(nested_cls.__name__)
            nested_fields = get_field_info(nested_cls, default_descriptions)
            print(f"{pad}  schema ({nested_cls.__name__}):")
            _print_fields(
                nested_fields,
                default_descriptions,
                indent=indent + 4,
                seen=seen,
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )


def print_yaml_entry(
    type_key: str,
    type_value: str,
    cls: type,
    default_descriptions: dict[str, str],
    uppercase_value: bool = False,
) -> None:
    """Print YAML-style output for a Pydantic model class.

    Args:
        type_key: The key name for the type (e.g., "sampler_type" or "column_type").
        type_value: The value of the type (e.g., "category" or "llm-text").
        cls: The Pydantic model class to print.
        default_descriptions: Fallback descriptions for fields without docstrings.
        uppercase_value: If True, print type_value in uppercase.
    """
    class_name = cls.__name__
    description = get_brief_description(cls)
    fields = get_field_info(cls, default_descriptions)

    display_value = type_value.upper() if uppercase_value else type_value

    print(f"{class_name}:")
    print(f"  {type_key}: {display_value}")
    print(f"  description: {description}")
    print("  fields:")

    _print_fields(fields, default_descriptions)


def print_all_entries(
    items: dict[str, type],
    type_key: str,
    header_title: str,
    default_descriptions: dict[str, str],
    uppercase_value: bool = False,
) -> None:
    """Print YAML-style output for all items.

    Args:
        items: Dict mapping type values to their classes.
        type_key: The key name for the type (e.g., "sampler_type" or "column_type").
        header_title: Title for the header comment.
        default_descriptions: Fallback descriptions for fields without docstrings.
        uppercase_value: If True, print type values in uppercase.
    """
    sorted_types = sorted(items.keys())

    print(f"# {header_title}")
    print(f"# {len(sorted_types)} types discovered from data_designer.config")
    print()

    for type_value in sorted_types:
        cls = items[type_value]
        print_yaml_entry(type_key, type_value, cls, default_descriptions, uppercase_value)
        print()


def print_single_entry(
    items: dict[str, type],
    lookup_key: str,
    type_key: str,
    default_descriptions: dict[str, str],
    case_insensitive: bool = False,
    uppercase_value: bool = False,
) -> None:
    """Print YAML-style output for a specific item.

    Args:
        items: Dict mapping type values to their classes.
        lookup_key: The type value to look up.
        type_key: The key name for the type (e.g., "sampler_type" or "column_type").
        default_descriptions: Fallback descriptions for fields without docstrings.
        case_insensitive: If True, perform case-insensitive lookup.
        uppercase_value: If True, print type value in uppercase.
    """
    normalized = lookup_key.lower() if case_insensitive else lookup_key

    if normalized not in items:
        available = ", ".join(sorted(items.keys()))
        print(f"Error: Unknown {type_key} '{lookup_key}'", file=sys.stderr)
        print(f"Available types: {available}", file=sys.stderr)
        sys.exit(1)

    cls = items[normalized]
    print_yaml_entry(type_key, normalized, cls, default_descriptions, uppercase_value)


def print_list_table(
    items: dict[str, type],
    type_label: str,
    class_label: str,
) -> None:
    """Print available types with their class names in a table.

    Args:
        items: Dict mapping type values to their classes.
        type_label: Label for the type column (e.g., "sampler_type" or "column_type").
        class_label: Label for the class column (e.g., "params_class" or "config_class").
    """
    sorted_items = sorted(items.items())

    # Calculate column widths
    type_width = max(len(type_value) for type_value, _ in sorted_items)
    type_width = max(type_width, len(type_label))

    # Print header
    print(f"{type_label:<{type_width}}  {class_label}")
    print(f"{'-' * type_width}  {'-' * max(len(class_label), 25)}")

    # Print rows
    for type_value, cls in sorted_items:
        print(f"{type_value:<{type_width}}  {cls.__name__}")


def print_help(
    items: dict[str, type],
    type_label: str,
    class_label: str,
    script_name: str,
    description: str,
    examples: list[str],
) -> None:
    """Print help message with available types.

    Args:
        items: Dict mapping type values to their classes.
        type_label: Label for the type (e.g., "sampler_type" or "column_type").
        class_label: Label for the class column in list output.
        script_name: Name of the script for usage examples.
        description: Brief description of what the script does.
        examples: List of example command lines.
    """
    available_types = sorted(items.keys())

    print(f"Usage: uv run {script_name} <{type_label}>")
    print()
    print(description)
    print()
    print("Options:")
    print("  -h, --help    Show this help message")
    print(f"  -l, --list    List {type_label}s and their {class_label}es")
    print()
    print("Arguments:")
    print(f"  {type_label}  Type to print (use 'all' for all types)")
    print()
    print("Examples:")
    for example in examples:
        print(f"  {example}")
    print()
    print(f"Available {type_label}s ({len(available_types)}):")
    print()
    print_list_table(items, type_label, class_label)


def run_cli(
    discover_fn: Callable[[], dict[str, type]],
    type_key: str,
    type_label: str,
    class_label: str,
    default_descriptions: dict[str, str],
    script_name: str,
    description: str,
    header_title: str,
    examples: list[str],
    case_insensitive: bool = False,
    uppercase_value: bool = False,
) -> None:
    """Run the CLI for a Pydantic info script.

    Args:
        discover_fn: Function that returns dict mapping type values to classes.
        type_key: The key name for the type in YAML output.
        type_label: Label for the type in help text.
        class_label: Label for the class column in list output.
        default_descriptions: Fallback descriptions for fields without docstrings.
        script_name: Name of the script for usage examples.
        description: Brief description of what the script does.
        header_title: Title for the header when printing all entries.
        examples: List of example command lines.
        case_insensitive: If True, perform case-insensitive lookup.
        uppercase_value: If True, print type values in uppercase.
    """
    if len(sys.argv) == 1:
        # No arguments: print help
        items = discover_fn()
        print_help(items, type_label, class_label, script_name, description, examples)
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg in ("-h", "--help"):
            items = discover_fn()
            print_help(items, type_label, class_label, script_name, description, examples)
        elif arg in ("-l", "--list"):
            items = discover_fn()
            print_list_table(items, type_label, class_label)
        elif arg == "all":
            items = discover_fn()
            print_all_entries(items, type_key, header_title, default_descriptions, uppercase_value)
        else:
            # Single argument: print specific type
            items = discover_fn()
            print_single_entry(
                items,
                arg,
                type_key,
                default_descriptions,
                case_insensitive,
                uppercase_value,
            )
    else:
        items = discover_fn()
        print_help(items, type_label, class_label, script_name, description, examples)
        sys.exit(1)
