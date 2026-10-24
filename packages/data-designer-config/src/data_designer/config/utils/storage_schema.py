# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Literal
from urllib.parse import unquote

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.errors import InvalidConfigError

JSON_STORAGE_KEY = "x-data-designer-storage"
JSON_STORAGE_METADATA_KEY = b"data_designer.json_storage"
JSON_STORAGE_RUN_KEY = "json_storage"

_SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_SCHEMA_ARRAYS = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SCHEMA_SINGLE = {
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
    "unevaluatedProperties",
    "unevaluatedItems",
    "additionalItems",
}


class JsonStoragePlan(ConfigBase):
    """Versioned JSON encoding paths; None selects every element of a list."""

    version: Literal[1] = 1
    columns: dict[str, list[list[str | None]]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_version(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data and type(data["version"]) is not int:
            raise ValueError("JSON storage version must be an integer")
        return data

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        for name, paths in self.columns.items():
            if not paths:
                raise ValueError(f"JSON storage column {name!r} has no encoding paths")
            containers: dict[tuple[str | None, ...], bool] = {}
            for index, path in enumerate(paths):
                for depth, step in enumerate(path):
                    prefix = tuple(path[:depth])
                    is_array = step is None
                    if prefix in containers and containers[prefix] != is_array:
                        raise ValueError(f"Conflicting list/object JSON storage paths in column {name!r}")
                    containers[prefix] = is_array
                for other in paths[index + 1 :]:
                    if path[: len(other)] == other or other[: len(path)] == path:
                        raise ValueError(f"Overlapping JSON storage paths in column {name!r}")
            self.columns[name] = sorted(paths, key=_storage_path_sort_key)
        return self

    def restrict_to_columns(self, names: list[str]) -> JsonStoragePlan:
        """Restrict a contract to the columns present in an artifact."""
        return JsonStoragePlan(columns={name: self.columns[name] for name in names if name in self.columns})


def compile_json_storage_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], list[list[str | None]]]:
    """Extract storage paths and return a model-facing schema without DD annotations.

    Supports finite properties/items traversal through explicitly typed containers
    and local references within one schema resource. Annotate a whole enclosing
    value when storage would otherwise depend on a union, unknown type, or recursion.
    Literal data in const, enum, defaults, and examples is never traversed as a schema.
    """
    cleaned = deepcopy(schema)
    annotated = False
    for node in _schema_nodes(cleaned):
        if JSON_STORAGE_KEY in node:
            if node[JSON_STORAGE_KEY] != "json":
                raise InvalidConfigError(f"{JSON_STORAGE_KEY} must be 'json'")
            annotated = True
            del node[JSON_STORAGE_KEY]
    if not annotated:
        return cleaned, []
    dialect = schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    if not isinstance(dialect, str) or dialect.rstrip("#") != "https://json-schema.org/draft/2020-12/schema":
        raise InvalidConfigError("JSON storage annotations require JSON Schema Draft 2020-12")
    paths: list[list[str | None]] = []
    _collect_paths(schema, schema, [], (), False, paths)
    if paths:
        paths = JsonStoragePlan(columns={"value": paths}).columns["value"]
    return cleaned, paths


def resolve_local_json_pointer(root: dict[str, Any], reference: str) -> Any:
    """Resolve an in-document URI-fragment JSON Pointer within one schema resource.

    Args:
        root: The schema document containing the reference.
        reference: A fragment-only reference, including ``#`` for the document root.

    Returns:
        The referenced value, decoding URI escapes and JSON Pointer tokens.

    Raises:
        InvalidConfigError: If the pointer is invalid, unresolved, or crosses a nested
            ``$id`` resource. Annotate an enclosing value to avoid that traversal.
    """
    if not isinstance(reference, str) or not reference.startswith("#"):
        raise InvalidConfigError("JSON storage supports only local JSON Pointer references")
    fragment = reference[1:]
    if re.search(r"%(?![0-9A-Fa-f]{2})", fragment):
        raise InvalidConfigError(f"Invalid JSON storage reference {reference!r}")
    try:
        pointer = unquote(fragment, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InvalidConfigError(f"Invalid JSON storage reference {reference!r}") from exc
    if not pointer:
        return root
    if not pointer.startswith("/"):
        raise InvalidConfigError("JSON storage supports only local JSON Pointer references")
    target: Any = root
    for token in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", token):
            raise InvalidConfigError(f"Invalid JSON storage reference {reference!r}")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict) and target is not root and "$id" in target:
            raise InvalidConfigError("Nested $id schema resources require an enclosing JSON-encoded field")
        try:
            if isinstance(target, list):
                if token != "0" and (
                    not token or token[0] not in "123456789" or not token.isascii() or not token.isdigit()
                ):
                    raise ValueError("Invalid array index")
                target = target[int(token)]
            else:
                target = target[token]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InvalidConfigError(f"Unresolved JSON storage reference {reference!r}") from exc
    return target


def _storage_path_sort_key(path: list[str | None]) -> tuple[tuple[bool, str], ...]:
    return tuple((step is not None, step or "") for step in path)


def _schema_nodes(schema: Any) -> list[dict[str, Any]]:
    if not isinstance(schema, dict):
        return []
    nodes = [schema]
    for key, value in schema.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            for child in value.values():
                nodes.extend(_schema_nodes(child))
        elif key in _SCHEMA_ARRAYS and isinstance(value, list):
            for child in value:
                nodes.extend(_schema_nodes(child))
        elif key in _SCHEMA_SINGLE:
            if isinstance(value, list):
                for child in value:
                    nodes.extend(_schema_nodes(child))
            else:
                nodes.extend(_schema_nodes(value))
    return nodes


def _collect_paths(
    node: Any,
    root: dict[str, Any],
    path: list[str | None],
    references: tuple[str, ...],
    unsupported: bool,
    paths: list[list[str | None]],
    inherited_types: set[str] | None = None,
    inherited_prefix_items: bool = False,
) -> None:
    if not isinstance(node, dict):
        return
    if JSON_STORAGE_KEY in node:
        if unsupported:
            raise InvalidConfigError(
                f"Unsupported branch-dependent or ambiguous-container JSON storage at {path!r}; "
                "annotate its enclosing field"
            )
        if sum(JSON_STORAGE_KEY in child for child in _schema_nodes(node)) > 1:
            raise InvalidConfigError(f"Overlapping JSON storage declarations at {path!r}")
        paths.append(path)
        return
    if node is not root and "$id" in node:
        raise InvalidConfigError("Nested $id schema resources require an enclosing JSON-encoded field")
    if "$dynamicRef" in node or "$recursiveRef" in node:
        raise InvalidConfigError(
            "Dynamic references are unsupported in JSON storage schemas; annotate an enclosing field"
        )
    reference = node.get("$ref")
    types = _get_possible_types(node, root, references)
    has_prefix_items = inherited_prefix_items or _has_prefix_items(node, root, references)
    if inherited_types is not None:
        types = inherited_types if types is None else types & inherited_types
    if reference is not None:
        if reference in references:
            raise InvalidConfigError("Recursive JSON storage traversal requires an enclosing JSON-encoded field")
        target = resolve_local_json_pointer(root, reference)
        _collect_paths(target, root, path, (*references, reference), unsupported, paths, types, has_prefix_items)
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            for name, child in value.items():
                _collect_paths(
                    child,
                    root,
                    [*path, name],
                    references,
                    unsupported or types is None or types - {"null"} != {"object"},
                    paths,
                )
        elif key == "items" and isinstance(value, dict):
            _collect_paths(
                value,
                root,
                [*path, None],
                references,
                unsupported or has_prefix_items or types is None or types - {"null"} != {"array"},
                paths,
            )
        elif key in _SCHEMA_ARRAYS and isinstance(value, list):
            for child in value:
                _collect_paths(child, root, path, references, True, paths)
        elif key in _SCHEMA_MAPS - {"properties", "$defs", "definitions"} and isinstance(value, dict):
            for child in value.values():
                _collect_paths(child, root, path, references, True, paths)
        elif key in _SCHEMA_SINGLE - {"items"}:
            _collect_paths(value, root, path, references, True, paths)
        elif key == "items" and isinstance(value, list):
            for child in value:
                _collect_paths(child, root, path, references, True, paths)


def _get_possible_types(node: dict[str, Any], root: dict[str, Any], references: tuple[str, ...]) -> set[str] | None:
    raw_types = node.get("type")
    types = set(raw_types) if isinstance(raw_types, list) else {raw_types} if isinstance(raw_types, str) else None
    if JSON_STORAGE_KEY in node:
        return types
    reference = node.get("$ref")
    if reference is not None and reference not in references:
        target = resolve_local_json_pointer(root, reference)
        if isinstance(target, dict):
            target_types = _get_possible_types(target, root, (*references, reference))
            if target_types is not None:
                types = target_types if types is None else types & target_types
    return types


def _has_prefix_items(node: dict[str, Any], root: dict[str, Any], references: tuple[str, ...]) -> bool:
    if node.get("prefixItems"):
        return True
    reference = node.get("$ref")
    if reference is None or reference in references:
        return False
    target = resolve_local_json_pointer(root, reference)
    return isinstance(target, dict) and _has_prefix_items(target, root, (*references, reference))
