# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

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
            for index, path in enumerate(paths):
                for other in paths[index + 1 :]:
                    if path[: len(other)] == other or other[: len(path)] == path:
                        raise ValueError(f"Overlapping JSON storage paths in column {name!r}")
        return self

    def for_columns(self, names: list[str]) -> JsonStoragePlan:
        """Restrict a contract to the columns present in an artifact."""
        return JsonStoragePlan(columns={name: self.columns[name] for name in names if name in self.columns})


def compile_json_storage_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], list[list[str | None]]]:
    """Extract storage paths and return a model-facing schema without DD annotations.

    Supports finite properties/items traversal and local references. Annotate a whole
    enclosing value when storage would otherwise depend on a union or recursive shape.
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
        JsonStoragePlan(columns={"value": paths})
    return cleaned, paths


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
) -> None:
    if not isinstance(node, dict):
        return
    if JSON_STORAGE_KEY in node:
        if unsupported:
            raise InvalidConfigError(
                f"Unsupported branch-dependent JSON storage at {path!r}; annotate its enclosing field"
            )
        if sum(JSON_STORAGE_KEY in child for child in _schema_nodes(node)) > 1:
            raise InvalidConfigError(f"Overlapping JSON storage declarations at {path!r}")
        paths.append(path)
        return
    if "$dynamicRef" in node or "$recursiveRef" in node:
        raise InvalidConfigError(
            "Dynamic references are unsupported in JSON storage schemas; annotate an enclosing field"
        )
    reference = node.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise InvalidConfigError("JSON storage supports only local JSON Pointer references")
        if reference in references:
            raise InvalidConfigError("Recursive JSON storage traversal requires an enclosing JSON-encoded field")
        target: Any = root
        try:
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError) as exc:
            raise InvalidConfigError(f"Unresolved JSON storage reference {reference!r}") from exc
        _collect_paths(target, root, path, (*references, reference), unsupported, paths)
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            for name, child in value.items():
                _collect_paths(child, root, [*path, name], references, unsupported, paths)
        elif key == "items" and isinstance(value, dict):
            _collect_paths(value, root, [*path, None], references, unsupported, paths)
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
