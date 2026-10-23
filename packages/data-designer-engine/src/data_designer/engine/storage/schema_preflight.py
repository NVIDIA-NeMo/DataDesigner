# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative checks for JSON schemas that native Parquet cannot always preserve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.storage_schema import compile_json_storage_schema, resolve_local_json_pointer


@dataclass(frozen=True)
class StorageSchemaRisk:
    path: tuple[str | None, ...]
    reason: str
    definite: bool = False


def find_storage_schema_risks(schema: dict[str, Any]) -> list[StorageSchemaRisk]:
    """Report proven failures separately from value-dependent risks.

    This is deliberately not a proof that an arbitrary JSON Schema is storable.
    Only required, unconditional locations can produce fatal findings. Nullable
    values, optional properties, and union branches receive warnings instead.
    """
    _, paths = compile_json_storage_schema(schema)
    dialect = schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    if not isinstance(dialect, str) or dialect.rstrip("#") != "https://json-schema.org/draft/2020-12/schema":
        return [StorageSchemaRisk((), "uses a schema dialect whose native storage cannot be verified")]
    encoded = [tuple(path) for path in paths]
    risks: dict[tuple[str | None, ...], StorageSchemaRisk] = {}

    def report(path: tuple[str | None, ...], reason: str, definite: bool = False) -> None:
        previous = risks.get(path)
        if previous is None or definite:
            risks[path] = StorageSchemaRisk(path, reason, definite)

    def visit(node: Any, path: tuple[str | None, ...], required: bool, refs: tuple[str, ...]) -> None:
        if any(path[: len(parent)] == parent for parent in encoded):
            return
        if node is True:
            report(path, "allows arbitrary JSON values")
            return
        if not isinstance(node, dict):
            return
        if node is not schema and "$id" in node:
            report(path, "uses a nested $id schema resource whose native storage cannot be verified")
            return

        reference = node.get("$ref")
        if reference is not None:
            if not isinstance(reference, str) or reference in refs:
                report(path, "uses a recursive or unresolved schema whose native storage cannot be verified")
                return
            try:
                target = resolve_local_json_pointer(schema, reference)
            except InvalidConfigError:
                report(path, "uses a reference whose native storage cannot be verified")
                return
            visit(target, path, required, (*refs, reference))

        # A constant is actual schema data, not a fabricated example. Probe only
        # when no descendant will be encoded by the storage contract.
        has_encoded_child = any(child[: len(path)] == path for child in encoded)
        constants = [node["const"]] if "const" in node else node.get("enum")
        if isinstance(constants, list) and constants and not has_encoded_child:
            # Bound backend probes for large enums. An incomplete check can warn,
            # but cannot justify rejecting a configuration.
            failures = [_constant_cannot_write(value) for value in constants[:64]]
            if any(failures):
                report(
                    path,
                    "declares constant values that native Parquet cannot serialize",
                    # Individually failing enum values can complement each
                    # other's struct fields within a batch. Only a single
                    # constant proves that every batch has the same failure.
                    required and len(constants) == 1,
                )
            if len({_literal_kind(value) for value in constants}) > 1:
                report(path, "permits different literal types that may fail native Parquet inference")
            if len(constants) > len(failures) and any(isinstance(value, (dict, list)) for value in constants):
                report(path, "has a large structured enum whose native storage was not fully checked")
            # Literals completely determine the value at this location.
            return

        raw_types = node.get("type")
        types = set(raw_types) if isinstance(raw_types, list) else {raw_types} if isinstance(raw_types, str) else set()
        nonnull_types = types - {"null"}
        categories = {"number" if kind == "integer" else kind for kind in nonnull_types}
        if len(categories) > 1:
            report(path, "permits different JSON types that may fail native Parquet inference")
        certain = required and "null" not in types and len(nonnull_types) == 1
        branches = {"anyOf", "oneOf", "allOf", "if", "then", "else", "not"}
        if branches.intersection(node):
            certain = False

        is_object = (not types or "object" in types) and (
            "object" in types or "properties" in node or "additionalProperties" in node
        )
        if is_object:
            object_certain = certain and nonnull_types == {"object"}
            properties = node.get("properties", {})
            closed_empty = (
                node.get("additionalProperties") is False and not properties and not node.get("patternProperties")
            )
            if node.get("maxProperties") == 0 or closed_empty:
                # Encoded descendants supply explicit physical fields even when
                # the logical object is empty.
                if not has_encoded_child:
                    report(
                        path, "requires an empty object, which native Parquet cannot store as a struct", object_certain
                    )
            elif (
                not properties or node.get("additionalProperties", False) is not False or node.get("patternProperties")
            ):
                report(path, "allows dynamic object keys that native struct storage may normalize or fail to serialize")
            required_names = node.get("required", [])
            for name, child in properties.items():
                visit(child, (*path, name), object_certain and name in required_names, refs)

        if (not types or "array" in types) and ("array" in types or "items" in node or "prefixItems" in node):
            # Prefix elements can supply struct fields for otherwise
            # unwritable trailing items, so inspecting those items alone
            # cannot prove the entire array will fail.
            items_required = (
                certain and nonnull_types == {"array"} and node.get("minItems", 0) > 0 and not node.get("prefixItems")
            )
            if "items" in node:
                visit(node["items"], (*path, None), items_required, refs)
            elif "prefixItems" not in node:
                report(path, "allows arbitrary array elements that may fail native Parquet inference")
            if "prefixItems" in node:
                report(path, "uses positional array schemas whose elements may need incompatible native types")
                for child in node["prefixItems"]:
                    visit(child, (*path, None), False, refs)

        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in node:
                children = node[keyword]
                if keyword == "allOf" or not _is_nullable_scalar_union(children):
                    report(path, "combines schema branches whose native storage depends on generated values")
                for child in children:
                    visit(child, path, False, refs)
        if any(keyword in node for keyword in ("if", "then", "else", "$dynamicRef", "$recursiveRef")):
            report(path, "uses conditional or dynamic schemas whose native storage cannot be verified")
        if not types and not any(
            key in node
            for key in (
                "$ref",
                "const",
                "enum",
                "properties",
                "additionalProperties",
                "items",
                "prefixItems",
                *branches,
            )
        ):
            report(path, "allows arbitrary JSON values")

    visit(schema, (), True, ())
    return list(risks.values())


def _is_nullable_scalar_union(branches: list[Any]) -> bool:
    types = [branch.get("type") if isinstance(branch, dict) else None for branch in branches]
    if not all(isinstance(kind, str) for kind in types):
        return False
    categories = {"number" if kind == "integer" else kind for kind in types if kind != "null"}
    return len(categories) == 1 and categories <= {"string", "number", "boolean"}


def _constant_cannot_write(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, float)):
        return False
    if isinstance(value, int) and -(2**63) <= value < 2**63:
        return False
    try:
        with lazy.pa.BufferOutputStream() as sink:
            lazy.pd.DataFrame({"value": [value]}).to_parquet(sink, index=False)
    except (lazy.pa.ArrowException, ValueError, TypeError, OverflowError):
        return True
    return False


def _literal_kind(value: Any) -> type:
    return float if isinstance(value, (int, float)) and not isinstance(value, bool) else type(value)
