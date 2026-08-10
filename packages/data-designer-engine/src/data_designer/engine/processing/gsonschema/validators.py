# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import re
from copy import deepcopy
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, overload

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.processing.gsonschema.exceptions import JSONSchemaValidationError
from data_designer.engine.processing.gsonschema.schema_transformers import forbid_additional_properties
from data_designer.engine.processing.gsonschema.types import DataObjectT, JSONSchemaT, T_primitive


@functools.lru_cache(maxsize=1)
def _get_default_validator() -> type:
    return lazy.jsonschema.Draft202012Validator


logger = logging.getLogger(__name__)


def prune_additional_properties(
    _, allow_additional_properties: bool, instance: DataObjectT, schema: JSONSchemaT
) -> None:
    """A JSONSchemaValidtor extension function to prune additional properties.

    Operates on an individual schema in-place.

    Args:
        allow_additional_properties (bool): The value of the `additionalProperties`
            field for this schema.
        instance (DataObjectT): The data object being validated.
        schema (JSONSchemaT): The schema for this object.

    Returns:
        Nothing (in place)
    """
    # Only act if the instance is a dict.
    if not isinstance(instance, dict) or allow_additional_properties:
        return

    # Allowed keys are those defined in the schema's "properties".
    allowed = schema.get("properties", {}).keys()

    # Iterate over a copy of keys so we can modify the dict in place.
    n_removed = 0
    for key in list(instance.keys()):
        if key not in allowed:
            instance.pop(key)
            n_removed += 1
            logger.info(f"Unspecified property removed from data object: {key}.")

    if n_removed > 0:
        logger.info(f"{n_removed} unspecified properties removed from data object.")


def _validate_one_of_with_discriminator(
    validator: Any, one_of: list[JSONSchemaT], instance: DataObjectT, schema: JSONSchemaT
) -> Any:
    """Validate a oneOf using the discriminator to select the correct variant.

    Standard oneOf tries all variants, which combined with in-place pruning
    can corrupt the instance (pruning from a failed variant removes properties
    needed by the correct variant). When a discriminator is present, this
    validator selects the matching variant directly.
    """
    discriminator = schema.get("discriminator")
    if not discriminator or not isinstance(discriminator, dict) or not isinstance(instance, dict):
        yield from lazy.jsonschema.Draft202012Validator.VALIDATORS["oneOf"](validator, one_of, instance, schema)
        return

    prop_name = discriminator.get("propertyName")
    mapping = discriminator.get("mapping", {})
    if not prop_name or prop_name not in instance or not mapping:
        yield from lazy.jsonschema.Draft202012Validator.VALIDATORS["oneOf"](validator, one_of, instance, schema)
        return

    matched_ref = mapping.get(str(instance[prop_name]))
    if matched_ref is None:
        yield lazy.jsonschema.ValidationError(
            f"{instance[prop_name]!r} is not a valid value for discriminator {prop_name!r}",
        )
        return

    matched_schema = {"$ref": matched_ref}
    errs = list(validator.descend(instance, matched_schema))
    yield from errs


def extend_jsonschema_validator_with_pruning(validator):
    """Modify behavior of a jsonschema.Validator to use pruning.

    Validators extended using this function will prune extra
    fields, rather than raising a ValidationError, when encountering
    extra, unspecified fiends when `additionalProperties: False` is
    set in the validating schema.

    When a oneOf has a discriminator, the discriminator is used to select
    the correct variant directly, preventing in-place pruning from
    corrupting the instance during failed variant checks.

    Args:
        validator (Type[jsonschema.Validator): A validator class
            to extend with pruning behavior.

    Returns:
        Type[jsonschema.Validator]: A validator class that will
            prune extra fields.
    """
    return lazy.jsonschema.validators.extend(
        validator,
        {
            "additionalProperties": prune_additional_properties,
            "oneOf": _validate_one_of_with_discriminator,
        },
    )


def _get_decimal_info_from_anyof(schema: dict) -> tuple[bool, int | None]:
    """Check if schema is a Decimal anyOf and extract decimal places.

    Returns (is_decimal, decimal_places) where decimal_places is None if no constraint.
    """
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return False, None

    has_number = any(item.get("type") == "number" for item in any_of)
    if not has_number:
        return False, None

    for item in any_of:
        if item.get("type") == "string" and "pattern" in item:
            match = re.search(r"\\d\{0,(\d+)\}", item["pattern"])
            if match:
                return True, int(match.group(1))
            return True, None  # Decimal without precision constraint
    return False, None


def _resolve_local_ref(schema: JSONSchemaT, root_schema: JSONSchemaT) -> JSONSchemaT:
    """Resolve local JSON Pointer references while preserving sibling keywords."""
    resolved_schema = schema
    seen_refs: set[str] = set()

    while isinstance(resolved_schema, dict) and (ref := resolved_schema.get("$ref")):
        if not isinstance(ref, str) or ref in seen_refs or not (ref == "#" or ref.startswith("#/")):
            break
        seen_refs.add(ref)

        target: Any = root_schema
        if ref != "#":
            for token in ref[2:].split("/"):
                token = token.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, dict) or token not in target:
                    return resolved_schema
                target = target[token]
        if not isinstance(target, dict):
            return resolved_schema

        siblings = {key: value for key, value in resolved_schema.items() if key != "$ref"}
        resolved_schema = target | siblings

    return resolved_schema


def _normalize_numeric_fields(
    obj: DataObjectT,
    schema: JSONSchemaT,
    root_schema: JSONSchemaT,
    validator: Any,
) -> DataObjectT:
    """Recursively canonicalize numeric values according to their JSON Schema types."""
    schema = _resolve_local_ref(schema, root_schema)

    is_decimal, decimal_places = _get_decimal_info_from_anyof(schema)
    if is_decimal and isinstance(obj, (int, float, str)) and not isinstance(obj, bool):
        value = Decimal(str(obj))
        if decimal_places is not None:
            value = value.quantize(Decimal(f"0.{'0' * decimal_places}"), rounding=ROUND_HALF_UP)
        return float(value)

    for keyword in ("oneOf", "anyOf"):
        alternatives = schema.get(keyword)
        if isinstance(alternatives, list):
            for alternative in alternatives:
                if validator.evolve(schema=alternative).is_valid(obj):
                    return _normalize_numeric_fields(obj, alternative, root_schema, validator)

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for subschema in all_of:
            obj = _normalize_numeric_fields(obj, subschema, root_schema, validator)

    schema_type = schema.get("type")
    schema_types = {schema_type} if isinstance(schema_type, str) else set(schema_type or [])
    if "integer" in schema_types and isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return int(obj)
    if "number" in schema_types and isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)

    if isinstance(obj, dict):
        properties = schema.get("properties", {})
        additional_properties = schema.get("additionalProperties", {})
        for key, value in obj.items():
            field_schema = properties.get(key, additional_properties if isinstance(additional_properties, dict) else {})
            obj[key] = _normalize_numeric_fields(value, field_schema, root_schema, validator)
        return obj

    if isinstance(obj, list):
        prefix_items = schema.get("prefixItems", [])
        item_schema = schema.get("items", {})
        for index, value in enumerate(obj):
            field_schema = prefix_items[index] if index < len(prefix_items) else item_schema
            obj[index] = _normalize_numeric_fields(value, field_schema, root_schema, validator)

    return obj


def normalize_numeric_fields(obj: DataObjectT, schema: JSONSchemaT) -> DataObjectT:
    """Normalize JSON Schema numbers and integers to stable Python numeric types."""
    validator = _get_default_validator()(schema)
    return _normalize_numeric_fields(obj, schema, schema, validator)


## We don't expect the outer data type (e.g. dict, list, or const) to be
## modified by the pruning action.
@overload
def validate(
    obj: dict[str, Any],
    schema: JSONSchemaT,
    pruning: bool = False,
    no_extra_properties: bool = False,
) -> dict[str, Any]: ...


@overload
def validate(
    obj: list[Any],
    schema: JSONSchemaT,
    pruning: bool = False,
    no_extra_properties: bool = False,
) -> list[Any]: ...


@overload
def validate(
    obj: T_primitive,
    schema: JSONSchemaT,
    pruning: bool = False,
    no_extra_properties: bool = False,
) -> T_primitive: ...


def validate(
    obj: DataObjectT,
    schema: JSONSchemaT,
    pruning: bool = False,
    no_extra_properties: bool = False,
) -> DataObjectT:
    """Validate a data object against a JSONSchema.

    Args:
        obj (DataObjectT): A data structure to validate against the
            schema.
        schema: (JSONSchemaT): A valid JSONSchema to use to validate
            the provided data object.
        pruning (bool): If set to `True`, then the default behavior for
            `additionalProperties: False` is set to prune non-specified
            properties instead of raising a ValidationError.
            Default: `False`.
        no_extra_properties (bool): If set to `True`, then
            `additionalProperties: False` is set on all the schema
            and all of its sub-schemas. This operation overrides any
            existing settings of `additionalProperties` within the
            schema. Default: `False`.

    Raises:
        JSONSchemaValidationError: This exception raised in the
            event that the JSONSchema doesn't match the provided
            schema.
    """
    final_object = deepcopy(obj)
    validator = _get_default_validator()
    if pruning:
        validator = extend_jsonschema_validator_with_pruning(validator)

    if no_extra_properties:
        schema = forbid_additional_properties(schema)

    try:
        validator(schema).validate(final_object)
    except lazy.jsonschema.ValidationError as exc:
        raise JSONSchemaValidationError(str(exc)) from exc

    final_object = normalize_numeric_fields(final_object, schema)

    return final_object
