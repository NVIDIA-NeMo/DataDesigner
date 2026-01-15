# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from copy import deepcopy
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, overload

from data_designer.engine.processing.gsonschema.exceptions import JSONSchemaValidationError
from data_designer.engine.processing.gsonschema.schema_transformers import forbid_additional_properties
from data_designer.engine.processing.gsonschema.types import DataObjectT, JSONSchemaT, T_primitive
from data_designer.lazy_heavy_imports import jsonschema

if TYPE_CHECKING:
    import jsonschema

DEFAULT_JSONSCHEMA_VALIDATOR = jsonschema.Draft202012Validator

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


def extend_jsonschema_validator_with_pruning(validator):
    """Modify behavior of a jsonschema.Validator to use pruning.

    Validators extended using this function will prune extra
    fields, rather than raising a ValidationError, when encountering
    extra, unspecified fiends when `additionalProperties: False` is
    set in the validating schema.

    Args:
        validator (Type[jsonschema.Validator): A validator class
            to extend with pruning behavior.

    Returns:
        Type[jsonschema.Validator]: A validator class that will
            prune extra fields.
    """
    return jsonschema.validators.extend(validator, {"additionalProperties": prune_additional_properties})


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


def normalize_decimal_fields(obj: DataObjectT, schema: JSONSchemaT) -> DataObjectT:
    """Normalize Decimal-like anyOf fields to floats with proper precision."""
    if not isinstance(obj, dict):
        return obj

    defs = schema.get("$defs", {})
    obj_schema = defs.get(schema.get("$ref", "")[len("#/$defs/") :], schema)
    props = obj_schema.get("properties", {})

    for key, value in obj.items():
        field_schema = props.get(key, {})
        if "$ref" in field_schema:
            field_schema = defs.get(field_schema["$ref"][len("#/$defs/") :], {})

        if isinstance(value, dict):
            obj[key] = normalize_decimal_fields(value, schema)
        elif isinstance(value, list):
            obj[key] = [normalize_decimal_fields(v, schema) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, (int, float, str)) and not isinstance(value, bool):
            is_decimal, decimal_places = _get_decimal_info_from_anyof(field_schema)
            if is_decimal:
                d = Decimal(str(value))
                if decimal_places is not None:
                    d = d.quantize(Decimal(f"0.{'0' * decimal_places}"), rounding=ROUND_HALF_UP)
                obj[key] = float(d)

    return obj


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
    validator = DEFAULT_JSONSCHEMA_VALIDATOR
    if pruning:
        validator = extend_jsonschema_validator_with_pruning(validator)

    if no_extra_properties:
        schema = forbid_additional_properties(schema)

    try:
        validator(schema).validate(final_object)
    except jsonschema.ValidationError as exc:
        raise JSONSchemaValidationError(str(exc)) from exc

    final_object = normalize_decimal_fields(final_object, schema)

    return final_object
