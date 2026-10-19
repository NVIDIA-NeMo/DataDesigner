# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterator
from copy import deepcopy
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, overload

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.processing.gsonschema.exceptions import JSONSchemaValidationError
from data_designer.engine.processing.gsonschema.schema_transformers import forbid_additional_properties
from data_designer.engine.processing.gsonschema.types import DataObjectT, JSONSchemaT, T_primitive


@functools.lru_cache(maxsize=1)
def _get_default_validator() -> type:
    return lazy.jsonschema.Draft202012Validator


logger = logging.getLogger(__name__)


def prune_additional_properties(
    validator: Any, allow_additional_properties: bool | JSONSchemaT, instance: DataObjectT, schema: JSONSchemaT
) -> Iterator[Any]:
    """Prune forbidden properties and validate explicitly allowed dictionary values.

    Operates on an individual schema in-place.

    Args:
        validator: The active JSON Schema validator.
        allow_additional_properties: The value of the `additionalProperties`
            field for this schema.
        instance (DataObjectT): The data object being validated.
        schema (JSONSchemaT): The schema for this object.

    Yields:
        Validation errors for schema-valued additional properties.
    """
    if allow_additional_properties is not False:
        yield from lazy.jsonschema.Draft202012Validator.VALIDATORS["additionalProperties"](
            validator, allow_additional_properties, instance, schema
        )
        return
    if not isinstance(instance, dict):
        return

    # Properties matched by a pattern are also declared fields.
    allowed = schema.get("properties", {}).keys()

    # Iterate over a copy of keys so we can modify the dict in place.
    n_removed = 0
    for key in list(instance.keys()):
        if key not in allowed and not any(re.search(pattern, key) for pattern in schema.get("patternProperties", {})):
            instance.pop(key)
            n_removed += 1
            logger.info(f"Unspecified property removed from data object: {key}.")

    if n_removed > 0:
        logger.info(f"{n_removed} unspecified properties removed from data object.")


def _validate_alternatives_with_pruning(
    validator: Any,
    alternatives: list[JSONSchemaT],
    instance: DataObjectT,
    keyword: Literal["anyOf", "oneOf"],
) -> Iterator[Any]:
    """Prefer unchanged matches, falling back to isolated pruning candidates."""
    errors = []
    unpruned_matches: list[DataObjectT] = []
    pruned_matches: list[DataObjectT] = []
    for index, subschema in enumerate(alternatives):
        candidate = deepcopy(instance)
        candidate_errors = list(validator.descend(candidate, subschema, schema_path=index))
        if candidate_errors:
            errors.extend(candidate_errors)
        elif candidate == instance:
            unpruned_matches.append(candidate)
            if keyword == "anyOf":
                return
        else:
            pruned_matches.append(candidate)

    matches = unpruned_matches or pruned_matches
    if not matches:
        yield lazy.jsonschema.ValidationError(
            f"{instance!r} is not valid under any of the given schemas", context=errors
        )
        return
    if keyword == "oneOf" and len(matches) > 1:
        yield lazy.jsonschema.ValidationError(f"{instance!r} is valid under more than one of the given schemas")
        return

    if keyword == "oneOf":
        # An explicit dialect selects the stock validator while evolve retains reference scope.
        strict_validator = validator.evolve(
            schema={"$schema": _get_default_validator().META_SCHEMA["$id"], "oneOf": alternatives}
        )
        final_errors = list(strict_validator.iter_errors(matches[0]))
        if final_errors:
            yield from final_errors
            return

    if isinstance(instance, dict):
        instance.clear()
        instance.update(matches[0])
    elif isinstance(instance, list):
        instance[:] = matches[0]


def _validate_any_of_with_pruning(
    validator: Any, any_of: list[JSONSchemaT], instance: DataObjectT, schema: JSONSchemaT
) -> Iterator[Any]:
    yield from _validate_alternatives_with_pruning(validator, any_of, instance, "anyOf")


def _validate_one_of_with_discriminator(
    validator: Any, one_of: list[JSONSchemaT], instance: DataObjectT, schema: JSONSchemaT
) -> Iterator[Any]:
    """Validate a oneOf using the discriminator to select the correct variant.

    Standard oneOf tries all variants, which combined with in-place pruning
    can corrupt the instance (pruning from a failed variant removes properties
    needed by the correct variant). When a discriminator is present, this
    validator selects the matching variant directly. Without a discriminator,
    each variant is validated on an independent copy. Matches that need no
    pruning take precedence over matches that remove fields.
    """
    discriminator = schema.get("discriminator")
    if not discriminator or not isinstance(discriminator, dict) or not isinstance(instance, dict):
        yield from _validate_alternatives_with_pruning(validator, one_of, instance, "oneOf")
        return

    prop_name = discriminator.get("propertyName")
    mapping = discriminator.get("mapping", {})
    if not prop_name or prop_name not in instance or not mapping:
        yield from _validate_alternatives_with_pruning(validator, one_of, instance, "oneOf")
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

    Alternatives are checked on independent copies so failed candidates cannot
    prune fields from later candidates. Unchanged matches take precedence over
    pruned matches. Selected oneOf results must match exactly one alternative
    without further pruning. Discriminated oneOf schemas select the correct variant directly.

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
            "anyOf": _validate_any_of_with_pruning,
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
            and its record sub-schemas where no explicit setting exists.
            Declared dictionary schemas and explicit `additionalProperties`
            settings are preserved. Default: `False`.

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

    final_object = normalize_decimal_fields(final_object, schema)

    return final_object
