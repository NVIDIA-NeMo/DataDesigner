# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from copy import deepcopy
from typing import Any, overload

from jsonschema import Draft202012Validator, ValidationError, validators

from data_designer.engine.processing.gsonschema.exceptions import JSONSchemaValidationError
from data_designer.engine.processing.gsonschema.schema_transformers import forbid_additional_properties
from data_designer.engine.processing.gsonschema.types import DataObjectT, JSONSchemaT, T_primitive

DEFAULT_JSONSCHEMA_VALIDATOR = Draft202012Validator

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
    return validators.extend(validator, {"additionalProperties": prune_additional_properties})


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
    except ValidationError as exc:
        raise JSONSchemaValidationError(str(exc)) from exc

    return final_object
