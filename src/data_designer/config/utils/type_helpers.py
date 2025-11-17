# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import inspect
from typing import Any, Literal, Type, get_args, get_origin

from pydantic import BaseModel

from .. import sampler_params
from .errors import InvalidDiscriminatorFieldError, InvalidEnumValueError, InvalidTypeUnionError


class StrEnum(str, Enum):
    pass


def create_str_enum_from_discriminated_type_union(
    enum_name: str,
    type_union: type,
    discriminator_field_name: str,
) -> StrEnum:
    """Create a string enum from a type union.

    The type union is assumed to be a union of configs (Pydantic models) that have a discriminator field,
    which must be a Literal string type - e.g., Literal["expression"].

    Args:
        enum_name: Name of the StrEnum.
        type_union: Type union of configs (Pydantic models).
        discriminator_field_name: Name of the discriminator field.

    Returns:
        StrEnum with values being the discriminator field values of the configs in the type union.

    Example:
        DataDesignerColumnType = create_str_enum_from_discriminated_type_union(
            enum_name="DataDesignerColumnType",
            type_union=ColumnConfigT,
            discriminator_field_name="column_type",
        )
    """
    discriminator_field_values = []
    for model in type_union.__args__:
        if not issubclass(model, BaseModel):
            raise InvalidTypeUnionError(f"🛑 {model} must be a subclass of pydantic.BaseModel.")
        if discriminator_field_name not in model.model_fields:
            raise InvalidDiscriminatorFieldError(f"🛑 '{discriminator_field_name}' is not a field of {model}.")
        if get_origin(model.model_fields[discriminator_field_name].annotation) is not Literal:
            raise InvalidDiscriminatorFieldError(f"🛑 '{discriminator_field_name}' must be a Literal type.")
        discriminator_field_values.extend(get_args(model.model_fields[discriminator_field_name].annotation))
    return StrEnum(enum_name, {v.replace("-", "_").upper(): v for v in set(discriminator_field_values)})


def get_sampler_params() -> dict[str, Type[BaseModel]]:
    """Returns a dictionary of sampler parameter classes."""
    params_cls_list = [
        params_cls
        for cls_name, params_cls in inspect.getmembers(sampler_params, inspect.isclass)
        if cls_name.endswith("SamplerParams")
    ]

    params_cls_dict = {}

    for source in sampler_params.SamplerType:
        source_name = source.value.replace("_", "")
        # Iterate in reverse order so the shortest match is first.
        # This is necessary for params that start with the same name.
        # For example, "bernoulli" and "bernoulli_mixture".
        params_cls_dict[source.value] = [
            params_cls
            for params_cls in reversed(params_cls_list)
            # Match param type string with parameter class.
            # For example, "gaussian" -> "GaussianSamplerParams"
            if source_name == params_cls.__name__.lower()[: len(source_name)]
            # Take the first match.
        ][0]

    return params_cls_dict


def resolve_string_enum(enum_instance: Any, enum_type: Type[Enum]) -> Enum:
    if not issubclass(enum_type, Enum):
        raise InvalidEnumValueError(f"🛑 `enum_type` must be a subclass of Enum. You provided: {enum_type}")
    invalid_enum_value_error = InvalidEnumValueError(
        f"🛑 '{enum_instance}' is not a valid string enum of type {type(enum_type)}. "
        f"Valid options are: {[option.value for option in enum_type]}"
    )
    if isinstance(enum_instance, enum_type):
        return enum_instance
    elif isinstance(enum_instance, str):
        try:
            return enum_type(enum_instance)
        except ValueError:
            raise invalid_enum_value_error
    else:
        raise invalid_enum_value_error


SAMPLER_PARAMS = get_sampler_params()
