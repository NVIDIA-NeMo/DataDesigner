# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Literal, get_args, get_origin

import data_designer.config as dd


def _extract_literal_discriminator_value(annotation: Any) -> str | None:
    """Extract the first literal discriminator value from a type annotation.

    Supports ``Literal["value"]`` and ``Literal[SomeEnum.MEMBER]``.
    Returns ``None`` when the annotation is not a literal discriminator.
    """
    if get_origin(annotation) is not Literal:
        return None

    args = get_args(annotation)
    if not args:
        return None

    value = args[0]
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _discover_configs_by_discriminator(
    class_name_suffix: str,
    discriminator_field: str,
    exclude_class_names: set[str] | None = None,
) -> dict[str, type]:
    """Discover config classes whose discriminator field is a Literal value.

    Args:
        class_name_suffix: Class-name suffix to select candidate classes.
        discriminator_field: Pydantic field name containing the discriminator.
        exclude_class_names: Optional set of class names to skip.

    Returns:
        Dict mapping discriminator values to config classes.
    """
    excluded = exclude_class_names or set()
    discovered: dict[str, type] = {}

    for name in dir(dd):
        if name in excluded or not name.endswith(class_name_suffix):
            continue

        obj = getattr(dd, name)
        if not (inspect.isclass(obj) and hasattr(obj, "model_fields")):
            continue
        if discriminator_field not in obj.model_fields:
            continue

        annotation = obj.model_fields[discriminator_field].annotation
        discriminator_value = _extract_literal_discriminator_value(annotation)
        if discriminator_value is not None:
            discovered[discriminator_value] = obj

    return discovered


def _discover_params_by_discriminator(
    params_class_suffix: str,
    discriminator_field: str,
    enum_name: str,
) -> dict[str, type]:
    """Discover params classes keyed by their literal discriminator value.

    Args:
        params_class_suffix: Class-name suffix to select params classes.
        discriminator_field: Field name that stores the literal discriminator.

    Returns:
        Dict mapping discriminator values to params classes.
    """
    discovered: dict[str, type] = {}
    normalized_name_map: dict[str, type] = {}

    for name in dir(dd):
        if not name.endswith(params_class_suffix):
            continue

        obj = getattr(dd, name)
        if not (inspect.isclass(obj) and hasattr(obj, "model_fields")):
            continue

        if discriminator_field in obj.model_fields:
            annotation = obj.model_fields[discriminator_field].annotation
            discriminator_value = _extract_literal_discriminator_value(annotation)
            if discriminator_value is not None:
                discovered[discriminator_value] = obj
                continue

        normalized_name = name.removesuffix(params_class_suffix).replace("_", "").lower()
        normalized_name_map[normalized_name] = obj

    enum_cls = getattr(dd, enum_name, None)
    if enum_cls is None or not (inspect.isclass(enum_cls) and issubclass(enum_cls, Enum)):
        return discovered

    for member in enum_cls:
        value = str(member.value)
        if value in discovered:
            continue
        normalized_value = value.replace("_", "").lower()
        params_cls = normalized_name_map.get(normalized_value)
        if params_cls is not None:
            discovered[value] = params_cls

    return discovered


def discover_column_configs() -> dict[str, type]:
    """Dynamically discover all ColumnConfig classes from data_designer.config.

    Returns:
        Dict mapping column_type literal values (e.g., 'llm-text') to their config classes.
    """
    return _discover_configs_by_discriminator(
        class_name_suffix="ColumnConfig",
        discriminator_field="column_type",
    )


def discover_sampler_types() -> dict[str, type]:
    """Dynamically discover sampler types and params classes from data_designer.config.

    Returns:
        Dict mapping sampler type names (e.g., 'category') to their params classes.
    """
    return _discover_params_by_discriminator(
        params_class_suffix="SamplerParams",
        discriminator_field="sampler_type",
        enum_name="SamplerType",
    )


def discover_validator_types() -> dict[str, type]:
    """Dynamically discover validator types and params classes from data_designer.config.

    Returns:
        Dict mapping validator type names to their params classes.
    """
    return _discover_params_by_discriminator(
        params_class_suffix="ValidatorParams",
        discriminator_field="validator_type",
        enum_name="ValidatorType",
    )


def discover_processor_configs() -> dict[str, type]:
    """Dynamically discover all ProcessorConfig classes from data_designer.config.

    Returns:
        Dict mapping processor_type values to their config classes.
    """
    return _discover_configs_by_discriminator(
        class_name_suffix="ProcessorConfig",
        discriminator_field="processor_type",
        exclude_class_names={"ProcessorConfig"},
    )


def _discover_by_modules(*module_suffixes: str) -> dict[str, type]:
    """Discover config types by filtering _LAZY_IMPORTS on source-module suffix.

    Args:
        module_suffixes: One or more module suffixes to match against
            (e.g., ``"models"``, ``"seed"``).

    Returns:
        Dict mapping class/object names to their resolved types.
    """
    lazy_imports: dict[str, tuple[str, str]] = getattr(dd, "_LAZY_IMPORTS", {})
    prefix = "data_designer.config."
    result: dict[str, type] = {}
    for name, (module_path, _attr) in lazy_imports.items():
        suffix = module_path.removeprefix(prefix) if module_path.startswith(prefix) else module_path
        if suffix in module_suffixes:
            obj = getattr(dd, name, None)
            if obj is not None:
                result[name] = obj
    return result


def discover_constraint_types() -> dict[str, type]:
    """Return constraint-related classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """
    return _discover_by_modules("sampler_constraints")
