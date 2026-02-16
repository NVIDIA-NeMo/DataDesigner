# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from enum import Enum
from typing import Any, Literal, get_args, get_origin

import data_designer
import data_designer.config as dd
import data_designer.interface as interface_mod
from data_designer.config.preview_results import PreviewResults
from data_designer.config.run_config import RunConfig

logger = logging.getLogger(__name__)


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


def _walk_namespace(
    package_path: list[str],
    prefix: str,
    max_depth: int,
    current_depth: int,
    import_errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Recursively walk a namespace package and build a tree of children nodes.

    Import failures are appended to import_errors as {"module": full_name, "message": str}.
    """
    if current_depth >= max_depth:
        return []

    children: list[dict[str, Any]] = []
    for importer, name, is_pkg in pkgutil.iter_modules(package_path):
        node: dict[str, Any] = {
            "name": name,
            "is_package": is_pkg,
            "children": [],
        }
        if is_pkg:
            full_name = f"{prefix}.{name}"
            try:
                sub_mod = importlib.import_module(full_name)
                sub_path = getattr(sub_mod, "__path__", [])
                node["children"] = _walk_namespace(
                    list(sub_path), full_name, max_depth, current_depth + 1, import_errors
                )
            except Exception as e:
                logger.debug("Failed to import %s during namespace discovery.", full_name, exc_info=True)
                import_errors.append({"module": full_name, "message": str(e)})
        children.append(node)

    children.sort(key=lambda n: (not n["is_package"], n["name"]))
    return children


def discover_namespace_tree(max_depth: int = 2) -> dict[str, Any]:
    """Walk the data_designer namespace and return install paths plus a module tree.

    Returns:
        Dict with ``paths`` (list of install directories) and ``tree`` (nested node dict).

    Raises:
        ValueError: If max_depth < 0.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0.")
    paths = list(data_designer.__path__)
    import_errors: list[dict[str, str]] = []
    tree: dict[str, Any] = {
        "name": "data_designer",
        "is_package": True,
        "children": _walk_namespace(paths, "data_designer", max_depth, 0, import_errors),
    }
    result: dict[str, Any] = {"paths": paths, "tree": tree}
    if import_errors:
        result["import_errors"] = import_errors
    return result


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


def discover_model_configs() -> dict[str, type]:
    """Return model-related configuration classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """
    return _discover_by_modules("models")


def discover_constraint_types() -> dict[str, type]:
    """Return constraint-related classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """
    return _discover_by_modules("sampler_constraints")


def discover_seed_types() -> dict[str, type]:
    """Return seed dataset-related classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """
    return _discover_by_modules("seed", "seed_source")


def discover_mcp_types() -> dict[str, type]:
    """Return MCP-related classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """
    return _discover_by_modules("mcp")


def discover_interface_classes() -> dict[str, type]:
    """Discover interface-layer classes plus config-layer types used in the interface workflow.

    Dynamically scans ``data_designer.interface.__all__`` for non-exception classes and
    adds ``PreviewResults`` and ``RunConfig`` from the config layer.

    Returns:
        Dict mapping class names to their types.
    """
    result: dict[str, type] = {}
    for name in getattr(interface_mod, "__all__", []):
        obj = getattr(interface_mod, name, None)
        if obj is not None and inspect.isclass(obj) and not issubclass(obj, Exception):
            result[name] = obj
    result["PreviewResults"] = PreviewResults
    result["RunConfig"] = RunConfig
    return result


_MODULE_CATEGORIES: dict[str, str] = {
    "column_configs": "Column Configs",
    "column_types": "Column Types",
    "config_builder": "Builder",
    "custom_column": "Custom Columns",
    "data_designer_config": "Core Config",
    "mcp": "MCP",
    "models": "Model Configs",
    "processors": "Processors",
    "run_config": "Runtime Config",
    "sampler_constraints": "Constraints",
    "sampler_params": "Sampler Params",
    "seed": "Seed Config",
    "seed_source": "Seed Sources",
    "validator_params": "Validator Params",
    "analysis.column_profilers": "Analysis",
    "utils": "Utilities",
    "version": "Utilities",
}


def _categorize_module(module_path: str) -> str:
    """Map a module path from _LAZY_IMPORTS to a human-readable category name."""
    prefix = "data_designer.config."
    suffix = module_path.removeprefix(prefix) if module_path.startswith(prefix) else module_path

    for key, category in _MODULE_CATEGORIES.items():
        if suffix == key or suffix.startswith(key + "."):
            return category
    return "Other"


def discover_importable_names() -> dict[str, list[dict[str, str]]]:
    """Discover all importable names from data_designer.config and data_designer.interface.

    Reads _LAZY_IMPORTS from the config module and __all__ from the interface module,
    grouping names by source-module category.

    Returns:
        Dict mapping category names to lists of ``{"name": str, "module": str}`` entries.
    """
    lazy_imports: dict[str, tuple[str, str]] = getattr(dd, "_LAZY_IMPORTS", {})

    categories: dict[str, list[dict[str, str]]] = {}
    for name, (module_path, _attr) in sorted(lazy_imports.items()):
        category = _categorize_module(module_path)
        categories.setdefault(category, []).append({"name": name, "module": "data_designer.config"})

    interface_all: list[str] = getattr(interface_mod, "__all__", [])
    if interface_all:
        categories["Interface"] = [{"name": n, "module": "data_designer.interface"} for n in sorted(interface_all)]

    return categories
