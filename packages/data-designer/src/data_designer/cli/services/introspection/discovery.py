# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import pkgutil
from enum import Enum
from typing import Any, Literal, get_args, get_origin

import data_designer
import data_designer.config as dd
import data_designer.interface as interface_mod
from data_designer.config.preview_results import PreviewResults
from data_designer.config.run_config import RunConfig
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.results import DatasetCreationResults


def _walk_namespace(package_path: list[str], prefix: str, max_depth: int, current_depth: int) -> list[dict[str, Any]]:
    """Recursively walk a namespace package and build a tree of children nodes."""
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
                node["children"] = _walk_namespace(list(sub_path), full_name, max_depth, current_depth + 1)
            except Exception:
                pass
        children.append(node)

    children.sort(key=lambda n: (not n["is_package"], n["name"]))
    return children


def discover_namespace_tree(max_depth: int = 2) -> dict[str, Any]:
    """Walk the data_designer namespace and return install paths plus a module tree.

    Returns:
        Dict with ``paths`` (list of install directories) and ``tree`` (nested node dict).
    """
    paths = list(data_designer.__path__)
    tree: dict[str, Any] = {
        "name": "data_designer",
        "is_package": True,
        "children": _walk_namespace(paths, "data_designer", max_depth, 0),
    }
    return {"paths": paths, "tree": tree}


def discover_column_configs() -> dict[str, type]:
    """Dynamically discover all ColumnConfig classes from data_designer.config.

    Returns:
        Dict mapping column_type literal values (e.g., 'llm-text') to their config classes.
    """

    column_configs: dict[str, type] = {}
    for name in dir(dd):
        if name.endswith("ColumnConfig"):
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                if "column_type" in obj.model_fields:
                    annotation = obj.model_fields["column_type"].annotation
                    if get_origin(annotation) is Literal:
                        args = get_args(annotation)
                        if args:
                            column_configs[args[0]] = obj
    return column_configs


def discover_sampler_types() -> dict[str, type]:
    """Dynamically discover all sampler types and their param classes from data_designer.config.

    Returns:
        Dict mapping sampler type names (e.g., 'category') to their params classes.
    """

    sampler_type_enum = getattr(dd, "SamplerType", None)
    if sampler_type_enum is None or not issubclass(sampler_type_enum, Enum):
        return {}

    params_classes: dict[str, type] = {}
    for name in dir(dd):
        if name.endswith("SamplerParams"):
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                normalized = name.replace("SamplerParams", "").lower()
                params_classes[normalized] = obj

    sampler_types: dict[str, type] = {}
    for member in sampler_type_enum:
        sampler_name = member.name.lower()
        normalized_name = sampler_name.replace("_", "")
        params_cls = params_classes.get(normalized_name)
        if params_cls is not None:
            sampler_types[sampler_name] = params_cls

    return sampler_types


def discover_validator_types() -> dict[str, type]:
    """Dynamically discover all validator types and their param classes from data_designer.config.

    Returns:
        Dict mapping validator type names to their params classes.
    """

    validator_type_enum = getattr(dd, "ValidatorType", None)
    if validator_type_enum is None or not issubclass(validator_type_enum, Enum):
        return {}

    params_classes: dict[str, type] = {}
    for name in dir(dd):
        if name.endswith("ValidatorParams"):
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                normalized = name.replace("ValidatorParams", "").lower()
                params_classes[normalized] = obj

    validator_types: dict[str, type] = {}
    for member in validator_type_enum:
        validator_name = member.name.lower()
        normalized_name = validator_name.replace("_", "")
        params_cls = params_classes.get(normalized_name)
        if params_cls is not None:
            validator_types[validator_name] = params_cls

    return validator_types


def discover_processor_configs() -> dict[str, type]:
    """Dynamically discover all ProcessorConfig classes from data_designer.config.

    Returns:
        Dict mapping processor_type values to their config classes.
    """

    processor_configs: dict[str, type] = {}
    for name in dir(dd):
        if name.endswith("ProcessorConfig") and name != "ProcessorConfig":
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                if "processor_type" in obj.model_fields:
                    annotation = obj.model_fields["processor_type"].annotation
                    if get_origin(annotation) is Literal:
                        args = get_args(annotation)
                        if args:
                            key = args[0].value if isinstance(args[0], Enum) else args[0]
                            processor_configs[key] = obj
    return processor_configs


def discover_model_configs() -> dict[str, type]:
    """Return model-related configuration classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """

    return {
        "ModelConfig": dd.ModelConfig,
        "ChatCompletionInferenceParams": dd.ChatCompletionInferenceParams,
        "EmbeddingInferenceParams": dd.EmbeddingInferenceParams,
        "ImageInferenceParams": dd.ImageInferenceParams,
        "ImageContext": dd.ImageContext,
        "UniformDistribution": dd.UniformDistribution,
        "ManualDistribution": dd.ManualDistribution,
    }


def discover_constraint_types() -> dict[str, type]:
    """Return constraint-related classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """

    return {
        "ScalarInequalityConstraint": dd.ScalarInequalityConstraint,
        "ColumnInequalityConstraint": dd.ColumnInequalityConstraint,
        "InequalityOperator": dd.InequalityOperator,
    }


def discover_seed_types() -> dict[str, type]:
    """Return seed dataset-related classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """

    return {
        "SeedConfig": dd.SeedConfig,
        "SamplingStrategy": dd.SamplingStrategy,
        "LocalFileSeedSource": dd.LocalFileSeedSource,
        "HuggingFaceSeedSource": dd.HuggingFaceSeedSource,
        "DataFrameSeedSource": dd.DataFrameSeedSource,
        "IndexRange": dd.IndexRange,
        "PartitionBlock": dd.PartitionBlock,
    }


def discover_mcp_types() -> dict[str, type]:
    """Return MCP-related classes from data_designer.config.

    Returns:
        Dict mapping class names to their types.
    """

    return {
        "MCPProvider": dd.MCPProvider,
        "LocalStdioMCPProvider": dd.LocalStdioMCPProvider,
        "ToolConfig": dd.ToolConfig,
    }


def discover_interface_classes() -> dict[str, type]:
    """Return the key interface-layer classes an agent uses after building a config.

    Returns:
        Dict mapping class names to their types for DataDesigner, DatasetCreationResults,
        PreviewResults, and RunConfig.
    """
    return {
        "DataDesigner": DataDesigner,
        "DatasetCreationResults": DatasetCreationResults,
        "PreviewResults": PreviewResults,
        "RunConfig": RunConfig,
    }


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
