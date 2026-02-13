# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from enum import Enum
from typing import Literal, get_args, get_origin


def discover_column_configs() -> dict[str, type]:
    """Dynamically discover all ColumnConfig classes from data_designer.config.

    Returns:
        Dict mapping column_type literal values (e.g., 'llm-text') to their config classes.
    """
    import data_designer.config as dd

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
    import data_designer.config as dd

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
    import data_designer.config as dd

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
    import data_designer.config as dd

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
    import data_designer.config as dd

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
    import data_designer.config as dd

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
    import data_designer.config as dd

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
    import data_designer.config as dd

    return {
        "MCPProvider": dd.MCPProvider,
        "LocalStdioMCPProvider": dd.LocalStdioMCPProvider,
        "ToolConfig": dd.ToolConfig,
    }
