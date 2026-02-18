# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON schema generation from DataDesigner Pydantic models.

Exposes Pydantic-derived JSON schemas to the frontend so that forms can be
rendered dynamically for each column type, model config, seed config, etc.
"""

from __future__ import annotations

from typing import Any

from data_designer.config.column_configs import (
    EmbeddingColumnConfig,
    ExpressionColumnConfig,
    ImageColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
)
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.mcp import ToolConfig
from data_designer.config.models import ModelConfig
from data_designer.config.processors import DropColumnsProcessorConfig, SchemaTransformProcessorConfig
from data_designer.config.sampler_constraints import ColumnInequalityConstraint, ScalarInequalityConstraint
from data_designer.config.sampler_params import SamplerType
from data_designer.config.seed import SeedConfig
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.trace_type import TraceType
from data_designer.config.validator_params import ValidatorType

_COLUMN_CONFIG_CLASSES = {
    "sampler": SamplerColumnConfig,
    "llm-text": LLMTextColumnConfig,
    "llm-code": LLMCodeColumnConfig,
    "llm-structured": LLMStructuredColumnConfig,
    "llm-judge": LLMJudgeColumnConfig,
    "expression": ExpressionColumnConfig,
    "validation": ValidationColumnConfig,
    "seed-dataset": SeedDatasetColumnConfig,
    "embedding": EmbeddingColumnConfig,
    "image": ImageColumnConfig,
}


def get_column_schemas() -> dict[str, Any]:
    """Return JSON schemas for all column types keyed by column_type discriminator."""
    return {key: cls.model_json_schema() for key, cls in _COLUMN_CONFIG_CLASSES.items()}


def get_model_config_schema() -> dict[str, Any]:
    return ModelConfig.model_json_schema()


def get_seed_config_schema() -> dict[str, Any]:
    return SeedConfig.model_json_schema()


def get_tool_config_schema() -> dict[str, Any]:
    return ToolConfig.model_json_schema()


def get_constraint_schemas() -> dict[str, Any]:
    return {
        "scalar_inequality": ScalarInequalityConstraint.model_json_schema(),
        "column_inequality": ColumnInequalityConstraint.model_json_schema(),
    }


def get_processor_schemas() -> dict[str, Any]:
    return {
        "drop_columns": DropColumnsProcessorConfig.model_json_schema(),
        "schema_transform": SchemaTransformProcessorConfig.model_json_schema(),
    }


def get_full_config_schema() -> dict[str, Any]:
    return DataDesignerConfig.model_json_schema()


def get_enum_values() -> dict[str, list[str]]:
    """Return enum values for dropdowns in the frontend."""
    return {
        "sampler_types": [e.value for e in SamplerType],
        "code_languages": [e.value for e in CodeLang],
        "trace_types": [e.value for e in TraceType],
        "validator_types": [e.value for e in ValidatorType],
    }
