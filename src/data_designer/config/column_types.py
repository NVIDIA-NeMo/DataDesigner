# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from typing_extensions import TypeAlias

from ..plugin_manager import PluginManager
from .column_configs import (
    EmbeddingColumnConfig,
    ExpressionColumnConfig,
    ImageGenerationColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
)
from .errors import InvalidColumnTypeError, InvalidConfigError
from .sampler_params import SamplerType
from .utils.type_helpers import SAMPLER_PARAMS, create_str_enum_from_discriminated_type_union, resolve_string_enum

plugin_manager = PluginManager()

ColumnConfigT: TypeAlias = Union[
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
    EmbeddingColumnConfig,
    ImageGenerationColumnConfig,
]
ColumnConfigT = plugin_manager.inject_into_column_config_type_union(ColumnConfigT)

DataDesignerColumnType = create_str_enum_from_discriminated_type_union(
    enum_name="DataDesignerColumnType",
    type_union=ColumnConfigT,
    discriminator_field_name="column_type",
)

COLUMN_TYPE_EMOJI_MAP = {
    "general": "⚛️",  # possible analysis column type
    DataDesignerColumnType.EXPRESSION: "🧩",
    DataDesignerColumnType.LLM_CODE: "💻",
    DataDesignerColumnType.LLM_JUDGE: "⚖️",
    DataDesignerColumnType.LLM_STRUCTURED: "🗂️",
    DataDesignerColumnType.LLM_TEXT: "📝",
    DataDesignerColumnType.SEED_DATASET: "🌱",
    DataDesignerColumnType.SAMPLER: "🎲",
    DataDesignerColumnType.VALIDATION: "🔍",
    DataDesignerColumnType.EMBEDDING: "🧬",
    DataDesignerColumnType.IMAGE_GENERATION: "🖼️",
}
COLUMN_TYPE_EMOJI_MAP.update(
    {DataDesignerColumnType(p.name): p.emoji for p in plugin_manager.get_column_generator_plugins()}
)


def column_type_used_in_execution_dag(column_type: Union[str, DataDesignerColumnType]) -> bool:
    """Return True if the column type is used in the workflow execution DAG."""
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    dag_column_types = {
        DataDesignerColumnType.EXPRESSION,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.VALIDATION,
        DataDesignerColumnType.EMBEDDING,
        DataDesignerColumnType.IMAGE_GENERATION,
    }
    dag_column_types.update(plugin_manager.get_plugin_column_types(DataDesignerColumnType))
    return column_type in dag_column_types


def column_type_is_llm_generated(column_type: Union[str, DataDesignerColumnType]) -> bool:
    """Return True if the column type is an LLM-generated column."""
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    llm_generated_column_types = {
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.EMBEDDING,
        DataDesignerColumnType.IMAGE_GENERATION,
    }
    llm_generated_column_types.update(
        plugin_manager.get_plugin_column_types(
            DataDesignerColumnType,
            required_resources=["model_registry"],
        )
    )
    return column_type in llm_generated_column_types


def get_column_config_from_kwargs(name: str, column_type: DataDesignerColumnType, **kwargs) -> ColumnConfigT:
    """Create a Data Designer column config object from kwargs.

    Args:
        name: Name of the column.
        column_type: Type of the column.
        **kwargs: Keyword arguments to pass to the column constructor.

    Returns:
        Data Designer column object of the appropriate type.
    """
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    if column_type == DataDesignerColumnType.LLM_TEXT:
        return LLMTextColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.LLM_CODE:
        return LLMCodeColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.LLM_STRUCTURED:
        return LLMStructuredColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.LLM_JUDGE:
        return LLMJudgeColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.VALIDATION:
        return ValidationColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.EXPRESSION:
        return ExpressionColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.SAMPLER:
        return SamplerColumnConfig(name=name, **_resolve_sampler_kwargs(name, kwargs))
    if column_type == DataDesignerColumnType.SEED_DATASET:
        return SeedDatasetColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.EMBEDDING:
        return EmbeddingColumnConfig(name=name, **kwargs)
    if column_type == DataDesignerColumnType.IMAGE_GENERATION:
        return ImageGenerationColumnConfig(name=name, **kwargs)
    if plugin := plugin_manager.get_column_generator_plugin_if_exists(column_type.value):
        return plugin.config_cls(name=name, **kwargs)
    raise InvalidColumnTypeError(f"🛑 {column_type} is not a valid column type.")  # pragma: no cover


def get_column_display_order() -> list[DataDesignerColumnType]:
    """Return the preferred display order of the column types."""
    display_order = [
        DataDesignerColumnType.SEED_DATASET,
        DataDesignerColumnType.SAMPLER,
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.EMBEDDING,
        DataDesignerColumnType.IMAGE_GENERATION,
        DataDesignerColumnType.VALIDATION,
        DataDesignerColumnType.EXPRESSION,
    ]
    display_order.extend(plugin_manager.get_plugin_column_types(DataDesignerColumnType))
    return display_order


def _resolve_sampler_kwargs(name: str, kwargs: dict) -> dict:
    if "sampler_type" not in kwargs:
        raise InvalidConfigError(f"🛑 `sampler_type` is required for sampler column '{name}'.")
    sampler_type = resolve_string_enum(kwargs["sampler_type"], SamplerType)

    # Handle params - it could be a dict or already a concrete object
    params_value = kwargs.get("params", {})
    expected_params_class = SAMPLER_PARAMS[sampler_type.value]

    if isinstance(params_value, expected_params_class):
        # params is already a concrete object of the right type
        params = params_value
    elif isinstance(params_value, dict):
        # params is a dictionary, create new instance
        params = expected_params_class(**params_value)
    else:
        # params is neither dict nor expected type
        raise InvalidConfigError(
            f"🛑 Invalid params for sampler column '{name}'. "
            f"Expected a dictionary or an instance of {expected_params_class.__name__}. "
            f"You provided {params_value=}."
        )

    return {
        "sampler_type": sampler_type,
        "params": params,
        **{k: v for k, v in kwargs.items() if k not in ["sampler_type", "params"]},
    }
