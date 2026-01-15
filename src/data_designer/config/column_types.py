# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing_extensions import TypeAlias

from data_designer.config.column_configs import (
    EmbeddingColumnConfig,
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
)
from data_designer.config.errors import InvalidConfigError
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.type_helpers import (
    SAMPLER_PARAMS,
    create_str_enum_from_discriminated_type_union,
    resolve_string_enum,
)
from data_designer.plugin_manager import PluginManager

plugin_manager = PluginManager()

ColumnConfigT: TypeAlias = (
    ExpressionColumnConfig
    | LLMCodeColumnConfig
    | LLMJudgeColumnConfig
    | LLMStructuredColumnConfig
    | LLMTextColumnConfig
    | SamplerColumnConfig
    | SeedDatasetColumnConfig
    | ValidationColumnConfig
    | EmbeddingColumnConfig
)
ColumnConfigT = plugin_manager.inject_into_column_config_type_union(ColumnConfigT)

DataDesignerColumnType = create_str_enum_from_discriminated_type_union(
    enum_name="DataDesignerColumnType",
    type_union=ColumnConfigT,
    discriminator_field_name="column_type",
)


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
    config_cls = get_column_config_cls_from_type(column_type)
    if column_type == DataDesignerColumnType.SAMPLER:
        kwargs = _resolve_sampler_kwargs(name, kwargs)
    return config_cls(name=name, **kwargs)


def get_column_config_cls_from_type(column_type: DataDesignerColumnType) -> type[ColumnConfigT]:
    """Get the column config class for a column type."""
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    if column_type in _COLUMN_TYPE_CONFIG_CLS_MAP:
        return _COLUMN_TYPE_CONFIG_CLS_MAP[column_type]
    if plugin := plugin_manager.get_column_generator_plugin_if_exists(column_type.value):
        return plugin.config_cls
    raise InvalidConfigError(f"🛑 {column_type} is not a valid column type.")


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
        DataDesignerColumnType.VALIDATION,
        DataDesignerColumnType.EXPRESSION,
    ]
    display_order.extend(plugin_manager.get_plugin_column_types(DataDesignerColumnType))
    return display_order


def get_column_emoji_from_type(column_type: DataDesignerColumnType) -> str:
    """Get the emoji for a column type."""
    config_cls = get_column_config_cls_from_type(resolve_string_enum(column_type, DataDesignerColumnType))
    return config_cls.get_column_emoji()


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


_COLUMN_TYPE_CONFIG_CLS_MAP = {
    DataDesignerColumnType.LLM_TEXT: LLMTextColumnConfig,
    DataDesignerColumnType.LLM_CODE: LLMCodeColumnConfig,
    DataDesignerColumnType.LLM_STRUCTURED: LLMStructuredColumnConfig,
    DataDesignerColumnType.LLM_JUDGE: LLMJudgeColumnConfig,
    DataDesignerColumnType.VALIDATION: ValidationColumnConfig,
    DataDesignerColumnType.EXPRESSION: ExpressionColumnConfig,
    DataDesignerColumnType.SAMPLER: SamplerColumnConfig,
    DataDesignerColumnType.SEED_DATASET: SeedDatasetColumnConfig,
    DataDesignerColumnType.EMBEDDING: EmbeddingColumnConfig,
}
