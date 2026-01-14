# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.utils.type_helpers import resolve_string_enum
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry
from data_designer.plugin_manager import PluginManager

plugin_manager = PluginManager()


def column_type_used_in_execution_dag(column_type: str | DataDesignerColumnType) -> bool:
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
    }
    dag_column_types.update(plugin_manager.get_plugin_column_types(DataDesignerColumnType))
    return column_type in dag_column_types


def column_type_is_model_generated(column_type: str | DataDesignerColumnType) -> bool:
    """Return True if the column type is a model-generated column."""
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    model_generated_column_types = {
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.EMBEDDING,
    }
    for plugin in plugin_manager.get_column_generator_plugins():
        if issubclass(plugin.impl_cls, ColumnGeneratorWithModelRegistry):
            model_generated_column_types.add(plugin.name)
    return column_type in model_generated_column_types
