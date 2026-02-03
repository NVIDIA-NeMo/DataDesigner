# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.column_types import DataDesignerColumnType
from data_designer.engine.column_generators.utils.generator_classification import (
    column_type_is_model_generated,
    column_type_used_in_execution_dag,
)


def test_column_type_is_model_generated() -> None:
    assert column_type_is_model_generated(DataDesignerColumnType.LLM_TEXT)
    assert column_type_is_model_generated(DataDesignerColumnType.LLM_CODE)
    assert column_type_is_model_generated(DataDesignerColumnType.LLM_STRUCTURED)
    assert column_type_is_model_generated(DataDesignerColumnType.LLM_JUDGE)
    assert column_type_is_model_generated(DataDesignerColumnType.EMBEDDING)
    assert not column_type_is_model_generated(DataDesignerColumnType.SAMPLER)
    assert not column_type_is_model_generated(DataDesignerColumnType.VALIDATION)
    assert not column_type_is_model_generated(DataDesignerColumnType.EXPRESSION)
    assert not column_type_is_model_generated(DataDesignerColumnType.SEED_DATASET)


def test_column_type_used_in_execution_dag() -> None:
    assert column_type_used_in_execution_dag(DataDesignerColumnType.EXPRESSION)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_CODE)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_JUDGE)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_STRUCTURED)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.LLM_TEXT)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.VALIDATION)
    assert column_type_used_in_execution_dag(DataDesignerColumnType.EMBEDDING)
    assert not column_type_used_in_execution_dag(DataDesignerColumnType.SAMPLER)
    assert not column_type_used_in_execution_dag(DataDesignerColumnType.SEED_DATASET)
