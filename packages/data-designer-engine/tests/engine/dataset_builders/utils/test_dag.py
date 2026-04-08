# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from data_designer.config.column_configs import (
    CustomColumnConfig,
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    Score,
    ValidationColumnConfig,
)
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
from data_designer.engine.dataset_builders.utils.errors import ConfigCompilationError, DAGCircularDependencyError
from data_designer.engine.dataset_builders.utils.execution_graph import topologically_sort_column_configs

MODEL_ALIAS = "stub-model-alias"


def test_dag_construction():
    column_configs = []
    column_configs.append(
        SamplerMultiColumnConfig(
            columns=[SamplerColumnConfig(name="test_id", sampler_type=SamplerType.UUID, params={})]
        )
    )
    column_configs.append(
        LLMCodeColumnConfig(
            name="test_code",
            prompt="Write some zig but call it Python.",
            code_lang=CodeLang.PYTHON,
            model_alias=MODEL_ALIAS,
        )
    )
    column_configs.append(
        LLMCodeColumnConfig(
            name="depends_on_validation",
            prompt="Write {{ test_validation.python_linter_score }}.",
            code_lang=CodeLang.PYTHON,
            model_alias=MODEL_ALIAS,
        )
    )
    column_configs.append(
        LLMJudgeColumnConfig(
            name="test_judge",
            prompt="Judge this {{ test_code }} {{ depends_on_validation }}",
            scores=[Score(name="test_score", description="test", options={0: "Not Good", 1: "Good"})],
            model_alias=MODEL_ALIAS,
        )
    )
    column_configs.append(
        ExpressionColumnConfig(
            name="uses_all_the_stuff", expr="{{ test_code }} {{ depends_on_validation }} {{ test_judge }}"
        )
    )
    column_configs.append(
        ExpressionColumnConfig(
            name="test_code_and_depends_on_validation_reasoning_traces",
            expr="{{ test_code__trace }} {{ depends_on_validation }}",
        )
    )
    column_configs.append(
        ValidationColumnConfig(
            name="test_validation",
            target_columns=["test_code"],
            validator_type="code",
            validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        )
    )

    sorted_column_configs = topologically_sort_column_configs(column_configs)

    assert sorted_column_configs[0].column_type == DataDesignerColumnType.SAMPLER

    names = [c.name for c in sorted_column_configs[1:]]
    assert names[0] == "test_code"
    assert names[1] == "test_validation"
    assert names[2] == "depends_on_validation"
    # test_judge and test_code_and_depends_on_validation_reasoning_traces have no mutual
    # dependency, so their relative order is not guaranteed by topological sort.
    assert set(names[3:5]) == {"test_judge", "test_code_and_depends_on_validation_reasoning_traces"}
    assert names[5] == "uses_all_the_stuff"


def test_circular_dependencies():
    column_configs = []
    column_configs.append(
        SamplerMultiColumnConfig(
            columns=[SamplerColumnConfig(name="test_id", sampler_type=SamplerType.UUID, params={})]
        )
    )
    column_configs.append(
        LLMTextColumnConfig(
            name="col_1",
            prompt="I need you {{ col_2 }}",
            model_alias=MODEL_ALIAS,
        )
    )
    column_configs.append(
        LLMTextColumnConfig(
            name="col_2",
            prompt="I need you {{ col_1 }}",
            model_alias=MODEL_ALIAS,
        )
    )
    with pytest.raises(DAGCircularDependencyError, match="cyclic dependencies"):
        topologically_sort_column_configs(column_configs)


def test_duplicate_side_effect_producers_raises() -> None:
    """Two custom columns declaring the same side-effect column is a configuration error."""

    @custom_column_generator(required_columns=["text"], side_effect_columns=["shared_col"])
    def gen_a(row: dict[str, Any]) -> dict[str, Any]:
        return row

    @custom_column_generator(required_columns=["text"], side_effect_columns=["shared_col"])
    def gen_b(row: dict[str, Any]) -> dict[str, Any]:
        return row

    column_configs = [
        LLMTextColumnConfig(name="text", prompt="hello", model_alias=MODEL_ALIAS),
        CustomColumnConfig(name="col_a", generator_function=gen_a),
        CustomColumnConfig(name="col_b", generator_function=gen_b),
    ]
    with pytest.raises(ConfigCompilationError, match="already produced by"):
        topologically_sort_column_configs(column_configs)
