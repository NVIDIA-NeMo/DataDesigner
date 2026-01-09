# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    Score,
    ValidationColumnConfig,
)
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
from data_designer.engine.dataset_builders.utils.dag import topologically_sort_column_configs
from data_designer.engine.dataset_builders.utils.errors import DAGCircularDependencyError

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
            expr="{{ test_code__reasoning_trace }} {{ depends_on_validation }}",
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

    assert [c.name for c in sorted_column_configs[1:]] == [
        "test_code",
        "test_validation",
        "depends_on_validation",
        "test_judge",
        "test_code_and_depends_on_validation_reasoning_traces",
        "uses_all_the_stuff",
    ]


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
