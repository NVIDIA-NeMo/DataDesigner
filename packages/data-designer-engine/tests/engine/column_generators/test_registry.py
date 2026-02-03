# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.column_types import DataDesignerColumnType
from data_designer.engine.column_generators.generators.expression import ExpressionColumnGenerator
from data_designer.engine.column_generators.generators.llm_completion import (
    LLMCodeCellGenerator,
    LLMJudgeCellGenerator,
    LLMStructuredCellGenerator,
    LLMTextCellGenerator,
)
from data_designer.engine.column_generators.generators.samplers import SamplerColumnGenerator
from data_designer.engine.column_generators.generators.seed_dataset import SeedDatasetColumnGenerator
from data_designer.engine.column_generators.generators.validation import ValidationColumnGenerator
from data_designer.engine.column_generators.registry import (
    ColumnGeneratorRegistry,
    create_default_column_generator_registry,
)


def test_column_generator_registry_create_default_registry_with_generators():
    registry = create_default_column_generator_registry()

    assert isinstance(registry, ColumnGeneratorRegistry)

    expected_generators = {
        DataDesignerColumnType.LLM_TEXT: LLMTextCellGenerator,
        DataDesignerColumnType.LLM_CODE: LLMCodeCellGenerator,
        DataDesignerColumnType.LLM_JUDGE: LLMJudgeCellGenerator,
        DataDesignerColumnType.EXPRESSION: ExpressionColumnGenerator,
        DataDesignerColumnType.SAMPLER: SamplerColumnGenerator,
        DataDesignerColumnType.SEED_DATASET: SeedDatasetColumnGenerator,
        DataDesignerColumnType.VALIDATION: ValidationColumnGenerator,
        DataDesignerColumnType.LLM_STRUCTURED: LLMStructuredCellGenerator,
    }

    for column_type, expected_generator in expected_generators.items():
        assert column_type in registry._registry
        assert registry._registry[column_type] == expected_generator
