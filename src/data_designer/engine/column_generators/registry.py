# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    ValidationColumnConfig,
)
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.engine.column_generators.generators.base import ColumnGenerator
from data_designer.engine.column_generators.generators.expression import ExpressionColumnGenerator
from data_designer.engine.column_generators.generators.llm_generators import (
    LLMCodeCellGenerator,
    LLMJudgeCellGenerator,
    LLMStructuredCellGenerator,
    LLMTextCellGenerator,
)
from data_designer.engine.column_generators.generators.samplers import SamplerColumnGenerator
from data_designer.engine.column_generators.generators.seed_dataset import SeedDatasetColumnGenerator
from data_designer.engine.column_generators.generators.validation import ValidationColumnGenerator
from data_designer.engine.dataset_builders.multi_column_configs import (
    SamplerMultiColumnConfig,
    SeedDatasetMultiColumnConfig,
)
from data_designer.engine.registry.base import TaskRegistry


class ColumnGeneratorRegistry(TaskRegistry[DataDesignerColumnType, ColumnGenerator, ConfigBase]): ...


def create_builtin_column_generator_registry() -> ColumnGeneratorRegistry:
    registry = ColumnGeneratorRegistry()
    registry.register(DataDesignerColumnType.LLM_TEXT, LLMTextCellGenerator, LLMTextColumnConfig)
    registry.register(DataDesignerColumnType.LLM_CODE, LLMCodeCellGenerator, LLMCodeColumnConfig)
    registry.register(DataDesignerColumnType.LLM_JUDGE, LLMJudgeCellGenerator, LLMJudgeColumnConfig)
    registry.register(DataDesignerColumnType.EXPRESSION, ExpressionColumnGenerator, ExpressionColumnConfig)
    registry.register(DataDesignerColumnType.SAMPLER, SamplerColumnGenerator, SamplerMultiColumnConfig)
    registry.register(DataDesignerColumnType.SEED_DATASET, SeedDatasetColumnGenerator, SeedDatasetMultiColumnConfig)
    registry.register(DataDesignerColumnType.VALIDATION, ValidationColumnGenerator, ValidationColumnConfig)
    registry.register(DataDesignerColumnType.LLM_STRUCTURED, LLMStructuredCellGenerator, LLMStructuredColumnConfig)
    return registry
