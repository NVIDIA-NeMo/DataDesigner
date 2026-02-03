# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig, SeedDatasetColumnConfig
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.seed import SamplingStrategy, SeedConfig
from data_designer.config.seed_source import HuggingFaceSeedSource
from data_designer.engine.dataset_builders.utils.config_compiler import (
    compile_dataset_builder_column_configs,
)
from data_designer.engine.dataset_builders.utils.errors import ConfigCompilationError


def test_compile_dataset_builder_column_configs_with_seed_columns():
    config = DataDesignerConfig(
        columns=[SeedDatasetColumnConfig(name="seed_col")],
        seed_config=SeedConfig(
            source=HuggingFaceSeedSource(path="hf://datasets/test/dataset"),
            sampling_strategy=SamplingStrategy.SHUFFLE,
        ),
    )

    compiled_configs = compile_dataset_builder_column_configs(config)

    assert len(compiled_configs) == 1
    assert compiled_configs[0].column_type == DataDesignerColumnType.SEED_DATASET


def test_compile_dataset_builder_column_configs_with_sampler_columns():
    config = DataDesignerConfig(
        columns=[SamplerColumnConfig(name="sampler_col", sampler_type="category", params={"values": ["col1", "col2"]})]
    )

    compiled_configs = compile_dataset_builder_column_configs(config)

    assert len(compiled_configs) == 1
    assert compiled_configs[0].column_type == DataDesignerColumnType.SAMPLER


def test_compile_dataset_builder_column_configs_with_generated_columns():
    config = DataDesignerConfig(
        columns=[LLMTextColumnConfig(name="text_col", prompt="Generate text", model_alias="test_model")]
    )

    compiled_configs = compile_dataset_builder_column_configs(config)

    assert len(compiled_configs) == 1
    assert compiled_configs[0].column_type == DataDesignerColumnType.LLM_TEXT


def test_compile_dataset_builder_column_configs_mixed_column_types():
    config = DataDesignerConfig(
        columns=[
            SeedDatasetColumnConfig(name="seed_col"),
            LLMTextColumnConfig(name="text_col", prompt="Generate text", model_alias="test_model"),
            SamplerColumnConfig(name="sampler_col", sampler_type="category", params={"values": ["col3", "col4"]}),
        ],
        seed_config=SeedConfig(
            source=HuggingFaceSeedSource(path="hf://datasets/test/dataset"),
            sampling_strategy=SamplingStrategy.SHUFFLE,
        ),
    )

    compiled_configs = compile_dataset_builder_column_configs(config)

    assert len(compiled_configs) == 3
    assert compiled_configs[0].column_type == DataDesignerColumnType.SEED_DATASET
    assert compiled_configs[1].column_type == DataDesignerColumnType.SAMPLER
    assert compiled_configs[2].column_type == DataDesignerColumnType.LLM_TEXT


def test_compile_dataset_builder_column_configs_seed_columns_without_seed_config():
    config = DataDesignerConfig(columns=[SeedDatasetColumnConfig(name="seed_col")])

    with pytest.raises(ConfigCompilationError, match="Seed column configs require a seed configuration"):
        compile_dataset_builder_column_configs(config)


def test_compile_dataset_builder_column_configs_empty_columns():
    config = DataDesignerConfig(
        columns=[LLMTextColumnConfig(name="dummy_col", prompt="dummy", model_alias="test_model")]
    )

    # Manually set columns to empty list after creation to test the function
    config.columns = []

    compiled_configs = compile_dataset_builder_column_configs(config)

    assert len(compiled_configs) == 0
