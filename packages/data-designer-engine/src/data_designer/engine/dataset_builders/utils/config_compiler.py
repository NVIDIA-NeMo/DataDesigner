# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import SeedDatasetColumnConfig
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.engine.dataset_builders.multi_column_configs import (
    DatasetBuilderColumnConfigT,
    SamplerMultiColumnConfig,
    SeedDatasetMultiColumnConfig,
)
from data_designer.engine.dataset_builders.utils.errors import ConfigCompilationError
from data_designer.engine.dataset_builders.utils.execution_graph import topologically_sort_column_configs


def compile_dataset_builder_column_configs(
    config: DataDesignerConfig, *, seed_column_names: list[str] | None = None
) -> list[DatasetBuilderColumnConfigT]:
    """Group generation tasks while preserving raw inputs for pre-batch processors.

    Args:
        config: Validated configuration with the schema after pre-batch processing.
        seed_column_names: Raw seed names from the reader, including names removed
            by pre-batch processors. Required to retain those seed-loading tasks.

    Returns:
        Ordered seed, sampler, and generated column tasks.

    Raises:
        ConfigCompilationError: If seed columns have no seed configuration.
    """
    seed_column_configs = []
    sampler_column_configs = []
    generated_column_configs = []

    for column_config in topologically_sort_column_configs(config.columns):
        if column_config.column_type == DataDesignerColumnType.SEED_DATASET:
            seed_column_configs.append(column_config)
        elif column_config.column_type == DataDesignerColumnType.SAMPLER:
            sampler_column_configs.append(column_config)
        else:
            generated_column_configs.append(column_config)

    # Raw seed columns must still be read before processors remove or replace them.
    effective_seed_names = {column.name for column in seed_column_configs}
    seed_column_configs.extend(
        SeedDatasetColumnConfig(name=name) for name in seed_column_names or [] if name not in effective_seed_names
    )

    compiled_column_configs = []

    if len(seed_column_configs) > 0:
        if config.seed_config is None:
            raise ConfigCompilationError("🛑 Seed column configs require a seed configuration.")
        compiled_column_configs.append(
            SeedDatasetMultiColumnConfig(
                columns=seed_column_configs,
                source=config.seed_config.source,
                sampling_strategy=config.seed_config.sampling_strategy,
                selection_strategy=config.seed_config.selection_strategy,
            )
        )

    if len(sampler_column_configs) > 0:
        compiled_column_configs.append(
            SamplerMultiColumnConfig(
                columns=sampler_column_configs,
                constraints=config.constraints or [],
            )
        )

    if len(generated_column_configs) > 0:
        compiled_column_configs.extend(generated_column_configs)

    return compiled_column_configs
