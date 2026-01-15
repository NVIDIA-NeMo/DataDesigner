# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from data_designer.config.column_configs import SeedDatasetColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.errors import InvalidConfigError
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.engine.validation import ViolationLevel, rich_print_violations, validate_data_designer_config

logger = logging.getLogger(__name__)


def compile_data_designer_config(
    config_builder: DataDesignerConfigBuilder, resource_provider: ResourceProvider
) -> DataDesignerConfig:
    config = config_builder.build()
    _resolve_and_add_seed_columns(config, resource_provider.seed_reader)
    _validate(config)

    return config


def _resolve_and_add_seed_columns(config: DataDesignerConfig, seed_reader: SeedReader | None) -> None:
    """Fetches the seed dataset column names, ensures there are no conflicts
    with other columns, and adds seed column configs to the DataDesignerConfig.
    """

    if not seed_reader:
        return

    seed_col_names = seed_reader.get_column_names()
    existing_columns = {column.name for column in config.columns}
    colliding_columns = {name for name in seed_col_names if name in existing_columns}
    if colliding_columns:
        raise InvalidConfigError(
            f"🛑 Seed dataset column(s) {colliding_columns} collide with existing column(s). "
            "Please remove the conflicting columns or use a seed dataset with different column names."
        )

    config.columns.extend([SeedDatasetColumnConfig(name=col_name) for col_name in seed_col_names])


def _validate(config: DataDesignerConfig) -> None:
    allowed_references = _get_allowed_references(config)
    violations = validate_data_designer_config(
        columns=config.columns,
        processor_configs=config.processors or [],
        allowed_references=allowed_references,
    )
    rich_print_violations(violations)
    if len([v for v in violations if v.level == ViolationLevel.ERROR]) > 0:
        raise InvalidConfigError(
            "🛑 Your configuration contains validation errors. Please address the indicated issues and try again."
        )
    if len(violations) == 0:
        logger.info("✅ Validation passed")


def _get_allowed_references(config: DataDesignerConfig) -> list[str]:
    refs = set[str]()
    for column_config in config.columns:
        refs.add(column_config.name)
        for side_effect_column in column_config.side_effect_columns:
            refs.add(side_effect_column)
    return list(refs)
