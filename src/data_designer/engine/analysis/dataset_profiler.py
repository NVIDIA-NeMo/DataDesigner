# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from data_designer.config.analysis.column_profilers import ColumnProfilerConfigT
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.analysis.column_profilers.base import ColumnConfigWithDataFrame, ColumnProfiler
from data_designer.engine.analysis.column_statistics import get_column_statistics_calculator
from data_designer.engine.analysis.errors import DatasetProfilerConfigurationError
from data_designer.engine.analysis.utils.column_statistics_calculations import has_pyarrow_backend
from data_designer.engine.dataset_builders.multi_column_configs import DatasetBuilderColumnConfigT, MultiColumnConfig
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import pa, pd

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

logger = logging.getLogger(__name__)


class DatasetProfilerConfig(ConfigBase):
    column_configs: Sequence[DatasetBuilderColumnConfigT] = Field(..., min_length=1)
    column_profiler_configs: Sequence[ColumnProfilerConfigT] | None = None

    @field_validator("column_configs")
    def flatten_and_validate_column_configs(cls, v: list[DatasetBuilderColumnConfigT]) -> list[ColumnConfigT]:
        column_configs = []
        for config in v:
            if isinstance(config, SingleColumnConfig) and not config.drop:
                column_configs.append(config)
            elif isinstance(config, MultiColumnConfig):
                column_configs.extend([c for c in config.columns if not c.drop])
        if len(column_configs) == 0:
            raise DatasetProfilerConfigurationError("All columns were dropped!")
        return column_configs


class DataDesignerDatasetProfiler:
    def __init__(self, config: DatasetProfilerConfig, resource_provider: ResourceProvider):
        self.config = config
        self.resource_provider = resource_provider
        self._validate_column_profiler_configs()

    @cached_property
    def column_names_from_configs(self) -> list[str]:
        return [c.name for c in self.config.column_configs]

    @cached_property
    def registry(self) -> DataDesignerRegistry:
        return DataDesignerRegistry()

    def profile_dataset(
        self,
        target_num_records: int,
        dataset: pd.DataFrame,
    ) -> DatasetProfilerResults:
        logger.info("📐 Measuring dataset column statistics:")

        self._validate_schema_consistency(list(dataset.columns))
        dataset = self._convert_to_pyarrow_backend_if_needed(dataset)

        column_statistics = []
        for c in self.config.column_configs:
            logger.info(f"  |-- {c.get_column_emoji()} column: '{c.name}'")
            column_statistics.append(
                get_column_statistics_calculator(c.column_type)(
                    column_config_with_df=ColumnConfigWithDataFrame(column_config=c, df=dataset)
                ).calculate()
            )

        column_profiles = []
        for profiler_config in self.config.column_profiler_configs or []:
            profiler = self._create_column_profiler(profiler_config)
            applicable_column_types = profiler.get_applicable_column_types()
            for c in self.config.column_configs:
                if c.column_type in applicable_column_types:
                    params = ColumnConfigWithDataFrame(column_config=c, df=dataset)
                    column_profiles.append(profiler.profile(params))
            if len(column_profiles) == 0:
                logger.warning(
                    f"⚠️ No applicable column types found for the '{profiler.name}' profiler. "
                    f"This profiler is applicable to the following column types: {applicable_column_types}"
                )

        return DatasetProfilerResults(
            num_records=len(dataset),
            target_num_records=target_num_records,
            side_effect_column_names=list(set(dataset.columns) - set(self.column_names_from_configs)),
            column_statistics=column_statistics,
            column_profiles=column_profiles if column_profiles else None,
        )

    def _convert_to_pyarrow_backend_if_needed(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not has_pyarrow_backend(dataset):
            try:
                dataset = pa.Table.from_pandas(dataset).to_pandas(types_mapper=pd.ArrowDtype)
            except Exception as e:
                # For ArrowTypeError, the second arg contains the more informative message
                if isinstance(e, pa.lib.ArrowTypeError) and len(e.args) > 1:
                    error_msg = str(e.args[1])
                else:
                    error_msg = str(e)
                for col in dataset.columns:
                    # Make sure column names are clear in the error message
                    error_msg = error_msg.replace(col, f"'{col}'")
                logger.warning("⚠️ Unable to convert the dataset to a PyArrow backend")
                logger.warning(f"  |-- Conversion Error Message: {error_msg}")
                logger.warning("  |-- This is often due to at least one column having mixed data types")
                logger.warning(
                    "  |-- Note: Reported data types will be inferred from the first non-null value of each column"
                )
        return dataset

    def _create_column_profiler(self, profiler_config: ColumnProfilerConfigT) -> ColumnProfiler:
        return self.registry.column_profilers.get_for_config_type(type(profiler_config))(
            config=profiler_config, resource_provider=self.resource_provider
        )

    def _validate_column_profiler_configs(self) -> None:
        if self.config.column_profiler_configs:
            if self.resource_provider.model_registry is None:
                raise DatasetProfilerConfigurationError("Model registry is required for column profiler configs")
            self._validate_model_configs()

    def _validate_model_configs(self) -> None:
        aliases = [alias for alias in self.resource_provider.model_registry.model_configs.keys()]
        for column_config in self.config.column_configs:
            if hasattr(column_config, "model_alias") and column_config.model_alias not in aliases:
                raise DatasetProfilerConfigurationError(
                    f"Model config '{column_config.model_alias}' not found in model configs"
                )

    def _validate_schema_consistency(self, dataset_column_names: list[str]) -> None:
        for column_name in self.column_names_from_configs:
            if column_name not in dataset_column_names:
                raise DatasetProfilerConfigurationError(f"Column '{column_name}' not found in dataset")
