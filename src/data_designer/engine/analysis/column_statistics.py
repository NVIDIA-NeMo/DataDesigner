# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel
from typing_extensions import Self

from data_designer.config.analysis.column_statistics import (
    DEFAULT_COLUMN_STATISTICS_MAP,
    ColumnStatisticsT,
    GeneralColumnStatistics,
)
from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType
from data_designer.config.sampler_params import SamplerType, is_numerical_sampler_type
from data_designer.engine.analysis.column_profilers.base import ColumnConfigWithDataFrame
from data_designer.engine.analysis.utils.column_statistics_calculations import (
    ColumnDistributionType,
    calculate_column_distribution,
    calculate_general_column_info,
    calculate_token_stats,
    calculate_validation_column_info,
)
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class GeneralColumnStatisticsCalculator(BaseModel):
    column_config_with_df: ColumnConfigWithDataFrame

    @property
    def column_config(self) -> ColumnConfigT:
        return self.column_config_with_df.column_config

    @property
    def df(self) -> pd.DataFrame:
        return self.column_config_with_df.df

    @property
    def column_statistics_type(self) -> type[ColumnStatisticsT]:
        return DEFAULT_COLUMN_STATISTICS_MAP.get(self.column_config.column_type, GeneralColumnStatistics)

    def calculate(self) -> Self:
        """Calculate all the column statistics fields for the given column configuration and dataset profiler.

        This method dynamically collects all class methods prefixed with 'calculate_' and invokes them to
        compute various column statistics, aggregating their results into a single statistics object.
        """
        calculate_methods = [
            name for name in dir(self) if name.startswith("calculate_") and callable(getattr(self, name))
        ]
        return self.column_statistics_type(
            column_name=self.column_config.name,
            **{k: v for name in calculate_methods for k, v in getattr(self, name)().items()},
        )

    def calculate_general_column_info(self) -> dict[str, Any]:
        return calculate_general_column_info(self.column_config.name, self.df)

    def __repr__(self) -> str:
        params = []
        for field, value in self.model_dump(mode="json").items():
            params.append(f"    {field}: {value}")
        params_str = "\n".join(params)
        return f"{self.__class__.__name__}(\n{params_str}\n)"


class LLMTextColumnStatisticsCalculator(GeneralColumnStatisticsCalculator):
    def calculate_token_stats(self) -> dict[str, Any]:
        return calculate_token_stats(self.column_config, self.df)


class LLMCodeColumnStatisticsCalculator(LLMTextColumnStatisticsCalculator): ...


class LLMStructuredColumnStatisticsCalculator(LLMTextColumnStatisticsCalculator): ...


class LLMJudgedColumnStatisticsCalculator(LLMTextColumnStatisticsCalculator): ...


class SamplerColumnStatisticsCalculator(GeneralColumnStatisticsCalculator):
    def calculate_sampler_distribution(self) -> dict[str, Any]:
        make_dist, dist_type = False, ColumnDistributionType.OTHER
        if self.column_config.sampler_type in [SamplerType.CATEGORY, SamplerType.SUBCATEGORY]:
            make_dist, dist_type = True, ColumnDistributionType.CATEGORICAL
        elif is_numerical_sampler_type(self.column_config.sampler_type):
            make_dist, dist_type = True, ColumnDistributionType.NUMERICAL
        return (
            {
                "sampler_type": SamplerType(self.column_config.sampler_type),
                **calculate_column_distribution(self.column_config.name, self.df, dist_type),
            }
            if make_dist
            else {
                "sampler_type": SamplerType(self.column_config.sampler_type),
                "distribution_type": dist_type,
                "distribution": None,
            }
        )


class SeedDatasetColumnStatisticsCalculator(GeneralColumnStatisticsCalculator): ...


class ValidationColumnStatisticsCalculator(GeneralColumnStatisticsCalculator):
    def calculate_validation_column_info(self) -> dict[str, Any]:
        return calculate_validation_column_info(self.column_config.name, self.df)


class ExpressionColumnStatisticsCalculator(GeneralColumnStatisticsCalculator): ...


ColumnStatisticsCalculatorT: TypeAlias = (
    ExpressionColumnStatisticsCalculator
    | ValidationColumnStatisticsCalculator
    | GeneralColumnStatisticsCalculator
    | LLMCodeColumnStatisticsCalculator
    | LLMJudgedColumnStatisticsCalculator
    | LLMStructuredColumnStatisticsCalculator
    | LLMTextColumnStatisticsCalculator
    | SamplerColumnStatisticsCalculator
    | SeedDatasetColumnStatisticsCalculator
)
DEFAULT_COLUMN_STATISTICS_CALCULATOR_MAP = {
    DataDesignerColumnType.EXPRESSION: ExpressionColumnStatisticsCalculator,
    DataDesignerColumnType.VALIDATION: ValidationColumnStatisticsCalculator,
    DataDesignerColumnType.LLM_CODE: LLMCodeColumnStatisticsCalculator,
    DataDesignerColumnType.LLM_JUDGE: LLMJudgedColumnStatisticsCalculator,
    DataDesignerColumnType.LLM_STRUCTURED: LLMStructuredColumnStatisticsCalculator,
    DataDesignerColumnType.LLM_TEXT: LLMTextColumnStatisticsCalculator,
    DataDesignerColumnType.SAMPLER: SamplerColumnStatisticsCalculator,
    DataDesignerColumnType.SEED_DATASET: SeedDatasetColumnStatisticsCalculator,
}


def get_column_statistics_calculator(column_type: DataDesignerColumnType) -> ColumnStatisticsCalculatorT:
    return DEFAULT_COLUMN_STATISTICS_CALCULATOR_MAP.get(column_type, GeneralColumnStatisticsCalculator)
