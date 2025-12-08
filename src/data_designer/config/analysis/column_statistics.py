# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union

from pandas import Series
from pydantic import BaseModel, ConfigDict, create_model, field_validator, model_validator
from typing_extensions import Self, TypeAlias

from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.constants import EPSILON
from data_designer.config.utils.numerical_helpers import is_float, is_int, prepare_number_for_reporting
from data_designer.plugin_manager import PluginManager


class MissingValue(str, Enum):
    CALCULATION_FAILED = "--"
    OUTPUT_FORMAT_ERROR = "output_format_error"


class ColumnDistributionType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TEXT = "text"
    OTHER = "other"
    UNKNOWN = "unknown"


class BaseColumnStatistics(BaseModel, ABC):
    model_config = ConfigDict(use_enum_values=True)

    @abstractmethod
    def create_report_row_data(self) -> dict[str, str]: ...


class GeneralColumnStatistics(BaseColumnStatistics):
    column_name: str
    num_records: Union[int, MissingValue]
    num_null: Union[int, MissingValue]
    num_unique: Union[int, MissingValue]
    pyarrow_dtype: str
    simple_dtype: str
    column_type: Literal["general"] = "general"

    @field_validator("num_null", "num_unique", "num_records", mode="before")
    def general_statistics_ensure_python_integers(cls, v: Union[int, MissingValue]) -> Union[int, MissingValue]:
        return v if isinstance(v, MissingValue) else prepare_number_for_reporting(v, int)

    @property
    def percent_null(self) -> Union[float, MissingValue]:
        return (
            self.num_null
            if self._is_missing_value(self.num_null)
            else prepare_number_for_reporting(100 * self.num_null / (self.num_records + EPSILON), float)
        )

    @property
    def percent_unique(self) -> Union[float, MissingValue]:
        return (
            self.num_unique
            if self._is_missing_value(self.num_unique)
            else prepare_number_for_reporting(100 * self.num_unique / (self.num_records + EPSILON), float)
        )

    @property
    def _general_display_row(self) -> dict[str, str]:
        pct_unique_str = f" ({self.percent_unique:.1f}%)" if not self._is_missing_value(self.percent_unique) else ""
        return {
            "column name": self.column_name,
            "data type": self.simple_dtype,
            "number unique values": f"{self.num_unique}{pct_unique_str}",
        }

    def create_report_row_data(self) -> dict[str, str]:
        return self._general_display_row

    def _is_missing_value(self, v: Union[float, int, MissingValue]) -> bool:
        return v in set(MissingValue)


class LLMTextColumnStatistics(GeneralColumnStatistics):
    completion_tokens_mean: Union[float, MissingValue]
    completion_tokens_median: Union[float, MissingValue]
    completion_tokens_stddev: Union[float, MissingValue]
    prompt_tokens_mean: Union[float, MissingValue]
    prompt_tokens_median: Union[float, MissingValue]
    prompt_tokens_stddev: Union[float, MissingValue]
    column_type: Literal[DataDesignerColumnType.LLM_TEXT.value] = DataDesignerColumnType.LLM_TEXT.value

    @field_validator(
        "completion_tokens_mean",
        "completion_tokens_median",
        "completion_tokens_stddev",
        "prompt_tokens_mean",
        "prompt_tokens_median",
        "prompt_tokens_stddev",
        mode="before",
    )
    def llm_column_ensure_python_floats(cls, v: Union[float, int, MissingValue]) -> Union[float, int, MissingValue]:
        return v if isinstance(v, MissingValue) else prepare_number_for_reporting(v, float)

    def create_report_row_data(self) -> dict[str, Any]:
        prompt_tokens_str = (
            f"{self.prompt_tokens_median:.1f} +/- {self.prompt_tokens_stddev:.1f}"
            if not self._is_missing_value(self.prompt_tokens_median)
            else "--"
        )
        completion_tokens_str = (
            f"{self.completion_tokens_median:.1f} +/- {self.completion_tokens_stddev:.1f}"
            if not self._is_missing_value(self.completion_tokens_median)
            else "--"
        )
        return {
            **self._general_display_row,
            "prompt tokens\nper record": prompt_tokens_str,
            "completion tokens\nper record": completion_tokens_str,
        }


class LLMCodeColumnStatistics(LLMTextColumnStatistics):
    column_type: Literal[DataDesignerColumnType.LLM_CODE.value] = DataDesignerColumnType.LLM_CODE.value


class LLMStructuredColumnStatistics(LLMTextColumnStatistics):
    column_type: Literal[DataDesignerColumnType.LLM_STRUCTURED.value] = DataDesignerColumnType.LLM_STRUCTURED.value


class LLMJudgedColumnStatistics(LLMTextColumnStatistics):
    column_type: Literal[DataDesignerColumnType.LLM_JUDGE.value] = DataDesignerColumnType.LLM_JUDGE.value


class SamplerColumnStatistics(GeneralColumnStatistics):
    sampler_type: SamplerType
    distribution_type: ColumnDistributionType
    distribution: Optional[Union[CategoricalDistribution, NumericalDistribution, MissingValue]]
    column_type: Literal[DataDesignerColumnType.SAMPLER.value] = DataDesignerColumnType.SAMPLER.value

    def create_report_row_data(self) -> dict[str, str]:
        return {
            **self._general_display_row,
            "sampler type": self.sampler_type,
        }


class SeedDatasetColumnStatistics(GeneralColumnStatistics):
    column_type: Literal[DataDesignerColumnType.SEED_DATASET.value] = DataDesignerColumnType.SEED_DATASET.value


class ExpressionColumnStatistics(GeneralColumnStatistics):
    column_type: Literal[DataDesignerColumnType.EXPRESSION.value] = DataDesignerColumnType.EXPRESSION.value


class ValidationColumnStatistics(GeneralColumnStatistics):
    num_valid_records: Union[int, MissingValue]
    column_type: Literal[DataDesignerColumnType.VALIDATION.value] = DataDesignerColumnType.VALIDATION.value

    @field_validator("num_valid_records", mode="before")
    def code_validation_column_ensure_python_integers(cls, v: Union[int, MissingValue]) -> Union[int, MissingValue]:
        return v if isinstance(v, MissingValue) else prepare_number_for_reporting(v, int)

    @property
    def percent_valid(self) -> Union[float, MissingValue]:
        return (
            self.num_valid_records
            if self._is_missing_value(self.num_valid_records)
            else prepare_number_for_reporting(100 * self.num_valid_records / (self.num_records + EPSILON), float)
        )

    def create_report_row_data(self) -> dict[str, str]:
        percent_valid_str = f"{self.percent_valid:.1f}%" if not self._is_missing_value(self.percent_valid) else "--"
        return {**self._general_display_row, "percent valid": percent_valid_str}


class CategoricalHistogramData(BaseModel):
    categories: list[Union[float, int, str]]
    counts: list[int]

    @model_validator(mode="after")
    def ensure_python_types(self) -> Self:
        """Ensure numerical values are Python objects rather than Numpy types."""
        self.categories = [(float(x) if is_float(x) else (int(x) if is_int(x) else str(x))) for x in self.categories]
        self.counts = [int(i) for i in self.counts]
        return self

    @classmethod
    def from_series(cls, series: Series) -> Self:
        counts = series.value_counts()
        return cls(categories=counts.index.tolist(), counts=counts.tolist())


class CategoricalDistribution(BaseModel):
    most_common_value: Union[str, int]
    least_common_value: Union[str, int]
    histogram: CategoricalHistogramData

    @field_validator("most_common_value", "least_common_value", mode="before")
    def ensure_python_types(cls, v: Union[str, int]) -> Union[str, int]:
        return str(v) if not is_int(v) else prepare_number_for_reporting(v, int)

    @classmethod
    def from_series(cls, series: Series) -> Self:
        counts = series.value_counts()
        return cls(
            most_common_value=counts.index[0],
            least_common_value=counts.index[-1],
            histogram=CategoricalHistogramData.from_series(series),
        )


class NumericalDistribution(BaseModel):
    min: Union[float, int]
    max: Union[float, int]
    mean: float
    stddev: float
    median: float

    @field_validator("min", "max", "mean", "stddev", "median", mode="before")
    def ensure_python_types(cls, v: Union[float, int]) -> Union[float, int]:
        return prepare_number_for_reporting(v, int if is_int(v) else float)

    @classmethod
    def from_series(cls, series: Series) -> Self:
        return cls(
            min=series.min(skipna=True),
            max=series.max(skipna=True),
            mean=series.mean(skipna=True),
            stddev=series.std(skipna=True),
            median=series.median(skipna=True),
        )


ColumnStatisticsT: TypeAlias = Union[
    GeneralColumnStatistics,
    LLMTextColumnStatistics,
    LLMCodeColumnStatistics,
    LLMStructuredColumnStatistics,
    LLMJudgedColumnStatistics,
    SamplerColumnStatistics,
    SeedDatasetColumnStatistics,
    ValidationColumnStatistics,
    ExpressionColumnStatistics,
]


DEFAULT_COLUMN_STATISTICS_MAP = {
    DataDesignerColumnType.EXPRESSION: ExpressionColumnStatistics,
    DataDesignerColumnType.LLM_CODE: LLMCodeColumnStatistics,
    DataDesignerColumnType.LLM_JUDGE: LLMJudgedColumnStatistics,
    DataDesignerColumnType.LLM_STRUCTURED: LLMStructuredColumnStatistics,
    DataDesignerColumnType.LLM_TEXT: LLMTextColumnStatistics,
    DataDesignerColumnType.SAMPLER: SamplerColumnStatistics,
    DataDesignerColumnType.SEED_DATASET: SeedDatasetColumnStatistics,
    DataDesignerColumnType.VALIDATION: ValidationColumnStatistics,
}

for plugin in PluginManager().get_column_generator_plugins():
    # Dynamically create a statistics class for this plugin using Pydantic's create_model
    plugin_stats_cls_name = f"{plugin.config_type_as_class_name}ColumnStatistics"

    # Create the class with proper Pydantic field
    plugin_stats_cls = create_model(
        plugin_stats_cls_name,
        __base__=GeneralColumnStatistics,
        column_type=(Literal[plugin.name], plugin.name),
    )

    # Add the plugin statistics class to the union
    ColumnStatisticsT |= plugin_stats_cls
    DEFAULT_COLUMN_STATISTICS_MAP[DataDesignerColumnType(plugin.name)] = plugin_stats_cls
