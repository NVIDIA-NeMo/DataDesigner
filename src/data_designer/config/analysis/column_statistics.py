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
    """Abstract base class for all column statistics types.

    Serves as a container for computed statistics across different column types in
    Data-Designer-generated datasets. Subclasses hold column-specific statistical results
    and provide methods for formatting these results for display in reports.
    """

    model_config = ConfigDict(use_enum_values=True)

    @abstractmethod
    def create_report_row_data(self) -> dict[str, str]:
        """Creates a formatted dictionary of statistics for display in reports.

        Returns:
            Dictionary mapping display labels to formatted statistic values.
        """
        ...


class GeneralColumnStatistics(BaseColumnStatistics):
    """Container for general statistics applicable to all column types.

    Holds core statistical measures that apply universally across all column types,
    including null counts, unique values, and data type information. Serves as the base
    for more specialized column statistics classes that store additional column-specific metrics.

    Attributes:
        column_name: Name of the column being analyzed.
        num_records: Total number of records in the column.
        num_null: Number of null/missing values in the column.
        num_unique: Number of distinct values in the column. If a value is not hashable, it is converted to a string.
        pyarrow_dtype: PyArrow data type of the column as a string.
        simple_dtype: Simplified human-readable data type label.
        column_type: Discriminator field, always "general" for this statistics type.
    """

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
    """Container for statistics on LLM-generated text columns.

    Inherits general statistics plus token usage metrics specific to LLM text generation.
    Stores both prompt and completion token consumption data.

    Attributes:
        completion_tokens_mean: Mean number of completion tokens generated per record.
        completion_tokens_median: Median number of completion tokens generated per record.
        completion_tokens_stddev: Standard deviation of completion tokens per record.
        prompt_tokens_mean: Mean number of prompt tokens used per record.
        prompt_tokens_median: Median number of prompt tokens used per record.
        prompt_tokens_stddev: Standard deviation of prompt tokens per record.
        column_type: Discriminator field, always "llm-text" for this statistics type.
    """

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
    """Container for statistics on LLM-generated code columns.

    Inherits all token usage metrics from LLMTextColumnStatistics. Stores
    statistics from columns that generate code snippets in specific programming languages.

    Attributes:
        column_type: Discriminator field, always "llm-code" for this statistics type.
    """

    column_type: Literal[DataDesignerColumnType.LLM_CODE.value] = DataDesignerColumnType.LLM_CODE.value


class LLMStructuredColumnStatistics(LLMTextColumnStatistics):
    """Container for statistics on LLM-generated structured JSON columns.

    Inherits all token usage metrics from LLMTextColumnStatistics. Stores statistics from
    columns that generate structured data conforming to JSON schemas or Pydantic models.

    Attributes:
        column_type: Discriminator field, always "llm-structured" for this statistics type.
    """

    column_type: Literal[DataDesignerColumnType.LLM_STRUCTURED.value] = DataDesignerColumnType.LLM_STRUCTURED.value


class LLMJudgedColumnStatistics(LLMTextColumnStatistics):
    """Container for statistics on LLM-as-a-judge quality assessment columns.

    Inherits all token usage metrics from LLMTextColumnStatistics. Stores statistics from
    columns that evaluate and score other generated content based on defined criteria.

    Attributes:
        column_type: Discriminator field, always "llm-judge" for this statistics type.
    """

    column_type: Literal[DataDesignerColumnType.LLM_JUDGE.value] = DataDesignerColumnType.LLM_JUDGE.value


class SamplerColumnStatistics(GeneralColumnStatistics):
    """Container for statistics on sampler-generated columns.

    Inherits general statistics plus sampler-specific information including the sampler type
    used and the empirical distribution of generated values. Stores both categorical and
    numerical distribution results.

    Attributes:
        sampler_type: Type of sampler used to generate this column (e.g., "uniform", "category",
            "gaussian", "person").
        distribution_type: Classification of the column's distribution (categorical, numerical,
            text, other, or unknown).
        distribution: Empirical distribution statistics for the generated values. Can be
            CategoricalDistribution (for discrete values), NumericalDistribution (for continuous
            values), or MissingValue if distribution could not be computed.
        column_type: Discriminator field, always "sampler" for this statistics type.
    """

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
    """Container for statistics on columns sourced from seed datasets.

    Inherits general statistics and stores statistics computed from columns that originate
    from existing data provided via the seed dataset functionality.

    Attributes:
        column_type: Discriminator field, always "seed-dataset" for this statistics type.
    """

    column_type: Literal[DataDesignerColumnType.SEED_DATASET.value] = DataDesignerColumnType.SEED_DATASET.value


class ExpressionColumnStatistics(GeneralColumnStatistics):
    """Container for statistics on expression-based derived columns.

    Inherits general statistics and stores statistics computed from columns that are derived
    from columns that are derived from Jinja2 expressions referencing other column values.

    Attributes:
        column_type: Discriminator field, always "expression" for this statistics type.
    """

    column_type: Literal[DataDesignerColumnType.EXPRESSION.value] = DataDesignerColumnType.EXPRESSION.value


class ValidationColumnStatistics(GeneralColumnStatistics):
    """Container for statistics on validation result columns.

    Inherits general statistics plus validation-specific metrics including the count and
    percentage of records that passed validation. Stores results from validation logic
    (Python, SQL, or remote) executed against target columns.

    Attributes:
        num_valid_records: Number of records that passed validation.
        column_type: Discriminator field, always "validation" for this statistics type.
    """

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
    """Container for categorical distribution histogram data.

    Stores the computed frequency distribution of categorical values.

    Attributes:
        categories: List of unique category values that appear in the data.
        counts: List of occurrence counts for each category.
    """

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
    """Container for computed categorical distribution statistics.

    Attributes:
        most_common_value: The category value that appears most frequently in the data.
        least_common_value: The category value that appears least frequently in the data.
        histogram: Complete frequency distribution showing all categories and their counts.
    """

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
    """Container for computed numerical distribution statistics.

    Attributes:
        min: Minimum value in the distribution.
        max: Maximum value in the distribution.
        mean: Arithmetic mean (average) of all values.
        stddev: Standard deviation measuring the spread of values around the mean.
        median: Median value of the distribution.
    """

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
