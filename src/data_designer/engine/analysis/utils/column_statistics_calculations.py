# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.dtypes.common import is_integer_dtype, is_numeric_dtype
import pyarrow as pa
import tiktoken

from data_designer.config.analysis.column_statistics import (
    CategoricalDistribution,
    ColumnDistributionType,
    MissingValue,
    NumericalDistribution,
)
from data_designer.config.column_configs import (
    LLMTextColumnConfig,
    SingleColumnConfig,
    ValidationColumnConfig,
)
from data_designer.engine.column_generators.generators.llm_generators import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)

RANDOM_SEED = 42
MAX_PROMPT_SAMPLE_SIZE = 1000
TOKENIZER = tiktoken.get_encoding("cl100k_base")
WARNING_PREFIX = "⚠️ Error during column profile calculation: "
TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD = 0.1

logger = logging.getLogger(__name__)


def calculate_column_distribution(
    column_config: SingleColumnConfig, df: pd.DataFrame, distribution_type: ColumnDistributionType
) -> dict[str, CategoricalDistribution | NumericalDistribution | MissingValue | None]:
    distribution_type = ColumnDistributionType(distribution_type)
    try:
        if distribution_type == ColumnDistributionType.CATEGORICAL:
            return {
                "distribution_type": ColumnDistributionType.CATEGORICAL,
                "distribution": CategoricalDistribution.from_series(df[column_config.name]),
            }

        if distribution_type == ColumnDistributionType.NUMERICAL:
            return {
                "distribution_type": ColumnDistributionType.NUMERICAL,
                "distribution": NumericalDistribution.from_series(df[column_config.name]),
            }
    except Exception as e:
        logger.warning(f"{WARNING_PREFIX} failed to calculate column distribution for '{column_config.name}' {e}")
        return {
            "distribution_type": ColumnDistributionType.UNKNOWN,
            "distribution": MissingValue.CALCULATION_FAILED,
        }


def calculate_general_column_info(column_config: SingleColumnConfig, df: pd.DataFrame) -> dict[str, Any]:
    try:
        _df = pd.DataFrame(df[column_config.name].apply(ensure_hashable))
        return {
            "pyarrow_dtype": str(df[column_config.name].dtype.pyarrow_dtype),
            "simple_dtype": convert_pyarrow_dtype_to_simple_dtype(df[column_config.name].dtype.pyarrow_dtype),
            "num_records": len(_df[column_config.name]),
            "num_null": _df[column_config.name].isnull().sum(),
            "num_unique": _df[column_config.name].nunique(),
        }
    except Exception as e:
        logger.warning(f"{WARNING_PREFIX} failed to calculate general column info for '{column_config.name}': {e}")
        return {
            "pyarrow_dtype": MissingValue.CALCULATION_FAILED,
            "simple_dtype": MissingValue.CALCULATION_FAILED,
            "num_records": MissingValue.CALCULATION_FAILED,
            "num_null": MissingValue.CALCULATION_FAILED,
            "num_unique": MissingValue.CALCULATION_FAILED,
        }


def calculate_prompt_token_stats(
    column_config: LLMTextColumnConfig, df: pd.DataFrame
) -> dict[str, float | MissingValue]:
    try:
        num_tokens = []
        num_samples = min(MAX_PROMPT_SAMPLE_SIZE, len(df))
        renderer = RecordBasedPromptRenderer(response_recipe=create_response_recipe(column_config))
        for record in df.sample(num_samples, random_state=RANDOM_SEED).to_dict(orient="records"):
            system_prompt = renderer.render(
                prompt_template=column_config.system_prompt, record=record, prompt_type=PromptType.SYSTEM_PROMPT
            )
            prompt = renderer.render(
                prompt_template=column_config.prompt, record=record, prompt_type=PromptType.USER_PROMPT
            )
            concatenated_prompt = str(system_prompt + "\n\n" + prompt)
            num_tokens.append(len(TOKENIZER.encode(concatenated_prompt, disallowed_special=())))
    except Exception as e:
        logger.warning(
            f"{WARNING_PREFIX} failed to calculate prompt token stats for column {column_config.name!r}: {e}"
        )
        return {
            "prompt_tokens_mean": MissingValue.CALCULATION_FAILED,
            "prompt_tokens_median": MissingValue.CALCULATION_FAILED,
            "prompt_tokens_stddev": MissingValue.CALCULATION_FAILED,
        }
    return {
        "prompt_tokens_mean": np.mean(num_tokens),
        "prompt_tokens_median": np.median(num_tokens),
        "prompt_tokens_stddev": np.std(num_tokens),
    }


def calculate_completion_token_stats(
    column_config: LLMTextColumnConfig, df: pd.DataFrame
) -> dict[str, float | MissingValue]:
    try:
        tokens_per_record = df[column_config.name].apply(
            lambda value: len(TOKENIZER.encode(str(value), disallowed_special=()))
        )
        return {
            "completion_tokens_mean": tokens_per_record.mean(),
            "completion_tokens_median": tokens_per_record.median(),
            "completion_tokens_stddev": tokens_per_record.std(),
        }
    except Exception as e:
        logger.warning(
            f"{WARNING_PREFIX} failed to calculate completion token stats for column {column_config.name}: {e}"
        )
        return {
            "completion_tokens_mean": MissingValue.CALCULATION_FAILED,
            "completion_tokens_median": MissingValue.CALCULATION_FAILED,
            "completion_tokens_stddev": MissingValue.CALCULATION_FAILED,
        }


def calculate_token_stats(column_config: LLMTextColumnConfig, df: pd.DataFrame) -> dict[str, float | MissingValue]:
    return {
        **calculate_prompt_token_stats(column_config, df),
        **calculate_completion_token_stats(column_config, df),
    }


def calculate_validation_column_info(column_config: ValidationColumnConfig, df: pd.DataFrame) -> dict[str, Any]:
    try:
        return {"num_valid_records": df[column_config.name].apply(lambda x: ensure_boolean(x["is_valid"])).sum()}
    except Exception as e:
        logger.warning(
            f"{WARNING_PREFIX} failed to calculate code validation column info for column {column_config.name}: {e}"
        )
        return {"num_valid_records": MissingValue.CALCULATION_FAILED}


def convert_pyarrow_dtype_to_simple_dtype(pyarrow_dtype: pa.DataType) -> str:
    if isinstance(pyarrow_dtype, pa.ListType):
        return f"list[{convert_pyarrow_dtype_to_simple_dtype(pyarrow_dtype.value_type)}]"
    if isinstance(pyarrow_dtype, pa.StructType):
        return "dict"
    pyarrow_dtype_str = str(pyarrow_dtype)
    if "int" in pyarrow_dtype_str:
        return "int"
    if "double" in pyarrow_dtype_str:
        return "float"
    if "float" in pyarrow_dtype_str:
        return "float"
    if "string" in pyarrow_dtype_str:
        return "string"
    if "timestamp" in pyarrow_dtype_str:
        return "timestamp"
    if "time" in pyarrow_dtype_str:
        return "time"
    if "date" in pyarrow_dtype_str:
        return "date"
    return pyarrow_dtype_str


def determine_column_distribution_type(column: Series) -> ColumnDistributionType:
    """Based on the logic used by Gretel's SQS report to determine column data type."""
    if len(column) == 0:
        return ColumnDistributionType.OTHER

    if isinstance(column.iloc[0], np.ndarray):
        return ColumnDistributionType.OTHER

    if isinstance(column.iloc[0], dict):
        return ColumnDistributionType.OTHER

    try:
        non_na_data = column.dropna()
        non_na_count = int(non_na_data.count())
        unique_count = int(non_na_data.nunique())
    except Exception:
        column = column.astype(str)
        non_na_data = column.dropna()
        non_na_count = int(non_na_data.count())
        unique_count = int(non_na_data.nunique())

    if non_na_count == 0:
        return ColumnDistributionType.OTHER

    if is_numeric_dtype(non_na_data.dtype):
        # Float values that are within 1e-8 of an integer are considered integers
        # Floats are considered numerical.
        if not np.allclose(non_na_data, non_na_data.astype(int), atol=1e-8):
            return ColumnDistributionType.NUMERICAL
        # We can visualize numeric data with histograms, but we will not use it for diversity calculations
        min_value = int(non_na_data.min())
        if unique_count <= 10 and min_value >= 0:
            return ColumnDistributionType.CATEGORICAL
        if unique_count == non_na_count and is_integer_dtype(non_na_data.dtype):
            # All unique integer values, potentially an ID column
            return ColumnDistributionType.OTHER
        return ColumnDistributionType.NUMERICAL

    # Check if the column is a date-like column before checking for categorical or text columns.
    try:
        pd.to_datetime(non_na_data, format="%Y-%m-%d")
        return ColumnDistributionType.OTHER
    except Exception:
        pass

    diff = non_na_count - unique_count
    diff_percent = diff / non_na_count
    if diff_percent >= 0.9 or (diff_percent >= 0.7 and len(non_na_data) <= 50):
        return ColumnDistributionType.CATEGORICAL

    space_count = sum(str(entry).strip().count(" ") for entry in non_na_data)
    if space_count / non_na_count > TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD:
        return ColumnDistributionType.TEXT

    if pd.api.types.is_string_dtype(non_na_data.dtype) and unique_count <= 10:
        # Check for string columns with a small number of unique values (categorical)
        return ColumnDistributionType.CATEGORICAL

    return ColumnDistributionType.OTHER


def ensure_hashable(x: Any) -> str:
    """
    Makes a best effort turn known unhashable types to a hashable
    string representation that preserves both structure and values.
    """
    if isinstance(x, (Number, bool)) or x is None:
        return x

    if isinstance(x, dict):
        # Sort by keys and convert key-value pairs to tuples
        return str(sorted([(str(k), ensure_hashable(v)) for k, v in x.items()]))

    if isinstance(x, (list, tuple, set, np.ndarray)):
        # Recursively make all elements hashable
        return str(sorted([ensure_hashable(e) for e in x]))

    return str(x)


def ensure_boolean(v: bool | str | int | None) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)) and v in [0, 1, 0.0, 1.0]:
        return bool(v)
    if isinstance(v, (str, np.str_)) and v.lower() in ["true", "false"]:
        return v.lower() == "true"
    if v is None:
        return False
    raise ValueError(f"Invalid boolean value: {v}")
