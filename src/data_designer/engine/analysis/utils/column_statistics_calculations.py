# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from numbers import Number
from typing import TYPE_CHECKING, Any

import tiktoken

from data_designer.config.analysis.column_statistics import (
    CategoricalDistribution,
    ColumnDistributionType,
    MissingValue,
    NumericalDistribution,
)
from data_designer.config.column_configs import (
    LLMTextColumnConfig,
)
from data_designer.engine.column_generators.utils.prompt_renderer import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)
from data_designer.lazy_heavy_imports import np, pa, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import pyarrow as pa

RANDOM_SEED = 42
MAX_PROMPT_SAMPLE_SIZE = 1000
TOKENIZER = tiktoken.get_encoding("cl100k_base")
WARNING_PREFIX = "⚠️ Error during column profile calculation: "
TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD = 0.1

logger = logging.getLogger(__name__)


def calculate_column_distribution(
    column_name: str, df: pd.DataFrame, distribution_type: ColumnDistributionType
) -> dict[str, CategoricalDistribution | NumericalDistribution | MissingValue | None]:
    distribution_type = ColumnDistributionType(distribution_type)
    try:
        if distribution_type == ColumnDistributionType.CATEGORICAL:
            return {
                "distribution_type": ColumnDistributionType.CATEGORICAL,
                "distribution": CategoricalDistribution.from_series(df[column_name]),
            }

        if distribution_type == ColumnDistributionType.NUMERICAL:
            return {
                "distribution_type": ColumnDistributionType.NUMERICAL,
                "distribution": NumericalDistribution.from_series(df[column_name]),
            }
    except Exception as e:
        logger.warning(f"{WARNING_PREFIX} failed to calculate column distribution for '{column_name}' {e}")
        return {
            "distribution_type": ColumnDistributionType.UNKNOWN,
            "distribution": MissingValue.CALCULATION_FAILED,
        }


def calculate_general_column_info(column_name: str, df: pd.DataFrame) -> dict[str, Any]:
    try:
        _df = pd.DataFrame(df[column_name].apply(ensure_hashable))

        if has_pyarrow_backend(df):
            pyarrow_dtype = str(df[column_name].dtype.pyarrow_dtype)
            simple_dtype = convert_pyarrow_dtype_to_simple_dtype(df[column_name].dtype.pyarrow_dtype)
        else:
            # We do not log a warning at the column-level because it would be too noisy.
            # However, there is a logged warning at the dataset-profiler level.
            try:
                simple_dtype = get_column_data_type_from_first_non_null_value(column_name, df)
            except Exception:
                simple_dtype = MissingValue.CALCULATION_FAILED
            pyarrow_dtype = "n/a"

        return {
            "pyarrow_dtype": pyarrow_dtype,
            "simple_dtype": simple_dtype,
            "num_records": len(_df[column_name]),
            "num_null": _df[column_name].isnull().sum(),
            "num_unique": _df[column_name].nunique(),
        }
    except Exception as e:
        logger.warning(f"{WARNING_PREFIX} failed to calculate general column info for '{column_name}': {e}")
        return {
            "pyarrow_dtype": MissingValue.CALCULATION_FAILED,
            "simple_dtype": MissingValue.CALCULATION_FAILED,
            "num_records": MissingValue.CALCULATION_FAILED,
            "num_null": MissingValue.CALCULATION_FAILED,
            "num_unique": MissingValue.CALCULATION_FAILED,
        }


def calculate_input_token_stats(
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
        logger.warning(f"{WARNING_PREFIX} failed to calculate input token stats for column {column_config.name!r}: {e}")
        return {
            "input_tokens_mean": MissingValue.CALCULATION_FAILED,
            "input_tokens_median": MissingValue.CALCULATION_FAILED,
            "input_tokens_stddev": MissingValue.CALCULATION_FAILED,
        }
    return {
        "input_tokens_mean": np.mean(num_tokens),
        "input_tokens_median": np.median(num_tokens),
        "input_tokens_stddev": np.std(num_tokens),
    }


def calculate_output_token_stats(
    column_config: LLMTextColumnConfig, df: pd.DataFrame
) -> dict[str, float | MissingValue]:
    try:
        tokens_per_record = df[column_config.name].apply(
            lambda value: len(TOKENIZER.encode(str(value), disallowed_special=()))
        )
        return {
            "output_tokens_mean": tokens_per_record.mean(),
            "output_tokens_median": tokens_per_record.median(),
            "output_tokens_stddev": tokens_per_record.std(),
        }
    except Exception as e:
        logger.warning(f"{WARNING_PREFIX} failed to calculate output token stats for column {column_config.name}: {e}")
        return {
            "output_tokens_mean": MissingValue.CALCULATION_FAILED,
            "output_tokens_median": MissingValue.CALCULATION_FAILED,
            "output_tokens_stddev": MissingValue.CALCULATION_FAILED,
        }


def calculate_token_stats(column_config: LLMTextColumnConfig, df: pd.DataFrame) -> dict[str, float | MissingValue]:
    return {
        **calculate_input_token_stats(column_config, df),
        **calculate_output_token_stats(column_config, df),
    }


def calculate_validation_column_info(column_name: str, df: pd.DataFrame) -> dict[str, Any]:
    try:
        return {"num_valid_records": df[column_name].apply(lambda x: ensure_boolean(x["is_valid"])).sum()}
    except Exception as e:
        logger.warning(
            f"{WARNING_PREFIX} failed to calculate code validation column info for column {column_name}: {e}"
        )
        return {"num_valid_records": MissingValue.CALCULATION_FAILED}


def convert_pyarrow_dtype_to_simple_dtype(pyarrow_dtype: pa.DataType) -> str:
    if isinstance(pyarrow_dtype, pa.ListType):
        return f"list[{convert_pyarrow_dtype_to_simple_dtype(pyarrow_dtype.value_type)}]"
    if isinstance(pyarrow_dtype, pa.StructType):
        return "dict"
    return convert_to_simple_dtype(str(pyarrow_dtype))


def convert_to_simple_dtype(dtype: str) -> str:
    if "int" in dtype:
        return "int"
    if "double" in dtype:
        return "float"
    if "float" in dtype:
        return "float"
    if "str" in dtype:
        return "string"
    if "timestamp" in dtype:
        return "timestamp"
    if "time" in dtype:
        return "time"
    if "date" in dtype:
        return "date"
    return dtype


def get_column_data_type_from_first_non_null_value(column_name: str, df: pd.DataFrame) -> str:
    df_no_nulls = df[column_name].dropna()
    if len(df_no_nulls) == 0:
        return MissingValue.CALCULATION_FAILED
    dtype = type(df_no_nulls.iloc[0]).__name__
    return convert_to_simple_dtype(dtype)


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


def has_pyarrow_backend(df: pd.DataFrame) -> bool:
    return all(isinstance(dtype, pd.ArrowDtype) for dtype in df.dtypes)
