# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
import json
import logging
import re
from typing import TYPE_CHECKING, Any, TypeVar

import data_designer.lazy_heavy_imports as lazy

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


def concat_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    _verify_columns_are_unique(datasets)
    _verify_dataset_lengths_are_equal(datasets)
    emoji = " + ".join(["💾"] * len(datasets))
    logger.info(f"({emoji}) Concatenating {len(datasets)} datasets")
    return lazy.pd.concat([df for df in datasets], axis=1)


def deserialize_json_string(value: str) -> Any:
    """Decode a JSON document, returning the original string if it is not valid JSON."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def prepare_template_context(data: T) -> Any:
    """Decode JSON objects and arrays stored as strings so templates can reach their fields.

    Native dictionaries and lists are processed recursively. A string is replaced only when it holds a JSON
    object or array, so `{{ column.key }}` works on a column that stored a JSON object, while scalar strings
    like "1.10", "true" or "null" stay text. The input is not mutated.

    Args:
        data: A record, a list, a string, or any other value.

    Returns:
        A copy of `data` with embedded JSON objects and arrays decoded.
    """
    if isinstance(data, str):
        loaded = deserialize_json_string(data)
        return loaded if isinstance(loaded, (dict, list)) else data

    if isinstance(data, list):
        return [prepare_template_context(item) for item in data]

    if isinstance(data, dict):
        return {key: prepare_template_context(value) for key, value in data.items()}

    return copy.deepcopy(data)


def parse_list_string(text: str) -> list[str]:
    """Parse a list from a string, handling JSON arrays, Python lists, and trailing commas."""
    text = text.strip()

    # Try JSON first
    try:
        list_obj = json.loads(text)
        if isinstance(list_obj, list):
            return _clean_whitespace(list_obj)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before closing brackets (common in JSON-like strings)
    text_cleaned = re.sub(r",\s*]", "]", text)
    text_cleaned = re.sub(r",\s*}", "}", text_cleaned)

    # Try JSON again with cleaned text
    try:
        list_obj = json.loads(text_cleaned)
        if isinstance(list_obj, list):
            return _clean_whitespace(list_obj)
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (handles single quotes)
    try:
        list_obj = ast.literal_eval(text_cleaned)
        if isinstance(list_obj, list):
            return _clean_whitespace(list_obj)
    except (ValueError, SyntaxError):
        pass

    # If all else fails, return the original text
    return [text.strip()]


def _clean_whitespace(texts: list[Any]) -> list[str]:
    return [str(text).strip() for text in texts]


def _verify_columns_are_unique(datasets: list[pd.DataFrame]) -> None:
    joined_columns = set()
    for df in datasets:
        columns = set(df.columns)
        overlapping_columns = joined_columns & columns
        if len(overlapping_columns) > 0:
            raise ValueError(
                f"🛑 Input datasets have overlapping columns: {overlapping_columns} "
                "Please ensure that the column names are unique."
            )
        joined_columns.update(columns)


def _verify_dataset_lengths_are_equal(datasets: list[pd.DataFrame]) -> None:
    if len(set([len(df) for df in datasets])) > 1:
        raise ValueError(
            "🛑 Input datasets have different lengths. Please ensure that the datasets have the same number of rows."
        )
