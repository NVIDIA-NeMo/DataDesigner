# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import logging
import re
from typing import Any, TypeVar, Union, overload

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def concat_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    _verify_columns_are_unique(datasets)
    _verify_dataset_lengths_are_equal(datasets)
    emoji = " + ".join(["💾"] * len(datasets))
    logger.info(f"({emoji}) Concatenating {len(datasets)} datasets")
    return pd.concat([df for df in datasets], axis=1)


# Overloads to help static type checker better understand
# the input/output types of the deserialize_json_values function.
@overload
def deserialize_json_values(data: str) -> Union[dict[str, Any], list[Any], Any]: ...


@overload
def deserialize_json_values(data: list[T]) -> list[Any]: ...


@overload
def deserialize_json_values(data: dict[K, V]) -> dict[K, Any]: ...


@overload
def deserialize_json_values(data: T) -> T: ...


def deserialize_json_values(data):
    """De-serialize JSON strings in various input formats.

    Args:
        data: Input data in one of four formats:
            - Single string (JSON string to deserialize)
            - List of strings (list of JSON strings to deserialize)
            - Dictionary (potentially with nested JSON strings to deserialize)
            - Some other object that can't be deserialized.


    Returns:
        Deserialized data in the corresponding format:
            - Dictionary (when input is a single string)
            - List of dictionaries (when input is a list of strings)
            - Dictionary (when input is a dictionary, with nested JSON strings deserialized)
            - The original object (if there is no deserialization to perform)
    """
    # Case 1: Single string input
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data

    # Case 2: List of strings input
    elif isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, str):
                try:
                    result.append(json.loads(item))
                except json.JSONDecodeError:
                    result.append(item)
            else:
                # If list contains non-string items, recursively process them
                result.append(deserialize_json_values(item))
        return result

    # Case 3: Dictionary input with potential nested JSON strings
    elif isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            elif isinstance(value, (dict, list)):
                # Recursively process nested dictionaries and lists
                result[key] = deserialize_json_values(value)
            else:
                result[key] = value
        return result

    # Fallback for other data types
    else:
        return data


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
        return _clean_whitespace(json.loads(text_cleaned))
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (handles single quotes)
    try:
        return _clean_whitespace(ast.literal_eval(text_cleaned))
    except (ValueError, SyntaxError):
        pass

    # If all else fails, return the original text
    return [text.strip()]


def _clean_whitespace(texts: list[str]) -> list[str]:
    return [text.strip() for text in texts]


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
