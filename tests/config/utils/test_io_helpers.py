# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import yaml

from data_designer.config.utils.io_helpers import serialize_data, smart_load_dataframe, smart_load_yaml
from data_designer.lazy_heavy_imports import np, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@patch("data_designer.config.utils.io_helpers.Path", autospec=True)
@patch("data_designer.config.utils.io_helpers.pd.read_csv", autospec=True)
@patch("data_designer.config.utils.io_helpers.pd.read_json", autospec=True)
@patch("data_designer.config.utils.io_helpers.pd.read_parquet", autospec=True)
def test_smart_load_dataframe(mock_read_parquet, mock_read_json, mock_read_csv, mock_path_cls, stub_dataframe):
    mock_read_parquet.return_value = stub_dataframe
    mock_read_json.return_value = stub_dataframe
    mock_read_csv.return_value = stub_dataframe

    # dataframe objects are passed through
    assert smart_load_dataframe(stub_dataframe).size == stub_dataframe.size

    # url based
    stub_base_url = "https://example.com/data.{extention}"
    url_csv = stub_base_url.format(extention="csv")
    smart_load_dataframe(url_csv)
    mock_read_csv.assert_called_once_with(url_csv)

    url_json = stub_base_url.format(extention="json")
    smart_load_dataframe(url_json)
    mock_read_json.assert_called_once_with(url_json, lines=True)

    url_parquet = stub_base_url.format(extention="parquet")
    smart_load_dataframe(url_parquet)
    mock_read_parquet.assert_called_once_with(url_parquet)

    url_unknown = stub_base_url.format(extention="unknown")
    with pytest.raises(ValueError):
        smart_load_dataframe(url_unknown)

    # local file based
    mock_read_csv.reset_mock()
    mock_read_json.reset_mock()
    mock_read_parquet.reset_mock()

    mock_path = MagicMock(autospec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix.lower.return_value = "csv"
    mock_path_cls.return_value = mock_path

    stub_base_path_str = "/some/path/to/data.{extension}"
    path_csv = stub_base_path_str.format(extension="csv")
    _ = smart_load_dataframe(path_csv)
    mock_read_csv.assert_called_once_with(mock_path)

    mock_path.reset_mock()
    mock_path.suffix.lower.return_value = "json"
    mock_path.exists.return_value = False
    path_json = stub_base_path_str.format(extension="json")
    with pytest.raises(FileNotFoundError):
        _ = smart_load_dataframe(Path(path_json))


def test_smart_load_yaml():
    stub_dict = {
        "hello": "world",
        "some": {"nested_strin": "string", "nested_date": date(2023, 10, 5), "nested_list": [1, 2, 3]},
    }
    stub_yaml_str = yaml.dump(stub_dict)

    assert smart_load_yaml(stub_dict) == stub_dict
    assert smart_load_yaml(stub_yaml_str) == stub_dict

    # create temp file
    with tempfile.NamedTemporaryFile(delete=True, suffix=".yaml") as temp_file:
        temp_file.write(stub_yaml_str.encode())
        temp_file.flush()
        assert smart_load_yaml(temp_file.name) == stub_dict

    with pytest.raises(FileNotFoundError, match="File not found"):
        smart_load_yaml("path/to/file/that/does/not/exist.yaml")

    with pytest.raises(ValueError, match="invalid yaml config format"):
        smart_load_yaml(1)

    with pytest.raises(ValueError, match="Loaded yaml must be a dict"):
        smart_load_yaml("invalid yaml with just a string")


@pytest.mark.parametrize(
    "test_case,input_data,expected_result,expected_error",
    [
        ("serialize_dict", {"key": "value", "number": 42}, '{"key": "value", "number": 42}', None),
        ("serialize_list", [1, 2, 3, "test"], '[1, 2, 3, "test"]', None),
        ("serialize_string", "test_string", "test_string", None),
        ("serialize_number_int", 42, "42", None),
        ("serialize_number_float", 3.14, "3.14", None),
        ("serialize_boolean", True, "True", None),
        (
            "serialize_complex_dict",
            {"nested": {"inner": "value"}, "list": [1, 2, 3], "string": "test"},
            '{"nested": {"inner": "value"}, "list": [1, 2, 3], "string": "test"}',
            None,
        ),
        ("serialize_invalid_type", object(), None, ValueError),
        ("serialize_none", None, None, ValueError),
        ("serialize_datetime", {"dt": datetime(2024, 1, 15, 10, 30, 45)}, '{"dt": "2024-01-15T10:30:45"}', None),
        ("serialize_date", {"d": date(2024, 1, 15)}, '{"d": "2024-01-15"}', None),
        (
            "serialize_timedelta",
            {"td": timedelta(days=1, hours=2, minutes=30, seconds=45)},
            '{"td": 95445.0}',
            None,
        ),
        ("serialize_decimal", {"dec": Decimal("123.45")}, '{"dec": 123.45}', None),
        ("serialize_bytes", {"b": b"hello"}, '{"b": "hello"}', None),
        ("serialize_set", {"s": {1, 2, 3}}, '{"s": [1, 2, 3]}', None),
        ("serialize_pd_series", {"series": pd.Series([1, 2, 3])}, '{"series": [1, 2, 3]}', None),
        ("serialize_pd_timestamp", {"ts": pd.Timestamp("2024-01-15 10:30:45")}, '{"ts": "2024-01-15T10:30:45"}', None),
        ("serialize_pd_na", {"na": pd.NA}, '{"na": null}', None),
        (
            "serialize_mixed_types",
            {
                "date": date(2024, 1, 15),
                "decimal": Decimal("99.99"),
                "bytes": b"data",
                "series": pd.Series([1, 2]),
            },
            '{"date": "2024-01-15", "decimal": 99.99, "bytes": "data", "series": [1, 2]}',
            None,
        ),
        ("serialize_np_int32", {"val": np.int32(42)}, '{"val": 42}', None),
        ("serialize_np_int64", {"val": np.int64(9999999999)}, '{"val": 9999999999}', None),
        ("serialize_np_float32", {"val": np.float32(3.14)}, '{"val": 3.140000104904175}', None),
        ("serialize_np_float64", {"val": np.float64(2.718281828)}, '{"val": 2.718281828}', None),
        ("serialize_np_bool_true", {"val": np.bool_(True)}, '{"val": true}', None),
        ("serialize_np_bool_false", {"val": np.bool_(False)}, '{"val": false}', None),
        ("serialize_np_array", {"arr": np.array([1, 2, 3, 4])}, '{"arr": [1, 2, 3, 4]}', None),
        (
            "serialize_np_datetime64",
            {"dt": np.datetime64("2024-01-15T10:30:45")},
            '{"dt": "2024-01-15T10:30:45"}',
            None,
        ),
        ("serialize_np_timedelta64", {"td": np.timedelta64(5, "D")}, '{"td": "5 days"}', None),
        (
            "serialize_numpy_mixed_types",
            {
                "int32": np.int32(100),
                "float64": np.float64(1.5),
                "bool": np.bool_(True),
                "array": np.array([10, 20]),
            },
            '{"int32": 100, "float64": 1.5, "bool": true, "array": [10, 20]}',
            None,
        ),
    ],
)
def test_serialize_data_scenarios(test_case, input_data, expected_result, expected_error):
    if expected_error:
        with pytest.raises(expected_error, match="Invalid data type"):
            serialize_data(input_data)
    else:
        result = serialize_data(input_data)
        assert result == expected_result
