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
import requests
import yaml

from data_designer.config.utils.io_helpers import (
    _maybe_rewrite_url,
    is_http_url,
    serialize_data,
    smart_load_dataframe,
    smart_load_yaml,
)
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


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_yaml_url(mock_requests: MagicMock) -> None:
    stub_dict = {"hello": "world", "nested": {"value": 1}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = yaml.dump(stub_dict).encode("utf-8")
    mock_requests.get.return_value = mock_response

    result = smart_load_yaml("https://example.com/config.yaml")

    assert result == stub_dict
    mock_requests.get.assert_called_once_with("https://example.com/config.yaml", timeout=10)


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_json_url(mock_requests: MagicMock) -> None:
    stub_dict = {"hello": "world", "nested": {"value": 1}}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"hello":"world","nested":{"value":1}}'
    mock_requests.get.return_value = mock_response

    result = smart_load_yaml("https://example.com/config.json")

    assert result == stub_dict
    mock_requests.get.assert_called_once_with("https://example.com/config.json", timeout=10)


def test_smart_load_yaml_from_url_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported config URL extension"):
        smart_load_yaml("https://example.com/config.txt")


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_url_fetch_failure(mock_requests: MagicMock) -> None:
    mock_requests.get.side_effect = requests.RequestException("connection failed")
    mock_requests.RequestException = requests.RequestException

    with pytest.raises(ValueError, match="Failed to fetch config URL"):
        smart_load_yaml("https://example.com/config.yaml")


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_url_parse_failure(mock_requests: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b":\n  - [\n"
    mock_requests.get.return_value = mock_response

    with pytest.raises(ValueError, match="Failed to parse config from URL"):
        smart_load_yaml("https://example.com/config.yaml")


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_url_string_payload_does_not_recurse(mock_requests: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"https://example.com/next.yaml"
    mock_requests.get.return_value = mock_response

    with pytest.raises(ValueError, match="Loaded yaml must be a dict"):
        smart_load_yaml("https://example.com/config.yaml")

    mock_requests.get.assert_called_once_with("https://example.com/config.yaml", timeout=10)


@pytest.mark.parametrize(
    "status_code,reason,error_text",
    [
        (401, "Unauthorized", "requires authentication"),
        (403, "Forbidden", "received 403 Forbidden"),
        (404, "Not Found", "received 404 Not Found"),
        (500, "Internal Server Error", r"received HTTP 500 \(Internal Server Error\)"),
    ],
)
@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_url_http_status_error(
    mock_requests: MagicMock, status_code: int, reason: str, error_text: str
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.reason = reason
    mock_requests.get.return_value = mock_response

    with pytest.raises(ValueError, match=error_text):
        smart_load_yaml("https://example.com/config.yaml")


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_url_exceeds_size_limit(mock_requests: MagicMock) -> None:
    from data_designer.config.utils.io_helpers import MAX_CONFIG_URL_SIZE_BYTES

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"x" * (MAX_CONFIG_URL_SIZE_BYTES + 1)
    mock_requests.get.return_value = mock_response

    with pytest.raises(ValueError, match="exceeds maximum size"):
        smart_load_yaml("https://example.com/config.yaml")


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_from_url_non_dict_payload(mock_requests: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"- item1\n- item2\n"
    mock_requests.get.return_value = mock_response

    with pytest.raises(ValueError, match="Failed to parse config from URL") as exc_info:
        smart_load_yaml("https://example.com/config.yaml")

    assert "list" in str(exc_info.value).lower()


def test_smart_load_yaml_from_url_no_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported config URL extension"):
        smart_load_yaml("https://example.com/config")


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com/config.yaml", True),
        ("http://example.com/config.yaml", True),
        ("https://example.com", True),
        ("http://localhost:8000", True),
        ("example.com", False),
        ("http://", False),
        ("https://", False),
        ("", False),
        ("ftp://example.com/config.yaml", False),
        ("not-a-url", False),
        ("/local/path/config.yaml", False),
    ],
)
def test_is_http_url(url: str, expected: bool) -> None:
    assert is_http_url(url) == expected


@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "https://github.com/org/repo/blob/main/config.yaml",
            "https://raw.githubusercontent.com/org/repo/main/config.yaml",
        ),
        (
            "https://www.github.com/org/repo/blob/main/config.yaml",
            "https://raw.githubusercontent.com/org/repo/main/config.yaml",
        ),
        (
            "https://github.com/org/repo/blob/main/deep/path/f.yaml",
            "https://raw.githubusercontent.com/org/repo/main/deep/path/f.yaml",
        ),
        (
            "https://github.com/org/repo/blob/main/f.yaml#L10-L20",
            "https://raw.githubusercontent.com/org/repo/main/f.yaml",
        ),
        (
            "https://github.com/org/repo/blob/main/f.yaml?token=abc",
            "https://raw.githubusercontent.com/org/repo/main/f.yaml?token=abc",
        ),
        (
            "https://github.com/blob/repo/blob/main/f.yaml",
            "https://raw.githubusercontent.com/blob/repo/main/f.yaml",
        ),
        (
            "https://github.com/org/repo/tree/main/some-dir",
            "https://github.com/org/repo/tree/main/some-dir",
        ),
        (
            "https://raw.githubusercontent.com/org/repo/main/f.yaml",
            "https://raw.githubusercontent.com/org/repo/main/f.yaml",
        ),
        (
            "https://example.com/config.yaml",
            "https://example.com/config.yaml",
        ),
        (
            "https://github.com/org/repo",
            "https://github.com/org/repo",
        ),
        (
            "https://huggingface.co/datasets/nabinnvidia/multi-lingual-greetings/blob/main/builder_config.json",
            "https://huggingface.co/datasets/nabinnvidia/multi-lingual-greetings/raw/main/builder_config.json",
        ),
        (
            "https://www.huggingface.co/datasets/nabinnvidia/multi-lingual-greetings/blob/main/builder_config.json",
            "https://huggingface.co/datasets/nabinnvidia/multi-lingual-greetings/raw/main/builder_config.json",
        ),
        (
            "https://huggingface.co/datasets/org/repo/blob/main/f.yaml#L10-L20",
            "https://huggingface.co/datasets/org/repo/raw/main/f.yaml",
        ),
        (
            "https://huggingface.co/datasets/org/repo/blob/main/f.yaml?download=1",
            "https://huggingface.co/datasets/org/repo/raw/main/f.yaml?download=1",
        ),
        (
            "https://huggingface.co/org/repo/blob/main/f.yaml",
            "https://huggingface.co/org/repo/raw/main/f.yaml",
        ),
        (
            "https://huggingface.co/datasets/org/repo/raw/main/f.yaml",
            "https://huggingface.co/datasets/org/repo/raw/main/f.yaml",
        ),
    ],
)
def test_maybe_rewrite_url(url: str, expected: str) -> None:
    assert _maybe_rewrite_url(url) == expected


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_rewrites_github_blob_url(mock_requests: MagicMock) -> None:
    stub_dict = {"hello": "world"}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = yaml.dump(stub_dict).encode("utf-8")
    mock_requests.get.return_value = mock_response

    result = smart_load_yaml("https://github.com/org/repo/blob/main/config.yaml")

    assert result == stub_dict
    mock_requests.get.assert_called_once_with("https://raw.githubusercontent.com/org/repo/main/config.yaml", timeout=10)


@patch("data_designer.config.utils.io_helpers.requests")
def test_smart_load_yaml_rewrites_huggingface_blob_url(mock_requests: MagicMock) -> None:
    stub_dict = {"hello": "world"}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = yaml.dump(stub_dict).encode("utf-8")
    mock_requests.get.return_value = mock_response

    result = smart_load_yaml("https://huggingface.co/datasets/org/repo/blob/main/config.yaml")

    assert result == stub_dict
    mock_requests.get.assert_called_once_with(
        "https://huggingface.co/datasets/org/repo/raw/main/config.yaml", timeout=10
    )


@patch("data_designer.config.utils.io_helpers.pd.read_csv", autospec=True)
def test_smart_load_dataframe_rewrites_github_blob_url(mock_read_csv: MagicMock, stub_dataframe: pd.DataFrame) -> None:
    mock_read_csv.return_value = stub_dataframe

    smart_load_dataframe("https://github.com/org/repo/blob/main/data.csv")

    mock_read_csv.assert_called_once_with("https://raw.githubusercontent.com/org/repo/main/data.csv")


@patch("data_designer.config.utils.io_helpers.pd.read_csv", autospec=True)
def test_smart_load_dataframe_rewrites_github_blob_url_with_token(
    mock_read_csv: MagicMock, stub_dataframe: pd.DataFrame
) -> None:
    mock_read_csv.return_value = stub_dataframe

    smart_load_dataframe("https://github.com/org/repo/blob/main/data.csv?token=secret123")

    mock_read_csv.assert_called_once_with("https://raw.githubusercontent.com/org/repo/main/data.csv?token=secret123")


@patch("data_designer.config.utils.io_helpers.pd.read_csv", autospec=True)
def test_smart_load_dataframe_rewrites_huggingface_blob_url(
    mock_read_csv: MagicMock, stub_dataframe: pd.DataFrame
) -> None:
    mock_read_csv.return_value = stub_dataframe

    smart_load_dataframe("https://huggingface.co/datasets/org/repo/blob/main/data.csv")

    mock_read_csv.assert_called_once_with("https://huggingface.co/datasets/org/repo/raw/main/data.csv")


def test_maybe_rewrite_github_url_log_does_not_leak_query(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="data_designer.config.utils.io_helpers"):
        _maybe_rewrite_url("https://github.com/org/repo/blob/main/f.yaml?token=secret123")

    assert len(caplog.records) == 1
    assert "secret123" not in caplog.records[0].message


def test_maybe_rewrite_huggingface_url_log_does_not_leak_query(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="data_designer.config.utils.io_helpers"):
        _maybe_rewrite_url("https://huggingface.co/datasets/org/repo/blob/main/f.yaml?token=secret123")

    assert len(caplog.records) == 1
    assert "secret123" not in caplog.records[0].message


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
