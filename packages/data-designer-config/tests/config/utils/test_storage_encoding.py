# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.utils.io_helpers import read_parquet_dataset
from data_designer.config.utils.storage_encoding import (
    decode_storage_table,
    encode_storage_table,
    read_storage_plan,
    validate_storage_plans,
)
from data_designer.config.utils.storage_schema import JSON_STORAGE_METADATA_KEY, JSON_STORAGE_RUN_KEY, JsonStoragePlan


@pytest.mark.parametrize(
    "value",
    [
        {},
        [],
        None,
        "",
        "null",
        '{"ordinary":"text"}',
        'quotes"\n雪',
        True,
        False,
        0,
        2**100,
        1.25,
        {"alpha": 1},
        {"beta": None},
        [1, "two", True, None],
        {"nested": {"items": [1, "two", {"value": None}]}},
    ],
)
def test_json_values_round_trip(value: Any) -> None:
    original = deepcopy(value)
    frame = pd.DataFrame({"payload": pd.Series([value], dtype=object)})
    plan = JsonStoragePlan(columns={"payload": [[]]})
    table = encode_storage_table(frame, plan)
    stream = io.BytesIO()
    pq.write_table(table, stream)
    stream.seek(0)
    restored = decode_storage_table(pq.read_table(stream))
    assert restored.iloc[0]["payload"] == original
    assert frame.iloc[0]["payload"] == original
    assert table.schema.field("payload").type == pa.string()
    assert restored.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


@pytest.mark.parametrize("values", [[None], [], [{"calls": []}], [{"calls": [None]}]])
def test_null_empty_batches_have_explicit_nested_leaf_type(values: list[Any]) -> None:
    plan = JsonStoragePlan(columns={"oracle": [["calls", None, "arguments"]]})
    frame = pd.DataFrame({"oracle": pd.Series(values, dtype=object)})
    table = encode_storage_table(frame, plan)
    assert table.schema.field("oracle").type.field("calls").type.value_type.field("arguments").type == pa.string()
    assert decode_storage_table(table)["oracle"].tolist() == values


def test_multiple_nested_fields_round_trip_without_mutation() -> None:
    value = {"calls": [{"arguments": {"alpha": 1}, "other": [1, "two"]}, {"arguments": {"beta": None}, "other": []}]}
    original = deepcopy(value)
    plan = JsonStoragePlan(columns={"oracle": [["calls", None, "arguments"], ["calls", None, "other"]]})
    table = encode_storage_table(pd.DataFrame({"oracle": [value], "ordinary": ['{"x":1}']}), plan)
    assert decode_storage_table(table).to_dict("records") == [{"oracle": original, "ordinary": '{"x":1}'}]
    assert value == original


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {1: "key"}, (1, 2), {1, 2}, Decimal("1.25"), datetime(2026, 1, 1)]
)
def test_non_json_values_fail_without_stringification(value: Any) -> None:
    with pytest.raises(InvalidFileFormatError, match="column 'payload', row 0"):
        encode_storage_table(
            pd.DataFrame({"payload": pd.Series([value], dtype=object)}), JsonStoragePlan(columns={"payload": [[]]})
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b'{"version":2,"columns":{}}',
        b'{"version":true,"columns":{}}',
        b'{"version":1,"columns":{"missing":[[]]}}',
        b'{"version":1,"columns":{"payload":[[],[]]}}',
    ],
)
def test_invalid_contract_rejected(payload: bytes) -> None:
    schema = pa.schema([("payload", pa.string())], metadata={JSON_STORAGE_METADATA_KEY: payload})
    with pytest.raises(InvalidFileFormatError, match="metadata"):
        read_storage_plan(schema)


@pytest.mark.parametrize("text", ["not json", "NaN", "Infinity", '{"x":'])
def test_corrupt_encoded_values_fail(text: str) -> None:
    table = encode_storage_table(pd.DataFrame({"payload": [{}]}), JsonStoragePlan(columns={"payload": [[]]}))
    table = table.set_column(0, "payload", pa.array([text]))
    with pytest.raises(InvalidFileFormatError, match="decode.*payload"):
        decode_storage_table(table)


def test_directory_preserves_dynamic_keys_across_batches(tmp_path: Path) -> None:
    plan = JsonStoragePlan(columns={"payload": [[]]})
    values = [{"alpha": 1}, {"beta": None}, {}, [1, "two"], None]
    for index, value in enumerate(values):
        pq.write_table(
            encode_storage_table(pd.DataFrame({"payload": pd.Series([value], dtype=object)}), plan),
            tmp_path / f"batch_{index}.parquet",
        )
    result = read_parquet_dataset(tmp_path)
    assert result["payload"].tolist() == values
    assert result.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


def test_incompatible_legacy_and_encoded_files_rejected(tmp_path: Path) -> None:
    pq.write_table(
        encode_storage_table(pd.DataFrame({"payload": [{}]}), JsonStoragePlan(columns={"payload": [[]]})),
        tmp_path / "a.parquet",
    )
    pd.DataFrame({"payload": ['{"x":1}']}).to_parquet(tmp_path / "b.parquet")
    with pytest.raises(InvalidFileFormatError, match="Incompatible"):
        read_parquet_dataset(tmp_path)


def test_legacy_strings_are_not_decoded(tmp_path: Path) -> None:
    path = tmp_path / "legacy.parquet"
    pd.DataFrame({"payload": ['{"x":1}']}).to_parquet(path)
    assert read_parquet_dataset(path).iloc[0]["payload"] == '{"x":1}'


def test_metadata_ignores_other_namespaces() -> None:
    schema = pa.schema([("payload", pa.string())], metadata={b"third-party": json.dumps({"encoding": "json"}).encode()})
    assert not validate_storage_plans([schema]).columns
