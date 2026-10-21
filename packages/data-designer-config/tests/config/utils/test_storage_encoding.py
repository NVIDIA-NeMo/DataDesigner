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
    decode_storage_dataframe,
    decode_storage_table,
    encode_storage_table,
    read_dataframe_storage_plan,
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


@pytest.mark.parametrize(
    ("dtype", "path"),
    [
        (pa.int64(), []),
        (pa.null(), []),
        (pa.binary(), []),
        (pa.struct([("actual", pa.string())]), ["missing"]),
        (pa.struct([("value", pa.string()), ("value", pa.string())]), ["value"]),
        (pa.struct([("value", pa.int64())]), ["value"]),
        (pa.list_(pa.int64()), [None]),
        (pa.string(), ["value"]),
        (pa.struct([("value", pa.string())]), [None]),
    ],
)
@pytest.mark.parametrize("num_rows", [0, 1])
def test_contract_validates_physical_paths_before_decoding_rows(
    dtype: pa.DataType, path: list[str | None], num_rows: int
) -> None:
    plan = JsonStoragePlan(columns={"payload": [path]})
    table = pa.Table.from_arrays([pa.array([None] * num_rows, type=dtype)], names=["payload"])
    table = table.replace_schema_metadata({JSON_STORAGE_METADATA_KEY: plan.model_dump_json().encode()})
    with pytest.raises(InvalidFileFormatError, match="metadata"):
        decode_storage_table(table)


def test_encoded_contract_rejects_duplicate_physical_columns() -> None:
    plan = JsonStoragePlan(columns={"payload": [[]]})
    schema = pa.schema(
        [("payload", pa.string()), ("payload", pa.string())],
        metadata={JSON_STORAGE_METADATA_KEY: plan.model_dump_json().encode()},
    )
    with pytest.raises(InvalidFileFormatError, match="metadata"):
        read_storage_plan(schema)


@pytest.mark.parametrize(
    "dtype", [pa.string(), pa.large_string(), pa.list_(pa.large_string()), pa.large_list(pa.string())]
)
def test_valid_utf8_physical_contracts_accept_null_only_batches(dtype: pa.DataType) -> None:
    path = [None] if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) else []
    plan = JsonStoragePlan(columns={"payload": [path]})
    table = pa.Table.from_arrays([pa.array([None], type=dtype)], names=["payload"])
    table = table.replace_schema_metadata({JSON_STORAGE_METADATA_KEY: plan.model_dump_json().encode()})
    assert decode_storage_table(table)["payload"].tolist() == [None]


def test_dataframe_contract_restricts_projected_columns_without_mutation() -> None:
    frame = pd.DataFrame({"kept": [{"x": 1}], "native": [1]})
    plan = JsonStoragePlan(columns={"kept": [[]], "removed": [[]]})
    frame.attrs[JSON_STORAGE_RUN_KEY] = plan.model_dump()
    assert read_dataframe_storage_plan(frame) == JsonStoragePlan(columns={"kept": [[]]})
    assert frame.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


@pytest.mark.parametrize(
    "contract",
    [None, [], "json", {"version": 2}, {"version": True}, {"columns": {"payload": []}}],
)
def test_dataframe_invalid_contract_has_canonical_error(contract: Any) -> None:
    frame = pd.DataFrame({"payload": [{}]})
    frame.attrs[JSON_STORAGE_RUN_KEY] = contract
    with pytest.raises(InvalidFileFormatError, match="DataFrame"):
        read_dataframe_storage_plan(frame)


@pytest.mark.parametrize("nested", [False, True])
def test_decoding_query_null_sentinels_preserves_null_values(nested: bool) -> None:
    plan = JsonStoragePlan(columns={"payload": [["arguments"]] if nested else [[]]})
    values = [float("nan"), {"arguments": float("nan")}] if nested else [float("nan"), '{"value":1}']
    frame = pd.DataFrame({"payload": values})
    expected = [None, {"arguments": None}] if nested else [None, {"value": 1}]
    assert decode_storage_dataframe(frame, plan)["payload"].tolist() == expected


def test_legacy_partitioned_directory_keeps_native_columns(tmp_path: Path) -> None:
    values = pd.DataFrame({"group": ["a", "b"], "value": [1, 2]})
    values.to_parquet(tmp_path, partition_cols=["group"], index=False)
    result = read_parquet_dataset(tmp_path).sort_values("value")
    assert result.to_dict("records") == values.to_dict("records")
    assert JSON_STORAGE_RUN_KEY not in result.attrs


def test_encoded_partitioned_directory_restores_json_and_partition_columns(tmp_path: Path) -> None:
    values = [{"alpha": 1}, {"beta": [1, "two"]}, None]
    frame = pd.DataFrame({"group": ["a", "b", "c"], "payload": pd.Series(values, dtype=object)})
    plan = JsonStoragePlan(columns={"payload": [[]]})
    pq.write_to_dataset(encode_storage_table(frame, plan), root_path=tmp_path, partition_cols=["group"])
    result = read_parquet_dataset(tmp_path)
    assert {row["group"]: row["payload"] for row in result.to_dict("records")} == dict(zip(["a", "b", "c"], values))
    assert result.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


def test_partitioned_directory_validates_every_fragment_contract(tmp_path: Path) -> None:
    plan = JsonStoragePlan(columns={"payload": [[]]})
    encoded = encode_storage_table(pd.DataFrame({"group": ["a"], "payload": [{"alpha": 1}]}), plan)
    pq.write_to_dataset(encoded, root_path=tmp_path, partition_cols=["group"])
    pd.DataFrame({"group": ["b"], "payload": ['{"beta":2}']}).to_parquet(
        tmp_path, partition_cols=["group"], index=False
    )
    with pytest.raises(InvalidFileFormatError, match="Incompatible"):
        read_parquet_dataset(tmp_path)
