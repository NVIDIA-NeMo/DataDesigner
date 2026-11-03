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


@pytest.mark.parametrize("counts", [(1, 3, 2), (3, 3, 3), (0, 0, 0)], ids=["variable", "uniform", "empty"])
def test_nested_null_side_effect_survives_parquet_read(tmp_path: Path, counts: tuple[int, ...]) -> None:
    values = [[{"entries": [{"label": "synthetic", "optional": None} for _ in range(count)]}] for count in counts]
    plan = JsonStoragePlan(columns={"payload": [[]]})
    frame = pd.DataFrame({"side_effect": values, "payload": [{"x": 1}] * len(values), "ordinary": [1, 2, 3]})
    table = encode_storage_table(frame, plan)
    path = tmp_path / "batch.parquet"
    pq.write_table(table, path)
    original = pq.read_table(path)
    original.validate(full=True)

    restored = read_parquet_dataset(path)

    restored["side_effect"].array.__arrow_array__().validate(full=True)
    assert restored["side_effect"].tolist() == values
    assert isinstance(restored["side_effect"].dtype, pd.ArrowDtype)
    assert isinstance(restored["ordinary"].dtype, pd.ArrowDtype)
    assert restored["ordinary"].tolist() == [1, 2, 3]
    assert restored["payload"].tolist() == [{"x": 1}] * len(values)
    assert restored.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()
    assert list(restored.columns) == list(frame.columns)
    assert original["side_effect"].to_pylist() == values


def test_multiple_nested_fields_round_trip_without_mutation() -> None:
    value = {"calls": [{"arguments": {"alpha": 1}, "other": [1, "two"]}, {"arguments": {"beta": None}, "other": []}]}
    original = deepcopy(value)
    plan = JsonStoragePlan(columns={"oracle": [["calls", None, "arguments"], ["calls", None, "other"]]})
    table = encode_storage_table(pd.DataFrame({"oracle": [value], "ordinary": ['{"x":1}']}), plan)
    assert decode_storage_table(table).to_dict("records") == [{"oracle": original, "ordinary": '{"x":1}'}]
    assert value == original


@pytest.mark.parametrize("size", [0, 3])
def test_all_nested_columns_preserve_shape_and_values(size: int) -> None:
    values = [[{"value": None}], [], None][:size]
    nested_type = pa.list_(pa.struct([("value", pa.null())]))
    table = pa.table({"first": pa.array(values, type=nested_type), "second": pa.array(values, type=nested_type)})

    restored = decode_storage_table(table)

    assert restored.shape == (size, 2)
    assert list(restored.columns) == ["first", "second"]
    for name in table.column_names:
        assert isinstance(restored[name].dtype, pd.ArrowDtype)
        assert restored[name].array.__arrow_array__().to_pylist() == values
        restored[name].array.__arrow_array__().validate(full=True)


@pytest.mark.parametrize("index_kind", ["range", "named", "multi"])
def test_nested_read_preserves_index_and_interleaved_column_order(index_kind: str) -> None:
    index = pd.RangeIndex(4, 10, 2, name="row")
    if index_kind == "named":
        index = pd.Index([7, 2, 7], name="row")
    elif index_kind == "multi":
        index = pd.MultiIndex.from_tuples([("a", 2), ("b", 1), ("a", 2)], names=["group", "row"])
    values = [[{"optional": None}], [], None]
    frame = pd.DataFrame({"nested_first": values, "ordinary": [1, 2, 3], "nested_last": values}, index=index)
    table = pa.Table.from_pandas(frame)

    restored = decode_storage_table(table)

    pd.testing.assert_index_equal(restored.index, table.to_pandas(types_mapper=pd.ArrowDtype).index)
    assert list(restored.columns) == list(frame.columns)
    assert restored["ordinary"].tolist() == [1, 2, 3]
    assert restored["nested_first"].array.__arrow_array__().to_pylist() == values
    assert restored["nested_last"].array.__arrow_array__().to_pylist() == values


def test_nested_read_preserves_sliced_chunked_arrays() -> None:
    values = [[{"optional": None}] * count for count in (1, 3, 2, 4)]
    array = pa.chunked_array([pa.array(values[:2]), pa.array(values[2:])]).slice(1, 2)
    table = pa.table({"nested": array, "ordinary": [1, 2]})

    restored = decode_storage_table(table)

    restored["nested"].array.__arrow_array__().validate(full=True)
    assert restored["nested"].tolist() == values[1:3]
    assert table["nested"].to_pylist() == values[1:3]


@pytest.mark.parametrize("reader", ["decoder", "public"])
@pytest.mark.parametrize("column_kind", ["integer", "multi"])
def test_nested_read_preserves_non_string_column_labels(tmp_path: Path, reader: str, column_kind: str) -> None:
    columns = pd.Index([0, 1], name="fields")
    if column_kind == "multi":
        columns = pd.MultiIndex.from_tuples([("nested", 0), ("ordinary", 1)], names=["kind", "number"])
    frame = pd.DataFrame([[["{}"], 10], [["{}"], 20]], columns=columns)
    table = pa.Table.from_pandas(frame)
    plan = JsonStoragePlan(columns={table.column_names[0]: [[None]]})
    table = table.replace_schema_metadata(
        {**table.schema.metadata, JSON_STORAGE_METADATA_KEY: plan.model_dump_json().encode()}
    )
    path = tmp_path / "batch.parquet"
    pq.write_table(table, path)

    restored = decode_storage_table(table) if reader == "decoder" else read_parquet_dataset(path)

    pd.testing.assert_index_equal(restored.columns, frame.columns)
    assert restored.iloc[:, 0].tolist() == [[{}], [{}]]
    assert restored.iloc[:, 1].tolist() == [10, 20]
    assert isinstance(restored.iloc[:, 1].dtype, pd.ArrowDtype)
    assert restored.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


@pytest.mark.parametrize("reader", ["decoder", "public"])
@pytest.mark.parametrize("index_name", [None, "row", "ordinary"])
def test_nested_read_keeps_tuple_index_out_of_data_columns(tmp_path: Path, reader: str, index_name: str | None) -> None:
    index = pd.Index([(1, 2), (3, 4)], tupleize_cols=False, name=index_name)
    frame = pd.DataFrame({"payload": ['{"x":1}', '{"x":2}'], "ordinary": [10, 20]}, index=index)
    table = pa.Table.from_pandas(frame)
    plan = JsonStoragePlan(columns={"payload": [[]]})
    table = table.replace_schema_metadata(
        {**table.schema.metadata, JSON_STORAGE_METADATA_KEY: plan.model_dump_json().encode()}
    )
    path = tmp_path / "batch.parquet"
    pq.write_table(table, path)

    restored = decode_storage_table(table) if reader == "decoder" else read_parquet_dataset(path)

    pd.testing.assert_index_equal(restored.columns, frame.columns)
    expected_index = table.to_pandas(types_mapper=pd.ArrowDtype).index if reader == "decoder" else pd.RangeIndex(2)
    pd.testing.assert_index_equal(restored.index, expected_index)
    assert restored["payload"].tolist() == [{"x": 1}, {"x": 2}]
    assert restored["ordinary"].tolist() == [10, 20]
    assert restored.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


@pytest.mark.parametrize("size", [0, 3])
@pytest.mark.parametrize("directory", [False, True], ids=["file", "directory"])
@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pa.list_(pa.int64()), [[1, 2], [], None]),
        (pa.large_list(pa.int64()), [[1, 2], [], None]),
        (pa.list_(pa.int64(), 2), [[1, 2], [3, 4], [5, 6]]),
        (pa.struct([("value", pa.int64())]), [{"value": 1}, {"value": None}, None]),
        (pa.map_(pa.string(), pa.int64()), [[("a", 1)], [], None]),
        (pa.list_(pa.struct([("optional", pa.null())])), [[{"optional": None}], [], None]),
        (pa.list_(pa.int64()), [None, None, None]),
    ],
    ids=["list", "large-list", "fixed-list", "struct", "map", "nested-null", "all-null"],
)
def test_annotated_nested_columns_survive_repeated_persisted_reads(
    tmp_path: Path, size: int, directory: bool, dtype: pa.DataType, values: list[Any]
) -> None:
    plan = JsonStoragePlan(columns={"payload": [[]]})
    table = pa.table(
        {
            "nested": pa.array(values[:size], type=dtype),
            "ordinary": pa.array(range(size), type=pa.int64()),
            "payload": pa.array(['{"x":1}'] * size, type=pa.string()),
        }
    ).replace_schema_metadata({JSON_STORAGE_METADATA_KEY: plan.model_dump_json().encode(), b"external": b"preserve"})

    for cycle in range(3):
        metadata = table.schema.metadata
        path = tmp_path / f"cycle_{cycle}"
        path.mkdir()
        pq.write_table(table, path / "batch.parquet")
        restored = read_parquet_dataset(path if directory else path / "batch.parquet")

        assert restored.shape == (size, 3)
        assert restored["nested"].array.__arrow_array__().to_pylist() == values[:size]
        restored["nested"].array.__arrow_array__().validate(full=True)
        assert isinstance(restored["nested"].dtype, pd.ArrowDtype)
        assert restored["ordinary"].tolist() == list(range(size))
        assert restored["payload"].tolist() == [{"x": 1}] * size
        assert restored.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()
        assert table.schema.metadata == metadata
        assert metadata[b"external"] == b"preserve"
        table = encode_storage_table(restored, plan)
        table = table.replace_schema_metadata({**table.schema.metadata, b"external": b"preserve"})


@pytest.mark.parametrize("size", [0, 3])
@pytest.mark.parametrize("column_kind", ["string", "integer", "multi"])
@pytest.mark.parametrize("index_kind", ["range", "named", "tuple", "datetime", "categorical", "multi"])
def test_nested_decoder_preserves_metadata_across_repeated_persistence(
    tmp_path: Path, size: int, column_kind: str, index_kind: str
) -> None:
    index = pd.RangeIndex(4, 10, 2, name="row")
    if index_kind == "named":
        index = pd.Index([7, 2, 7], name="row")
    elif index_kind == "tuple":
        index = pd.Index([(1, 2), (3, 4), (1, 2)], tupleize_cols=False, name="row")
    elif index_kind == "datetime":
        index = pd.date_range("2026-01-01", periods=3, tz="Asia/Kolkata", name="row")
    elif index_kind == "categorical":
        index = pd.CategoricalIndex(["a", "b", "a"], name="row")
    elif index_kind == "multi":
        index = pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("a", 1)], names=["group", "row"])
    columns = pd.Index(["nested", "ordinary"], name="fields")
    if column_kind == "integer":
        columns = pd.Index([0, 1], name="fields")
    elif column_kind == "multi":
        columns = pd.MultiIndex.from_tuples([("nested", 0), ("ordinary", 1)], names=["kind", "number"])
    frame = pd.DataFrame([[[1, 2], 10], [[], 20], [None, 30]], index=index, columns=columns).iloc[:size]
    frame.attrs["external"] = {"keep": [1, 2]}
    table = pa.Table.from_pandas(frame)
    expected = None

    for cycle in range(3):
        path = tmp_path / f"cycle_{cycle}.parquet"
        pq.write_table(table, path)
        persisted = pq.read_table(path)
        original_metadata = persisted.schema.metadata
        if expected is None:
            expected = persisted.to_pandas(types_mapper=pd.ArrowDtype)
        restored = decode_storage_table(persisted)

        pd.testing.assert_frame_equal(restored, expected)
        assert restored.attrs == frame.attrs
        assert persisted.schema.metadata == original_metadata
        table = pa.Table.from_pandas(restored)


@pytest.mark.parametrize("size", [0, 3])
@pytest.mark.parametrize("keep_nested", [False, True])
@pytest.mark.parametrize("directory", [False, True], ids=["file", "directory"])
def test_projected_nested_metadata_keeps_only_surviving_fields(
    tmp_path: Path, size: int, keep_nested: bool, directory: bool
) -> None:
    values = [[1, 2], [], None][:size]
    array = pa.array(values, type=pa.list_(pa.int64()))
    frame = pd.DataFrame(
        {
            "keep": pd.arrays.ArrowExtensionArray(array),
            "drop": pd.arrays.ArrowExtensionArray(array),
            "ordinary": list(range(size)),
            "payload": pd.Series([{}] * size, dtype=object),
        }
    )
    plan = JsonStoragePlan(columns={"payload": [[]]})
    columns = ["keep", "ordinary", "payload"] if keep_nested else ["ordinary", "payload"]
    table = encode_storage_table(frame, plan).select(columns)
    original_metadata = table.schema.metadata

    for cycle in range(3):
        path = tmp_path / f"cycle_{cycle}"
        path.mkdir()
        pq.write_table(table, path / "batch.parquet")
        restored = read_parquet_dataset(path if directory else path / "batch.parquet")

        assert list(restored.columns) == columns
        assert restored["ordinary"].tolist() == list(range(size))
        assert restored["payload"].tolist() == [{}] * size
        assert restored.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()
        if keep_nested:
            assert restored["keep"].array.__arrow_array__().to_pylist() == values
        if cycle == 0:
            assert table.schema.metadata == original_metadata
        table = encode_storage_table(restored, plan)


@pytest.mark.parametrize("size", [0, 3])
@pytest.mark.parametrize("directory", [False, True], ids=["file", "directory"])
def test_all_nested_annotated_columns_survive_repeated_persistence(tmp_path: Path, size: int, directory: bool) -> None:
    plan = JsonStoragePlan(columns={"protected": [["payload"]]})
    frame = pd.DataFrame(
        {"protected": [{"payload": {"a": 1}}, {"payload": None}, {"payload": []}], "native": [[1, 2], [], None]}
    ).iloc[:size]
    expected = frame.to_dict("records")

    for cycle in range(3):
        table = encode_storage_table(frame, plan)
        path = tmp_path / f"cycle_{cycle}"
        path.mkdir()
        pq.write_table(table, path / "batch.parquet")
        frame = read_parquet_dataset(path if directory else path / "batch.parquet")

        assert len(frame) == size
        assert frame.to_dict("records") == expected
        assert frame.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()
        frame["native"].array.__arrow_array__().validate(full=True)


@pytest.mark.parametrize("payload", [b'{"version":99,"columns":{}}', b'{"version":1,"columns":{"missing":[[]]}}'])
def test_metadata_filter_does_not_relax_invalid_json_contracts(payload: bytes) -> None:
    table = pa.table({"native": pa.array([[1, 2]], type=pa.list_(pa.int64()))})
    table = table.replace_schema_metadata({JSON_STORAGE_METADATA_KEY: payload, b"external": b"preserve"})
    metadata = table.schema.metadata

    with pytest.raises(InvalidFileFormatError, match="metadata"):
        decode_storage_table(table)

    assert table.schema.metadata == metadata
    assert table["native"].to_pylist() == [[1, 2]]


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


@pytest.mark.parametrize("as_directory", [True, False], ids=["directory", "file"])
@pytest.mark.parametrize(
    ("dtype_backend", "count_check", "label_check"),
    [
        ("pyarrow", lambda dtype: isinstance(dtype, pd.ArrowDtype), lambda dtype: isinstance(dtype, pd.ArrowDtype)),
        ("numpy_nullable", lambda dtype: dtype == pd.Int64Dtype(), lambda dtype: dtype == pd.StringDtype()),
        (None, lambda dtype: dtype == "int64", lambda dtype: pd.api.types.is_string_dtype(dtype)),
    ],
)
def test_encoded_dataset_honors_dtype_backend_for_native_columns(
    tmp_path: Path, as_directory: bool, dtype_backend: Any, count_check: Any, label_check: Any
) -> None:
    values = [{"alpha": 1}, {"beta": [1, "two"]}]
    frame = pd.DataFrame({"count": [1, 2], "label": ["a", "b"], "payload": pd.Series(values, dtype=object)})
    plan = JsonStoragePlan(columns={"payload": [[]]})
    path = tmp_path / "batch_0.parquet"
    pq.write_table(encode_storage_table(frame, plan), path)

    result = read_parquet_dataset(tmp_path if as_directory else path, dtype_backend=dtype_backend)

    assert count_check(result["count"].dtype)
    assert label_check(result["label"].dtype)
    assert result["payload"].dtype == object
    assert result["payload"].tolist() == values
    assert result.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()


def test_decode_storage_table_rejects_unknown_dtype_backend() -> None:
    table = pa.Table.from_pylist([{"payload": "1"}]).replace_schema_metadata(
        {JSON_STORAGE_METADATA_KEY: JsonStoragePlan(columns={"payload": [[]]}).model_dump_json().encode()}
    )
    with pytest.raises(ValueError, match="dtype_backend must be"):
        decode_storage_table(table, dtype_backend="bogus")  # type: ignore[arg-type]


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


def test_deeply_malformed_metadata_has_canonical_format_error() -> None:
    payload = b'{"version":1,"columns":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}"
    schema = pa.schema([("payload", pa.string())], metadata={JSON_STORAGE_METADATA_KEY: payload})
    with pytest.raises(InvalidFileFormatError, match="metadata"):
        read_storage_plan(schema)


@pytest.mark.parametrize("mixed", [False, True])
def test_native_nan_siblings_keep_pandas_null_semantics(mixed: bool) -> None:
    values = [{"arguments": {"x": 1}, "native": float("nan")}]
    if mixed:
        values.append({"arguments": {"x": 2}, "native": 1.0})
    frame = pd.DataFrame({"payload": values})
    plan = JsonStoragePlan(columns={"payload": [["arguments"]]})
    table = encode_storage_table(frame, plan)
    stream = io.BytesIO()
    pq.write_table(table, stream)
    stream.seek(0)
    restored = decode_storage_table(pq.read_table(stream))["payload"].tolist()
    expected = [{"arguments": {"x": 1}, "native": None}]
    if mixed:
        expected.append({"arguments": {"x": 2}, "native": 1.0})
    assert restored == expected
    assert pd.isna(frame.iloc[0]["payload"]["native"])


def test_native_null_normalization_does_not_relax_annotated_json_values() -> None:
    frame = pd.DataFrame({"payload": [{"arguments": {"x": float("nan")}, "native": float("nan")}]})
    with pytest.raises(InvalidFileFormatError, match="column 'payload', row 0"):
        encode_storage_table(frame, JsonStoragePlan(columns={"payload": [["arguments"]]}))
