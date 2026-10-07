# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.io_helpers import read_parquet_dataset
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage


@pytest.fixture
def storage(tmp_path: Path) -> ArtifactStorage:
    return ArtifactStorage(artifact_path=tmp_path)


def _checkpoint(storage: ArtifactStorage, batch_number: int, rows: list[dict]) -> Path:
    storage.write_batch_to_parquet_file(batch_number, lazy.pd.DataFrame(rows), BatchStage.PARTIAL_RESULT)
    return storage.move_partial_result_to_final_file_path(batch_number)


def _batch_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.parquet"))


def _assert_consistent(directory: Path) -> None:
    files = _batch_files(directory)
    schemas = [lazy.pq.read_schema(f) for f in files]
    assert all(schema.equals(schemas[0], check_metadata=True) for schema in schemas)
    lazy.pa.concat_tables([lazy.pq.read_table(f) for f in files], promote_options="default")
    lazy.pq.read_table(directory)
    lazy.pq.ParquetDataset(directory).read()


def test_int_then_float_column_is_widened_across_batches(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"value": 1}, {"value": 2}])
    _checkpoint(storage, 1, [{"value": 1.5}, {"value": 2.5}])

    _assert_consistent(storage.final_dataset_path)
    assert lazy.pq.read_schema(_batch_files(storage.final_dataset_path)[0]).field("value").type == lazy.pa.float64()
    assert storage.load_dataset()["value"].tolist() == [1.0, 2.0, 1.5, 2.5]


def test_float_then_int_column_keeps_float_type(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"value": 1.5}])
    _checkpoint(storage, 1, [{"value": 2}])

    _assert_consistent(storage.final_dataset_path)
    assert lazy.pq.read_schema(storage.final_dataset_path / "batch_00001.parquet").field("value").type == (
        lazy.pa.float64()
    )


def test_all_null_column_in_first_batch_takes_later_concrete_type(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"id": 1, "note": None}, {"id": 2, "note": None}])
    _checkpoint(storage, 1, [{"id": 3, "note": "hello"}])

    _assert_consistent(storage.final_dataset_path)
    assert lazy.pq.read_schema(storage.final_dataset_path / "batch_00000.parquet").field("note").type in (
        lazy.pa.string(),
        lazy.pa.large_string(),
    )
    assert storage.load_dataset()["note"].isna().tolist() == [True, True, False]


def test_all_null_column_in_later_batch_takes_earlier_concrete_type(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"id": 1, "note": "hello"}])
    _checkpoint(storage, 1, [{"id": 2, "note": None}])

    _assert_consistent(storage.final_dataset_path)
    assert not lazy.pa.types.is_null(
        lazy.pq.read_schema(storage.final_dataset_path / "batch_00001.parquet").field("note").type
    )


def test_nested_struct_with_differing_keys_is_unioned(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"payload": {"a": 1}}])
    _checkpoint(storage, 1, [{"payload": {"a": 2.5, "b": "x"}}])
    _checkpoint(storage, 2, [{"payload": {"c": [1, 2]}}])

    _assert_consistent(storage.final_dataset_path)
    rows = storage.load_dataset()["payload"].tolist()
    assert rows[0] == {"a": 1.0, "b": None, "c": None}
    assert rows[1] == {"a": 2.5, "b": "x", "c": None}
    assert rows[2] == {"a": None, "b": None, "c": [1, 2]}


def test_column_missing_from_later_batch_is_filled_with_nulls(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"a": 1, "b": "x"}])
    _checkpoint(storage, 1, [{"a": 2}])

    _assert_consistent(storage.final_dataset_path)
    assert storage.load_dataset()["b"].isna().tolist() == [False, True]


def test_column_added_in_later_batch_is_added_to_earlier_batches(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"a": 1}])
    _checkpoint(storage, 1, [{"a": 2, "b": "x"}])

    _assert_consistent(storage.final_dataset_path)


def test_widening_does_not_touch_batch_layout(storage: ArtifactStorage) -> None:
    for batch_number, value in enumerate([1, 2, 3.5]):
        _checkpoint(storage, batch_number, [{"value": value}])

    assert [f.name for f in _batch_files(storage.final_dataset_path)] == [
        "batch_00000.parquet",
        "batch_00001.parquet",
        "batch_00002.parquet",
    ]
    assert not list(storage.final_dataset_path.glob("*.tmp*")) and not list(storage.final_dataset_path.glob("*.widen*"))
    assert not _batch_files(storage.partial_results_path)


def test_incompatible_types_fail_with_column_types_and_batch_number(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"value": 1}])

    with pytest.raises(ArtifactStorageError, match=r"column 'value'.*int64.*string.*batch 1"):
        _checkpoint(storage, 1, [{"value": "text"}])
    assert [f.name for f in _batch_files(storage.final_dataset_path)] == ["batch_00000.parquet"]


def test_lossy_int_to_float_widening_fails_loudly(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"value": 2**60 + 1}])

    with pytest.raises(ArtifactStorageError, match=r"'value' of batch 0.*batch 1.*losing data"):
        _checkpoint(storage, 1, [{"value": 1.5}])
    assert [f.name for f in _batch_files(storage.final_dataset_path)] == ["batch_00000.parquet"]
    assert lazy.pq.read_table(storage.final_dataset_path / "batch_00000.parquet")["value"].to_pylist() == [2**60 + 1]


def test_dropped_columns_batches_share_one_schema(storage: ArtifactStorage) -> None:
    storage.write_batch_to_parquet_file(0, lazy.pd.DataFrame({"x": [1], "y": [None]}), BatchStage.DROPPED_COLUMNS)
    storage.write_batch_to_parquet_file(1, lazy.pd.DataFrame({"x": [1.5], "y": ["s"]}), BatchStage.DROPPED_COLUMNS)

    _assert_consistent(storage.dropped_columns_dataset_path)


def test_processor_output_batches_share_one_schema(storage: ArtifactStorage) -> None:
    for batch_number, value in enumerate([1, 2.5]):
        storage.write_batch_to_parquet_file(
            batch_number,
            lazy.pd.DataFrame({"x": [value]}),
            BatchStage.PROCESSORS_OUTPUTS,
            subfolder="proc",
        )

    _assert_consistent(storage.processors_outputs_path / "proc")


def test_single_file_processor_outputs_are_not_unified_with_each_other(storage: ArtifactStorage) -> None:
    storage.write_parquet_file("a.parquet", lazy.pd.DataFrame({"x": [1]}), BatchStage.PROCESSORS_OUTPUTS)
    storage.write_parquet_file("b.parquet", lazy.pd.DataFrame({"y": ["s"]}), BatchStage.PROCESSORS_OUTPUTS)

    assert lazy.pq.read_schema(storage.processors_outputs_path / "a.parquet").names == ["x"]
    assert lazy.pq.read_schema(storage.processors_outputs_path / "b.parquet").names == ["y"]


def test_after_generation_rechunk_shares_one_schema(storage: ArtifactStorage) -> None:
    df = lazy.pd.DataFrame({"x": [1, 2, 3.5, 4.5], "n": [None, None, "a", "b"]})
    for i in range(0, len(df), 2):
        storage.write_batch_to_parquet_file(i // 2, df.iloc[i : i + 2], BatchStage.FINAL_RESULT)

    _assert_consistent(storage.final_dataset_path)


def test_read_parquet_dataset_round_trips_consistent_output(storage: ArtifactStorage) -> None:
    _checkpoint(storage, 0, [{"id": 1, "value": 1, "tags": ["a"]}])
    _checkpoint(storage, 1, [{"id": 2, "value": 2.5, "tags": ["b", "c"]}])

    df = read_parquet_dataset(storage.final_dataset_path)
    assert df["id"].tolist() == [1, 2]
    assert df["value"].tolist() == [1.0, 2.5]
    assert [list(v) for v in df["tags"]] == [["a"], ["b", "c"]]


def test_resume_widens_batches_written_by_earlier_session(tmp_path: Path) -> None:
    first = ArtifactStorage(artifact_path=tmp_path)
    _checkpoint(first, 0, [{"value": 1}])

    resumed = ArtifactStorage(artifact_path=tmp_path)
    _checkpoint(resumed, 1, [{"value": 2.5}])

    _assert_consistent(resumed.final_dataset_path)
