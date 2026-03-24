# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager


def _mock_artifact_storage() -> MagicMock:
    storage = MagicMock()
    storage.dataset_name = "test_dataset"
    storage.get_file_paths.return_value = {"parquet-files": []}
    return storage


def test_init_row_group() -> None:
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 3)

    row = mgr.get_row(0, 0)
    assert row == {}


def test_update_cell() -> None:
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 2)

    mgr.update_cell(0, 0, "col_a", "val_0")
    mgr.update_cell(0, 1, "col_a", "val_1")

    assert mgr.get_row(0, 0) == {"col_a": "val_0"}
    assert mgr.get_row(0, 1) == {"col_a": "val_1"}


def test_update_cells() -> None:
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 1)

    mgr.update_cells(0, 0, {"col_a": "a", "col_b": "b"})

    assert mgr.get_row(0, 0) == {"col_a": "a", "col_b": "b"}


def test_update_batch() -> None:
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 3)

    mgr.update_batch(0, "col_a", ["x", "y", "z"])

    assert mgr.get_row(0, 0) == {"col_a": "x"}
    assert mgr.get_row(0, 1) == {"col_a": "y"}
    assert mgr.get_row(0, 2) == {"col_a": "z"}


def test_get_dataframe_excludes_dropped() -> None:
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 3)

    mgr.update_batch(0, "val", [1, 2, 3])
    mgr.drop_row(0, 1)

    df = mgr.get_dataframe(0)
    assert len(df) == 2
    assert list(df["val"]) == [1, 3]


def test_drop_row_and_is_dropped() -> None:
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 3)

    assert not mgr.is_dropped(0, 1)
    mgr.drop_row(0, 1)
    assert mgr.is_dropped(0, 1)
    assert not mgr.is_dropped(0, 0)


def test_concurrent_row_groups() -> None:
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 2)
    mgr.init_row_group(1, 3)

    mgr.update_cell(0, 0, "a", "rg0_r0")
    mgr.update_cell(1, 0, "a", "rg1_r0")
    mgr.update_cell(1, 2, "a", "rg1_r2")

    assert mgr.get_row(0, 0) == {"a": "rg0_r0"}
    assert mgr.get_row(1, 0) == {"a": "rg1_r0"}
    assert mgr.get_row(1, 2) == {"a": "rg1_r2"}


def test_checkpoint_frees_memory() -> None:
    storage = _mock_artifact_storage()
    storage.write_batch_to_parquet_file.return_value = "/fake/path.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake/final.parquet"

    mgr = RowGroupBufferManager(storage)
    mgr.init_row_group(0, 2)
    mgr.update_batch(0, "col", ["a", "b"])

    mgr.checkpoint_row_group(0)

    with pytest.raises(KeyError):
        mgr.get_row(0, 0)
    assert mgr.actual_num_records == 2


def test_checkpoint_calls_on_complete() -> None:
    storage = _mock_artifact_storage()
    storage.write_batch_to_parquet_file.return_value = "/fake/path.parquet"
    storage.move_partial_result_to_final_file_path.return_value = "/fake/final.parquet"

    callback = Mock()

    mgr = RowGroupBufferManager(storage)
    mgr.init_row_group(0, 1)
    mgr.update_cell(0, 0, "col", "val")

    mgr.checkpoint_row_group(0, on_complete=callback)

    callback.assert_called_once_with("/fake/final.parquet")


def test_replace_dataframe_same_size() -> None:
    """replace_dataframe with same number of rows replaces data in-place."""
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 3)
    mgr.update_batch(0, "col", ["a", "b", "c"])

    df = lazy.pd.DataFrame({"col": ["x", "y", "z"]})
    mgr.replace_dataframe(0, df)

    assert mgr.get_row(0, 0) == {"col": "x"}
    assert mgr.get_row(0, 1) == {"col": "y"}
    assert mgr.get_row(0, 2) == {"col": "z"}


def test_replace_dataframe_with_dropped_rows() -> None:
    """replace_dataframe skips dropped rows and replaces only active slots."""
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 4)
    mgr.update_batch(0, "col", ["a", "b", "c", "d"])
    mgr.drop_row(0, 1)  # drop row 1

    # 3 active rows: indices 0, 2, 3
    df = lazy.pd.DataFrame({"col": ["x", "y", "z"]})
    mgr.replace_dataframe(0, df)

    assert mgr.get_row(0, 0) == {"col": "x"}
    assert mgr.is_dropped(0, 1)
    assert mgr.get_row(0, 2) == {"col": "y"}
    assert mgr.get_row(0, 3) == {"col": "z"}


def test_replace_dataframe_fewer_rows_marks_trailing_dropped() -> None:
    """replace_dataframe with fewer rows marks trailing active slots as dropped."""
    mgr = RowGroupBufferManager(_mock_artifact_storage())
    mgr.init_row_group(0, 4)
    mgr.update_batch(0, "col", ["a", "b", "c", "d"])

    # Only 2 rows - should drop indices 2 and 3
    df = lazy.pd.DataFrame({"col": ["x", "y"]})
    mgr.replace_dataframe(0, df)

    assert mgr.get_row(0, 0) == {"col": "x"}
    assert mgr.get_row(0, 1) == {"col": "y"}
    assert mgr.is_dropped(0, 2)
    assert mgr.is_dropped(0, 3)

    # get_dataframe should only return the 2 active rows
    result_df = mgr.get_dataframe(0)
    assert len(result_df) == 2


def test_checkpoint_calls_on_complete_when_all_rows_dropped() -> None:
    storage = _mock_artifact_storage()
    callback = Mock()

    mgr = RowGroupBufferManager(storage)
    mgr.init_row_group(0, 2)
    mgr.drop_row(0, 0)
    mgr.drop_row(0, 1)

    mgr.checkpoint_row_group(0, on_complete=callback)

    callback.assert_called_once_with(None)
    storage.write_batch_to_parquet_file.assert_not_called()
