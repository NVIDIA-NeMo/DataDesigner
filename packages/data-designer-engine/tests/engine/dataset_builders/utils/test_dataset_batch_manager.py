# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.dataset_builders.utils.dataset_batch_manager import DatasetBatchManager
from data_designer.engine.dataset_builders.utils.errors import DatasetBatchManagementError
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def stub_batch_manager(artifact_storage):
    return DatasetBatchManager(artifact_storage)


@pytest.fixture
def stub_batch_manager_with_data(artifact_storage):
    manager = DatasetBatchManager(artifact_storage)
    manager.start(num_records=10, buffer_size=3)
    return manager


def test_initial_state_properties(stub_batch_manager):
    assert stub_batch_manager.num_batches == 0
    assert stub_batch_manager.num_records_in_buffer == 0
    assert stub_batch_manager.buffer_is_empty is True

    with pytest.raises(DatasetBatchManagementError, match="num_records_list.*not set"):
        _ = stub_batch_manager.num_records_list

    with pytest.raises(DatasetBatchManagementError, match="buffer_size.*not set"):
        _ = stub_batch_manager.buffer_size

    with pytest.raises(DatasetBatchManagementError, match="Invalid batch number"):
        _ = stub_batch_manager.num_records_batch


# Test start method
def test_start_with_valid_parameters(stub_batch_manager):
    stub_batch_manager.start(num_records=10, buffer_size=3)

    assert stub_batch_manager.num_batches == 4
    assert stub_batch_manager.buffer_size == 3
    assert stub_batch_manager.num_records_list == [3, 3, 3, 1]
    assert stub_batch_manager._current_batch_number == 0
    assert stub_batch_manager.buffer_is_empty is True


def test_start_with_exact_buffer_size(stub_batch_manager):
    """Test start method when num_records is exactly divisible by buffer_size."""
    stub_batch_manager.start(num_records=9, buffer_size=3)

    assert stub_batch_manager.num_batches == 3
    assert stub_batch_manager.num_records_list == [3, 3, 3]


def test_start_with_single_batch(stub_batch_manager):
    stub_batch_manager.start(num_records=2, buffer_size=5)

    assert stub_batch_manager.num_batches == 1
    assert stub_batch_manager.num_records_list == [2]


def test_start_with_invalid_num_records(stub_batch_manager):
    with pytest.raises(DatasetBatchManagementError, match="num_records must be positive"):
        stub_batch_manager.start(num_records=0, buffer_size=3)

    with pytest.raises(DatasetBatchManagementError, match="num_records must be positive"):
        stub_batch_manager.start(num_records=-1, buffer_size=3)


def test_start_with_invalid_buffer_size(stub_batch_manager):
    with pytest.raises(DatasetBatchManagementError, match="buffer_size must be positive"):
        stub_batch_manager.start(num_records=10, buffer_size=0)

    with pytest.raises(DatasetBatchManagementError, match="buffer_size must be positive"):
        stub_batch_manager.start(num_records=10, buffer_size=-1)


# Test buffer management
def test_add_single_record(stub_batch_manager_with_data):
    record = {"id": 1, "name": "test"}
    stub_batch_manager_with_data.add_record(record)

    assert stub_batch_manager_with_data.num_records_in_buffer == 1
    assert stub_batch_manager_with_data.buffer_is_empty is False
    assert stub_batch_manager_with_data._buffer[0] == record


def test_add_multiple_records(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    assert stub_batch_manager_with_data.num_records_in_buffer == 3
    assert stub_batch_manager_with_data._buffer == records


def test_add_records_exceeds_buffer_size(stub_batch_manager_with_data):
    invalid_buffer_size = stub_batch_manager_with_data.buffer_size + 1
    records = [{"id": i, "name": f"test{i}"} for i in range(invalid_buffer_size)]

    with pytest.raises(DatasetBatchManagementError, match="Buffer size exceeded"):
        stub_batch_manager_with_data.add_records(records)


def test_drop_records(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    stub_batch_manager_with_data.drop_records([0, 2])

    assert stub_batch_manager_with_data.num_records_in_buffer == 1
    assert stub_batch_manager_with_data._buffer[0] == {"id": 1, "name": "test1"}


def test_drop_records_with_empty_list(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    stub_batch_manager_with_data.drop_records([])

    assert stub_batch_manager_with_data.num_records_in_buffer == 3


def test_update_record(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    new_record = {"id": 1, "name": "updated"}
    stub_batch_manager_with_data.update_record(1, new_record)

    assert stub_batch_manager_with_data._buffer[1] == new_record


def test_update_record_invalid_index(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    with pytest.raises(IndexError, match="Index.*out of bounds"):
        stub_batch_manager_with_data.update_record(5, {"id": 5, "name": "test"})

    with pytest.raises(IndexError, match="Index.*out of bounds"):
        stub_batch_manager_with_data.update_record(-1, {"id": -1, "name": "test"})


def test_update_records(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    new_records = [{"id": i, "name": f"updated{i}"} for i in range(3)]
    stub_batch_manager_with_data.update_records(new_records)

    assert stub_batch_manager_with_data._buffer == new_records


def test_update_records_wrong_length(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    wrong_length_records = [{"id": i, "name": f"test{i}"} for i in range(2)]

    with pytest.raises(DatasetBatchManagementError, match="Number of records to update.*must match"):
        stub_batch_manager_with_data.update_records(wrong_length_records)


# Test write method
def test_write_empty_buffer(stub_batch_manager_with_data):
    result = stub_batch_manager_with_data.write()
    assert result is None


def test_write_with_data(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    result = stub_batch_manager_with_data.write()

    assert result is not None
    assert result.exists()
    assert (
        result.name
        == stub_batch_manager_with_data.artifact_storage.create_batch_file_path(0, BatchStage.PARTIAL_RESULT).name
    )

    df = pd.read_parquet(result)
    expected_df = pd.DataFrame(records)
    pd.testing.assert_frame_equal(df, expected_df)


def test_write_creates_partial_results_dir(stub_batch_manager_with_data):
    records = [{"id": 1, "name": "test"}]
    stub_batch_manager_with_data.add_records(records)

    stub_batch_manager_with_data.write()

    assert stub_batch_manager_with_data.artifact_storage.partial_results_path.exists()


# Test finish_batch method
def test_finish_batch_basic_functionality(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    result = stub_batch_manager_with_data.finish_batch()

    assert result.exists()
    assert result.parent == stub_batch_manager_with_data.artifact_storage.final_dataset_path
    assert (
        result.name
        == stub_batch_manager_with_data.artifact_storage.create_batch_file_path(0, BatchStage.PARTIAL_RESULT).name
    )

    # Verify metadata file was created
    assert stub_batch_manager_with_data.artifact_storage.metadata_file_path.exists()

    # Verify buffer was cleared
    assert stub_batch_manager_with_data.buffer_is_empty
    assert stub_batch_manager_with_data._current_batch_number == 1


def test_finish_batch_all_batches_processed(stub_batch_manager_with_data):
    # Process all batches with data
    for i in range(stub_batch_manager_with_data.num_batches):
        # Add some data for each batch
        records = [{"id": j, "name": f"batch{i}_{j}"} for j in range(3)]
        stub_batch_manager_with_data.add_records(records)
        stub_batch_manager_with_data.finish_batch()

    with pytest.raises(DatasetBatchManagementError, match="All batches have been processed"):
        stub_batch_manager_with_data.finish_batch()


def test_finish_batch_empty_buffer_logs_warning_and_returns_none(
    stub_batch_manager_with_data, caplog: pytest.LogCaptureFixture
) -> None:
    result = stub_batch_manager_with_data.finish_batch()

    assert result is None
    assert "finished without any results to write" in caplog.text
    assert stub_batch_manager_with_data.buffer_is_empty
    assert stub_batch_manager_with_data.get_current_batch_number() == 1
    assert not stub_batch_manager_with_data.artifact_storage.metadata_file_path.exists()


def test_finish_batch_empty_buffer_does_not_call_on_complete(stub_batch_manager_with_data) -> None:
    on_complete = Mock()

    result = stub_batch_manager_with_data.finish_batch(on_complete=on_complete)

    assert result is None
    on_complete.assert_not_called()


def test_finish_batch_metadata_content(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    stub_batch_manager_with_data.finish_batch()

    with open(stub_batch_manager_with_data.artifact_storage.metadata_file_path) as f:
        metadata = json.load(f)

    assert metadata["target_num_records"] == 10
    assert metadata["total_num_batches"] == 4
    assert metadata["buffer_size"] == 3
    assert metadata["num_completed_batches"] == 1
    assert metadata["dataset_name"] == stub_batch_manager_with_data.artifact_storage.dataset_name
    assert "schema" in metadata
    assert "file_paths" in metadata
    assert isinstance(metadata["file_paths"], dict)
    assert "parquet-files" in metadata["file_paths"]
    assert isinstance(metadata["file_paths"]["parquet-files"], list)
    assert len(metadata["file_paths"]["parquet-files"]) == 1
    assert metadata["file_paths"]["parquet-files"][0] == "parquet-files/batch_00000.parquet"

    # processor-files key should not exist if no processor files
    assert "processor-files" not in metadata["file_paths"]


# Test finish method
def test_finish_with_empty_partial_results(stub_batch_manager_with_data):
    # Accessing the partial results path to trigger the creation of the directory
    stub_batch_manager_with_data.artifact_storage.mkdir_if_needed(
        stub_batch_manager_with_data.artifact_storage.partial_results_path
    )

    stub_batch_manager_with_data.finish()

    # Directory should be removed
    assert not (
        stub_batch_manager_with_data.artifact_storage.base_dataset_path
        / stub_batch_manager_with_data.artifact_storage.partial_results_folder_name
    ).exists()


def test_finish_with_partial_results(stub_batch_manager_with_data, caplog):
    (
        stub_batch_manager_with_data.artifact_storage.mkdir_if_needed(
            stub_batch_manager_with_data.artifact_storage.partial_results_path
        )
        / "test_file.txt"
    ).touch()

    stub_batch_manager_with_data.finish()

    # Directory should still exist
    assert stub_batch_manager_with_data.artifact_storage.partial_results_path.exists()

    # Warning should be logged
    assert "Dataset writing finished with partial results" in caplog.text


# Test reset method
def test_reset_without_delete_files(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)
    stub_batch_manager_with_data._current_batch_number = 2

    stub_batch_manager_with_data.reset()

    assert stub_batch_manager_with_data._current_batch_number == 0
    assert stub_batch_manager_with_data.buffer_is_empty
    # final_dataset_path doesn't exist until finish_batch is called
    assert not (
        stub_batch_manager_with_data.artifact_storage.base_dataset_path
        / stub_batch_manager_with_data.artifact_storage.final_dataset_folder_name
    ).exists()


def test_reset_with_delete_files(stub_batch_manager_with_data):
    (
        stub_batch_manager_with_data.artifact_storage.mkdir_if_needed(
            stub_batch_manager_with_data.artifact_storage.final_dataset_path
        )
        / "test.parquet"
    ).touch()
    (stub_batch_manager_with_data.artifact_storage.metadata_file_path).touch()

    stub_batch_manager_with_data.reset(delete_files=True)

    assert stub_batch_manager_with_data._current_batch_number == 0
    assert stub_batch_manager_with_data.buffer_is_empty
    assert not stub_batch_manager_with_data.artifact_storage.final_dataset_path.exists()
    assert not stub_batch_manager_with_data.artifact_storage.metadata_file_path.exists()


# Test error handling and edge cases
def test_num_records_batch_invalid_batch_number(stub_batch_manager_with_data):
    stub_batch_manager_with_data._current_batch_number = 10  # Beyond num_batches

    with pytest.raises(DatasetBatchManagementError, match="Invalid batch number"):
        _ = stub_batch_manager_with_data.num_records_batch


def test_write_exception_handling(stub_batch_manager_with_data):
    records = [{"id": i, "name": f"test{i}"} for i in range(3)]
    stub_batch_manager_with_data.add_records(records)

    with patch("pandas.DataFrame.to_parquet", side_effect=Exception("Write error")):
        with pytest.raises(DatasetBatchManagementError, match="Failed to write batch"):
            stub_batch_manager_with_data.write()


def test_reset_delete_files_exception_handling(stub_batch_manager_with_data):
    stub_batch_manager_with_data.artifact_storage.mkdir_if_needed(
        stub_batch_manager_with_data.artifact_storage.final_dataset_path
    )
    (stub_batch_manager_with_data.artifact_storage.final_dataset_path / "test.parquet").touch()

    with patch("shutil.rmtree", side_effect=OSError("Delete error")):
        with pytest.raises(DatasetBatchManagementError, match="Failed to delete directory"):
            stub_batch_manager_with_data.reset(delete_files=True)


def test_full_workflow(stub_batch_manager):
    stub_batch_manager.start(num_records=7, buffer_size=3)

    assert stub_batch_manager.num_batches == 3
    assert stub_batch_manager.num_records_list == [3, 3, 1]

    # Process first batch
    records1 = [{"id": i, "name": f"batch1_{i}"} for i in range(3)]
    stub_batch_manager.add_records(records1)
    result1 = stub_batch_manager.finish_batch()

    assert result1.exists()
    assert stub_batch_manager.get_current_batch_number() == 1
    assert stub_batch_manager.buffer_is_empty

    # Process second batch
    records2 = [{"id": i, "name": f"batch2_{i}"} for i in range(3)]
    stub_batch_manager.add_records(records2)
    result2 = stub_batch_manager.finish_batch()

    assert result2.exists()
    assert stub_batch_manager.get_current_batch_number() == 2

    # Process final batch
    records3 = [{"id": 0, "name": "batch3_0"}]
    stub_batch_manager.add_records(records3)
    result3 = stub_batch_manager.finish_batch()

    assert result3.exists()
    assert stub_batch_manager.get_current_batch_number() == 3

    # Finish
    stub_batch_manager.finish()

    # Verify all files exist
    assert stub_batch_manager.artifact_storage.metadata_file_path.exists()
    assert len(list(stub_batch_manager.artifact_storage.final_dataset_path.glob("*.parquet"))) == 3
