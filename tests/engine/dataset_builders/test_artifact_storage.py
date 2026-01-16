# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pyarrow import ArrowNotImplementedError

from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage, BatchStage
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def stub_artifact_storage(tmp_path):
    return ArtifactStorage(artifact_path=tmp_path)


@pytest.fixture
def stub_custom_artifact_storage(tmp_path):
    return ArtifactStorage(
        artifact_path=tmp_path,
        dataset_name="custom_dataset",
        final_dataset_folder_name="final-files",
        partial_results_folder_name="temp-files",
        dropped_columns_folder_name="dropped-files",
    )


def test_artifact_storage_artifact_path_must_exist():
    with pytest.raises(ArtifactStorageError):
        ArtifactStorage(artifact_path="non/existent/path")


def test_artifact_storage_custom_names(stub_custom_artifact_storage):
    assert "custom_dataset" in str(stub_custom_artifact_storage.base_dataset_path)
    assert "final-files" in str(stub_custom_artifact_storage.final_dataset_path)
    assert "temp-files" in str(stub_custom_artifact_storage.partial_results_path)
    assert "dropped-files" in str(stub_custom_artifact_storage.dropped_columns_dataset_path)


@pytest.mark.parametrize(
    "batch_number,stage,expected_name,expected_parent_attr",
    [
        (0, BatchStage.PARTIAL_RESULT, "batch_00000.parquet", "partial_results_path"),
        (42, BatchStage.FINAL_RESULT, "batch_00042.parquet", "final_dataset_path"),
        (123, BatchStage.DROPPED_COLUMNS, "batch_00123.parquet", "dropped_columns_dataset_path"),
    ],
)
def test_artifact_storage_create_batch_file_path(
    stub_artifact_storage, batch_number, stage, expected_name, expected_parent_attr
):
    path = stub_artifact_storage.create_batch_file_path(batch_number, stage)
    assert path.name == expected_name
    assert path.parent == getattr(stub_artifact_storage, expected_parent_attr)


def test_artifact_storage_create_batch_file_path_negative_batch_number(stub_artifact_storage):
    with pytest.raises(ArtifactStorageError, match="Batch number must be non-negative"):
        stub_artifact_storage.create_batch_file_path(-1, BatchStage.PARTIAL_RESULT)


def test_artifact_storage_write_parquet_file(stub_artifact_storage, stub_sample_dataframe):
    file_path = stub_artifact_storage.write_parquet_file(
        "test.parquet", stub_sample_dataframe, BatchStage.PARTIAL_RESULT
    )
    assert file_path.exists()
    assert file_path.parent == stub_artifact_storage.partial_results_path

    read_df = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(stub_sample_dataframe, read_df)


def test_artifact_storage_write_batch_to_parquet_file(stub_artifact_storage, stub_sample_dataframe):
    file_path = stub_artifact_storage.write_batch_to_parquet_file(5, stub_sample_dataframe, BatchStage.FINAL_RESULT)
    assert file_path.exists()
    assert file_path.name == "batch_00005.parquet"
    assert file_path.parent == stub_artifact_storage.final_dataset_path


def test_artifact_storage_move_partial_result_to_final_file_path(stub_artifact_storage, stub_sample_dataframe):
    partial_path = stub_artifact_storage.write_batch_to_parquet_file(
        10, stub_sample_dataframe, BatchStage.PARTIAL_RESULT
    )
    assert partial_path.exists()

    final_path = stub_artifact_storage.move_partial_result_to_final_file_path(10)
    assert final_path.exists()
    assert not partial_path.exists()  # Original should be gone
    assert final_path.parent == stub_artifact_storage.final_dataset_path

    read_df = pd.read_parquet(final_path)
    pd.testing.assert_frame_equal(stub_sample_dataframe, read_df)


def test_artifact_storage_move_partial_result_to_final_file_path_not_found(stub_artifact_storage):
    with pytest.raises(ArtifactStorageError, match="Partial result file not found"):
        stub_artifact_storage.move_partial_result_to_final_file_path(999)


def test_artifact_storage_write_metadata(stub_artifact_storage):
    metadata = {"dataset_name": "test", "rows": 100, "columns": 5}
    file_path = stub_artifact_storage.write_metadata(metadata)

    assert file_path.exists()
    assert file_path == stub_artifact_storage.metadata_file_path

    with open(file_path, "r") as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata == metadata


def test_artifact_storage_metadata_file_path_property(stub_artifact_storage):
    expected_path = stub_artifact_storage.base_dataset_path / "metadata.json"
    assert stub_artifact_storage.metadata_file_path == expected_path


@pytest.mark.parametrize(
    "params,expected_error",
    [
        ({"dataset_name": ""}, "Directory names must be non-empty strings"),
        ({"final_dataset_folder_name": ""}, "Directory names must be non-empty strings"),
        ({"partial_results_folder_name": ""}, "Directory names must be non-empty strings"),
        ({"dropped_columns_folder_name": ""}, "Directory names must be non-empty strings"),
        ({"dataset_name": "same_name", "final_dataset_folder_name": "same_name"}, "Folder names must be unique"),
        (
            {"partial_results_folder_name": "duplicate", "dropped_columns_folder_name": "duplicate"},
            "Folder names must be unique",
        ),
        ({"dataset_name": "test", "final_dataset_folder_name": "test"}, "Folder names must be unique"),
    ],
)
def test_artifact_storage_invalid_folder_names_validation(tmp_path, params, expected_error):
    with pytest.raises(ArtifactStorageError, match=expected_error):
        ArtifactStorage(artifact_path=tmp_path, **params)


@pytest.mark.parametrize("invalid_char", ["<", ">", ":", '"', "/", "\\", "|", "?", "*"])
def test_artifact_storage_invalid_characters_in_folder_names(tmp_path, invalid_char):
    invalid_params = [
        {"dataset_name": f"invalid{invalid_char}name"},
        {"final_dataset_folder_name": f"invalid{invalid_char}name"},
        {"partial_results_folder_name": f"invalid{invalid_char}name"},
        {"dropped_columns_folder_name": f"invalid{invalid_char}name"},
    ]

    for params in invalid_params:
        with pytest.raises(ArtifactStorageError, match="contains invalid characters"):
            ArtifactStorage(artifact_path=tmp_path, **params)


def test_artifact_storage_read_parquet_files(stub_artifact_storage):
    df1 = pd.DataFrame([{"id": 1, "data": {"some_list": ["yes"]}}, {"id": 2, "data": {"some_list": ["no"]}}])
    df2 = pd.DataFrame({"id": 3, "data": {"some_list": []}})

    stub_artifact_storage.write_parquet_file("test1.parquet", df1, BatchStage.PARTIAL_RESULT)
    stub_artifact_storage.write_parquet_file("test2.parquet", df2, BatchStage.PARTIAL_RESULT)

    # pd.read_parquet is not able to combine the two parquet files due to mismatching schemas
    with pytest.raises(ArrowNotImplementedError) as exc:
        pd.read_parquet(stub_artifact_storage.partial_results_path)
    assert "Unsupported cast" in str(exc.value)

    read_df1 = stub_artifact_storage.read_parquet_files(stub_artifact_storage.partial_results_path / "test1.parquet")
    read_df2 = stub_artifact_storage.read_parquet_files(stub_artifact_storage.partial_results_path / "test2.parquet")
    read_df = stub_artifact_storage.read_parquet_files(stub_artifact_storage.partial_results_path)

    pd.testing.assert_frame_equal(pd.concat([read_df1, read_df2], ignore_index=True), read_df)


def test_artifact_storage_path_validation(stub_artifact_storage):
    assert stub_artifact_storage.artifact_path.is_absolute()
    assert stub_artifact_storage.base_dataset_path.is_absolute()
    assert stub_artifact_storage.partial_results_path.is_absolute()
    assert stub_artifact_storage.final_dataset_path.is_absolute()
    assert stub_artifact_storage.dropped_columns_dataset_path.is_absolute()


def test_artifact_storage_file_operations(stub_artifact_storage):
    df = pd.DataFrame({"test": [1, 2, 3]})

    file_path = stub_artifact_storage.write_parquet_file("test.parquet", df, BatchStage.PARTIAL_RESULT)
    assert file_path.exists()

    read_df = stub_artifact_storage.read_parquet_files(file_path)
    pd.testing.assert_frame_equal(df, read_df, check_dtype=False)


@pytest.mark.parametrize("batch_number", range(5))
def test_artifact_storage_batch_numbering(stub_artifact_storage, batch_number):
    path = stub_artifact_storage.create_batch_file_path(batch_number, BatchStage.FINAL_RESULT)
    expected_name = f"batch_{batch_number:05d}.parquet"
    assert path.name == expected_name


@patch("data_designer.engine.dataset_builders.artifact_storage.datetime")
def test_artifact_storage_resolved_dataset_name(mock_datetime, tmp_path):
    mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 3, 4)

    # dataset path does not exist yet
    assert ArtifactStorage(artifact_path=tmp_path).resolved_dataset_name == "dataset"

    # dataset path exists but is empty
    af_storage = ArtifactStorage(artifact_path=tmp_path)
    (af_storage.artifact_path / af_storage.dataset_name).mkdir()
    assert af_storage.resolved_dataset_name == "dataset"

    # dataset path exists and is not empty
    af_storage = ArtifactStorage(artifact_path=tmp_path)
    (af_storage.artifact_path / af_storage.dataset_name / "stub_file.txt").touch()
    print(af_storage.resolved_dataset_name)
    assert af_storage.resolved_dataset_name == "dataset_01-01-2025_120304"


def test_get_parquet_file_paths_empty(stub_artifact_storage):
    """Test get_parquet_file_paths when no parquet files exist."""
    paths = stub_artifact_storage.get_parquet_file_paths()
    assert paths == []


def test_get_parquet_file_paths_with_files(stub_artifact_storage):
    """Test get_parquet_file_paths returns relative paths to parquet files."""
    # Create some parquet files
    stub_artifact_storage.mkdir_if_needed(stub_artifact_storage.final_dataset_path)
    (stub_artifact_storage.final_dataset_path / "batch_00000.parquet").touch()
    (stub_artifact_storage.final_dataset_path / "batch_00001.parquet").touch()
    (stub_artifact_storage.final_dataset_path / "batch_00002.parquet").touch()

    paths = stub_artifact_storage.get_parquet_file_paths()

    assert len(paths) == 3
    assert "parquet-files/batch_00000.parquet" in paths
    assert "parquet-files/batch_00001.parquet" in paths
    assert "parquet-files/batch_00002.parquet" in paths
    # Ensure paths are relative
    assert all(not path.startswith("/") for path in paths)


def test_get_processor_file_paths_empty(stub_artifact_storage):
    """Test get_processor_file_paths when no processor files exist."""
    paths = stub_artifact_storage.get_processor_file_paths()
    assert paths == {}


def test_get_processor_file_paths_with_files(stub_artifact_storage):
    """Test get_processor_file_paths returns files organized by processor name."""
    # Create processor output directories and files
    processor1_dir = stub_artifact_storage.processors_outputs_path / "processor1"
    processor2_dir = stub_artifact_storage.processors_outputs_path / "processor2"
    stub_artifact_storage.mkdir_if_needed(processor1_dir)
    stub_artifact_storage.mkdir_if_needed(processor2_dir)

    (processor1_dir / "batch_00000.parquet").touch()
    (processor1_dir / "batch_00001.parquet").touch()
    (processor2_dir / "batch_00000.parquet").touch()
    (processor2_dir / "batch_00001.parquet").touch()
    (processor2_dir / "batch_00002.parquet").touch()

    paths = stub_artifact_storage.get_processor_file_paths()

    assert "processor1" in paths
    assert "processor2" in paths
    assert len(paths["processor1"]) == 2
    assert len(paths["processor2"]) == 3


def test_read_metadata_success(stub_artifact_storage):
    """Test read_metadata successfully reads metadata file."""
    metadata = {"key1": "value1", "key2": 123}
    stub_artifact_storage.write_metadata(metadata)

    read_data = stub_artifact_storage.read_metadata()

    assert read_data == metadata


def test_read_metadata_file_not_found(stub_artifact_storage):
    """Test read_metadata raises FileNotFoundError when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        stub_artifact_storage.read_metadata()


def test_write_metadata_creates_directory(stub_artifact_storage):
    """Test write_metadata creates base_dataset_path if it doesn't exist."""
    assert not stub_artifact_storage.base_dataset_path.exists()

    metadata = {"test": "data"}
    file_path = stub_artifact_storage.write_metadata(metadata)

    assert stub_artifact_storage.base_dataset_path.exists()
    assert file_path.exists()
    assert file_path == stub_artifact_storage.metadata_file_path


def test_write_metadata_content_and_formatting(stub_artifact_storage):
    """Test write_metadata writes properly formatted JSON."""
    metadata = {"key1": "value1", "key2": [1, 2, 3]}
    stub_artifact_storage.write_metadata(metadata)

    with open(stub_artifact_storage.metadata_file_path, "r") as f:
        content = f.read()
        loaded_data = json.loads(content)

    assert loaded_data == metadata
    # Check indentation (4 spaces)
    assert "    " in content


def test_update_metadata_creates_new_file(stub_artifact_storage):
    """Test update_metadata creates new file if metadata doesn't exist."""
    updates = {"new_key": "new_value"}
    file_path = stub_artifact_storage.update_metadata(updates)

    assert file_path.exists()
    metadata = stub_artifact_storage.read_metadata()
    assert metadata == updates


def test_update_metadata_merges_with_existing(stub_artifact_storage):
    """Test update_metadata merges new fields with existing metadata."""
    initial_metadata = {"key1": "value1", "key2": "value2"}
    stub_artifact_storage.write_metadata(initial_metadata)

    updates = {"key2": "updated_value2", "key3": "value3"}
    stub_artifact_storage.update_metadata(updates)

    final_metadata = stub_artifact_storage.read_metadata()

    assert final_metadata["key1"] == "value1"
    assert final_metadata["key2"] == "updated_value2"  # Updated
    assert final_metadata["key3"] == "value3"  # New key


def test_update_metadata_with_nested_structures(stub_artifact_storage):
    """Test update_metadata with complex nested data structures."""
    initial_metadata = {
        "simple": "value",
        "nested": {"a": 1, "b": 2},
        "list": [1, 2, 3],
    }
    stub_artifact_storage.write_metadata(initial_metadata)

    updates = {
        "nested": {"c": 3},  # This will replace the entire nested dict
        "new_list": [4, 5, 6],
    }
    stub_artifact_storage.update_metadata(updates)

    final_metadata = stub_artifact_storage.read_metadata()

    assert final_metadata["simple"] == "value"
    assert final_metadata["nested"] == {"c": 3}  # Replaced, not merged
    assert final_metadata["list"] == [1, 2, 3]  # Unchanged
    assert final_metadata["new_list"] == [4, 5, 6]
