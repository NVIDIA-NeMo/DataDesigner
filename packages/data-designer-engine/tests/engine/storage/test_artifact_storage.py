# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from pyarrow import ArrowNotImplementedError

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.io_helpers import load_processor_dataset
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.engine.storage.artifact_storage import (
    ArtifactStorage,
    BatchStage,
    ResumeMode,
    _get_selection_publication_file_name,
)


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


def test_artifact_storage_rejects_collision_with_selection_directories(tmp_path) -> None:
    with pytest.raises(ArtifactStorageError, match="unique"):
        ArtifactStorage(artifact_path=tmp_path, final_dataset_folder_name="selection-accepted")


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


def test_configure_selection_batch_file_width_covers_entire_candidate_budget(stub_artifact_storage) -> None:
    stub_artifact_storage.configure_selection_batch_file_width(
        max_candidate_records=100_001,
        candidate_batch_size=1,
    )

    assert stub_artifact_storage.create_batch_file_path(0, BatchStage.DROPPED_COLUMNS).name == "batch_000000.parquet"
    assert stub_artifact_storage.selection_partition_path(99_999).name == "batch_099999.parquet"
    assert stub_artifact_storage.selection_partition_path(100_000).name == "batch_100000.parquet"
    assert stub_artifact_storage.selection_checkpoint_path(100_000).name == "batch_100000.json"


def test_selection_candidate_artifact_migration_normalizes_legacy_boundary_names(
    stub_artifact_storage: ArtifactStorage,
) -> None:
    processor_name = "formatted"
    for batch_id in (99_999, 100_000):
        dataframe = lazy.pd.DataFrame({"value": [batch_id]})
        partition = stub_artifact_storage.write_selection_partition(batch_id, dataframe)
        stub_artifact_storage.write_selection_checkpoint(
            batch_id,
            {
                "candidate_batch_id": batch_id,
                "accepted_partition": partition.relative_to(stub_artifact_storage.base_dataset_path).as_posix(),
            },
        )
        stub_artifact_storage.write_batch_to_parquet_file(
            batch_id,
            dataframe.rename(columns={"value": "dropped"}),
            BatchStage.DROPPED_COLUMNS,
        )
        stub_artifact_storage.write_batch_to_parquet_file(
            batch_id,
            dataframe,
            BatchStage.PROCESSORS_OUTPUTS,
            subfolder=processor_name,
        )

    stub_artifact_storage.configure_selection_batch_file_width(
        max_candidate_records=100_001,
        candidate_batch_size=1,
    )

    assert stub_artifact_storage.requires_selection_candidate_artifact_migration()
    assert stub_artifact_storage.normalize_selection_candidate_artifact_width()

    expected_names = ["batch_099999.parquet", "batch_100000.parquet"]
    assert sorted(path.name for path in stub_artifact_storage.selection_accepted_path.glob("batch_*.parquet")) == (
        expected_names
    )
    assert sorted(path.name for path in stub_artifact_storage.dropped_columns_dataset_path.glob("batch_*.parquet")) == (
        expected_names
    )
    assert (
        sorted(
            path.name
            for path in (stub_artifact_storage.processors_outputs_path / processor_name).glob("batch_*.parquet")
        )
        == expected_names
    )
    checkpoints = stub_artifact_storage.read_selection_checkpoints()
    assert [checkpoint["accepted_partition"] for checkpoint in checkpoints] == [
        "selection-accepted/batch_099999.parquet",
        "selection-accepted/batch_100000.parquet",
    ]
    assert not stub_artifact_storage.requires_selection_candidate_artifact_migration()
    assert not stub_artifact_storage.normalize_selection_candidate_artifact_width()


def test_selection_candidate_artifact_migration_keeps_zero_prefix_side_artifacts_aligned(
    stub_artifact_storage: ArtifactStorage,
) -> None:
    processor_name = "formatted"
    for batch_id in range(3):
        stub_artifact_storage.write_selection_checkpoint(
            batch_id,
            {"candidate_batch_id": batch_id, "accepted_partition": None},
        )
        stub_artifact_storage.write_batch_to_parquet_file(
            batch_id,
            lazy.pd.DataFrame({"dropped": lazy.pd.Series(dtype="string")}),
            BatchStage.DROPPED_COLUMNS,
        )
        stub_artifact_storage.write_batch_to_parquet_file(
            batch_id,
            lazy.pd.DataFrame({"value": lazy.pd.Series(dtype="int64")}),
            BatchStage.PROCESSORS_OUTPUTS,
            subfolder=processor_name,
        )

    stub_artifact_storage.configure_selection_batch_file_width(
        max_candidate_records=100_001,
        candidate_batch_size=1,
    )
    assert stub_artifact_storage.normalize_selection_candidate_artifact_width()

    stub_artifact_storage.write_batch_to_parquet_file(
        0,
        lazy.pd.DataFrame({"value": [3]}),
        BatchStage.FINAL_RESULT,
    )
    stub_artifact_storage.write_batch_to_parquet_file(
        3,
        lazy.pd.DataFrame({"dropped": ["accepted"]}),
        BatchStage.DROPPED_COLUMNS,
    )
    stub_artifact_storage.write_batch_to_parquet_file(
        3,
        lazy.pd.DataFrame({"value": [3]}),
        BatchStage.PROCESSORS_OUTPUTS,
        subfolder=processor_name,
    )

    assert stub_artifact_storage.load_dataset_with_dropped_columns().to_dict(orient="records") == [
        {"value": 3, "dropped": "accepted"}
    ]
    assert stub_artifact_storage.load_processor_dataset(processor_name)["value"].tolist() == [3]
    assert all(
        len(path.stem.removeprefix("batch_")) == 6
        for path in stub_artifact_storage.dropped_columns_dataset_path.glob("batch_*.parquet")
    )


@pytest.mark.parametrize("interrupt_phase", ["staging", "committing"])
def test_selection_candidate_artifact_migration_recovers_after_interruption(
    stub_artifact_storage: ArtifactStorage,
    interrupt_phase: str,
) -> None:
    partition = stub_artifact_storage.write_selection_partition(0, lazy.pd.DataFrame({"value": [1]}))
    stub_artifact_storage.write_selection_checkpoint(
        0,
        {
            "candidate_batch_id": 0,
            "accepted_partition": partition.relative_to(stub_artifact_storage.base_dataset_path).as_posix(),
        },
    )
    stub_artifact_storage.final_dataset_path.mkdir(parents=True)
    for batch_id in (99_999, 100_000):
        lazy.pd.DataFrame({"value": [batch_id]}).to_parquet(
            stub_artifact_storage.final_dataset_path / f"batch_{batch_id}.parquet",
            index=False,
        )
    stub_artifact_storage.configure_selection_batch_file_width(
        max_candidate_records=100_001,
        candidate_batch_size=1,
    )
    original_replace = os.replace
    publication_replace_interrupted = False

    def interrupt_publication_artifact(source: Path | str, target: Path | str) -> None:
        nonlocal publication_replace_interrupted
        source_path = Path(source)
        target_path = Path(target)
        source_name = source_path.name
        target_name = target_path.name
        is_matching_phase = (
            interrupt_phase == "staging"
            and source_path.parent == stub_artifact_storage.final_dataset_path
            and ".selection-migration-" in target_name
        ) or (
            interrupt_phase == "committing"
            and target_path.parent == stub_artifact_storage.final_dataset_path
            and ".selection-migration-" in source_name
            and ".selection-migration-" not in target_name
        )
        if is_matching_phase and not publication_replace_interrupted:
            publication_replace_interrupted = True
            raise OSError("simulated migration interruption")
        original_replace(source, target)

    with (
        patch(
            "data_designer.engine.storage.artifact_storage.os.replace",
            side_effect=interrupt_publication_artifact,
        ),
        pytest.raises(OSError, match="simulated migration interruption"),
    ):
        stub_artifact_storage.normalize_selection_candidate_artifact_width()

    assert stub_artifact_storage.selection_artifact_migration_path.is_file()
    journal = json.loads(stub_artifact_storage.selection_artifact_migration_path.read_text(encoding="utf-8"))
    assert journal["phase"] == ("planned" if interrupt_phase == "staging" else "committing")
    assert any(operation["source"].startswith("parquet-files/") for operation in journal["operations"])
    assert stub_artifact_storage.normalize_selection_candidate_artifact_width()
    assert not stub_artifact_storage.selection_artifact_migration_path.exists()
    assert stub_artifact_storage.selection_partition_path(0).is_file()
    assert stub_artifact_storage.read_selection_checkpoints()[0]["accepted_partition"] == (
        "selection-accepted/batch_000000.parquet"
    )
    assert sorted(path.name for path in stub_artifact_storage.final_dataset_path.glob("batch_*.parquet")) == [
        "batch_00000.parquet",
        "batch_00001.parquet",
    ]
    assert stub_artifact_storage.load_dataset()["value"].tolist() == [99_999, 100_000]
    assert not stub_artifact_storage.normalize_selection_candidate_artifact_width()


def test_selection_candidate_artifact_migration_ignores_current_publication_names(
    stub_artifact_storage: ArtifactStorage,
) -> None:
    stub_artifact_storage.final_dataset_path.mkdir(parents=True)
    for batch_id in range(2):
        lazy.pd.DataFrame({"value": [batch_id]}).to_parquet(
            stub_artifact_storage.final_dataset_path / f"batch_{batch_id:05d}.parquet",
            index=False,
        )
    stub_artifact_storage.configure_selection_batch_file_width(
        max_candidate_records=100_001,
        candidate_batch_size=1,
    )

    assert not stub_artifact_storage.requires_selection_candidate_artifact_migration()
    assert not stub_artifact_storage.normalize_selection_candidate_artifact_width()


@pytest.mark.parametrize(
    "max_candidate_records,candidate_batch_size",
    [(0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_configure_selection_batch_file_width_rejects_nonpositive_limits(
    stub_artifact_storage,
    max_candidate_records: int,
    candidate_batch_size: int,
) -> None:
    with pytest.raises(ArtifactStorageError, match="candidate limits must be positive"):
        stub_artifact_storage.configure_selection_batch_file_width(
            max_candidate_records=max_candidate_records,
            candidate_batch_size=candidate_batch_size,
        )


def test_artifact_storage_write_parquet_file(stub_artifact_storage, stub_sample_dataframe):
    file_path = stub_artifact_storage.write_parquet_file(
        "test.parquet", stub_sample_dataframe, BatchStage.PARTIAL_RESULT
    )
    assert file_path.exists()
    assert file_path.parent == stub_artifact_storage.partial_results_path

    read_df = lazy.pd.read_parquet(file_path)
    lazy.pd.testing.assert_frame_equal(stub_sample_dataframe, read_df)


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

    read_df = lazy.pd.read_parquet(final_path)
    lazy.pd.testing.assert_frame_equal(stub_sample_dataframe, read_df)


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


def test_artifact_storage_write_metadata_includes_defaults(stub_artifact_storage):
    """Metadata defaults are included in each checkpoint write."""
    stub_artifact_storage.set_metadata_defaults({"config_hash": "sha256:abc", "config_hash_version": 1})

    file_path = stub_artifact_storage.write_metadata({"dataset_name": "test", "rows": 100})

    with open(file_path, "r") as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata == {
        "config_hash": "sha256:abc",
        "config_hash_version": 1,
        "dataset_name": "test",
        "rows": 100,
    }


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
    df1 = lazy.pd.DataFrame([{"id": 1, "data": {"some_list": ["yes"]}}, {"id": 2, "data": {"some_list": ["no"]}}])
    df2 = lazy.pd.DataFrame({"id": 3, "data": {"some_list": []}})

    stub_artifact_storage.write_parquet_file("test1.parquet", df1, BatchStage.PARTIAL_RESULT)
    stub_artifact_storage.write_parquet_file("test2.parquet", df2, BatchStage.PARTIAL_RESULT)

    # pd.read_parquet is not able to combine the two parquet files due to mismatching schemas
    with pytest.raises(ArrowNotImplementedError) as exc:
        lazy.pd.read_parquet(stub_artifact_storage.partial_results_path)
    assert "Unsupported cast" in str(exc.value)

    read_df1 = stub_artifact_storage.read_parquet_files(stub_artifact_storage.partial_results_path / "test1.parquet")
    read_df2 = stub_artifact_storage.read_parquet_files(stub_artifact_storage.partial_results_path / "test2.parquet")
    read_df = stub_artifact_storage.read_parquet_files(stub_artifact_storage.partial_results_path)

    lazy.pd.testing.assert_frame_equal(lazy.pd.concat([read_df1, read_df2], ignore_index=True), read_df)


def test_artifact_storage_path_validation(stub_artifact_storage):
    assert stub_artifact_storage.artifact_path.is_absolute()
    assert stub_artifact_storage.base_dataset_path.is_absolute()
    assert stub_artifact_storage.partial_results_path.is_absolute()
    assert stub_artifact_storage.final_dataset_path.is_absolute()
    assert stub_artifact_storage.dropped_columns_dataset_path.is_absolute()


def test_artifact_storage_file_operations(stub_artifact_storage):
    df = lazy.pd.DataFrame({"test": [1, 2, 3]})

    file_path = stub_artifact_storage.write_parquet_file("test.parquet", df, BatchStage.PARTIAL_RESULT)
    assert file_path.exists()

    read_df = stub_artifact_storage.read_parquet_files(file_path)
    lazy.pd.testing.assert_frame_equal(df, read_df, check_dtype=False)


@pytest.mark.parametrize("batch_number", range(5))
def test_artifact_storage_batch_numbering(stub_artifact_storage, batch_number):
    path = stub_artifact_storage.create_batch_file_path(batch_number, BatchStage.FINAL_RESULT)
    expected_name = f"batch_{batch_number:05d}.parquet"
    assert path.name == expected_name


@patch("data_designer.engine.storage.artifact_storage.datetime")
def test_artifact_storage_resolved_dataset_name(mock_datetime, tmp_path):
    mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 3, 4)

    # dataset path does not exist yet
    assert ArtifactStorage(artifact_path=tmp_path).resolved_dataset_name == "dataset"

    # dataset path exists but is empty
    af_storage = ArtifactStorage(artifact_path=tmp_path)
    (af_storage.artifact_path / af_storage.dataset_name).mkdir()
    assert af_storage.resolved_dataset_name == "dataset"

    # dataset path exists and is not empty (create file BEFORE constructing ArtifactStorage)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    (dataset_dir / "stub_file.txt").touch()
    af_storage = ArtifactStorage(artifact_path=tmp_path)
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


def test_get_processor_file_paths_with_single_files(stub_artifact_storage):
    """Test get_processor_file_paths picks up single parquet files."""
    stub_artifact_storage.mkdir_if_needed(stub_artifact_storage.processors_outputs_path)
    (stub_artifact_storage.processors_outputs_path / "preview.parquet").touch()

    paths = stub_artifact_storage.get_processor_file_paths()
    assert "preview" in paths
    assert len(paths["preview"]) == 1


def test_list_processor_names(stub_artifact_storage):
    assert stub_artifact_storage.list_processor_names() == []

    # Directory-based processor
    proc_dir = stub_artifact_storage.processors_outputs_path / "batched"
    stub_artifact_storage.mkdir_if_needed(proc_dir)
    (proc_dir / "batch_00000.parquet").touch()
    # Single-file processor
    (stub_artifact_storage.processors_outputs_path / "preview.parquet").touch()
    # Duplicate: both dir and file with same name
    dup_dir = stub_artifact_storage.processors_outputs_path / "both"
    dup_dir.mkdir()
    (dup_dir / "batch_00000.parquet").touch()
    (stub_artifact_storage.processors_outputs_path / "both.parquet").touch()

    assert stub_artifact_storage.list_processor_names() == ["batched", "both", "preview"]


@pytest.mark.parametrize("write_as_dir", [True, False], ids=["directory", "single_file"])
def test_load_processor_dataset(stub_artifact_storage, stub_sample_dataframe, write_as_dir):
    if write_as_dir:
        stub_artifact_storage.write_batch_to_parquet_file(
            0, stub_sample_dataframe, BatchStage.PROCESSORS_OUTPUTS, subfolder="chat_format"
        )
    else:
        stub_artifact_storage.write_parquet_file(
            "chat_format.parquet", stub_sample_dataframe, BatchStage.PROCESSORS_OUTPUTS
        )

    result = stub_artifact_storage.load_processor_dataset("chat_format")
    lazy.pd.testing.assert_frame_equal(result, stub_sample_dataframe, check_dtype=False)


def test_load_processor_dataset_not_found(stub_artifact_storage):
    with pytest.raises(ArtifactStorageError, match="No artifacts found"):
        stub_artifact_storage.load_processor_dataset("nonexistent")


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


def test_standalone_load_processor_dataset_raises_file_not_found(tmp_path):
    """Standalone function raises FileNotFoundError (not ArtifactStorageError)."""
    with pytest.raises(FileNotFoundError, match="No artifacts found"):
        load_processor_dataset(tmp_path, "nonexistent")


# ---------------------------------------------------------------------------
# Resume flag tests
# ---------------------------------------------------------------------------


def test_resolved_dataset_name_creates_timestamped_copy_when_folder_exists(tmp_path):
    """Default behaviour: existing non-empty folder gets a timestamped sibling."""
    existing = tmp_path / "dataset"
    existing.mkdir()
    (existing / "some_file.txt").write_text("x")

    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset")
    name = storage.resolved_dataset_name
    assert name != "dataset"
    assert name.startswith("dataset_")


def test_resolved_dataset_name_resume_uses_existing_folder(tmp_path):
    """With resume=ALWAYS, an existing non-empty folder is used as-is."""
    existing = tmp_path / "dataset"
    existing.mkdir()
    (existing / "some_file.txt").write_text("x")

    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.ALWAYS)
    assert storage.resolved_dataset_name == "dataset"


def test_resolved_dataset_name_resume_raises_when_no_existing_folder(tmp_path):
    """With resume=ALWAYS, missing dataset folder raises ArtifactStorageError."""
    with pytest.raises(ArtifactStorageError, match="Cannot resume"):
        ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.ALWAYS)


def test_resolved_dataset_name_resume_raises_when_folder_is_empty(tmp_path):
    """With resume=ALWAYS, an empty existing folder raises ArtifactStorageError."""
    (tmp_path / "dataset").mkdir()

    with pytest.raises(ArtifactStorageError, match="Cannot resume"):
        ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.ALWAYS)


def test_resolved_dataset_name_if_possible_uses_existing_folder(tmp_path):
    """With resume=IF_POSSIBLE, an existing non-empty folder is used as-is."""
    existing = tmp_path / "dataset"
    existing.mkdir()
    (existing / "some_file.txt").write_text("x")

    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.IF_POSSIBLE)
    assert storage.resolved_dataset_name == "dataset"


def test_resolved_dataset_name_if_possible_uses_clean_name_when_no_existing_folder(tmp_path):
    """With resume=IF_POSSIBLE, a missing dataset folder results in a fresh run (no error)."""
    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.IF_POSSIBLE)
    assert storage.resolved_dataset_name == "dataset"


def test_clear_partial_results_removes_partial_folder(tmp_path, stub_sample_dataframe):
    """clear_partial_results() deletes the partial results directory and its contents."""
    storage = ArtifactStorage(artifact_path=tmp_path)
    storage.write_batch_to_parquet_file(0, stub_sample_dataframe, BatchStage.PARTIAL_RESULT)
    assert storage.partial_results_path.exists()

    storage.clear_partial_results()
    assert not storage.partial_results_path.exists()


def test_clear_partial_results_is_noop_when_no_partial_folder(tmp_path):
    """clear_partial_results() does not raise when the partial results folder is absent."""
    storage = ArtifactStorage(artifact_path=tmp_path)
    assert not storage.partial_results_path.exists()
    storage.clear_partial_results()  # must not raise


def test_selection_checkpoint_and_partition_round_trip(stub_artifact_storage, stub_sample_dataframe) -> None:
    partition = stub_artifact_storage.write_selection_partition(0, stub_sample_dataframe.iloc[:2])
    marker = {
        "candidate_batch_id": 0,
        "row_group_id": 0,
        "candidate_start_offset": 0,
        "candidate_records": 4,
        "accepted_records": 2,
        "rejected_records": 2,
        "null_predicate_records": 0,
        "failed_generation_records": 0,
        "trimmed_accepted_records": 0,
        "accepted_partition": str(partition.relative_to(stub_artifact_storage.base_dataset_path)),
    }
    stub_artifact_storage.write_selection_checkpoint(0, marker)

    assert lazy.pq.read_metadata(partition).num_rows == 2
    assert stub_artifact_storage.read_selection_checkpoints() == [marker]


def test_selection_checkpoints_are_sorted_by_numeric_batch_id(stub_artifact_storage) -> None:
    stub_artifact_storage.write_selection_checkpoint(100_000, {"candidate_batch_id": 100_000})
    stub_artifact_storage.write_selection_checkpoint(10_001, {"candidate_batch_id": 10_001})

    assert [marker["candidate_batch_id"] for marker in stub_artifact_storage.read_selection_checkpoints()] == [
        10_001,
        100_000,
    ]


def test_materialize_selection_dataset_from_partitions(stub_artifact_storage, stub_sample_dataframe) -> None:
    stub_artifact_storage.write_selection_partition(0, stub_sample_dataframe.iloc[:2])
    stub_artifact_storage.write_selection_partition(2, stub_sample_dataframe.iloc[2:])

    published = stub_artifact_storage.materialize_selection_dataset()

    assert sorted(path.name for path in published.glob("*.parquet")) == [
        "batch_00000.parquet",
        "batch_00001.parquet",
    ]
    assert len(stub_artifact_storage.load_dataset()) == 4


def test_materialize_selection_dataset_preserves_numeric_order_after_five_digits(
    stub_artifact_storage,
) -> None:
    stub_artifact_storage.write_selection_partition(100_000, lazy.pd.DataFrame({"value": [2]}))
    stub_artifact_storage.write_selection_partition(10_001, lazy.pd.DataFrame({"value": [1]}))

    published = stub_artifact_storage.materialize_selection_dataset()

    assert sorted(path.name for path in published.glob("*.parquet")) == [
        "batch_00000.parquet",
        "batch_00001.parquet",
    ]
    assert stub_artifact_storage.load_dataset()["value"].tolist() == [1, 2]


def test_selection_publication_names_widen_together_and_preserve_metadata_order(stub_artifact_storage) -> None:
    assert _get_selection_publication_file_name(99_999, num_partitions=100_000) == "batch_99999.parquet"

    boundary_names = [
        _get_selection_publication_file_name(batch_id, num_partitions=100_001) for batch_id in (99_999, 100_000)
    ]
    assert boundary_names == ["batch_099999.parquet", "batch_100000.parquet"]
    assert sorted(boundary_names) == boundary_names

    stub_artifact_storage.final_dataset_path.mkdir(parents=True)
    for name in reversed(boundary_names):
        (stub_artifact_storage.final_dataset_path / name).touch()

    assert stub_artifact_storage.get_parquet_file_paths() == [f"parquet-files/{name}" for name in boundary_names]


def test_fixed_width_selection_side_artifacts_remain_aligned_when_storage_is_reconstructed(tmp_path) -> None:
    storage = ArtifactStorage(artifact_path=tmp_path)
    storage.final_dataset_path.mkdir(parents=True)
    storage.dropped_columns_dataset_path.mkdir()

    lazy.pd.DataFrame({"value": [99_999]}).to_parquet(
        storage.final_dataset_path / "batch_099999.parquet",
        index=False,
    )
    lazy.pd.DataFrame({"value": [100_000]}).to_parquet(
        storage.final_dataset_path / "batch_100000.parquet",
        index=False,
    )
    lazy.pd.DataFrame({"dropped": ["five-digit"]}).to_parquet(
        storage.dropped_columns_dataset_path / "batch_099999.parquet",
        index=False,
    )
    lazy.pd.DataFrame({"dropped": ["six-digit"]}).to_parquet(
        storage.dropped_columns_dataset_path / "batch_100000.parquet",
        index=False,
    )

    reconstructed = ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.ALWAYS)
    result = reconstructed.load_dataset_with_dropped_columns()

    assert result[["value", "dropped"]].to_dict(orient="records") == [
        {"value": 99_999, "dropped": "five-digit"},
        {"value": 100_000, "dropped": "six-digit"},
    ]


def test_materialize_empty_selection_uses_schema_anchor(stub_artifact_storage, stub_sample_dataframe) -> None:
    stub_artifact_storage.configure_selection_batch_file_width(
        max_candidate_records=100_001,
        candidate_batch_size=1,
    )
    stub_artifact_storage.write_selection_schema(stub_sample_dataframe.iloc[0:0])

    published = stub_artifact_storage.materialize_selection_dataset()
    dataset = lazy.pd.read_parquet(published / "batch_00000.parquet")

    assert dataset.empty
    assert dataset.columns.tolist() == stub_sample_dataframe.columns.tolist()


def test_selection_media_staging_promotes_only_referenced_files(stub_artifact_storage) -> None:
    stub_artifact_storage.begin_selection_media_batch(3)
    staging = stub_artifact_storage.selection_media_staging_path / "batch_00003" / "images" / "picture"
    staging.mkdir(parents=True)
    (staging / "accepted.png").write_bytes(b"accepted")
    (staging / "rejected.png").write_bytes(b"rejected")

    promoted = stub_artifact_storage.promote_selection_media(
        lazy.pd.DataFrame({"image": ["images/picture/accepted.png"]}),
        3,
    )

    relative = promoted.loc[0, "image"]
    assert relative == "images/selection_batch_00003/picture/accepted.png"
    assert (stub_artifact_storage.base_dataset_path / relative).read_bytes() == b"accepted"
    assert not stub_artifact_storage.selection_media_staging_path.joinpath("batch_00003").exists()
    assert not stub_artifact_storage.base_dataset_path.joinpath(
        "images/selection_batch_00003/picture/rejected.png"
    ).exists()


def test_selection_media_paths_use_configured_width_across_five_digit_boundary(
    stub_artifact_storage: ArtifactStorage,
) -> None:
    stub_artifact_storage.configure_selection_batch_file_width(
        max_candidate_records=100_001,
        candidate_batch_size=1,
    )

    promoted_paths: list[str] = []
    for batch_id, expected_name in ((99_999, "099999"), (100_000, "100000")):
        stub_artifact_storage.begin_selection_media_batch(batch_id)
        staging = stub_artifact_storage.selection_media_staging_path / f"batch_{expected_name}" / "images" / "picture"
        staging.mkdir(parents=True)
        (staging / "accepted.png").write_bytes(str(batch_id).encode())

        promoted = stub_artifact_storage.promote_selection_media(
            lazy.pd.DataFrame({"image": ["images/picture/accepted.png"]}),
            batch_id,
        )
        promoted_paths.append(promoted.loc[0, "image"])

    assert promoted_paths == [
        "images/selection_batch_099999/picture/accepted.png",
        "images/selection_batch_100000/picture/accepted.png",
    ]


def test_selection_media_promotion_rewrites_duplicate_references(stub_artifact_storage) -> None:
    stub_artifact_storage.begin_selection_media_batch(3)
    staging = stub_artifact_storage.selection_media_staging_path / "batch_00003" / "images" / "picture"
    staging.mkdir(parents=True)
    (staging / "shared.png").write_bytes(b"shared")

    promoted = stub_artifact_storage.promote_selection_media(
        lazy.pd.DataFrame(
            {
                "primary": ["images/picture/shared.png"],
                "references": [["images/picture/shared.png", {"copy": "images/picture/shared.png"}]],
            }
        ),
        3,
    )

    expected = "images/selection_batch_00003/picture/shared.png"
    assert promoted.loc[0, "primary"] == expected
    assert promoted.loc[0, "references"] == [expected, {"copy": expected}]
    assert (stub_artifact_storage.base_dataset_path / expected).read_bytes() == b"shared"


def test_clean_uncommitted_selection_batch_removes_promoted_media_and_side_artifacts(
    stub_artifact_storage, stub_sample_dataframe
) -> None:
    stub_artifact_storage.begin_selection_media_batch(5)
    staging = stub_artifact_storage.selection_media_staging_path / "batch_00005" / "images" / "picture"
    staging.mkdir(parents=True)
    (staging / "accepted.png").write_bytes(b"accepted")
    promoted = stub_artifact_storage.promote_selection_media(
        lazy.pd.DataFrame({"image": ["images/picture/accepted.png"]}),
        5,
    )
    stub_artifact_storage.write_batch_to_parquet_file(
        5,
        stub_sample_dataframe,
        BatchStage.DROPPED_COLUMNS,
    )
    stub_artifact_storage.write_batch_to_parquet_file(
        5,
        stub_sample_dataframe,
        BatchStage.PROCESSORS_OUTPUTS,
        subfolder="schema",
    )

    stub_artifact_storage.clean_uncommitted_selection_batch(5)

    assert not (stub_artifact_storage.base_dataset_path / promoted.loc[0, "image"]).exists()
    assert not (stub_artifact_storage.dropped_columns_dataset_path / "batch_00005.parquet").exists()
    assert not (stub_artifact_storage.processors_outputs_path / "schema" / "batch_00005.parquet").exists()


def test_selection_media_promotion_does_not_follow_path_traversal(stub_artifact_storage) -> None:
    stub_artifact_storage.begin_selection_media_batch(4)
    outside = stub_artifact_storage.selection_media_staging_path / "sensitive.png"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"sensitive")

    promoted = stub_artifact_storage.promote_selection_media(
        lazy.pd.DataFrame({"image": ["images/../../sensitive.png"]}),
        4,
    )

    assert promoted.loc[0, "image"] == "images/../../sensitive.png"
    assert outside.read_bytes() == b"sensitive"
