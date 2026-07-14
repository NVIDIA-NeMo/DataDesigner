# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

from data_designer.integrations.huggingface.client import HuggingFaceHubClient, HuggingFaceHubClientUploadError


@pytest.fixture
def mock_hf_api() -> MagicMock:
    """Mock HfApi for testing."""
    with patch("data_designer.integrations.huggingface.client.HfApi") as mock:
        api_instance = MagicMock()
        api_instance.repo_info.return_value.sha = "a" * 40
        mock.return_value = api_instance
        yield api_instance


@pytest.fixture
def mock_dataset_card() -> MagicMock:
    """Mock DataDesignerDatasetCard for testing."""
    with patch("data_designer.integrations.huggingface.client.DataDesignerDatasetCard") as mock:
        card_instance = MagicMock()
        mock.from_metadata.return_value = card_instance
        yield mock


@pytest.fixture
def sample_dataset_path(tmp_path: Path) -> Path:
    """Create a sample dataset directory structure.

    Structure mirrors actual DataDesigner output:
    - parquet-files/: Main dataset batch files
    - processors-files/{processor_name}/: Processor output batch files (same structure)
    - metadata.json: Dataset metadata
    - builder_config.json: Configuration
    """
    base_path = tmp_path / "dataset"
    base_path.mkdir()

    # Create parquet-files directory with batch files
    parquet_dir = base_path / "parquet-files"
    parquet_dir.mkdir()
    (parquet_dir / "batch_00000.parquet").write_text("dummy parquet data")
    (parquet_dir / "batch_00001.parquet").write_text("dummy parquet data")

    # Create processors-files directory with same structure as main parquet-files
    processors_dir = base_path / "processors-files"
    processors_dir.mkdir()
    processor1_dir = processors_dir / "processor1"
    processor1_dir.mkdir()
    (processor1_dir / "batch_00000.parquet").write_text("dummy processor output")
    (processor1_dir / "batch_00001.parquet").write_text("dummy processor output")

    processor2_dir = processors_dir / "processor2"
    processor2_dir.mkdir()
    (processor2_dir / "batch_00000.parquet").write_text("dummy processor output")

    # Create metadata.json with matching column statistics
    metadata = {
        "target_num_records": 100,
        "total_num_batches": 2,
        "buffer_size": 50,
        "schema": {"col1": "string"},
        "file_paths": {
            "parquet-files": ["parquet-files/batch_00000.parquet", "parquet-files/batch_00001.parquet"],
            "processor-files": {
                "processor1": ["processors-files/processor1/batch_00000.parquet"],
                "processor2": ["processors-files/processor2/batch_00000.parquet"],
            },
        },
        "num_completed_batches": 2,
        "dataset_name": "dataset",
        "column_statistics": [
            {
                "column_name": "col1",
                "num_records": 100,
                "num_unique": 100,
                "num_null": 0,
                "simple_dtype": "string",
                "pyarrow_dtype": "string",
                "column_type": "sampler",
                "sampler_type": "uuid",
            }
        ],
    }
    (base_path / "metadata.json").write_text(json.dumps(metadata))

    # Create builder_config.json with realistic BuilderConfig structure
    builder_config = {
        "data_designer": {
            "columns": [
                {
                    "name": "col1",
                    "column_type": "sampler",
                    "sampler_type": "uuid",
                    "params": {},
                }
            ],
            "model_configs": [],
            "constraints": None,
            "seed_config": None,
            "profilers": None,
        }
    }
    (base_path / "builder_config.json").write_text(json.dumps(builder_config))

    return base_path


def _make_record_selection_publishable(dataset_path: Path, *, publication_id: str) -> dict:
    metadata_path = dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": metadata.get("target_num_records", 0),
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": publication_id,
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data", "images"]},
        }
    )
    metadata_path.write_text(json.dumps(metadata))
    return metadata


def test_client_initialization() -> None:
    """Test HuggingFaceHubClient initialization."""
    with patch("data_designer.integrations.huggingface.client.HfApi"):
        client = HuggingFaceHubClient(token="test-token")
        assert client.has_token is True


def test_client_initialization_no_token() -> None:
    """Test HuggingFaceHubClient initialization without token."""
    with patch("data_designer.integrations.huggingface.client.HfApi"):
        client = HuggingFaceHubClient()
        assert client.has_token is False


def test_upload_dataset_creates_repo(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset creates a repository."""
    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=sample_dataset_path,
        description="Test dataset",
    )

    # Verify repo creation was called
    mock_hf_api.create_repo.assert_called_once()
    assert mock_hf_api.create_repo.call_args.kwargs["repo_id"] == "test/dataset"


def test_upload_dataset_uploads_parquet_files(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset uploads parquet files."""
    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=sample_dataset_path,
        description="Test dataset",
    )

    # Check that upload_folder was called for parquet files
    calls = [call for call in mock_hf_api.upload_folder.call_args_list if call.kwargs["path_in_repo"] == "data"]
    assert len(calls) >= 1


def test_upload_dataset_uploads_processor_outputs(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset uploads processor outputs."""
    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=sample_dataset_path,
        description="Test dataset",
    )

    # Check that upload_folder was called for processor outputs
    calls = [call for call in mock_hf_api.upload_folder.call_args_list if "processor1" in call.kwargs["path_in_repo"]]
    assert len(calls) >= 1


def test_ordinary_upload_includes_single_file_processor_and_rewrites_metadata(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
) -> None:
    processor_path = sample_dataset_path / "processors-files" / "summary.parquet"
    processor_path.write_bytes(b"single-file processor output")
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["file_paths"]["processor-files"]["summary"] = ["processors-files/summary.parquet"]
    metadata_path.write_text(json.dumps(metadata))
    uploaded_files: dict[str, bytes] = {}

    def capture_upload(*, path_or_fileobj: str, path_in_repo: str, **_: object) -> None:
        uploaded_files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    mock_hf_api.upload_file.side_effect = capture_upload
    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/dataset", sample_dataset_path, "Test dataset")

    assert uploaded_files["summary/summary.parquet"] == b"single-file processor output"
    uploaded_metadata = json.loads(uploaded_files["metadata.json"])
    assert uploaded_metadata["file_paths"]["processor-files"]["summary"] == ["summary/summary.parquet"]


def test_ordinary_upload_prefers_processor_directory_over_same_stem_file(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
) -> None:
    (sample_dataset_path / "processors-files" / "processor1.parquet").write_bytes(b"shadowed single file")

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/dataset", sample_dataset_path, "Test dataset")

    single_file_paths = [call.kwargs["path_in_repo"] for call in mock_hf_api.upload_file.call_args_list]
    assert "processor1/processor1.parquet" not in single_file_paths
    directory_calls = [
        call for call in mock_hf_api.upload_folder.call_args_list if call.kwargs["path_in_repo"] == "processor1"
    ]
    assert len(directory_calls) == 1


def test_upload_dataset_uploads_config_files(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset uploads builder_config.json and metadata.json."""
    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=sample_dataset_path,
        description="Test dataset",
    )

    # Check that upload_file was called for config files
    upload_file_calls = mock_hf_api.upload_file.call_args_list
    uploaded_files = [call.kwargs["path_in_repo"] for call in upload_file_calls]
    assert "builder_config.json" in uploaded_files
    assert "metadata.json" in uploaded_files


def test_upload_dataset_returns_url(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset returns the correct URL."""
    client = HuggingFaceHubClient(token="test-token")

    url = client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=sample_dataset_path,
        description="Test dataset",
    )

    assert url == "https://huggingface.co/datasets/test/dataset"


def test_upload_dataset_with_private_repo(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test upload_dataset with private repository."""
    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=sample_dataset_path,
        description="Test dataset",
        private=True,
    )

    mock_hf_api.create_repo.assert_called_once_with(
        repo_id="test/dataset",
        repo_type="dataset",
        exist_ok=True,
        private=True,
    )


def test_upload_dataset_card_missing_metadata(tmp_path: Path) -> None:
    """Test upload fails when metadata.json is missing."""
    client = HuggingFaceHubClient(token="test-token")

    # Create directory without metadata.json
    base_path = tmp_path / "dataset"
    base_path.mkdir()

    with pytest.raises(HuggingFaceHubClientUploadError, match="Required file not found"):
        client.upload_dataset(
            repo_id="test/dataset",
            base_dataset_path=base_path,
            description="Test description",
        )


def test_upload_dataset_card_calls_push_to_hub(mock_hf_api: MagicMock, sample_dataset_path: Path) -> None:
    """Test upload_dataset generates and pushes dataset card."""
    client = HuggingFaceHubClient(token="test-token")

    with patch("data_designer.integrations.huggingface.client.DataDesignerDatasetCard") as mock_card_class:
        mock_card = MagicMock()
        mock_card_class.from_metadata.return_value = mock_card

        client.upload_dataset(
            repo_id="test/dataset",
            base_dataset_path=sample_dataset_path,
            description="Test description",
        )

        # Verify card was created and pushed
        mock_card_class.from_metadata.assert_called_once()
        mock_card.push_to_hub.assert_called_once()


def test_upload_dataset_without_processors(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, tmp_path: Path
) -> None:
    """Test upload_dataset when no processor outputs exist."""
    # Create dataset path without processors directory
    base_path = tmp_path / "dataset"
    base_path.mkdir()

    parquet_dir = base_path / "parquet-files"
    parquet_dir.mkdir()
    (parquet_dir / "batch_00000.parquet").write_text("dummy data")

    metadata = {"target_num_records": 10, "schema": {"col1": "string"}, "column_statistics": []}
    (base_path / "metadata.json").write_text(json.dumps(metadata))

    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=base_path,
        description="Test dataset",
    )

    # Should only upload parquet files, not processors
    folder_calls = mock_hf_api.upload_folder.call_args_list
    data_calls = [call for call in folder_calls if call.kwargs["path_in_repo"] == "data"]
    processor_calls = [call for call in folder_calls if "processor" in call.kwargs["path_in_repo"]]

    assert len(data_calls) == 1  # Main parquet files uploaded
    assert len(processor_calls) == 0  # No processor files


def test_upload_dataset_without_builder_config(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, tmp_path: Path
) -> None:
    """Test upload_dataset when builder_config.json doesn't exist."""
    base_path = tmp_path / "dataset"
    base_path.mkdir()

    parquet_dir = base_path / "parquet-files"
    parquet_dir.mkdir()
    (parquet_dir / "batch_00000.parquet").write_text("dummy data")

    metadata = {"target_num_records": 10, "schema": {"col1": "string"}, "column_statistics": []}
    (base_path / "metadata.json").write_text(json.dumps(metadata))

    # No builder_config.json file

    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=base_path,
        description="Test dataset",
    )

    # Should only upload metadata.json, not builder_config.json
    file_calls = mock_hf_api.upload_file.call_args_list
    uploaded_files = [call.kwargs["path_in_repo"] for call in file_calls]

    assert len(uploaded_files) == 1  # Only metadata.json
    assert "metadata.json" in uploaded_files
    assert "builder_config.json" not in uploaded_files


def test_upload_dataset_multiple_processors(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that multiple processor outputs are uploaded correctly."""
    client = HuggingFaceHubClient(token="test-token")

    client.upload_dataset(
        repo_id="test/dataset",
        base_dataset_path=sample_dataset_path,
        description="Test dataset",
    )

    # Check that both processors were uploaded
    folder_calls = mock_hf_api.upload_folder.call_args_list
    processor_calls = [call for call in folder_calls if "processor" in call.kwargs["path_in_repo"]]

    assert len(processor_calls) >= 2
    processor_paths = [call.kwargs["path_in_repo"] for call in processor_calls]
    assert any("processor1" in path for path in processor_paths)
    assert any("processor2" in path for path in processor_paths)


# Error handling and validation tests


def test_validate_repo_id_invalid_format(sample_dataset_path: Path) -> None:
    """Test upload fails with invalid repo_id formats."""
    client = HuggingFaceHubClient(token="test-token")

    # Missing slash
    with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid repo_id format"):
        client.upload_dataset("my-dataset", sample_dataset_path, "Test")

    # Too many slashes (caught by regex)
    with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid repo_id format"):
        client.upload_dataset("user/org/dataset", sample_dataset_path, "Test")

    # Invalid characters (space)
    with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid repo_id format"):
        client.upload_dataset("user/my dataset", sample_dataset_path, "Test")

    # Empty string
    with pytest.raises(HuggingFaceHubClientUploadError, match="must be a non-empty string"):
        client.upload_dataset("", sample_dataset_path, "Test")


def test_validate_dataset_path_not_exists(tmp_path: Path) -> None:
    """Test upload fails when dataset path doesn't exist."""
    client = HuggingFaceHubClient(token="test-token")
    non_existent = tmp_path / "does-not-exist"

    with pytest.raises(HuggingFaceHubClientUploadError, match="does not exist"):
        client.upload_dataset("test/dataset", non_existent, "Test")


def test_validate_dataset_path_is_file(tmp_path: Path) -> None:
    """Test upload fails when dataset path is a file."""
    client = HuggingFaceHubClient(token="test-token")
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory")

    with pytest.raises(HuggingFaceHubClientUploadError, match="not a directory"):
        client.upload_dataset("test/dataset", file_path, "Test")


def test_validate_dataset_path_missing_metadata(tmp_path: Path) -> None:
    """Test upload fails when metadata.json is missing."""
    client = HuggingFaceHubClient(token="test-token")
    base_path = tmp_path / "dataset"
    base_path.mkdir()

    with pytest.raises(HuggingFaceHubClientUploadError, match="Required file not found"):
        client.upload_dataset("test/dataset", base_path, "Test")


def test_validate_dataset_path_missing_parquet_folder(tmp_path: Path) -> None:
    """Test upload fails when parquet-files directory is missing."""
    client = HuggingFaceHubClient(token="test-token")
    base_path = tmp_path / "dataset"
    base_path.mkdir()
    (base_path / "metadata.json").write_text('{"target_num_records": 10}')

    with pytest.raises(HuggingFaceHubClientUploadError, match="Required directory not found"):
        client.upload_dataset("test/dataset", base_path, "Test")


def test_validate_dataset_path_empty_parquet_folder(tmp_path: Path) -> None:
    """Test upload fails when parquet-files directory is empty."""
    client = HuggingFaceHubClient(token="test-token")
    base_path = tmp_path / "dataset"
    base_path.mkdir()
    (base_path / "metadata.json").write_text('{"target_num_records": 10}')
    parquet_dir = base_path / "parquet-files"
    parquet_dir.mkdir()

    with pytest.raises(HuggingFaceHubClientUploadError, match="directory is empty"):
        client.upload_dataset("test/dataset", base_path, "Test")


def test_validate_dataset_path_invalid_metadata_json(tmp_path: Path) -> None:
    """Test upload fails when metadata.json contains invalid JSON."""
    client = HuggingFaceHubClient(token="test-token")
    base_path = tmp_path / "dataset"
    base_path.mkdir()
    (base_path / "metadata.json").write_text("invalid json {{{")
    parquet_dir = base_path / "parquet-files"
    parquet_dir.mkdir()
    (parquet_dir / "batch_00000.parquet").write_text("data")

    with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid JSON"):
        client.upload_dataset("test/dataset", base_path, "Test")


@pytest.mark.parametrize("metadata", [[], ["not", "an", "object"], None, True])
def test_validate_dataset_path_rejects_non_object_metadata(tmp_path: Path, metadata: object) -> None:
    base_path = tmp_path / "dataset"
    parquet_path = base_path / "parquet-files" / "batch_00000.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_text("data")
    (base_path / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(HuggingFaceHubClientUploadError, match="metadata.json must contain a JSON object"):
        HuggingFaceHubClient._validate_dataset_path(base_path)


def test_validate_dataset_path_invalid_builder_config_json(tmp_path: Path) -> None:
    """Test upload fails when builder_config.json contains invalid JSON."""
    client = HuggingFaceHubClient(token="test-token")
    base_path = tmp_path / "dataset"
    base_path.mkdir()
    (base_path / "metadata.json").write_text('{"target_num_records": 10}')
    (base_path / "builder_config.json").write_text("invalid json {{{")
    parquet_dir = base_path / "parquet-files"
    parquet_dir.mkdir()
    (parquet_dir / "batch_00000.parquet").write_text("data")

    with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid JSON"):
        client.upload_dataset("test/dataset", base_path, "Test")


@pytest.mark.parametrize("builder_config", [[], ["not", "an", "object"], None, True])
def test_validate_dataset_path_rejects_non_object_builder_config_before_network(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    builder_config: object,
) -> None:
    (sample_dataset_path / "builder_config.json").write_text(json.dumps(builder_config))

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="builder_config.json must contain a JSON object"):
        client.upload_dataset("test/dataset", sample_dataset_path, "Test")

    mock_hf_api.repo_exists.assert_not_called()
    mock_hf_api.create_repo.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "post_generation_state": "pending",
            "record_selection": {
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
        },
        {
            "post_generation_state": "complete",
            "record_selection": {
                "selection_satisfied": False,
                "selection_exhausted": True,
                "on_exhausted": "raise",
            },
        },
    ],
)
def test_validate_dataset_path_rejects_unpublishable_record_selection(tmp_path: Path, metadata: dict) -> None:
    base_path = tmp_path / "dataset"
    parquet_path = base_path / "parquet-files" / "batch_00000.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_text("placeholder")
    (base_path / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(HuggingFaceHubClientUploadError, match="selection and publication are complete"):
        HuggingFaceHubClient._validate_dataset_path(base_path)


@pytest.mark.parametrize("record_selection", [None, [], ["invalid"], "invalid", 0, False])
def test_validate_dataset_path_rejects_non_object_record_selection_before_network(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    record_selection: object,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["record_selection"] = record_selection
    metadata_path.write_text(json.dumps(metadata))

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="record_selection.*JSON object"):
        client.upload_dataset("test/dataset", sample_dataset_path, "Test")

    assert mock_hf_api.method_calls == []


@pytest.mark.parametrize("publication_id", [None, "", "   ", 123])
def test_record_selection_upload_requires_nonempty_publication_id_before_network(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    publication_id: object,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": publication_id,
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata))

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="non-empty publication_id"):
        client.upload_dataset("test/dataset", sample_dataset_path, "Test")

    assert mock_hf_api.method_calls == []


@pytest.mark.parametrize("record_selection", [False, True], ids=["legacy", "record-selection"])
@pytest.mark.parametrize(
    "symlink_location",
    [
        "metadata",
        "builder_config",
        "final_parquet",
        "images_file",
        "images_directory",
        "processor_file",
        "processor_directory",
    ],
)
def test_upload_rejects_managed_symlinks_before_network(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    tmp_path: Path,
    record_selection: bool,
    symlink_location: str,
) -> None:
    if record_selection:
        _make_record_selection_publishable(
            sample_dataset_path,
            publication_id=f"symlink-{symlink_location}",
        )

    outside_path = tmp_path / "outside"
    outside_path.mkdir()
    outside_file = outside_path / "private.txt"
    outside_file.write_text("must not be uploaded")

    if symlink_location == "metadata":
        local_path = sample_dataset_path / "metadata.json"
        target_path = outside_path / "metadata.json"
        target_path.write_bytes(local_path.read_bytes())
        local_path.unlink()
        local_path.symlink_to(target_path)
    elif symlink_location == "builder_config":
        local_path = sample_dataset_path / "builder_config.json"
        target_path = outside_path / "builder_config.json"
        target_path.write_bytes(local_path.read_bytes())
        local_path.unlink()
        local_path.symlink_to(target_path)
    elif symlink_location == "final_parquet":
        local_path = sample_dataset_path / "parquet-files" / "batch_00000.parquet"
        local_path.unlink()
        local_path.symlink_to(outside_file)
    elif symlink_location == "images_file":
        images_path = sample_dataset_path / "images"
        images_path.mkdir()
        (images_path / "private.txt").symlink_to(outside_file)
    elif symlink_location == "images_directory":
        (sample_dataset_path / "images").symlink_to(outside_path, target_is_directory=True)
    elif symlink_location == "processor_file":
        local_path = sample_dataset_path / "processors-files" / "processor1" / "batch_00000.parquet"
        local_path.unlink()
        local_path.symlink_to(outside_file)
    else:
        (sample_dataset_path / "processors-files" / "linked").symlink_to(
            outside_path,
            target_is_directory=True,
        )

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="symbolic link"):
        client.upload_dataset("test/dataset", sample_dataset_path, "Test")

    assert mock_hf_api.method_calls == []


@pytest.mark.parametrize("raced_filename", ["metadata.json", "builder_config.json"])
def test_record_selection_staging_rejects_symlink_swapped_in_after_scan(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    tmp_path: Path,
    raced_filename: str,
) -> None:
    metadata = _make_record_selection_publishable(
        sample_dataset_path,
        publication_id=f"before-{raced_filename}",
    )
    external_path = tmp_path / f"external-{raced_filename}"
    if raced_filename == "metadata.json":
        external_metadata = deepcopy(metadata)
        external_metadata["record_selection"]["publication_id"] = "external-publication"
        external_path.write_text(json.dumps(external_metadata))
    else:
        external_path.write_text(json.dumps({"external": "builder config"}))

    local_path = sample_dataset_path / raced_filename
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    real_validate_symlinks = HuggingFaceHubClient._validate_managed_upload_symlinks
    scan_count = 0

    def validate_then_swap(base_dataset_path: Path) -> None:
        nonlocal scan_count
        real_validate_symlinks(base_dataset_path)
        scan_count += 1
        if scan_count == 1:
            local_path.unlink()
            local_path.symlink_to(external_path)

    client = HuggingFaceHubClient(token="test-token")
    with patch.object(HuggingFaceHubClient, "_validate_managed_upload_symlinks", side_effect=validate_then_swap):
        with pytest.raises(HuggingFaceHubClientUploadError, match="symbolic link|changed while it was being staged"):
            client._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert not (staging_directory / "dataset" / raced_filename).is_file()
    assert mock_hf_api.method_calls == []


@pytest.mark.parametrize("race_target", ["file", "ancestor-directory"])
def test_record_selection_staging_rejects_managed_tree_symlink_swap_during_copy(
    sample_dataset_path: Path,
    tmp_path: Path,
    race_target: str,
) -> None:
    _make_record_selection_publishable(
        sample_dataset_path,
        publication_id=f"tree-race-{race_target}",
    )
    external_path = tmp_path / "external"
    external_path.mkdir()
    (external_path / "secret.txt").write_text("SECRET")
    if race_target == "file":
        local_path = sample_dataset_path / "parquet-files" / "batch_00000.parquet"
        external_target = external_path / "secret.txt"
    else:
        local_path = sample_dataset_path / "processors-files" / "processor1"
        external_target = external_path
    original_path = local_path.with_name(f"{local_path.name}-original")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    real_open = os.open
    swapped = False

    def open_then_swap(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd is not None and path == local_path.name and not swapped:
            if local_path.is_dir():
                local_path.rename(original_path)
                local_path.symlink_to(external_target, target_is_directory=True)
            else:
                local_path.rename(original_path)
                local_path.symlink_to(external_target)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with patch("data_designer.integrations.huggingface.client.os.open", side_effect=open_then_swap):
        with pytest.raises(HuggingFaceHubClientUploadError, match="symbolic link|changed while it was being staged"):
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert swapped
    staged_dataset_path = staging_directory / "dataset"
    assert all(path.read_bytes() != b"SECRET" for path in staged_dataset_path.rglob("*") if path.is_file())


def test_record_selection_staging_rejects_symlinked_parent_swapped_after_scan(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="ancestor-race")
    live_parent = tmp_path / "live"
    live_parent.mkdir()
    live_dataset_path = live_parent / "dataset"
    sample_dataset_path.rename(live_dataset_path)
    external_parent = tmp_path / "external"
    external_dataset_path = external_parent / "dataset"
    shutil.copytree(live_dataset_path, external_dataset_path)
    (external_dataset_path / "parquet-files" / "batch_00000.parquet").write_text("SECRET")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    original_parent = tmp_path / "live-original"
    real_validate_symlinks = HuggingFaceHubClient._validate_managed_upload_symlinks

    def validate_then_swap(base_dataset_path: Path) -> None:
        real_validate_symlinks(base_dataset_path)
        live_parent.rename(original_parent)
        live_parent.symlink_to(external_parent, target_is_directory=True)

    with patch.object(HuggingFaceHubClient, "_validate_managed_upload_symlinks", side_effect=validate_then_swap):
        with pytest.raises(HuggingFaceHubClientUploadError, match="symbolic link|changed while it was being staged"):
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=live_dataset_path,
                staging_directory=staging_directory,
            )

    staged_dataset_path = staging_directory / "dataset"
    assert all(path.read_bytes() != b"SECRET" for path in staged_dataset_path.rglob("*") if path.is_file())


def test_record_selection_staging_rejects_in_place_file_mutation(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="in-place-race")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    local_path = sample_dataset_path / "parquet-files" / "batch_00000.parquet"
    real_copyfileobj = shutil.copyfileobj
    mutated = False

    def copy_then_mutate(source: object, destination: object, length: int = 0) -> None:
        nonlocal mutated
        real_copyfileobj(source, destination, length)
        if not mutated:
            local_path.write_text("changed in place")
            mutated = True

    with patch("data_designer.integrations.huggingface.client.shutil.copyfileobj", side_effect=copy_then_mutate):
        with pytest.raises(HuggingFaceHubClientUploadError, match="changed while it was being staged"):
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert mutated


def test_record_selection_staging_rejects_prior_file_mutation_during_later_copy(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="cross-file-race")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    prior_path = sample_dataset_path / "parquet-files" / "batch_00000.parquet"
    real_copyfileobj = shutil.copyfileobj
    copy_count = 0

    def mutate_prior_during_next_copy(source: object, destination: object, length: int = 0) -> None:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            prior_path.write_text("changed after its copy completed")
        real_copyfileobj(source, destination, length)

    with patch(
        "data_designer.integrations.huggingface.client.shutil.copyfileobj",
        side_effect=mutate_prior_during_next_copy,
    ):
        with pytest.raises(HuggingFaceHubClientUploadError, match="changed while it was being staged"):
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert copy_count >= 2


def test_record_selection_staging_rejects_later_file_rewrite_after_snapshot(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="snapshot-baseline-race")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    later_path = sample_dataset_path / "parquet-files" / "batch_00001.parquet"
    real_copy_file = HuggingFaceHubClient._copy_managed_file_at
    mutated = False

    def rewrite_before_copy(
        parent_directory_fd: int,
        name: str,
        destination: Path,
        display_path: Path,
        expected_stat: os.stat_result | None = None,
    ) -> None:
        nonlocal mutated
        if display_path == later_path and not mutated:
            later_path.write_text("changed after snapshot capture")
            mutated = True
        real_copy_file(
            parent_directory_fd,
            name,
            destination,
            display_path,
            expected_stat=expected_stat,
        )

    with patch.object(HuggingFaceHubClient, "_copy_managed_file_at", side_effect=rewrite_before_copy):
        with pytest.raises(HuggingFaceHubClientUploadError, match="changed while it was being staged"):
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert mutated


def test_record_selection_staging_rejects_top_level_directory_swap_and_restore(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="top-level-race")
    parquet_path = sample_dataset_path / "parquet-files"
    original_path = sample_dataset_path / "parquet-files-original"
    malicious_path = tmp_path / "parquet-files-malicious"
    shutil.copytree(parquet_path, malicious_path)
    (malicious_path / "batch_00000.parquet").write_text("MALICIOUS")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    real_copy_directory = HuggingFaceHubClient._copy_managed_directory_at
    swapped = False

    def swap_copy_and_restore(
        parent_directory_fd: int,
        name: str,
        destination: Path,
        display_path: Path,
        expected_stat: os.stat_result | None = None,
        **copy_kwargs: object,
    ) -> None:
        nonlocal swapped
        if display_path == parquet_path and not swapped:
            swapped = True
            parquet_path.rename(original_path)
            malicious_path.rename(parquet_path)
            try:
                real_copy_directory(
                    parent_directory_fd,
                    name,
                    destination,
                    display_path,
                    expected_stat=expected_stat,
                    **copy_kwargs,
                )
            finally:
                parquet_path.rename(malicious_path)
                original_path.rename(parquet_path)
            return
        real_copy_directory(
            parent_directory_fd,
            name,
            destination,
            display_path,
            expected_stat=expected_stat,
            **copy_kwargs,
        )

    with patch.object(HuggingFaceHubClient, "_copy_managed_directory_at", side_effect=swap_copy_and_restore):
        with pytest.raises(HuggingFaceHubClientUploadError, match="changed while it was being staged"):
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert swapped


def test_record_selection_staging_rejects_late_optional_directory(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="late-optional-directory")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    real_copy_optional = HuggingFaceHubClient._copy_optional_managed_directory_at
    added = False

    def copy_then_add(
        parent_directory_fd: int,
        name: str,
        destination: Path,
        display_path: Path,
        **copy_kwargs: object,
    ) -> None:
        nonlocal added
        real_copy_optional(parent_directory_fd, name, destination, display_path, **copy_kwargs)
        if name == "images" and not added:
            images_path = sample_dataset_path / "images"
            images_path.mkdir()
            (images_path / "late.txt").write_text("late")
            added = True

    with patch.object(
        HuggingFaceHubClient,
        "_copy_optional_managed_directory_at",
        side_effect=copy_then_add,
    ):
        with pytest.raises(HuggingFaceHubClientUploadError, match="changed while it was being staged"):
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert added


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_record_selection_staging_rejects_special_files_without_blocking(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="special-file")
    images_path = sample_dataset_path / "images"
    images_path.mkdir()
    os.mkfifo(images_path / "pipe")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()

    with pytest.raises(HuggingFaceHubClientUploadError, match="regular file or directory"):
        HuggingFaceHubClient._stage_record_selection_dataset(
            base_dataset_path=sample_dataset_path,
            staging_directory=staging_directory,
        )


def test_upload_uses_pinned_metadata_for_record_selection_branch(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="pinned-selection")
    live_parent = tmp_path / "live-upload"
    live_parent.mkdir()
    live_dataset_path = live_parent / "dataset"
    sample_dataset_path.rename(live_dataset_path)
    external_parent = tmp_path / "external-upload"
    external_dataset_path = external_parent / "dataset"
    shutil.copytree(live_dataset_path, external_dataset_path)
    external_metadata_path = external_dataset_path / "metadata.json"
    external_metadata = json.loads(external_metadata_path.read_text())
    external_metadata.pop("record_selection")
    external_metadata.pop("publication")
    external_metadata.pop("post_generation_state")
    external_metadata_path.write_text(json.dumps(external_metadata))
    (external_dataset_path / "parquet-files" / "batch_00000.parquet").write_text("SECRET")
    original_parent = tmp_path / "live-upload-original"
    real_open_directory = HuggingFaceHubClient._open_managed_directory
    swapped = False

    def open_then_swap(path: Path) -> tuple[int, os.stat_result]:
        nonlocal swapped
        descriptor, opened_stat = real_open_directory(path)
        if not swapped:
            live_parent.rename(original_parent)
            live_parent.symlink_to(external_parent, target_is_directory=True)
            swapped = True
        return descriptor, opened_stat

    client = HuggingFaceHubClient(token="test-token")
    with patch.object(HuggingFaceHubClient, "_open_managed_directory", side_effect=open_then_swap):
        with pytest.raises(HuggingFaceHubClientUploadError, match="changed while it was being staged"):
            client.upload_dataset("test/selection", live_dataset_path, "Selection dataset")

    assert swapped
    assert mock_hf_api.method_calls == []


@pytest.mark.parametrize("fail_copy", [False, True], ids=["success", "copy-failure"])
def test_record_selection_staging_closes_source_descriptors(
    sample_dataset_path: Path,
    tmp_path: Path,
    fail_copy: bool,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id=f"fd-close-{fail_copy}")
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir()
    real_open = os.open
    opened_descriptors: list[int] = []

    def record_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptors.append(descriptor)
        return descriptor

    copy_context = (
        patch(
            "data_designer.integrations.huggingface.client.shutil.copyfileobj",
            side_effect=RuntimeError("copy failed"),
        )
        if fail_copy
        else nullcontext()
    )
    with patch("data_designer.integrations.huggingface.client.os.open", side_effect=record_open), copy_context:
        if fail_copy:
            with pytest.raises(RuntimeError, match="copy failed"):
                HuggingFaceHubClient._stage_record_selection_dataset(
                    base_dataset_path=sample_dataset_path,
                    staging_directory=staging_directory,
                )
        else:
            HuggingFaceHubClient._stage_record_selection_dataset(
                base_dataset_path=sample_dataset_path,
                staging_directory=staging_directory,
            )

    assert opened_descriptors
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_open_managed_directory_accepts_macos_var_alias(sample_dataset_path: Path) -> None:
    canonical_prefix = "/private/var/"
    if not str(sample_dataset_path).startswith(canonical_prefix) or not Path("/var").is_symlink():
        pytest.skip("macOS /var alias is not present")
    alias_path = Path(f"/var/{str(sample_dataset_path).removeprefix(canonical_prefix)}")

    descriptor, _ = HuggingFaceHubClient._open_managed_directory(alias_path)

    os.close(descriptor)


def test_open_managed_directory_accepts_stable_ancestor_symlink(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    canonical_parent = tmp_path / "canonical"
    canonical_parent.mkdir()
    canonical_dataset_path = canonical_parent / "dataset"
    sample_dataset_path.rename(canonical_dataset_path)
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(canonical_parent, target_is_directory=True)

    descriptor, opened_stat = HuggingFaceHubClient._open_managed_directory(alias_parent / "dataset")

    try:
        assert (opened_stat.st_dev, opened_stat.st_ino) == (
            canonical_dataset_path.stat().st_dev,
            canonical_dataset_path.stat().st_ino,
        )
    finally:
        os.close(descriptor)


def test_open_managed_directory_normalizes_resolve_runtime_error(sample_dataset_path: Path) -> None:
    with (
        patch.object(Path, "resolve", side_effect=RuntimeError("symlink loop")),
        pytest.raises(HuggingFaceHubClientUploadError, match="must be a directory"),
    ):
        HuggingFaceHubClient._open_managed_directory(sample_dataset_path)


def test_open_managed_directory_rejects_final_directory_swap_after_lstat(
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "dataset-original"
    malicious_path = tmp_path / "dataset-malicious"
    shutil.copytree(sample_dataset_path, malicious_path)
    real_lstat = os.lstat
    swapped = False

    def lstat_then_swap(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> os.stat_result:
        nonlocal swapped
        path_stat = real_lstat(path)
        if Path(path) == sample_dataset_path and not swapped:
            sample_dataset_path.rename(original_path)
            malicious_path.rename(sample_dataset_path)
            swapped = True
        return path_stat

    with (
        patch("data_designer.integrations.huggingface.client.os.lstat", side_effect=lstat_then_swap),
        pytest.raises(HuggingFaceHubClientUploadError, match="changed while it was being staged"),
    ):
        HuggingFaceHubClient._open_managed_directory(sample_dataset_path)

    assert swapped


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_read_managed_regular_file_rejects_fifo_swap_without_blocking(tmp_path: Path) -> None:
    managed_path = tmp_path / "metadata.json"
    managed_path.write_text("{}")
    real_open = os.open
    swapped = False

    def swap_then_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if Path(path) == managed_path and not swapped:
            managed_path.unlink()
            os.mkfifo(managed_path)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with (
        patch("data_designer.integrations.huggingface.client.os.open", side_effect=swap_then_open),
        pytest.raises(HuggingFaceHubClientUploadError, match="regular file"),
    ):
        HuggingFaceHubClient._read_managed_regular_file(managed_path)

    assert swapped


@pytest.mark.parametrize("processor_name", [".", "..", "nested/name", "nested\\name"])
def test_processor_hub_prefix_rejects_traversing_components(processor_name: str) -> None:
    with pytest.raises(HuggingFaceHubClientUploadError, match="non-traversing"):
        HuggingFaceHubClient._validate_processor_hub_prefix(processor_name)


@pytest.mark.skipif(os.name == "nt", reason="Backslash is a path separator on Windows")
@pytest.mark.parametrize("record_selection", [False, True], ids=["legacy", "record-selection"])
def test_upload_rejects_invalid_managed_hub_path_before_network(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    record_selection: bool,
) -> None:
    if record_selection:
        _make_record_selection_publishable(sample_dataset_path, publication_id="invalid-hub-path")
    images_path = sample_dataset_path / "images"
    images_path.mkdir()
    (images_path / "bad\\name.png").write_bytes(b"image")

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="valid relative Hub path"):
        client.upload_dataset("test/dataset", sample_dataset_path, "Test")

    assert mock_hf_api.method_calls == []


@pytest.mark.parametrize(
    ("file_group", "invalid_path"),
    [
        ("parquet-files", "parquet-files/../README.md"),
        ("processor-files", "processors-files/safe/../README.md"),
        ("processor-files", "processors-files/other/batch.parquet"),
    ],
)
def test_upload_rejects_invalid_metadata_artifact_path_before_network(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    file_group: str,
    invalid_path: str,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if file_group == "parquet-files":
        metadata["file_paths"][file_group] = [invalid_path]
    else:
        metadata["file_paths"][file_group] = {"safe": [invalid_path]}
    metadata_path.write_text(json.dumps(metadata))

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="valid relative Hub path|outside"):
        client.upload_dataset("test/dataset", sample_dataset_path, "Test")

    assert mock_hf_api.method_calls == []


@pytest.mark.parametrize(
    "processor_name",
    ["data", "images", "README.md", "metadata.json", "builder_config.json"],
)
@pytest.mark.parametrize("artifact_layout", ["directory", "single_file"])
@pytest.mark.parametrize("record_selection", [False, True], ids=["legacy", "record-selection"])
def test_upload_rejects_reserved_processor_hub_names_before_network(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    processor_name: str,
    artifact_layout: str,
    record_selection: bool,
) -> None:
    if record_selection:
        metadata = _make_record_selection_publishable(
            sample_dataset_path,
            publication_id=f"reserved-{processor_name}-{artifact_layout}",
        )
    else:
        metadata = json.loads((sample_dataset_path / "metadata.json").read_text())
    processors_path = sample_dataset_path / "processors-files"
    if artifact_layout == "directory":
        (processors_path / "processor1").rename(processors_path / processor_name)
        metadata["file_paths"]["processor-files"].pop("processor1")
        metadata["file_paths"]["processor-files"][processor_name] = [
            f"processors-files/{processor_name}/batch_00000.parquet"
        ]
    else:
        (processors_path / f"{processor_name}.parquet").write_text("reserved processor output")
        metadata["file_paths"]["processor-files"][processor_name] = [f"processors-files/{processor_name}.parquet"]
    (sample_dataset_path / "metadata.json").write_text(json.dumps(metadata))

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="reserved Hub dataset path"):
        client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    assert mock_hf_api.method_calls == []


def test_add_hub_file_rejects_duplicate_destination() -> None:
    additions: dict[str, Path | bytes] = {}
    HuggingFaceHubClient._add_hub_file(additions, path_in_repo="data/batch.parquet", source=b"main")

    with pytest.raises(HuggingFaceHubClientUploadError, match="same Hub path"):
        HuggingFaceHubClient._add_hub_file(
            additions,
            path_in_repo="data/batch.parquet",
            source=b"processor",
        )

    assert additions == {"data/batch.parquet": b"main"}


@pytest.mark.parametrize(
    "paths",
    [
        ["README.md", "README.md/batch.parquet"],
        ["processor/output.parquet", "processor"],
        ["a", "a-foo", "a/b"],
    ],
    ids=["descendant", "ancestor", "interleaved-sort-order"],
)
def test_validate_hub_path_collisions_rejects_file_tree_conflicts(paths: list[str]) -> None:
    additions = {path: b"content" for path in paths}

    with pytest.raises(HuggingFaceHubClientUploadError, match="conflicting Hub paths"):
        HuggingFaceHubClient._validate_hub_path_collisions(additions)


def test_record_selection_upload_atomically_replaces_managed_hub_files(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-atomic-replace",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
                "candidate_records_generated": 120,
                "max_candidate_records": 200,
                "acceptance_rate": 100 / 120,
            },
            "publication": {
                "managed_hub_prefixes": ["data", "images", "processor1", "processor2"],
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata))

    remote_metadata = tmp_path / "remote-metadata.json"
    remote_metadata.write_text(
        json.dumps(
            {
                "publication": {"managed_hub_prefixes": ["data", "images", "legacy_processor"]},
                "file_paths": {
                    "processor-files": {
                        "legacy_processor": ["legacy_processor/batch_00000.parquet"],
                    }
                },
            }
        )
    )
    mock_hf_api.hf_hub_download.return_value = str(remote_metadata)
    mock_hf_api.list_repo_files.return_value = [
        "data/batch_00000.parquet",
        "data/obsolete.parquet",
        "images/obsolete.png",
        "legacy_processor/batch_00000.parquet",
        "notes/user-managed.txt",
    ]

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    mock_hf_api.create_commit.assert_called_once()
    assert mock_hf_api.hf_hub_download.call_args.kwargs["revision"] == "a" * 40
    assert mock_hf_api.list_repo_files.call_args.kwargs["revision"] == "a" * 40
    assert mock_hf_api.create_commit.call_args.kwargs["parent_commit"] == "a" * 40
    operations = mock_hf_api.create_commit.call_args.kwargs["operations"]
    deleted = {
        operation.path_in_repo for operation in operations if type(operation).__name__ == "CommitOperationDelete"
    }
    added = {operation.path_in_repo for operation in operations if type(operation).__name__ == "CommitOperationAdd"}
    assert deleted == {
        "data/obsolete.parquet",
        "images/obsolete.png",
        "legacy_processor/batch_00000.parquet",
    }
    assert "notes/user-managed.txt" not in deleted
    assert {
        "data/batch_00000.parquet",
        "data/batch_00001.parquet",
        "processor1/batch_00000.parquet",
        "processor2/batch_00000.parquet",
        "metadata.json",
        "builder_config.json",
        "README.md",
    } <= added
    mock_hf_api.upload_folder.assert_not_called()


@pytest.mark.parametrize("malicious_origin", ["local", "remote"])
def test_record_selection_upload_does_not_trust_notes_deletion_prefix_from_metadata(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
    tmp_path: Path,
    malicious_origin: str,
) -> None:
    metadata = _make_record_selection_publishable(
        sample_dataset_path,
        publication_id=f"untrusted-{malicious_origin}-prefix",
    )
    if malicious_origin == "local":
        metadata["publication"]["managed_hub_prefixes"].append("notes")
        (sample_dataset_path / "metadata.json").write_text(json.dumps(metadata))

    remote_metadata: dict = {
        "publication": {"managed_hub_prefixes": ["data", "images"]},
        "file_paths": {"processor-files": {}},
    }
    if malicious_origin == "remote":
        remote_metadata["publication"]["managed_hub_prefixes"].append("notes")
        remote_metadata["file_paths"]["processor-files"]["notes"] = ["notes/batch_00000.parquet"]
    remote_metadata_path = tmp_path / "remote-metadata.json"
    remote_metadata_path.write_text(json.dumps(remote_metadata))
    mock_hf_api.hf_hub_download.return_value = str(remote_metadata_path)
    mock_hf_api.list_repo_files.return_value = [
        "data/obsolete.parquet",
        "metadata.json",
        "README.md",
        "notes/batch_00000.parquet",
        "notes/user-managed.txt",
    ]

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    operations = mock_hf_api.create_commit.call_args.kwargs["operations"]
    deleted = {
        operation.path_in_repo for operation in operations if type(operation).__name__ == "CommitOperationDelete"
    }
    assert "data/obsolete.parquet" in deleted
    assert "notes/user-managed.txt" not in deleted
    if malicious_origin == "remote":
        assert "notes/batch_00000.parquet" in deleted
    else:
        assert "notes/batch_00000.parquet" not in deleted

    additions = {
        operation.path_in_repo: operation for operation in operations if isinstance(operation, CommitOperationAdd)
    }
    hub_metadata = json.loads(additions["metadata.json"].path_or_fileobj)
    assert set(hub_metadata["publication"]["managed_hub_prefixes"]) == {
        "data",
        "images",
        "processor1",
        "processor2",
    }
    card_metadata = mock_dataset_card.from_metadata.call_args.kwargs["metadata"]
    assert set(card_metadata["publication"]["managed_hub_prefixes"]) == {
        "data",
        "images",
        "processor1",
        "processor2",
    }


def test_record_selection_upload_deletes_stale_builder_config_only_when_current_snapshot_omits_it(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
    tmp_path: Path,
) -> None:
    _make_record_selection_publishable(sample_dataset_path, publication_id="without-builder-config")
    (sample_dataset_path / "builder_config.json").unlink()
    remote_metadata_path = tmp_path / "remote-metadata.json"
    remote_metadata_path.write_text(json.dumps({"file_paths": {"processor-files": {}}}))
    mock_hf_api.hf_hub_download.return_value = str(remote_metadata_path)
    mock_hf_api.list_repo_files.return_value = [
        "builder_config.json",
        "metadata.json",
        "README.md",
        "notes/user-managed.txt",
    ]

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    operations = mock_hf_api.create_commit.call_args.kwargs["operations"]
    deleted = {
        operation.path_in_repo for operation in operations if type(operation).__name__ == "CommitOperationDelete"
    }
    added = {operation.path_in_repo for operation in operations if isinstance(operation, CommitOperationAdd)}
    assert deleted == {"builder_config.json"}
    assert "builder_config.json" not in added
    assert {"metadata.json", "README.md"} <= added
    assert "notes/user-managed.txt" not in deleted


def test_record_selection_upload_stages_validated_snapshot_before_repo_setup(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-staged-snapshot",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data", "processor1", "processor2"]},
        }
    )
    metadata_path.write_text(json.dumps(metadata))
    parquet_path = sample_dataset_path / "parquet-files" / "batch_00000.parquet"
    parquet_path.write_text("validated snapshot")
    mock_hf_api.hf_hub_download.side_effect = RemoteEntryNotFoundError(
        "metadata not found",
        response=MagicMock(),
    )
    mock_hf_api.list_repo_files.return_value = []
    published_files: dict[str, str] = {}

    def mutate_live_metadata(**_: object) -> None:
        mutated_metadata = deepcopy(metadata)
        mutated_metadata["post_generation_state"] = "pending"
        metadata_path.write_text(json.dumps(mutated_metadata))
        parquet_path.write_text("mutated during repo setup")

    def capture_staged_files(*, operations: list[object], **_: object) -> None:
        for operation in operations:
            if isinstance(operation, CommitOperationAdd) and operation.path_in_repo == "data/batch_00000.parquet":
                published_files[operation.path_in_repo] = Path(cast(str, operation.path_or_fileobj)).read_text()

    mock_hf_api.create_repo.side_effect = mutate_live_metadata
    mock_hf_api.create_commit.side_effect = capture_staged_files

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    assert json.loads(metadata_path.read_text())["post_generation_state"] == "pending"
    assert mock_dataset_card.from_metadata.call_args.kwargs["metadata"]["post_generation_state"] == "complete"
    assert published_files == {"data/batch_00000.parquet": "validated snapshot"}
    operations = mock_hf_api.create_commit.call_args.kwargs["operations"]
    metadata_operation = next(operation for operation in operations if operation.path_in_repo == "metadata.json")
    assert json.loads(metadata_operation.path_or_fileobj)["post_generation_state"] == "complete"


def test_record_selection_upload_rejects_metadata_mutation_during_staging(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-metadata-mutation",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data", "processor1", "processor2"]},
        }
    )
    metadata_path.write_text(json.dumps(metadata))
    real_read_file = HuggingFaceHubClient._read_managed_regular_file_at
    metadata_read_count = 0

    def read_and_mutate_metadata(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
        **read_kwargs: object,
    ) -> bytes:
        nonlocal metadata_read_count
        if display_path == metadata_path:
            metadata_read_count += 1
            if metadata_read_count == 3:
                mutated_metadata = deepcopy(metadata)
                mutated_metadata["post_generation_state"] = "pending"
                metadata_path.write_text(json.dumps(mutated_metadata))
        return real_read_file(parent_directory_fd, name, display_path, **read_kwargs)

    client = HuggingFaceHubClient(token="test-token")
    with patch.object(HuggingFaceHubClient, "_read_managed_regular_file_at", side_effect=read_and_mutate_metadata):
        with pytest.raises(
            HuggingFaceHubClientUploadError,
            match="metadata changed while staging|input changed while it was being staged",
        ):
            client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    mock_hf_api.repo_exists.assert_not_called()
    mock_hf_api.create_repo.assert_not_called()
    mock_hf_api.repo_info.assert_not_called()


def test_record_selection_upload_rejects_complete_republication_during_staging(
    mock_hf_api: MagicMock,
    sample_dataset_path: Path,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-before",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data", "processor1", "processor2"]},
        }
    )
    metadata_path.write_text(json.dumps(metadata))
    real_read_file = HuggingFaceHubClient._read_managed_regular_file_at
    metadata_read_count = 0

    def read_and_republish(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
        **read_kwargs: object,
    ) -> bytes:
        nonlocal metadata_read_count
        if display_path == metadata_path:
            metadata_read_count += 1
            if metadata_read_count == 3:
                republished_metadata = deepcopy(metadata)
                republished_metadata["record_selection"]["publication_id"] = "publication-after"
                metadata_path.write_text(json.dumps(republished_metadata))
        return real_read_file(parent_directory_fd, name, display_path, **read_kwargs)

    client = HuggingFaceHubClient(token="test-token")
    with patch.object(HuggingFaceHubClient, "_read_managed_regular_file_at", side_effect=read_and_republish):
        with pytest.raises(
            HuggingFaceHubClientUploadError,
            match="metadata changed while staging|input changed while it was being staged",
        ):
            client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    mock_hf_api.repo_exists.assert_not_called()
    mock_hf_api.create_repo.assert_not_called()
    mock_hf_api.repo_info.assert_not_called()


def test_record_selection_upload_normalizes_local_card_failure_before_network(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-card-failure",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data", "processor1", "processor2"]},
        }
    )
    metadata_path.write_text(json.dumps(metadata))
    mock_dataset_card.from_metadata.side_effect = TypeError("invalid nested card metadata")

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(
        HuggingFaceHubClientUploadError,
        match="Failed to prepare the record-selection dataset.*invalid nested card metadata",
    ):
        client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    mock_hf_api.repo_exists.assert_not_called()
    mock_hf_api.create_repo.assert_not_called()


def test_record_selection_upload_includes_single_file_processor(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-single-file",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data", "summary"]},
        }
    )
    metadata["file_paths"]["processor-files"] = {
        "summary": ["processors-files/summary.parquet"],
    }
    metadata_path.write_text(json.dumps(metadata))
    (sample_dataset_path / "processors-files" / "summary.parquet").write_text("single-file output")
    mock_hf_api.hf_hub_download.side_effect = RemoteEntryNotFoundError(
        "metadata not found",
        response=MagicMock(),
    )
    mock_hf_api.list_repo_files.return_value = []

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    operations = mock_hf_api.create_commit.call_args.kwargs["operations"]
    additions = {operation.path_in_repo: operation for operation in operations if hasattr(operation, "path_or_fileobj")}
    assert "summary/summary.parquet" in additions
    hub_metadata = json.loads(additions["metadata.json"].path_or_fileobj)
    assert hub_metadata["file_paths"]["processor-files"]["summary"] == ["summary/summary.parquet"]


def test_record_selection_upload_preserves_fixed_width_processor_order_at_six_digits(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
) -> None:
    processor_path = sample_dataset_path / "processors-files" / "processor1"
    for path in processor_path.glob("*.parquet"):
        path.unlink()
    (processor_path / "batch_099999.parquet").write_text("candidate 99999")
    (processor_path / "batch_100000.parquet").write_text("candidate 100000")

    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    processor_paths = [
        "processors-files/processor1/batch_099999.parquet",
        "processors-files/processor1/batch_100000.parquet",
    ]
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-six-digit-boundary",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data", "processor1", "processor2"]},
        }
    )
    metadata["file_paths"]["processor-files"]["processor1"] = processor_paths
    metadata_path.write_text(json.dumps(metadata))
    mock_hf_api.hf_hub_download.side_effect = RemoteEntryNotFoundError(
        "metadata not found",
        response=MagicMock(),
    )
    mock_hf_api.list_repo_files.return_value = []

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")

    operations = mock_hf_api.create_commit.call_args.kwargs["operations"]
    processor_additions = [
        operation.path_in_repo
        for operation in operations
        if isinstance(operation, CommitOperationAdd) and operation.path_in_repo.startswith("processor1/")
    ]
    assert processor_additions == [
        "processor1/batch_099999.parquet",
        "processor1/batch_100000.parquet",
    ]
    metadata_operation = next(operation for operation in operations if operation.path_in_repo == "metadata.json")
    hub_metadata = json.loads(metadata_operation.path_or_fileobj)
    assert hub_metadata["file_paths"]["processor-files"]["processor1"] == processor_additions
    assert (
        mock_dataset_card.from_metadata.call_args.kwargs["metadata"]["file_paths"]["processor-files"]["processor1"]
        == processor_paths
    )


@pytest.mark.parametrize("status_code", [409, 412])
def test_record_selection_upload_reports_concurrent_hub_update(
    mock_hf_api: MagicMock,
    mock_dataset_card: MagicMock,
    sample_dataset_path: Path,
    status_code: int,
) -> None:
    metadata_path = sample_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "actual_num_records": 100,
            "post_generation_state": "complete",
            "record_selection": {
                "publication_id": "publication-concurrent-hub-update",
                "selection_satisfied": True,
                "selection_exhausted": False,
                "on_exhausted": "raise",
            },
            "publication": {"managed_hub_prefixes": ["data"]},
        }
    )
    metadata_path.write_text(json.dumps(metadata))
    mock_hf_api.hf_hub_download.side_effect = RemoteEntryNotFoundError(
        "metadata not found",
        response=MagicMock(),
    )
    mock_hf_api.list_repo_files.return_value = []
    response = MagicMock(status_code=status_code)
    mock_hf_api.create_commit.side_effect = HfHubHTTPError("parent commit changed", response=response)

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="changed while.*retry"):
        client.upload_dataset("test/selection", sample_dataset_path, "Selection dataset")


def test_load_remote_managed_processor_paths_tolerates_missing_metadata(mock_hf_api: MagicMock) -> None:
    mock_hf_api.hf_hub_download.side_effect = RemoteEntryNotFoundError(
        "metadata not found",
        response=MagicMock(),
    )

    client = HuggingFaceHubClient(token="test-token")

    assert client._load_remote_managed_processor_paths("test/new-or-legacy", remote_files=set()) == set()


def test_load_remote_managed_processor_paths_surfaces_download_failure(mock_hf_api: MagicMock) -> None:
    mock_hf_api.hf_hub_download.side_effect = RuntimeError("service unavailable")

    client = HuggingFaceHubClient(token="test-token")

    with pytest.raises(HuggingFaceHubClientUploadError, match="Failed to download existing Hub metadata"):
        client._load_remote_managed_processor_paths("test/dataset", remote_files=set())


@pytest.mark.parametrize(
    "remote_contents",
    [
        "not-json",
        "[]",
        json.dumps({"file_paths": {"processor-files": 3}}),
        json.dumps({"file_paths": {"processor-files": {".": ["./batch.parquet"]}}}),
        json.dumps({"file_paths": {"processor-files": {"..": ["../batch.parquet"]}}}),
        json.dumps({"file_paths": {"processor-files": {"bad\\name": ["bad\\name/batch.parquet"]}}}),
        json.dumps({"file_paths": {"processor-files": {"safe": ["safe/../README.md"]}}}),
        json.dumps({"file_paths": {"processor-files": {"safe": ["safe//batch.parquet"]}}}),
        json.dumps({"file_paths": {"processor-files": {"safe": ["safe/bad\\name.parquet"]}}}),
        json.dumps({"file_paths": {"processor-files": {"bad\0name": []}}}),
        None,
    ],
)
def test_load_remote_managed_processor_paths_surfaces_read_failure(
    mock_hf_api: MagicMock,
    tmp_path: Path,
    remote_contents: str | None,
) -> None:
    remote_metadata = tmp_path / "remote-metadata.json"
    if remote_contents is not None:
        remote_metadata.write_text(remote_contents)
    mock_hf_api.hf_hub_download.return_value = str(remote_metadata)

    client = HuggingFaceHubClient(token="test-token")

    with pytest.raises(HuggingFaceHubClientUploadError, match="Failed to read existing Hub metadata"):
        client._load_remote_managed_processor_paths("test/dataset", remote_files=set())


def test_upload_dataset_uploads_images_folder(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset uploads images when images folder exists with subfolders."""
    # Create images directory with column subfolders (matches MediaStorage structure)
    images_dir = sample_dataset_path / "images"
    col_dir = images_dir / "my_image_column"
    col_dir.mkdir(parents=True)
    (col_dir / "uuid1.png").write_bytes(b"fake png data")
    (col_dir / "uuid2.png").write_bytes(b"fake png data")

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset(repo_id="test/dataset", base_dataset_path=sample_dataset_path, description="Test dataset")

    # Check that upload_folder was called for images
    image_calls = [call for call in mock_hf_api.upload_folder.call_args_list if call.kwargs["path_in_repo"] == "images"]
    assert len(image_calls) == 1
    assert image_calls[0].kwargs["folder_path"] == str(images_dir)
    assert image_calls[0].kwargs["repo_type"] == "dataset"


def test_upload_dataset_skips_images_when_folder_missing(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset skips images upload when images folder doesn't exist."""
    # sample_dataset_path has no images/ directory by default
    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset(repo_id="test/dataset", base_dataset_path=sample_dataset_path, description="Test dataset")

    # No upload_folder call should target "images"
    image_calls = [call for call in mock_hf_api.upload_folder.call_args_list if call.kwargs["path_in_repo"] == "images"]
    assert len(image_calls) == 0


def test_upload_dataset_skips_images_when_folder_empty(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset skips images upload when images folder exists but is empty."""
    images_dir = sample_dataset_path / "images"
    images_dir.mkdir()

    client = HuggingFaceHubClient(token="test-token")
    client.upload_dataset(repo_id="test/dataset", base_dataset_path=sample_dataset_path, description="Test dataset")

    image_calls = [call for call in mock_hf_api.upload_folder.call_args_list if call.kwargs["path_in_repo"] == "images"]
    assert len(image_calls) == 0


def test_upload_dataset_images_upload_failure(
    mock_hf_api: MagicMock, mock_dataset_card: MagicMock, sample_dataset_path: Path
) -> None:
    """Test that upload_dataset raises error when images upload fails."""
    # Create images directory with a file
    images_dir = sample_dataset_path / "images"
    col_dir = images_dir / "col"
    col_dir.mkdir(parents=True)
    (col_dir / "img.png").write_bytes(b"fake")

    # Make upload_folder fail only for images
    def failing_upload_folder(**kwargs):
        if kwargs.get("path_in_repo") == "images":
            raise Exception("Network error")

    mock_hf_api.upload_folder.side_effect = failing_upload_folder

    client = HuggingFaceHubClient(token="test-token")
    with pytest.raises(HuggingFaceHubClientUploadError, match="Failed to upload images"):
        client.upload_dataset(repo_id="test/dataset", base_dataset_path=sample_dataset_path, description="Test dataset")


def test_upload_dataset_invalid_repo_id(mock_hf_api: MagicMock, sample_dataset_path: Path) -> None:
    """Test upload_dataset fails with invalid repo_id."""
    client = HuggingFaceHubClient(token="test-token")

    with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid repo_id format"):
        client.upload_dataset(
            repo_id="invalid-repo-id",  # Missing slash
            base_dataset_path=sample_dataset_path,
            description="Test dataset",
        )


def test_upload_dataset_authentication_error(mock_hf_api: MagicMock, sample_dataset_path: Path) -> None:
    """Test upload_dataset handles authentication errors."""
    client = HuggingFaceHubClient(token="invalid-token")

    # Mock 401 authentication error
    error_response = MagicMock()
    error_response.status_code = 401
    mock_hf_api.create_repo.side_effect = HfHubHTTPError("Unauthorized", response=error_response)

    with pytest.raises(HuggingFaceHubClientUploadError, match="Authentication failed"):
        client.upload_dataset(
            repo_id="test/dataset",
            base_dataset_path=sample_dataset_path,
            description="Test dataset",
        )


def test_upload_dataset_permission_error(mock_hf_api: MagicMock, sample_dataset_path: Path) -> None:
    """Test upload_dataset handles permission errors."""
    client = HuggingFaceHubClient(token="test-token")

    # Mock 403 permission error
    error_response = MagicMock()
    error_response.status_code = 403
    mock_hf_api.create_repo.side_effect = HfHubHTTPError("Forbidden", response=error_response)

    with pytest.raises(HuggingFaceHubClientUploadError, match="Permission denied"):
        client.upload_dataset(
            repo_id="test/dataset",
            base_dataset_path=sample_dataset_path,
            description="Test dataset",
        )


def test_upload_dataset_card_invalid_json(tmp_path: Path) -> None:
    """Test upload fails when metadata.json contains invalid JSON."""
    client = HuggingFaceHubClient(token="test-token")
    base_path = tmp_path / "dataset"
    base_path.mkdir()
    (base_path / "metadata.json").write_text("invalid json")

    # Create parquet directory so validation reaches the metadata JSON check
    parquet_dir = base_path / "parquet-files"
    parquet_dir.mkdir()
    (parquet_dir / "batch_00000.parquet").write_text("data")

    with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid JSON"):
        client.upload_dataset(
            repo_id="test/dataset",
            base_dataset_path=base_path,
            description="Test description",
        )


def test_update_metadata_paths(tmp_path: Path) -> None:
    """Test that _update_metadata_paths correctly updates file paths for HuggingFace Hub."""
    metadata = {
        "target_num_records": 100,
        "file_paths": {
            "parquet-files": [
                "parquet-files/batch_00000.parquet",
                "parquet-files/batch_00001.parquet",
            ],
            "processor-files": {
                "processor1": ["processors-files/processor1/batch_00000.parquet"],
                "processor2": ["processors-files/processor2/batch_00000.parquet"],
            },
        },
    }

    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    updated = HuggingFaceHubClient._update_metadata_paths(metadata_path)

    assert updated["file_paths"]["data"] == [
        "data/batch_00000.parquet",
        "data/batch_00001.parquet",
    ]
    assert updated["file_paths"]["processor-files"]["processor1"] == ["processor1/batch_00000.parquet"]
    assert updated["file_paths"]["processor-files"]["processor2"] == ["processor2/batch_00000.parquet"]
    assert "parquet-files" not in updated["file_paths"]


# push_to_hub_from_folder tests


def test_push_to_hub_from_folder_delegates_all_params() -> None:
    """Test that push_to_hub_from_folder forwards all parameters to HfApi and upload_dataset."""
    with patch("data_designer.integrations.huggingface.client.HfApi") as mock_hf_api_cls:
        mock_hf_api_cls.return_value = MagicMock()

        with patch.object(
            HuggingFaceHubClient, "upload_dataset", return_value="https://huggingface.co/datasets/test/dataset"
        ) as mock_upload:
            url = HuggingFaceHubClient.push_to_hub_from_folder(
                dataset_path="/some/path",
                repo_id="test/dataset",
                description="Test description",
                token="my-token",
                private=True,
                tags=["tag1", "tag2"],
            )

            assert url == "https://huggingface.co/datasets/test/dataset"
            mock_hf_api_cls.assert_called_once_with(token="my-token")
            mock_upload.assert_called_once_with(
                repo_id="test/dataset",
                base_dataset_path=Path("/some/path"),
                description="Test description",
                private=True,
                tags=["tag1", "tag2"],
            )


def test_push_to_hub_from_folder_converts_str_path_to_path() -> None:
    """Test that a string dataset_path is converted to Path before delegation."""
    with patch("data_designer.integrations.huggingface.client.HfApi"):
        with patch.object(HuggingFaceHubClient, "upload_dataset", return_value="https://example.com") as mock_upload:
            HuggingFaceHubClient.push_to_hub_from_folder(
                dataset_path="/string/path",
                repo_id="test/dataset",
                description="Test",
                token="t",
            )

            assert mock_upload.call_args.kwargs["base_dataset_path"] == Path("/string/path")
            assert isinstance(mock_upload.call_args.kwargs["base_dataset_path"], Path)


def test_push_to_hub_from_folder_default_optional_params() -> None:
    """Test defaults: token=None, private=False, tags=None."""
    with patch("data_designer.integrations.huggingface.client.HfApi") as mock_hf_api_cls:
        mock_hf_api_cls.return_value = MagicMock()

        with patch.object(HuggingFaceHubClient, "upload_dataset", return_value="https://example.com") as mock_upload:
            HuggingFaceHubClient.push_to_hub_from_folder(
                dataset_path="/some/path",
                repo_id="test/dataset",
                description="Test",
            )

            mock_hf_api_cls.assert_called_once_with(token=None)
            mock_upload.assert_called_once_with(
                repo_id="test/dataset",
                base_dataset_path=Path("/some/path"),
                description="Test",
                private=False,
                tags=None,
            )


def test_push_to_hub_from_folder_propagates_errors() -> None:
    """Test that errors from upload_dataset propagate through push_to_hub_from_folder."""
    with patch("data_designer.integrations.huggingface.client.HfApi"):
        with pytest.raises(HuggingFaceHubClientUploadError, match="Invalid repo_id format"):
            HuggingFaceHubClient.push_to_hub_from_folder(
                dataset_path="/any/path",
                repo_id="invalid-no-slash",
                description="Test",
            )
