# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import HfHubHTTPError

from data_designer.integrations.huggingface.client import HuggingFaceHubClient, HuggingFaceHubClientUploadError


@pytest.fixture
def mock_hf_api() -> MagicMock:
    """Mock HfApi for testing."""
    with patch("data_designer.integrations.huggingface.client.HfApi") as mock:
        api_instance = MagicMock()
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
