# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HuggingFaceHubClient."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.integrations.huggingface.client import (
    HuggingFaceHubClient,
    parse_size_category,
    pydantic_to_dict,
    resolve_hf_token,
)


class TestResolveHfToken:
    """Tests for resolve_hf_token function."""

    def test_resolve_token_provided(self) -> None:
        """Test that provided token is returned."""
        assert resolve_hf_token("test-token") == "test-token"

    def test_resolve_token_from_hub(self) -> None:
        """Test that token is resolved from huggingface_hub."""
        with patch("data_designer.integrations.huggingface.client.get_token", return_value="hub-token"):
            assert resolve_hf_token(None) == "hub-token"

    def test_resolve_token_none(self) -> None:
        """Test that None is returned when no token is available."""
        with patch("data_designer.integrations.huggingface.client.get_token", return_value=None):
            assert resolve_hf_token(None) is None

    def test_resolve_token_exception(self) -> None:
        """Test that exceptions are handled gracefully."""
        with patch("data_designer.integrations.huggingface.client.get_token", side_effect=Exception()):
            assert resolve_hf_token(None) is None


class TestParseSizeCategory:
    """Tests for parse_size_category function."""

    def test_small_dataset(self) -> None:
        """Test small dataset category."""
        assert parse_size_category(500) == "n<1K"

    def test_medium_dataset(self) -> None:
        """Test medium dataset category."""
        assert parse_size_category(5000) == "1K<n<10K"
        assert parse_size_category(50000) == "10K<n<100K"

    def test_large_dataset(self) -> None:
        """Test large dataset category."""
        assert parse_size_category(500000) == "100K<n<1M"
        assert parse_size_category(5000000) == "1M<n<10M"

    def test_very_large_dataset(self) -> None:
        """Test very large dataset category."""
        assert parse_size_category(50000000) == "10M<n<100M"
        assert parse_size_category(500000000) == "100M<n<1B"

    def test_extremely_large_dataset(self) -> None:
        """Test extremely large dataset category."""
        assert parse_size_category(5000000000) == "1B<n<10B"
        assert parse_size_category(50000000000) == "10B<n<100B"
        assert parse_size_category(500000000000) == "100B<n<1T"
        assert parse_size_category(5000000000000) == "n>1T"


class TestPydanticToDict:
    """Tests for pydantic_to_dict function."""

    def test_pydantic_model(self) -> None:
        """Test conversion of Pydantic model."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        obj = TestModel(name="test", value=42)
        result = pydantic_to_dict(obj)
        assert result == {"name": "test", "value": 42}

    def test_non_pydantic_object(self) -> None:
        """Test that non-Pydantic objects are returned as-is."""
        obj = {"key": "value"}
        assert pydantic_to_dict(obj) == obj


class TestHuggingFaceHubClient:
    """Tests for HuggingFaceHubClient class."""

    @pytest.fixture
    def mock_dataset_provider(self) -> Mock:
        """Create a mock dataset provider."""
        provider = Mock()
        provider.load_dataset.return_value = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        return provider

    @pytest.fixture
    def mock_artifact_storage(self) -> Mock:
        """Create a mock artifact storage."""
        storage = Mock(spec=ArtifactStorage)
        storage.base_dataset_path = Path("/tmp/test")
        storage.processors_outputs_path = Path("/tmp/test/processors")
        storage.metadata_file_path = Path("/tmp/test/metadata.json")
        return storage

    @pytest.fixture
    def mock_artifact_storage_provider(self, mock_artifact_storage: Mock) -> Mock:
        """Create a mock artifact storage provider."""
        provider = Mock()
        provider.artifact_storage = mock_artifact_storage
        return provider

    @pytest.fixture
    def mock_analysis(self) -> Mock:
        """Create a mock analysis."""
        analysis = Mock(spec=DatasetProfilerResults)
        analysis.num_records = 3
        analysis.target_num_records = 3
        analysis.percent_complete = 100.0
        analysis.column_types = []
        analysis.column_statistics = []
        analysis.get_column_statistics_by_type.return_value = []
        return analysis

    @pytest.fixture
    def mock_config_builder(self) -> Mock:
        """Create a mock config builder."""
        builder = Mock(spec=DataDesignerConfigBuilder)
        builder.get_column_configs.return_value = []
        return builder

    @pytest.fixture
    def client(
        self,
        mock_dataset_provider: Mock,
        mock_artifact_storage_provider: Mock,
        mock_analysis: Mock,
        mock_config_builder: Mock,
    ) -> HuggingFaceHubClient:
        """Create a HuggingFaceHubClient instance."""
        return HuggingFaceHubClient(
            dataset_provider=mock_dataset_provider,
            artifact_storage_provider=mock_artifact_storage_provider,
            analysis=mock_analysis,
            config_builder=mock_config_builder,
        )

    def test_init(self, client: HuggingFaceHubClient) -> None:
        """Test client initialization."""
        assert client._dataset_provider is not None
        assert client._artifact_storage_provider is not None
        assert client._analysis is not None
        assert client._config_builder is not None

    @patch("data_designer.integrations.huggingface.client.Dataset")
    @patch("data_designer.integrations.huggingface.client.resolve_hf_token")
    def test_push_to_hub_basic(
        self,
        mock_resolve_token: Mock,
        mock_dataset_class: Mock,
        client: HuggingFaceHubClient,
    ) -> None:
        """Test basic push_to_hub functionality."""
        mock_resolve_token.return_value = "test-token"
        mock_hf_dataset = Mock()
        mock_dataset_class.from_pandas.return_value = mock_hf_dataset

        with patch.object(client, "_upload_additional_artifacts"), patch.object(client, "_upload_dataset_card"):
            client.push_to_hub("test-user/test-dataset", token="test-token", generate_card=False)

        mock_dataset_class.from_pandas.assert_called_once()
        mock_hf_dataset.push_to_hub.assert_called_once_with("test-user/test-dataset", token="test-token")

    @patch("data_designer.integrations.huggingface.client.HfApi")
    def test_upload_analysis(
        self,
        mock_hf_api_class: Mock,
        client: HuggingFaceHubClient,
        mock_analysis: Mock,
    ) -> None:
        """Test uploading analysis results."""
        mock_hf_api = Mock()
        mock_hf_api_class.return_value = mock_hf_api
        mock_analysis.model_dump.return_value = {"num_records": 3}

        client._upload_analysis(mock_hf_api, "test-user/test-dataset")

        mock_hf_api.upload_file.assert_called_once()
        call_args = mock_hf_api.upload_file.call_args
        assert call_args.kwargs["path_in_repo"] == "analysis.json"
        assert call_args.kwargs["repo_id"] == "test-user/test-dataset"

    @patch("data_designer.integrations.huggingface.client.HfApi")
    def test_upload_analysis_none(
        self,
        mock_hf_api_class: Mock,
        mock_dataset_provider: Mock,
        mock_artifact_storage_provider: Mock,
        mock_config_builder: Mock,
    ) -> None:
        """Test uploading analysis when analysis is None."""
        client = HuggingFaceHubClient(
            dataset_provider=mock_dataset_provider,
            artifact_storage_provider=mock_artifact_storage_provider,
            analysis=None,
            config_builder=mock_config_builder,
        )
        mock_hf_api = Mock()
        mock_hf_api_class.return_value = mock_hf_api

        client._upload_analysis(mock_hf_api, "test-user/test-dataset")

        mock_hf_api.upload_file.assert_not_called()

    def test_sanitize_metadata_file_paths_absolute(
        self,
        client: HuggingFaceHubClient,
        mock_artifact_storage: Mock,
    ) -> None:
        """Test sanitizing absolute file paths."""
        mock_artifact_storage.base_dataset_path = Path("/base/path")
        metadata = {
            "file_paths": [
                "/base/path/data/file1.parquet",
                "/base/path/data/file2.parquet",
            ]
        }

        result = client._sanitize_metadata_file_paths(metadata, mock_artifact_storage)

        assert result["file_paths"] == ["data/data/file1.parquet", "data/data/file2.parquet"]

    def test_sanitize_metadata_file_paths_parquet_files(
        self,
        client: HuggingFaceHubClient,
        mock_artifact_storage: Mock,
    ) -> None:
        """Test sanitizing parquet file paths."""
        mock_artifact_storage.base_dataset_path = Path("/base/path")
        metadata = {
            "file_paths": [
                "/some/path/parquet-files/file1.parquet",
            ]
        }

        result = client._sanitize_metadata_file_paths(metadata, mock_artifact_storage)

        assert result["file_paths"] == ["data/parquet-files/file1.parquet"]

    def test_sanitize_metadata_file_paths_no_file_paths(
        self,
        client: HuggingFaceHubClient,
        mock_artifact_storage: Mock,
    ) -> None:
        """Test sanitizing metadata without file_paths."""
        metadata = {"other_key": "value"}

        result = client._sanitize_metadata_file_paths(metadata, mock_artifact_storage)

        assert result == metadata

    def test_build_column_info(self, client: HuggingFaceHubClient) -> None:
        """Test building column information."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        column_names = set(df.columns)

        result = client._build_column_info(df, column_names)

        assert "col1" in result
        assert "col2" in result
        assert isinstance(result["col1"], str)
        assert isinstance(result["col2"], str)

    def test_find_unconfigured_columns(self, client: HuggingFaceHubClient) -> None:
        """Test finding unconfigured columns."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"], "col3": [1.0, 2.0]})
        column_names = {"col1", "col2", "col3"}
        mock_config1 = Mock()
        mock_config1.name = "col1"
        mock_config2 = Mock()
        mock_config2.name = "col2"
        mock_configs = [mock_config1, mock_config2]

        result = client._find_unconfigured_columns(df, column_names, mock_configs)

        assert "col3" in result
        assert "col1" not in result
        assert "col2" not in result
        assert isinstance(result["col3"], str)

    def test_build_sample_records(self, client: HuggingFaceHubClient) -> None:
        """Test building sample records."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        result = client._build_sample_records(df)

        assert len(result) == 3
        assert all(isinstance(r, dict) for r in result)
        assert result[0]["col1"] == 1
        assert result[0]["col2"] == "a"

    def test_build_sample_records_empty(self, client: HuggingFaceHubClient) -> None:
        """Test building sample records from empty dataset."""
        df = pd.DataFrame()

        result = client._build_sample_records(df)

        assert result == []

    def test_build_config_types_summary(self, client: HuggingFaceHubClient) -> None:
        """Test building config types summary."""

        class Config1:
            pass

        class Config2:
            pass

        mock_configs = [
            type("Config1Instance", (Config1,), {})(),
            type("Config1Instance2", (Config1,), {})(),
            type("Config2Instance", (Config2,), {})(),
        ]

        result = client._build_config_types_summary(mock_configs)

        assert result["Config1Instance"] == 1
        assert result["Config1Instance2"] == 1
        assert result["Config2Instance"] == 1

    @patch("data_designer.integrations.huggingface.client.load_dataset")
    @patch("data_designer.integrations.huggingface.client.resolve_hf_token")
    def test_pull_from_hub_basic(
        self,
        mock_resolve_token: Mock,
        mock_load_dataset: Mock,
    ) -> None:
        """Test basic pull_from_hub functionality."""
        mock_resolve_token.return_value = "test-token"
        mock_hf_dataset = Mock()
        mock_hf_dataset.to_pandas.return_value = pd.DataFrame({"col1": [1, 2, 3]})
        mock_load_dataset.return_value = mock_hf_dataset

        with (
            patch.object(HuggingFaceHubClient, "_load_analysis_from_hub", return_value=None),
            patch.object(HuggingFaceHubClient, "_load_processors_from_hub", return_value=(None, None)),
            patch.object(HuggingFaceHubClient, "_load_configs_from_hub", return_value=(None, None, None)),
        ):
            result = HuggingFaceHubClient.pull_from_hub(
                "test-user/test-dataset",
                token="test-token",
                include_analysis=False,
                include_processors=False,
                include_configs=False,
            )

        assert result.dataset is not None
        assert len(result.dataset) == 3

    @patch("data_designer.integrations.huggingface.client.hf_hub_download")
    def test_load_analysis_from_hub_success(
        self,
        mock_hf_hub_download: Mock,
    ) -> None:
        """Test loading analysis from hub successfully."""
        with TemporaryDirectory() as tmpdir:
            analysis_path = Path(tmpdir) / "analysis.json"
            analysis_data = {
                "num_records": 10,
                "target_num_records": 10,
                "column_statistics": [
                    {
                        "column_name": "test_col",
                        "num_records": 10,
                        "num_null": 0,
                        "num_unique": 5,
                        "pyarrow_dtype": "string",
                        "simple_dtype": "string",
                        "column_type": "general",
                    }
                ],
            }
            with open(analysis_path, "w") as f:
                json.dump(analysis_data, f)

            mock_hf_hub_download.return_value = str(analysis_path)

            result = HuggingFaceHubClient._load_analysis_from_hub("test-user/test-dataset", "test-token")

            assert result is not None
            assert result.num_records == 10
            assert len(result.column_statistics) == 1

    @patch("data_designer.integrations.huggingface.client.hf_hub_download")
    def test_load_analysis_from_hub_not_found(
        self,
        mock_hf_hub_download: Mock,
    ) -> None:
        """Test loading analysis when file is not found."""
        from huggingface_hub.utils import HfHubHTTPError

        mock_hf_hub_download.side_effect = HfHubHTTPError("Not found", response=Mock(status_code=404))

        result = HuggingFaceHubClient._load_analysis_from_hub("test-user/test-dataset", "test-token")

        assert result is None

    def test_group_processor_files(self) -> None:
        """Test grouping processor files by processor name."""
        processor_files = [
            "processors/processor1/file1.parquet",
            "processors/processor1/file2.txt",
            "processors/processor2/file3.parquet",
        ]

        result = HuggingFaceHubClient._group_processor_files(processor_files)

        assert "processor1" in result
        assert "processor2" in result
        assert len(result["processor1"]) == 2
        assert len(result["processor2"]) == 1
