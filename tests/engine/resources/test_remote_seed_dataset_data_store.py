# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.engine.resources.seed_dataset_data_store import HfHubSeedDatasetDataStore


@pytest.mark.skipif(not os.environ.get("PUBLIC_HF_TOKEN"), reason="PUBLIC_HF_TOKEN environment variable not set")
def test_hf_hub_seed_dataset_data_store_integration_public_huggingface_file():
    hf_store = HfHubSeedDatasetDataStore(endpoint="https://huggingface.co", token=os.environ["PUBLIC_HF_TOKEN"])

    dataset = hf_store.load_dataset("hf://datasets/HuggingFaceFW/fineweb-2/data/aba_Latn/train/000_00000.parquet")

    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    assert len(dataset.columns) > 0


@pytest.mark.skipif(not os.environ.get("PUBLIC_HF_TOKEN"), reason="PUBLIC_HF_TOKEN environment variable not set")
def test_hf_hub_seed_dataset_data_store_integration_public_huggingface_directory():
    hf_store = HfHubSeedDatasetDataStore(endpoint="https://huggingface.co", token=os.environ["PUBLIC_HF_TOKEN"])

    dataset = hf_store.load_dataset("hf://datasets/HuggingFaceFW/fineweb-2/data/aba_Latn")

    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    assert len(dataset.columns) > 0


@pytest.mark.skipif(
    not os.environ.get("NVIDIA_DATASTORE_TOKEN"), reason="NVIDIA_DATASTORE_TOKEN environment variable not set"
)
def test_hf_hub_seed_dataset_data_store_integration_nvidia_datastore_file():
    datastore = HfHubSeedDatasetDataStore(
        endpoint="https://datastore.int.aire.nvidia.com/v1/hf",
        token=os.environ.get("NVIDIA_DATASTORE_TOKEN"),
    )

    dataset = datastore.load_dataset("hf://datasets/anesterenko/tmp-repo-777/train/folder_with_file/000_00000.parquet")

    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    assert len(dataset.columns) > 0


@pytest.mark.skipif(
    not os.environ.get("NVIDIA_DATASTORE_TOKEN"), reason="NVIDIA_DATASTORE_TOKEN environment variable not set"
)
def test_hf_hub_seed_dataset_data_store_integration_nvidia_datastore_directory():
    datastore = HfHubSeedDatasetDataStore(
        endpoint="https://datastore.int.aire.nvidia.com/v1/hf",
        token=os.environ.get("NVIDIA_DATASTORE_TOKEN"),
    )

    dataset = datastore.load_dataset("hf://datasets/anesterenko/tmp-repo-777/train/folder_with_important_files")

    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    assert len(dataset.columns) > 0


def test_hf_hub_seed_dataset_data_store_integration_public_huggingface_no_token():
    hf_store = HfHubSeedDatasetDataStore(endpoint="https://huggingface.co", token=None)

    dataset = hf_store.load_dataset("hf://datasets/HuggingFaceFW/fineweb-2/data/aba_Latn/train/000_00000.parquet")

    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0


def test_hf_hub_seed_dataset_data_store_integration_nvidia_datastore_no_token():
    datastore = HfHubSeedDatasetDataStore(
        endpoint="https://datastore.int.aire.nvidia.com/v1/hf",
        token=None,
    )

    dataset = datastore.load_dataset("hf://datasets/anesterenko/tmp-repo-777/train/folder_with_file/000_00000.parquet")

    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0


def test_hf_hub_seed_dataset_data_store_integration_invalid_dataset_path():
    hf_store = HfHubSeedDatasetDataStore(endpoint="https://huggingface.co", token=None)

    with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
        hf_store.load_dataset("hf://datasets/nonexistent/repo/file.parquet")


def test_hf_hub_seed_dataset_data_store_integration_malformed_file_id():
    hf_store = HfHubSeedDatasetDataStore(endpoint="https://huggingface.co", token=None)

    with pytest.raises(Exception):  # Should raise MalformedFileIdError
        hf_store.load_dataset("hf://datasets/invalid")


@pytest.mark.skipif(not os.environ.get("PUBLIC_HF_TOKEN"), reason="PUBLIC_HF_TOKEN environment variable not set")
def test_hf_hub_seed_dataset_data_store_integration_duckdb_connection():
    hf_store = HfHubSeedDatasetDataStore(endpoint="https://huggingface.co", token=os.environ["PUBLIC_HF_TOKEN"])

    conn = hf_store.create_duckdb_connection()

    result = conn.execute("SELECT 1 as test").fetchone()
    assert result[0] == 1

    conn.close()


@pytest.mark.skipif(not os.environ.get("PUBLIC_HF_TOKEN"), reason="PUBLIC_HF_TOKEN environment variable not set")
def test_hf_hub_seed_dataset_data_store_integration_dataset_uri_generation():
    hf_store = HfHubSeedDatasetDataStore(endpoint="https://huggingface.co", token=os.environ["PUBLIC_HF_TOKEN"])

    file_id = "hf://datasets/HuggingFaceFW/fineweb-2/data/aba_Latn/train/000_00000.parquet"
    uri = hf_store.get_dataset_uri(file_id)

    assert uri == file_id  # Should return the same URI for HF datasets


@pytest.fixture
def stub_hfapi():
    with patch("data_designer.engine.resources.seed_dataset_data_store.HfApi") as mock_api:
        mock_instance = Mock()
        mock_api.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def stub_hffs():
    with patch("data_designer.engine.resources.seed_dataset_data_store.HfFileSystem") as mock_fs:
        mock_instance = Mock()
        mock_fs.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def stub_remote_store(stub_hfapi, stub_hffs):
    return HfHubSeedDatasetDataStore(endpoint="https://test.endpoint", token="test_token")


def test_hf_hub_seed_dataset_data_store_mocked_duckdb_connection_with_filesystem(stub_remote_store, stub_hffs):
    mock_conn = Mock()

    with patch("data_designer.engine.resources.seed_dataset_data_store.duckdb") as mock_duckdb:
        mock_duckdb.connect.return_value = mock_conn

        conn = stub_remote_store.create_duckdb_connection()

        mock_conn.register_filesystem.assert_called_once_with(stub_hffs)
        assert conn == mock_conn


def test_hf_hub_seed_dataset_data_store_mocked_dataset_uri_generation(stub_remote_store):
    file_id = "hf://datasets/test_namespace/test_dataset/test_file.parquet"
    uri = stub_remote_store.get_dataset_uri(file_id)

    assert uri == file_id  # Should return the same URI


def test_hf_hub_seed_dataset_data_store_mocked_load_dataset_file_success(stub_remote_store, stub_hfapi):
    file_id = "hf://datasets/test_namespace/test_dataset/test_file.parquet"
    stub_hfapi.repo_exists.return_value = True
    stub_hfapi.file_exists.return_value = True

    with patch("data_designer.engine.resources.seed_dataset_data_store.tempfile") as mock_tempfile:
        with patch("data_designer.engine.resources.seed_dataset_data_store.load_dataset") as mock_load_dataset:
            mock_tempfile.TemporaryDirectory.return_value.__enter__.return_value = "/tmp/test"

            mock_hf_dataset = Mock()
            mock_hf_dataset.to_pandas.return_value = pd.DataFrame({"a": [1, 2, 3]})
            mock_load_dataset.return_value = mock_hf_dataset

            result = stub_remote_store.load_dataset(file_id)

            stub_hfapi.repo_exists.assert_called_once_with("test_namespace/test_dataset", repo_type="dataset")
            stub_hfapi.file_exists.assert_called_once_with(
                "test_namespace/test_dataset", "test_file.parquet", repo_type="dataset"
            )
            stub_hfapi.hf_hub_download.assert_called_once_with(
                repo_id="test_namespace/test_dataset",
                filename="test_file.parquet",
                local_dir="/tmp/test",
                repo_type="dataset",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
