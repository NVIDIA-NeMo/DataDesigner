# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.config.errors import InvalidFilePathError
from data_designer.engine.resources.seed_dataset_data_store import (
    _HF_DATASETS_PREFIX,
    HfHubSeedDatasetDataStore,
    LocalSeedDatasetDataStore,
)

SEED_DATASET_DATA_STORE_MODULE = "data_designer.engine.resources.seed_dataset_data_store"


@pytest.fixture
def stub_sample_dataframe():
    return pd.DataFrame(data={"a": [1, 2, 3]})


@pytest.fixture
def stub_hfapi():
    with patch(f"{SEED_DATASET_DATA_STORE_MODULE}.HfApi") as mock_api:
        mock_instance = Mock()
        mock_api.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def stub_remote_store(stub_hfapi):
    return HfHubSeedDatasetDataStore(endpoint="https://test.endpoint", token="test_token")


@pytest.fixture
def stub_temp_base_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_local_seed_dataset_data_store_init():
    datastore = LocalSeedDatasetDataStore()
    assert datastore.get_dataset_uri("test.csv") == "test.csv"


@pytest.mark.parametrize(
    "filename,format_func",
    [
        ("test.csv", lambda df, path: df.to_csv(path, index=False)),
        ("test.parquet", lambda df, path: df.to_parquet(path, index=False)),
        ("test.CSV", lambda df, path: df.to_csv(path, index=False)),  # Case insensitive
    ],
)
def test_local_load_dataset_supported_formats(filename, format_func, stub_sample_dataframe, stub_temp_base_dir):
    format_func(stub_sample_dataframe, stub_temp_base_dir / filename)

    datastore = LocalSeedDatasetDataStore()
    dataset = datastore.load_dataset(stub_temp_base_dir / filename)
    pd.testing.assert_frame_equal(stub_sample_dataframe, dataset)


@pytest.mark.parametrize(
    "test_case,filename,expected_error",
    [
        ("unsupported_format", "test.txt", InvalidFilePathError),
        ("file_not_found", "nonexistent.csv", InvalidFilePathError),
    ],
)
def test_local_load_dataset_error_cases(test_case, filename, expected_error, stub_temp_base_dir):
    datastore = LocalSeedDatasetDataStore()

    if test_case == "unsupported_format":
        with open(stub_temp_base_dir / filename, "w") as f:
            f.write("This is not a supported format")

    with pytest.raises(expected_error):
        datastore.load_dataset(filename)


def test_hfhub_seed_dataset_data_store_init(stub_hfapi):
    store = HfHubSeedDatasetDataStore(endpoint="https://custom.endpoint", token="custom_token")
    assert store.hfapi == stub_hfapi


@pytest.mark.parametrize(
    "error_type,repo_exists_return,expected_error",
    [
        (FileNotFoundError, False, "Repo test_namespace/test_dataset does not exist"),
        (
            FileNotFoundError,
            lambda repo_id, repo_type: repo_type == "model",
            "Repo test_namespace/test_dataset is a model repo, not a dataset repo",
        ),
        (FileNotFoundError, True, "File file.parquet does not exist in repo test_namespace/test_dataset"),
    ],
)
def test_load_dataset_errors(stub_remote_store, stub_hfapi, error_type, repo_exists_return, expected_error):
    file_id = f"{_HF_DATASETS_PREFIX}test_namespace/test_dataset/file.parquet"

    if callable(repo_exists_return):
        stub_hfapi.repo_exists.side_effect = repo_exists_return
    else:
        stub_hfapi.repo_exists.return_value = repo_exists_return

    if repo_exists_return is True:
        stub_hfapi.file_exists.return_value = False

    with pytest.raises(error_type, match=expected_error):
        stub_remote_store.load_dataset(file_id)


@patch(f"{SEED_DATASET_DATA_STORE_MODULE}.load_dataset", autospec=True)
@patch(f"{SEED_DATASET_DATA_STORE_MODULE}.tempfile", autospec=True)
def test_load_dataset_file_success(
    mock_tempfile, mock_load_dataset, stub_remote_store, stub_hfapi, stub_sample_dataframe
):
    file_id = f"{_HF_DATASETS_PREFIX}test_namespace/test_dataset/file.parquet"
    stub_hfapi.repo_exists.return_value = True
    stub_hfapi.file_exists.return_value = True

    mock_temp_dir = "/tmp/test_dir"
    mock_tempfile.TemporaryDirectory.return_value.__enter__.return_value = mock_temp_dir

    mock_hf_dataset = Mock()
    mock_hf_dataset.to_pandas.return_value = stub_sample_dataframe
    mock_load_dataset.return_value = mock_hf_dataset

    result = stub_remote_store.load_dataset(file_id)

    stub_hfapi.file_exists.assert_called_once_with("test_namespace/test_dataset", "file.parquet", repo_type="dataset")
    stub_hfapi.hf_hub_download.assert_called_once_with(
        repo_id="test_namespace/test_dataset", filename="file.parquet", local_dir=mock_temp_dir, repo_type="dataset"
    )

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, stub_sample_dataframe)


@patch(f"{SEED_DATASET_DATA_STORE_MODULE}.load_dataset", autospec=True)
@patch(f"{SEED_DATASET_DATA_STORE_MODULE}.tempfile", autospec=True)
@patch(f"{SEED_DATASET_DATA_STORE_MODULE}.os.path.exists", autospec=True)
@pytest.mark.parametrize("dir_exists", [True, False])
def test_load_dataset_directory_success(
    mock_exists, mock_tempfile, mock_load_dataset, stub_remote_store, stub_hfapi, dir_exists, stub_sample_dataframe
):
    dir_id = f"{_HF_DATASETS_PREFIX}test_namespace/test_dataset/directory"
    stub_hfapi.repo_exists.return_value = True
    mock_temp_dir = "/tmp/test_dir"
    mock_tempfile.TemporaryDirectory.return_value.__enter__.return_value = mock_temp_dir
    mock_exists.return_value = dir_exists

    mock_hf_dataset = Mock()
    mock_hf_dataset.to_pandas.return_value = stub_sample_dataframe
    mock_load_dataset.return_value = mock_hf_dataset

    result = stub_remote_store.load_dataset(dir_id)

    stub_hfapi.snapshot_download.assert_called_once_with(
        repo_id="test_namespace/test_dataset", local_dir=mock_temp_dir, repo_type="dataset"
    )

    expected_path = f"{mock_temp_dir}/directory" if dir_exists else mock_temp_dir
    mock_load_dataset.assert_called_once_with(path=expected_path)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, stub_sample_dataframe)
