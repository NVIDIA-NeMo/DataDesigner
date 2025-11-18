# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tempfile
from unittest.mock import Mock

import pandas as pd
import pytest

from data_designer.engine.resources.managed_assets import LocalDatasetManager


@pytest.fixture
def stub_temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def stub_local_dataset_manager(stub_temp_dir):
    return LocalDatasetManager(stub_temp_dir)


@pytest.fixture
def stub_managed_dataset_repository():
    mock_repo = Mock()
    mock_repo.query.return_value = Mock()  # Return a mock DataFrame
    return mock_repo


@pytest.fixture
def stub_sample_dataframe():
    return pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})


@pytest.fixture
def stub_artifact_storage():
    mock_storage = Mock()
    mock_storage.write_parquet_file = Mock()
    mock_storage.write_configs = Mock()
    return mock_storage


@pytest.fixture
def stub_model_registry():
    mock_registry = Mock()
    mock_registry.get_model = Mock()
    mock_registry.get_model_config = Mock()
    return mock_registry


@pytest.fixture
def stub_secret_resolver():
    mock_resolver = Mock()
    mock_resolver.resolve = Mock(return_value="resolved_secret")
    return mock_resolver
