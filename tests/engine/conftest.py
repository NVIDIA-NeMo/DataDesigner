# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pandas as pd
import pytest

from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.resources.managed_storage import ManagedBlobStorage
from data_designer.engine.resources.resource_provider import ResourceProvider


@pytest.fixture
def artifact_storage(tmp_path):
    return ArtifactStorage(artifact_path=tmp_path)


@pytest.fixture
def stub_model_facade():
    mock_facade = Mock(spec=ModelFacade)
    mock_facade.model_alias = "test_model"
    mock_facade.generate.return_value = ("Generated summary text", None)
    return mock_facade


@pytest.fixture
def stub_resource_provider(tmp_path, stub_model_facade):
    mock_provider = Mock(spec=ResourceProvider)
    mock_model_registry = Mock(spec=ModelRegistry)
    mock_model_registry.get_model.return_value = stub_model_facade
    mock_model_registry.model_configs = {}  # Add empty model_configs dict
    mock_provider.model_registry = mock_model_registry
    mock_provider.artifact_storage = ArtifactStorage(artifact_path=tmp_path)
    mock_provider.blob_storage = Mock(spec=ManagedBlobStorage)
    mock_provider.datastore = Mock()
    return mock_provider


@pytest.fixture
def stub_sample_dataframe():
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": ["a", "b", "c", "d"],
            "col3": [True, False, True, False],
            "category": ["A", "B", "A", "B"],
            "other_col": [1, 2, 3, 4],
        }
    )
