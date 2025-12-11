# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Type
from unittest.mock import Mock

import pandas as pd
import pytest

from data_designer.config.base import ConfigBase
from data_designer.engine.configurable_task import (
    ConfigurableTask,
    ConfigurableTaskMetadata,
    DataT,
    TaskConfigT,
)
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider, ResourceType


def test_configurable_task_metadata_creation():
    metadata = ConfigurableTaskMetadata(
        name="test_task", description="Test task description", required_resources=[ResourceType.MODEL_REGISTRY]
    )

    assert metadata.name == "test_task"
    assert metadata.description == "Test task description"
    assert metadata.required_resources == [ResourceType.MODEL_REGISTRY]


def test_configurable_task_metadata_with_no_resources():
    metadata = ConfigurableTaskMetadata(name="test_task", description="Test task description", required_resources=None)

    assert metadata.name == "test_task"
    assert metadata.description == "Test task description"
    assert metadata.required_resources is None


def test_configurable_task_generic_type_variables():
    assert DataT.__constraints__ == (dict, pd.DataFrame)

    assert TaskConfigT.__bound__ == ConfigBase


def test_configurable_task_concrete_implementation():
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        @classmethod
        def get_config_type(cls) -> Type[TestConfig]:
            return TestConfig

        @classmethod
        def metadata(cls) -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(name="test_task", description="Test task", required_resources=None)

        def _validate(self) -> None:
            pass

        def _initialize(self) -> None:
            pass

    config = TestConfig(value="test")
    mock_artifact_storage = Mock(spec=ArtifactStorage)
    mock_artifact_storage.dataset_name = "test_dataset"
    mock_artifact_storage.final_dataset_folder_name = "final_dataset"
    mock_artifact_storage.partial_results_folder_name = "partial_results"
    mock_artifact_storage.dropped_columns_folder_name = "dropped_columns"
    mock_artifact_storage.processors_outputs_folder_name = "processors_outputs"
    resource_provider = ResourceProvider(artifact_storage=mock_artifact_storage)

    task = TestTask(config=config, resource_provider=resource_provider)

    assert task._config == config
    assert task._resource_provider == resource_provider


def test_configurable_task_config_validation():
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        @classmethod
        def get_config_type(cls) -> Type[TestConfig]:
            return TestConfig

        @classmethod
        def metadata(cls) -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(name="test_task", description="Test task", required_resources=None)

        def _validate(self) -> None:
            if self._config.value == "invalid":
                raise ValueError("Invalid config")

    config = TestConfig(value="test")
    mock_artifact_storage = Mock(spec=ArtifactStorage)
    mock_artifact_storage.dataset_name = "test_dataset"
    mock_artifact_storage.final_dataset_folder_name = "final_dataset"
    mock_artifact_storage.partial_results_folder_name = "partial_results"
    mock_artifact_storage.dropped_columns_folder_name = "dropped_columns"
    mock_artifact_storage.processors_outputs_folder_name = "processors_outputs"
    resource_provider = ResourceProvider(artifact_storage=mock_artifact_storage)

    task = TestTask(config=config, resource_provider=resource_provider)
    assert task._config.value == "test"

    invalid_config = TestConfig(value="invalid")
    with pytest.raises(ValueError, match="Invalid config"):
        TestTask(config=invalid_config, resource_provider=resource_provider)


def test_configurable_task_resource_validation():
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        @classmethod
        def get_config_type(cls) -> Type[TestConfig]:
            return TestConfig

        @classmethod
        def metadata(cls) -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(
                name="test_task", description="Test task", required_resources=[ResourceType.MODEL_REGISTRY]
            )

        def _validate(self) -> None:
            pass

        def _initialize(self) -> None:
            pass

    config = TestConfig(value="test")

    mock_artifact_storage = Mock(spec=ArtifactStorage)
    mock_artifact_storage.dataset_name = "test_dataset"
    mock_artifact_storage.final_dataset_folder_name = "final_dataset"
    mock_artifact_storage.partial_results_folder_name = "partial_results"
    mock_artifact_storage.dropped_columns_folder_name = "dropped_columns"
    mock_artifact_storage.processors_outputs_folder_name = "processors_outputs"
    mock_model_registry = Mock(spec=ModelRegistry)
    resource_provider = ResourceProvider(artifact_storage=mock_artifact_storage, model_registry=mock_model_registry)
    task = TestTask(config=config, resource_provider=resource_provider)
    assert task._resource_provider == resource_provider


def test_configurable_task_resource_provider_is_none():
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        @classmethod
        def metadata(cls) -> ConfigurableTaskMetadata:
            return ConfigurableTaskMetadata(name="test_task", description="Test task", required_resources=None)

        def _validate(self) -> None:
            pass

        def _initialize(self) -> None:
            pass

    config = TestConfig(value="test")
    task = TestTask(config=config, resource_provider=None)
    assert task._resource_provider is None
