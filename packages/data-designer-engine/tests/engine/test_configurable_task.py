# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from data_designer.config.base import ConfigBase
from data_designer.engine.configurable_task import ConfigurableTask, DataT, TaskConfigT
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


def test_configurable_task_generic_type_variables() -> None:
    assert DataT.__constraints__ == (dict, pd.DataFrame)

    assert TaskConfigT.__bound__ == ConfigBase


def test_configurable_task_concrete_implementation(tmp_path) -> None:
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        @classmethod
        def get_config_type(cls) -> type[TestConfig]:
            return TestConfig

        def _validate(self) -> None:
            pass

        def _initialize(self) -> None:
            pass

    config = TestConfig(value="test")
    artifact_storage = ArtifactStorage(artifact_path=tmp_path)
    resource_provider = ResourceProvider(artifact_storage=artifact_storage)

    task = TestTask(config=config, resource_provider=resource_provider)

    assert task._config == config
    assert task._resource_provider == resource_provider


def test_configurable_task_config_validation(tmp_path) -> None:
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        @classmethod
        def get_config_type(cls) -> type[TestConfig]:
            return TestConfig

        def _validate(self) -> None:
            if self._config.value == "invalid":
                raise ValueError("Invalid config")

    config = TestConfig(value="test")
    artifact_storage = ArtifactStorage(artifact_path=tmp_path)
    resource_provider = ResourceProvider(artifact_storage=artifact_storage)

    task = TestTask(config=config, resource_provider=resource_provider)
    assert task._config.value == "test"

    invalid_config = TestConfig(value="invalid")
    with pytest.raises(ValueError, match="Invalid config"):
        TestTask(config=invalid_config, resource_provider=resource_provider)


def test_configurable_task_resource_validation(tmp_path) -> None:
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        @classmethod
        def get_config_type(cls) -> type[TestConfig]:
            return TestConfig

        def _validate(self) -> None:
            pass

        def _initialize(self) -> None:
            pass

    config = TestConfig(value="test")

    artifact_storage = ArtifactStorage(artifact_path=tmp_path)
    mock_model_registry = Mock(spec=ModelRegistry)
    resource_provider = ResourceProvider(artifact_storage=artifact_storage, model_registry=mock_model_registry)
    task = TestTask(config=config, resource_provider=resource_provider)
    assert task._resource_provider == resource_provider


def test_configurable_task_resource_provider_is_none() -> None:
    class TestConfig(ConfigBase):
        value: str

    class TestTask(ConfigurableTask[TestConfig]):
        def _validate(self) -> None:
            pass

        def _initialize(self) -> None:
            pass

    config = TestConfig(value="test")
    task = TestTask(config=config, resource_provider=None)
    assert task._resource_provider is None
