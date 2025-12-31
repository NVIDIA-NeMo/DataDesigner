# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata

MODULE_NAME = __name__


class ValidTestConfig(SingleColumnConfig):
    """Valid config for testing plugin creation."""

    column_type: Literal["test-generator"] = "test-generator"
    name: str


class ValidTestTask(ConfigurableTask[ValidTestConfig]):
    """Valid task for testing plugin creation."""

    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_generator",
            description="Test generator",
            required_resources=None,
        )


class ConfigWithoutDiscriminator(ConfigBase):
    some_field: str


class ConfigWithStringField(ConfigBase):
    column_type: str = "test-generator"


class ConfigWithNonStringDefault(ConfigBase):
    column_type: Literal["test-generator"] = 123  # type: ignore


class ConfigWithInvalidKey(ConfigBase):
    column_type: Literal["invalid-key-!@#"] = "invalid-key-!@#"


class StubPluginConfigA(SingleColumnConfig):
    column_type: Literal["test-plugin-a"] = "test-plugin-a"


class StubPluginConfigB(SingleColumnConfig):
    column_type: Literal["test-plugin-b"] = "test-plugin-b"


class StubPluginTaskA(ConfigurableTask[StubPluginConfigA]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_a",
            description="Test plugin A",
            required_resources=None,
        )


class StubPluginTaskB(ConfigurableTask[StubPluginConfigB]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_b",
            description="Test plugin B",
            required_resources=None,
        )
