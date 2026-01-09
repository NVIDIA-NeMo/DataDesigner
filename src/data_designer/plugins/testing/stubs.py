# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata
from data_designer.plugins.plugin import Plugin, PluginType

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
        )


class StubPluginTaskB(ConfigurableTask[StubPluginConfigB]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_b",
            description="Test plugin B",
        )


# Stub plugins requiring different combinations of resources


class StubPluginConfigModels(SingleColumnConfig):
    column_type: Literal["test-plugin-models"] = "test-plugin-models"


class StubPluginConfigModelsAndBlobs(SingleColumnConfig):
    column_type: Literal["test-plugin-models-and-blobs"] = "test-plugin-models-and-blobs"


class StubPluginConfigBlobsAndSeeds(SingleColumnConfig):
    column_type: Literal["test-plugin-blobs-and-seeds"] = "test-plugin-blobs-and-seeds"


class StubPluginTaskModels(ConfigurableTask[StubPluginConfigModels]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_models",
            description="Test plugin requiring models",
        )


class StubPluginTaskModelsAndBlobs(ConfigurableTask[StubPluginConfigModelsAndBlobs]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_models_and_blobs",
            description="Test plugin requiring models and blobs",
        )


class StubPluginTaskBlobsAndSeeds(ConfigurableTask[StubPluginConfigBlobsAndSeeds]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="test_plugin_blobs_and_seeds",
            description="Test plugin requiring blobs and seeds",
        )


plugin_none = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigA",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskA",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugin_models = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigModels",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskModels",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugin_models_and_blobs = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigModelsAndBlobs",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskModelsAndBlobs",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

plugin_blobs_and_seeds = Plugin(
    config_qualified_name=f"{MODULE_NAME}.StubPluginConfigBlobsAndSeeds",
    impl_qualified_name=f"{MODULE_NAME}.StubPluginTaskBlobsAndSeeds",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
