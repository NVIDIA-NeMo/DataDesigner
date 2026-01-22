# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SingleColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.plugins.plugin import Plugin, PluginType

MODULE_NAME = __name__


class StubHuggingFaceSeedReader(SeedReader):
    """Stub seed reader for testing."""

    def get_column_names(self) -> list[str]:
        return ["age", "city"]

    def get_dataset_uri(self) -> str:
        return "unused in these tests"

    def create_duckdb_connection(self):
        pass

    def get_seed_type(self) -> str:
        return "hf"


class ValidTestConfig(SingleColumnConfig):
    """Valid config for testing plugin creation."""

    column_type: Literal["test-generator"] = "test-generator"
    name: str


class ValidTestTask(ColumnGeneratorCellByCell[ValidTestConfig]):
    """Valid task for testing plugin creation."""

    def generate(self, data: dict) -> dict:
        return data


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


class StubPluginTaskA(ColumnGeneratorCellByCell[StubPluginConfigA]):
    def generate(self, data: dict) -> dict:
        return data


class StubPluginTaskB(ColumnGeneratorCellByCell[StubPluginConfigB]):
    def generate(self, data: dict) -> dict:
        return data


# Stub plugins requiring different combinations of resources


class StubPluginConfigModels(SingleColumnConfig):
    column_type: Literal["test-plugin-models"] = "test-plugin-models"


class StubPluginConfigModelsAndBlobs(SingleColumnConfig):
    column_type: Literal["test-plugin-models-and-blobs"] = "test-plugin-models-and-blobs"


class StubPluginConfigBlobsAndSeeds(SingleColumnConfig):
    column_type: Literal["test-plugin-blobs-and-seeds"] = "test-plugin-blobs-and-seeds"


class StubPluginTaskModels(ColumnGeneratorCellByCell[StubPluginConfigModels]):
    def generate(self, data: dict) -> dict:
        return data


class StubPluginTaskModelsAndBlobs(ColumnGeneratorCellByCell[StubPluginConfigModelsAndBlobs]):
    def generate(self, data: dict) -> dict:
        return data


class StubPluginTaskBlobsAndSeeds(ColumnGeneratorCellByCell[StubPluginConfigBlobsAndSeeds]):
    def generate(self, data: dict) -> dict:
        return data


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
