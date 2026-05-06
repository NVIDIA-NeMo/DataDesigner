# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

import pytest

from data_designer.config.base import ConfigBase, ProcessorConfig, SingleColumnConfig
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.plugins.plugin import Plugin, PluginType

MODULE_NAME = __name__


class ValidColumnGeneratorConfig(SingleColumnConfig):
    name: str = "valid-column-generator"
    column_type: Literal["valid-column-generator"] = "valid-column-generator"

    @property
    def required_columns(self) -> list[str]:
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class ValidColumnGenerator(ConfigurableTask[ValidColumnGeneratorConfig]):
    pass


class ValidSeedReaderConfig(ConfigBase):
    seed_type: Literal["valid-seed-reader"] = "valid-seed-reader"


class ValidSeedReader(SeedReader):
    def get_dataset_uri(self) -> str:
        return "unused"

    def create_duckdb_connection(self) -> Any:
        raise NotImplementedError


class ValidProcessorConfig(ProcessorConfig):
    name: str = "valid-processor"
    processor_type: Literal["valid-processor"] = "valid-processor"


class ValidProcessor(Processor[ValidProcessorConfig]):
    def process_before_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class NonProcessor:
    pass


class TaskButNotProcessor(ConfigurableTask[ValidProcessorConfig]):
    pass


@pytest.mark.parametrize(
    ("plugin_type", "config_class_name", "implementation_class_name"),
    [
        (PluginType.COLUMN_GENERATOR, "ValidColumnGeneratorConfig", "ValidColumnGenerator"),
        (PluginType.SEED_READER, "ValidSeedReaderConfig", "ValidSeedReader"),
        (PluginType.PROCESSOR, "ValidProcessorConfig", "ValidProcessor"),
    ],
)
def test_assert_valid_plugin_accepts_supported_plugin_types(
    plugin_type: PluginType,
    config_class_name: str,
    implementation_class_name: str,
) -> None:
    plugin = Plugin(
        config_qualified_name=f"{MODULE_NAME}.{config_class_name}",
        impl_qualified_name=f"{MODULE_NAME}.{implementation_class_name}",
        plugin_type=plugin_type,
    )

    assert_valid_plugin(plugin)


@pytest.mark.parametrize(
    ("plugin_type", "config_class_name", "expected_message"),
    [
        (
            PluginType.COLUMN_GENERATOR,
            "ValidColumnGeneratorConfig",
            "Column generator plugin impl class must be a subclass of ConfigurableTask",
        ),
        (
            PluginType.SEED_READER,
            "ValidSeedReaderConfig",
            "Seed reader plugin impl class must be a subclass of SeedReader",
        ),
        (
            PluginType.PROCESSOR,
            "ValidProcessorConfig",
            "Processor plugin impl class must be a subclass of Processor",
        ),
    ],
)
def test_assert_valid_plugin_rejects_invalid_impl_for_supported_plugin_types(
    plugin_type: PluginType,
    config_class_name: str,
    expected_message: str,
) -> None:
    plugin = Plugin(
        config_qualified_name=f"{MODULE_NAME}.{config_class_name}",
        impl_qualified_name=f"{MODULE_NAME}.NonProcessor",
        plugin_type=plugin_type,
    )

    with pytest.raises(AssertionError, match=expected_message):
        assert_valid_plugin(plugin)


def test_assert_valid_plugin_rejects_processor_plugin_with_configurable_task_impl() -> None:
    plugin = Plugin(
        config_qualified_name=f"{MODULE_NAME}.ValidProcessorConfig",
        impl_qualified_name=f"{MODULE_NAME}.TaskButNotProcessor",
        plugin_type=PluginType.PROCESSOR,
    )

    with pytest.raises(AssertionError, match="Processor plugin impl class must be a subclass of Processor"):
        assert_valid_plugin(plugin)
