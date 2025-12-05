# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest
from pydantic import ValidationError

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SamplerColumnConfig, SingleColumnConfig
from data_designer.engine.column_generators.generators.samplers import SamplerColumnGenerator
from data_designer.engine.configurable_task import ConfigurableTask, ConfigurableTaskMetadata
from data_designer.plugins.plugin import Plugin, PluginType


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


@pytest.fixture
def valid_plugin() -> Plugin:
    """Fixture providing a valid plugin instance for testing."""
    return Plugin(
        task_cls=ValidTestTask,
        config_cls=ValidTestConfig,
        plugin_type=PluginType.COLUMN_GENERATOR,
    )


# =============================================================================
# PluginType Tests
# =============================================================================


def test_plugin_type_discriminator_field_for_column_generator() -> None:
    """Test that COLUMN_GENERATOR type returns 'column_type' as discriminator field."""
    assert PluginType.COLUMN_GENERATOR.discriminator_field == "column_type"


def test_plugin_type_all_types_have_discriminator_fields() -> None:
    """Test that all plugin types have valid discriminator fields."""
    for plugin_type in PluginType:
        assert isinstance(plugin_type.discriminator_field, str)
        assert len(plugin_type.discriminator_field) > 0


# =============================================================================
# Plugin Creation and Properties Tests
# =============================================================================


def test_create_plugin_with_valid_inputs(valid_plugin: Plugin) -> None:
    """Test that Plugin can be created with valid task, config, and plugin type."""
    assert valid_plugin.task_cls == ValidTestTask
    assert valid_plugin.config_cls == ValidTestConfig
    assert valid_plugin.plugin_type == PluginType.COLUMN_GENERATOR


def test_plugin_name_derived_from_config_default(valid_plugin: Plugin) -> None:
    """Test that plugin.name returns the discriminator field's default value."""
    assert valid_plugin.name == "test-generator"


def test_plugin_discriminator_field_from_type(valid_plugin: Plugin) -> None:
    """Test that plugin.discriminator_field returns the correct field name."""
    assert valid_plugin.discriminator_field == "column_type"


def test_plugin_requires_all_fields() -> None:
    """Test that Plugin creation fails without required fields."""
    with pytest.raises(ValidationError):
        Plugin()  # type: ignore

    with pytest.raises(ValidationError):
        Plugin(task_cls=ValidTestTask)  # type: ignore

    with pytest.raises(ValidationError):
        Plugin(config_cls=ValidTestConfig)  # type: ignore


# =============================================================================
# Plugin Validation Tests
# =============================================================================


def test_validation_fails_when_config_missing_discriminator_field() -> None:
    """Test validation fails when config lacks the required discriminator field."""

    class ConfigWithoutDiscriminator(ConfigBase):
        some_field: str

    with pytest.raises(ValueError, match="Discriminator field 'column_type' not found in config class"):
        Plugin(
            task_cls=ValidTestTask,
            config_cls=ConfigWithoutDiscriminator,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_validation_fails_when_discriminator_field_not_literal_type() -> None:
    """Test validation fails when discriminator field is not a Literal type."""

    class ConfigWithStringField(ConfigBase):
        column_type: str = "test-generator"

    with pytest.raises(ValueError, match="Field 'column_type' of .* must be a Literal type"):
        Plugin(
            task_cls=ValidTestTask,
            config_cls=ConfigWithStringField,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_validation_fails_when_discriminator_default_not_string() -> None:
    """Test validation fails when discriminator field default is not a string."""

    class ConfigWithNonStringDefault(ConfigBase):
        column_type: Literal["test-generator"] = 123  # type: ignore

    with pytest.raises(ValueError, match="The default of 'column_type' must be a string"):
        Plugin(
            task_cls=ValidTestTask,
            config_cls=ConfigWithNonStringDefault,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_validation_fails_with_invalid_enum_key_conversion() -> None:
    """Test validation fails when default value cannot be converted to valid Python identifier."""

    class ConfigWithInvalidKey(ConfigBase):
        column_type: Literal["invalid-key-!@#"] = "invalid-key-!@#"

    with pytest.raises(ValueError, match="cannot be converted to a valid enum key"):
        Plugin(
            task_cls=ValidTestTask,
            config_cls=ConfigWithInvalidKey,
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_works_with_real_sampler_column_generator() -> None:
    """Test that Plugin works with actual SamplerColumnGenerator from the codebase."""
    plugin = Plugin(
        task_cls=SamplerColumnGenerator,
        config_cls=SamplerColumnConfig,
        plugin_type=PluginType.COLUMN_GENERATOR,
    )

    assert plugin.name == "sampler"
    assert plugin.discriminator_field == "column_type"
    assert plugin.task_cls == SamplerColumnGenerator
    assert plugin.config_cls == SamplerColumnConfig


def test_plugin_preserves_type_information(valid_plugin: Plugin) -> None:
    """Test that Plugin correctly stores and provides access to type information."""
    assert isinstance(valid_plugin.task_cls, type)
    assert isinstance(valid_plugin.config_cls, type)
    assert issubclass(valid_plugin.task_cls, ConfigurableTask)
    assert issubclass(valid_plugin.config_cls, ConfigBase)
