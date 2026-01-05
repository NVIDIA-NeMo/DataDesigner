# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.engine.column_generators.generators.samplers import SamplerColumnGenerator
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.plugins.errors import PluginLoadError
from data_designer.plugins.plugin import Plugin, PluginType
from data_designer.plugins.testing.stubs import MODULE_NAME, ValidTestConfig, ValidTestTask
from data_designer.plugins.testing.utils import assert_valid_plugin


@pytest.fixture
def valid_plugin() -> Plugin:
    """Fixture providing a valid plugin instance for testing."""
    return Plugin(
        task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
        config_qualified_name=f"{MODULE_NAME}.ValidTestConfig",
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


# =============================================================================
# Plugin Validation Tests
# =============================================================================


def test_validation_fails_when_config_missing_discriminator_field() -> None:
    """Test validation fails when config lacks the required discriminator field."""

    with pytest.raises(ValueError, match="Discriminator field 'column_type' not found in config class"):
        Plugin(
            task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
            config_qualified_name=f"{MODULE_NAME}.ConfigWithoutDiscriminator",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_validation_fails_when_discriminator_field_not_literal_type() -> None:
    """Test validation fails when discriminator field is not a Literal type."""
    with pytest.raises(ValueError, match="Field 'column_type' of .* must be a Literal type"):
        Plugin(
            task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
            config_qualified_name=f"{MODULE_NAME}.ConfigWithStringField",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_validation_fails_when_discriminator_default_not_string() -> None:
    """Test validation fails when discriminator field default is not a string."""
    with pytest.raises(ValueError, match="The default of 'column_type' must be a string"):
        Plugin(
            task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
            config_qualified_name=f"{MODULE_NAME}.ConfigWithNonStringDefault",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_validation_fails_with_invalid_enum_key_conversion() -> None:
    """Test validation fails when default value cannot be converted to valid Python identifier."""
    with pytest.raises(ValueError, match="cannot be converted to a valid enum key"):
        Plugin(
            task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
            config_qualified_name=f"{MODULE_NAME}.ConfigWithInvalidKey",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_validation_fails_with_invalid_modules() -> None:
    """Test validation fails when task or config class modules are invalid."""
    with pytest.raises(PluginLoadError, match="Could not find module"):
        Plugin(
            task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
            config_qualified_name="invalid.module.ValidTestConfig",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )

    with pytest.raises(PluginLoadError, match="Could not find module"):
        Plugin(
            task_qualified_name="invalid.module.ValidTestTask",
            config_qualified_name=f"{MODULE_NAME}.ValidTestConfig",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )

    with pytest.raises(PluginLoadError, match="Expected a fully-qualified object name"):
        Plugin(
            task_qualified_name="ValidTestTask",
            config_qualified_name="ValidTestConfig",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )

    with pytest.raises(PluginLoadError, match="Could not find class"):
        Plugin(
            task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
            config_qualified_name=f"{MODULE_NAME}.NotADefinedClass",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )

    with pytest.raises(PluginLoadError, match="Could not find class"):
        Plugin(
            task_qualified_name=f"{MODULE_NAME}.NotADefinedClass",
            config_qualified_name=f"{MODULE_NAME}.ValidTestConfig",
            plugin_type=PluginType.COLUMN_GENERATOR,
        )


def test_helper_utility_identifies_invalid_classes() -> None:
    """Test the helper utility provides deeper validation of config classes."""
    valid_plugin = Plugin(
        task_qualified_name=f"{MODULE_NAME}.ValidTestTask",
        config_qualified_name=f"{MODULE_NAME}.ValidTestConfig",
        plugin_type=PluginType.COLUMN_GENERATOR,
    )
    assert_valid_plugin(valid_plugin)

    plugin_with_improper_task_class_type = Plugin(
        task_qualified_name=f"{MODULE_NAME}.ValidTestConfig",
        config_qualified_name=f"{MODULE_NAME}.ValidTestConfig",
        plugin_type=PluginType.COLUMN_GENERATOR,
    )
    with pytest.raises(AssertionError):
        assert_valid_plugin(plugin_with_improper_task_class_type)


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_works_with_real_sampler_column_generator() -> None:
    """Test that Plugin works with actual SamplerColumnGenerator from the codebase."""
    plugin = Plugin(
        task_qualified_name="data_designer.engine.column_generators.generators.samplers.SamplerColumnGenerator",
        config_qualified_name="data_designer.config.column_configs.SamplerColumnConfig",
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
