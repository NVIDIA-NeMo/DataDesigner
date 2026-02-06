# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import ImageGenerationColumnConfig
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.column_generators.generators.image import ImageCellGenerator
from data_designer.engine.processing.ginja.exceptions import UserTemplateError


@pytest.fixture
def stub_image_column_config():
    return ImageGenerationColumnConfig(
        name="test_image", prompt="A {{ style }} image of {{ subject }}", model_alias="test_model"
    )


@pytest.fixture
def stub_base64_images() -> list[str]:
    return ["base64_image_1", "base64_image_2"]


def test_image_cell_generator_generation_strategy(
    stub_image_column_config: ImageGenerationColumnConfig, stub_resource_provider: None
) -> None:
    generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
    assert generator.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


def test_image_cell_generator_multimedia_storage_property(
    stub_image_column_config: ImageGenerationColumnConfig, stub_resource_provider: None
) -> None:
    generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
    # Should return multimedia_storage from artifact_storage (None by default in stub)
    assert generator.multimedia_storage is None


def test_image_cell_generator_generate_with_storage(
    stub_image_column_config, stub_resource_provider, stub_base64_images
):
    """Test generate with multimedia storage (create mode) - saves to disk."""
    # Setup mock multimedia storage
    mock_storage = Mock()
    mock_storage.save_base64_image.side_effect = ["images/uuid1.png", "images/uuid2.png"]
    stub_resource_provider.artifact_storage.multimedia_storage = mock_storage

    with patch.object(
        stub_resource_provider.model_registry.get_model.return_value,
        "generate_image",
        return_value=stub_base64_images,
    ) as mock_generate:
        generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
        data = generator.generate(data={"style": "photorealistic", "subject": "cat"})

        # Check that column was added with relative paths
        assert stub_image_column_config.name in data
        assert data[stub_image_column_config.name] == ["images/uuid1.png", "images/uuid2.png"]

        # Verify model was called with rendered prompt
        mock_generate.assert_called_once_with(prompt="A photorealistic image of cat")

        # Verify storage was called for each image
        assert mock_storage.save_base64_image.call_count == 2
        mock_storage.save_base64_image.assert_any_call("base64_image_1")
        mock_storage.save_base64_image.assert_any_call("base64_image_2")


def test_image_cell_generator_generate_without_storage(
    stub_image_column_config, stub_resource_provider, stub_base64_images
):
    """Test generate without multimedia storage (preview mode) - stores base64 directly."""
    # Ensure multimedia_storage is None (preview mode)
    stub_resource_provider.artifact_storage.multimedia_storage = None

    with patch.object(
        stub_resource_provider.model_registry.get_model.return_value,
        "generate_image",
        return_value=stub_base64_images,
    ) as mock_generate:
        generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
        data = generator.generate(data={"style": "watercolor", "subject": "dog"})

        # Check that column was added with base64 data
        assert stub_image_column_config.name in data
        assert data[stub_image_column_config.name] == stub_base64_images

        # Verify model was called with rendered prompt
        mock_generate.assert_called_once_with(prompt="A watercolor image of dog")


def test_image_cell_generator_missing_columns_error(stub_image_column_config, stub_resource_provider):
    """Test that missing required columns raises ValueError."""
    generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)

    with pytest.raises(ValueError, match="columns.*missing"):
        # Missing 'subject' column
        generator.generate(data={"style": "photorealistic"})


def test_image_cell_generator_empty_prompt_error(stub_resource_provider):
    """Test that empty rendered prompt raises UserTemplateError."""
    # Create config with template that renders to empty string
    config = ImageGenerationColumnConfig(name="test_image", prompt="{{ empty }}", model_alias="test_model")

    generator = ImageCellGenerator(config=config, resource_provider=stub_resource_provider)

    with pytest.raises(UserTemplateError):
        generator.generate(data={"empty": ""})


def test_image_cell_generator_whitespace_only_prompt_error(stub_resource_provider):
    """Test that whitespace-only rendered prompt raises ValueError."""
    config = ImageGenerationColumnConfig(name="test_image", prompt="{{ spaces }}", model_alias="test_model")

    generator = ImageCellGenerator(config=config, resource_provider=stub_resource_provider)

    with pytest.raises(ValueError, match="empty"):
        generator.generate(data={"spaces": "   "})
