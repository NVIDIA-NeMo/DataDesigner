# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import ImageColumnConfig
from data_designer.config.models import ImageContext, ImageFormat, ModalityDataType
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.column_generators.generators.image import ImageCellGenerator
from data_designer.engine.processing.ginja.exceptions import UserTemplateError


@pytest.fixture
def stub_image_column_config():
    return ImageColumnConfig(name="test_image", prompt="A {{ style }} image of {{ subject }}", model_alias="test_model")


@pytest.fixture
def stub_base64_images() -> list[str]:
    return ["base64_image_1", "base64_image_2"]


def test_image_cell_generator_generation_strategy(
    stub_image_column_config: ImageColumnConfig, stub_resource_provider: None
) -> None:
    generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
    assert generator.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


def test_image_cell_generator_media_storage_property(
    stub_image_column_config: ImageColumnConfig, stub_resource_provider: None
) -> None:
    generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
    # Should return media_storage from artifact_storage (always exists)
    assert generator.media_storage is not None


def test_image_cell_generator_generate_with_storage(
    stub_image_column_config, stub_resource_provider, stub_base64_images
):
    """Test generate with media storage (create mode) - saves to disk."""
    # Setup mock media storage
    mock_storage = Mock()
    mock_storage.save_base64_image.side_effect = [
        "images/test_image/uuid1.png",
        "images/test_image/uuid2.png",
    ]
    stub_resource_provider.artifact_storage.media_storage = mock_storage

    with patch.object(
        stub_resource_provider.model_registry.get_model.return_value,
        "generate_image",
        return_value=stub_base64_images,
    ) as mock_generate:
        generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
        data = generator.generate(data={"style": "photorealistic", "subject": "cat"})

        # Check that column was added with relative paths (organized in subfolder)
        assert stub_image_column_config.name in data
        assert data[stub_image_column_config.name] == [
            "images/test_image/uuid1.png",
            "images/test_image/uuid2.png",
        ]

        # Verify model was called with rendered prompt
        mock_generate.assert_called_once_with(prompt="A photorealistic image of cat", multi_modal_context=None)

        # Verify storage was called for each image with subfolder name
        assert mock_storage.save_base64_image.call_count == 2
        mock_storage.save_base64_image.assert_any_call("base64_image_1", subfolder_name="test_image")
        mock_storage.save_base64_image.assert_any_call("base64_image_2", subfolder_name="test_image")


def test_image_cell_generator_generate_in_dataframe_mode(
    stub_image_column_config, stub_resource_provider, stub_base64_images
):
    """Test generate with media storage in DATAFRAME mode - stores base64 directly."""
    # Mock save_base64_image to return base64 directly (simulating DATAFRAME mode)
    mock_storage = Mock()
    mock_storage.save_base64_image.side_effect = stub_base64_images
    stub_resource_provider.artifact_storage.media_storage = mock_storage

    with patch.object(
        stub_resource_provider.model_registry.get_model.return_value,
        "generate_image",
        return_value=stub_base64_images,
    ) as mock_generate:
        generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)
        data = generator.generate(data={"style": "watercolor", "subject": "dog"})

        # Check that column was added with base64 data (simulating DATAFRAME mode)
        assert stub_image_column_config.name in data
        assert data[stub_image_column_config.name] == stub_base64_images

        # Verify model was called with rendered prompt
        mock_generate.assert_called_once_with(prompt="A watercolor image of dog", multi_modal_context=None)

        # Verify storage was called for each image with subfolder name (even in DATAFRAME mode)
        assert mock_storage.save_base64_image.call_count == 2
        mock_storage.save_base64_image.assert_any_call("base64_image_1", subfolder_name="test_image")
        mock_storage.save_base64_image.assert_any_call("base64_image_2", subfolder_name="test_image")


def test_image_cell_generator_missing_columns_error(stub_image_column_config, stub_resource_provider):
    """Test that missing required columns raises ValueError."""
    generator = ImageCellGenerator(config=stub_image_column_config, resource_provider=stub_resource_provider)

    with pytest.raises(ValueError, match="columns.*missing"):
        # Missing 'subject' column
        generator.generate(data={"style": "photorealistic"})


def test_image_cell_generator_empty_prompt_error(stub_resource_provider):
    """Test that empty rendered prompt raises UserTemplateError."""
    # Create config with template that renders to empty string
    config = ImageColumnConfig(name="test_image", prompt="{{ empty }}", model_alias="test_model")

    generator = ImageCellGenerator(config=config, resource_provider=stub_resource_provider)

    with pytest.raises(UserTemplateError):
        generator.generate(data={"empty": ""})


def test_image_cell_generator_whitespace_only_prompt_error(stub_resource_provider):
    """Test that whitespace-only rendered prompt raises ValueError."""
    config = ImageColumnConfig(name="test_image", prompt="{{ spaces }}", model_alias="test_model")

    generator = ImageCellGenerator(config=config, resource_provider=stub_resource_provider)

    with pytest.raises(ValueError, match="empty"):
        generator.generate(data={"spaces": "   "})


def test_image_cell_generator_with_multi_modal_context(stub_resource_provider):
    """Test generate with multi-modal context for autoregressive models."""
    # Create image context that references a column with URL
    image_context = ImageContext(column_name="reference_image", data_type=ModalityDataType.URL)

    config = ImageColumnConfig(
        name="test_image",
        prompt="Generate a similar image to the reference",
        model_alias="test_model",
        multi_modal_context=[image_context],
    )

    # Setup mock media storage
    mock_storage = Mock()
    mock_storage.save_base64_image.return_value = "images/generated.png"
    stub_resource_provider.artifact_storage.media_storage = mock_storage

    stub_base64_images = ["base64_generated_image"]

    with patch.object(
        stub_resource_provider.model_registry.get_model.return_value,
        "generate_image",
        return_value=stub_base64_images,
    ) as mock_generate:
        generator = ImageCellGenerator(config=config, resource_provider=stub_resource_provider)
        data = generator.generate(data={"reference_image": "https://example.com/image.png"})

        # Check that column was added
        assert config.name in data
        assert data[config.name] == ["images/generated.png"]

        # Verify model was called with prompt and multi_modal_context
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        assert call_args.kwargs["prompt"] == "Generate a similar image to the reference"
        assert call_args.kwargs["multi_modal_context"] is not None
        assert len(call_args.kwargs["multi_modal_context"]) == 1
        assert call_args.kwargs["multi_modal_context"][0]["type"] == "image_url"
        assert call_args.kwargs["multi_modal_context"][0]["image_url"] == "https://example.com/image.png"


def test_image_cell_generator_with_base64_multi_modal_context(stub_resource_provider):
    """Test generate with base64 multi-modal context."""
    # Create image context that references a column with base64 data
    image_context = ImageContext(
        column_name="reference_image", data_type=ModalityDataType.BASE64, image_format=ImageFormat.PNG
    )

    config = ImageColumnConfig(
        name="test_image",
        prompt="Generate a variation of this image",
        model_alias="test_model",
        multi_modal_context=[image_context],
    )

    # Setup mock media storage
    mock_storage = Mock()
    mock_storage.save_base64_image.return_value = "images/generated.png"
    stub_resource_provider.artifact_storage.media_storage = mock_storage

    stub_base64_images = ["base64_generated_image"]

    with patch.object(
        stub_resource_provider.model_registry.get_model.return_value,
        "generate_image",
        return_value=stub_base64_images,
    ) as mock_generate:
        generator = ImageCellGenerator(config=config, resource_provider=stub_resource_provider)
        data = generator.generate(data={"reference_image": "iVBORw0KGgoAAAANS"})

        # Check that column was added
        assert config.name in data
        assert data[config.name] == ["images/generated.png"]

        # Verify model was called with prompt and multi_modal_context
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        assert call_args.kwargs["prompt"] == "Generate a variation of this image"
        assert call_args.kwargs["multi_modal_context"] is not None
        assert len(call_args.kwargs["multi_modal_context"]) == 1
        assert call_args.kwargs["multi_modal_context"][0]["type"] == "image_url"
        # Should be formatted as data URI
        assert "data:image/png;base64," in call_args.kwargs["multi_modal_context"][0]["image_url"]["url"]
