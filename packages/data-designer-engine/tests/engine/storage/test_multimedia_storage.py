# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io

# Explicitly import PIL.Image submodule to make it accessible as PIL.Image
# Python doesn't automatically import submodules when you import a package,
# so `import PIL` alone doesn't give you access to PIL.Image
import PIL.Image  # noqa: E402
import pytest

from data_designer.engine.storage.multimedia_storage import IMAGES_SUBDIR, MultimediaStorage
from data_designer.lazy_heavy_imports import PIL


@pytest.fixture
def multimedia_storage(tmp_path):
    """Create a MultimediaStorage instance with a temporary directory."""
    return MultimediaStorage(base_path=tmp_path)


@pytest.fixture
def sample_base64_png() -> str:
    """Create a valid 1x1 PNG as base64."""
    img = PIL.Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    return base64.b64encode(png_bytes).decode()


@pytest.fixture
def sample_base64_jpg() -> str:
    """Create a valid 1x1 JPEG as base64."""
    img = PIL.Image.new("RGB", (1, 1), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpg_bytes = buf.getvalue()
    return base64.b64encode(jpg_bytes).decode()


def test_multimedia_storage_init(tmp_path):
    """Test MultimediaStorage initialization."""
    storage = MultimediaStorage(base_path=tmp_path)
    assert storage.base_path == tmp_path
    assert storage.images_dir == tmp_path / IMAGES_SUBDIR
    assert storage.images_subdir == IMAGES_SUBDIR
    assert storage.validate_images is True
    # Should create images directory on init
    assert storage.images_dir.exists()


def test_multimedia_storage_init_custom_subdir(tmp_path):
    """Test MultimediaStorage initialization with custom subdirectory."""
    custom_subdir = "custom_images"
    storage = MultimediaStorage(base_path=tmp_path, images_subdir=custom_subdir, validate_images=False)
    assert storage.images_subdir == custom_subdir
    assert storage.images_dir == tmp_path / custom_subdir
    assert storage.validate_images is False
    assert storage.images_dir.exists()


def test_save_base64_image_png(multimedia_storage, sample_base64_png):
    """Test saving a PNG image from base64."""
    relative_path = multimedia_storage.save_base64_image(sample_base64_png)

    # Check return value format
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")
    assert relative_path.endswith(".png")

    # Check file exists on disk
    full_path = multimedia_storage.base_path / relative_path
    assert full_path.exists()

    # Verify file content
    saved_bytes = full_path.read_bytes()
    expected_bytes = base64.b64decode(sample_base64_png)
    assert saved_bytes == expected_bytes


def test_save_base64_image_jpg(multimedia_storage, sample_base64_jpg):
    """Test saving a JPEG image from base64."""
    relative_path = multimedia_storage.save_base64_image(sample_base64_jpg)

    # Check return value format
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")
    assert relative_path.endswith(".jpg")

    # Check file exists on disk
    full_path = multimedia_storage.base_path / relative_path
    assert full_path.exists()


def test_save_base64_image_with_data_uri(multimedia_storage, sample_base64_png):
    """Test saving image from data URI format."""
    data_uri = f"data:image/png;base64,{sample_base64_png}"
    relative_path = multimedia_storage.save_base64_image(data_uri)

    # Should successfully extract base64 and save
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")
    assert relative_path.endswith(".png")

    # Verify file exists and content is correct
    full_path = multimedia_storage.base_path / relative_path
    assert full_path.exists()
    saved_bytes = full_path.read_bytes()
    expected_bytes = base64.b64decode(sample_base64_png)
    assert saved_bytes == expected_bytes


def test_save_base64_image_invalid_base64_raises_error(multimedia_storage):
    """Test that invalid base64 data raises ValueError."""
    with pytest.raises(ValueError, match="Invalid base64"):
        multimedia_storage.save_base64_image("not-valid-base64!!!")


def test_save_base64_image_multiple_images_unique_filenames(multimedia_storage, sample_base64_png):
    """Test that multiple images get unique filenames."""
    path1 = multimedia_storage.save_base64_image(sample_base64_png)
    path2 = multimedia_storage.save_base64_image(sample_base64_png)

    # Paths should be different (different UUIDs)
    assert path1 != path2

    # Both files should exist
    assert (multimedia_storage.base_path / path1).exists()
    assert (multimedia_storage.base_path / path2).exists()


def test_save_base64_image_validation_enabled(tmp_path, sample_base64_png):
    """Test that validation is performed when enabled."""
    storage = MultimediaStorage(base_path=tmp_path, validate_images=True)
    # Should succeed with valid image
    relative_path = storage.save_base64_image(sample_base64_png)
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")


def test_save_base64_image_validation_corrupted_image_raises_error(tmp_path):
    """Test that corrupted image fails validation and is cleaned up."""
    storage = MultimediaStorage(base_path=tmp_path, validate_images=True)

    # Create base64 of invalid image data
    corrupted_bytes = b"not a valid image"
    corrupted_base64 = base64.b64encode(corrupted_bytes).decode()

    with pytest.raises(ValueError, match="Image validation failed"):
        storage.save_base64_image(corrupted_base64)

    # Check that no files were left behind
    assert len(list(storage.images_dir.iterdir())) == 0


def test_save_base64_image_validation_disabled(tmp_path):
    """Test that validation can be disabled."""
    storage = MultimediaStorage(base_path=tmp_path, validate_images=False)

    # Create base64 of invalid image data
    corrupted_bytes = b"not a valid image"
    corrupted_base64 = base64.b64encode(corrupted_bytes).decode()

    # Should succeed without validation
    relative_path = storage.save_base64_image(corrupted_base64)
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")

    # File should exist even though it's invalid
    full_path = storage.base_path / relative_path
    assert full_path.exists()


def test_cleanup(multimedia_storage, sample_base64_png):
    """Test cleanup removes images directory."""
    # Save an image first
    multimedia_storage.save_base64_image(sample_base64_png)
    assert multimedia_storage.images_dir.exists()
    assert len(list(multimedia_storage.images_dir.iterdir())) > 0

    # Cleanup should remove directory
    multimedia_storage.cleanup()
    assert not multimedia_storage.images_dir.exists()
