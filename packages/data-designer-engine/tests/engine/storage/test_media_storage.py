# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io

import pytest

from data_designer.engine.storage.media_storage import IMAGES_SUBDIR, MediaStorage, StorageMode
from data_designer.lazy_heavy_imports import Image


@pytest.fixture
def media_storage(tmp_path):
    """Create a MediaStorage instance with a temporary directory."""
    return MediaStorage(base_path=tmp_path)


@pytest.fixture
def sample_base64_png() -> str:
    """Create a valid 1x1 PNG as base64."""
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    return base64.b64encode(png_bytes).decode()


@pytest.fixture
def sample_base64_jpg() -> str:
    """Create a valid 1x1 JPEG as base64."""
    img = Image.new("RGB", (1, 1), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpg_bytes = buf.getvalue()
    return base64.b64encode(jpg_bytes).decode()


def test_media_storage_init(tmp_path):
    """Test MediaStorage initialization."""
    storage = MediaStorage(base_path=tmp_path)
    assert storage.base_path == tmp_path
    assert storage.images_dir == tmp_path / IMAGES_SUBDIR
    assert storage.images_subdir == IMAGES_SUBDIR
    assert storage.mode == StorageMode.DISK
    # Directory should NOT exist until first save (lazy initialization)
    assert not storage.images_dir.exists()


def test_media_storage_init_custom_subdir(tmp_path):
    """Test MediaStorage initialization with custom subdirectory and mode."""
    custom_subdir = "custom_images"
    storage = MediaStorage(base_path=tmp_path, images_subdir=custom_subdir, mode=StorageMode.DATAFRAME)
    assert storage.images_subdir == custom_subdir
    assert storage.images_dir == tmp_path / custom_subdir
    assert storage.mode == StorageMode.DATAFRAME
    # Directory should NOT exist until first save (lazy initialization)
    assert not storage.images_dir.exists()


def test_save_base64_image_png(media_storage, sample_base64_png):
    """Test saving a PNG image from base64."""
    relative_path = media_storage.save_base64_image(sample_base64_png)

    # Check return value format
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")
    assert relative_path.endswith(".png")

    # Check file exists on disk
    full_path = media_storage.base_path / relative_path
    assert full_path.exists()

    # Verify file content
    saved_bytes = full_path.read_bytes()
    expected_bytes = base64.b64decode(sample_base64_png)
    assert saved_bytes == expected_bytes


def test_save_base64_image_jpg(media_storage, sample_base64_jpg):
    """Test saving a JPEG image from base64."""
    relative_path = media_storage.save_base64_image(sample_base64_jpg)

    # Check return value format
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")
    assert relative_path.endswith(".jpg")

    # Check file exists on disk
    full_path = media_storage.base_path / relative_path
    assert full_path.exists()


def test_save_base64_image_with_data_uri(media_storage, sample_base64_png):
    """Test saving image from data URI format."""
    data_uri = f"data:image/png;base64,{sample_base64_png}"
    relative_path = media_storage.save_base64_image(data_uri)

    # Should successfully extract base64 and save
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")
    assert relative_path.endswith(".png")

    # Verify file exists and content is correct
    full_path = media_storage.base_path / relative_path
    assert full_path.exists()
    saved_bytes = full_path.read_bytes()
    expected_bytes = base64.b64decode(sample_base64_png)
    assert saved_bytes == expected_bytes


def test_save_base64_image_invalid_base64_raises_error(media_storage):
    """Test that invalid base64 data raises ValueError."""
    with pytest.raises(ValueError, match="Invalid base64"):
        media_storage.save_base64_image("not-valid-base64!!!")


def test_save_base64_image_multiple_images_unique_filenames(media_storage, sample_base64_png):
    """Test that multiple images get unique filenames."""
    path1 = media_storage.save_base64_image(sample_base64_png)
    path2 = media_storage.save_base64_image(sample_base64_png)

    # Paths should be different (different UUIDs)
    assert path1 != path2

    # Both files should exist
    assert (media_storage.base_path / path1).exists()
    assert (media_storage.base_path / path2).exists()


def test_save_base64_image_disk_mode_validates(tmp_path, sample_base64_png):
    """Test that DISK mode validates images."""
    storage = MediaStorage(base_path=tmp_path, mode=StorageMode.DISK)
    # Should succeed with valid image
    relative_path = storage.save_base64_image(sample_base64_png)
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/")


def test_save_base64_image_disk_mode_corrupted_image_raises_error(tmp_path):
    """Test that DISK mode validates and rejects corrupted images."""
    storage = MediaStorage(base_path=tmp_path, mode=StorageMode.DISK)

    # Create base64 of invalid image data
    corrupted_bytes = b"not a valid image"
    corrupted_base64 = base64.b64encode(corrupted_bytes).decode()

    with pytest.raises(ValueError, match="Image validation failed"):
        storage.save_base64_image(corrupted_base64)

    # Check that no files were left behind (cleanup on validation failure)
    assert len(list(storage.images_dir.iterdir())) == 0


def test_save_base64_image_dataframe_mode_returns_base64(tmp_path, sample_base64_png):
    """Test that DATAFRAME mode returns base64 directly without disk operations."""
    storage = MediaStorage(base_path=tmp_path, mode=StorageMode.DATAFRAME)

    # Should return the same base64 data
    result = storage.save_base64_image(sample_base64_png)
    assert result == sample_base64_png

    # Directory should not be created in DATAFRAME mode (lazy initialization)
    assert not storage.images_dir.exists()


def test_cleanup(media_storage, sample_base64_png):
    """Test cleanup removes images directory."""
    # Save an image first
    media_storage.save_base64_image(sample_base64_png)
    assert media_storage.images_dir.exists()
    assert len(list(media_storage.images_dir.iterdir())) > 0

    # Cleanup should remove directory
    media_storage.cleanup()
    assert not media_storage.images_dir.exists()
