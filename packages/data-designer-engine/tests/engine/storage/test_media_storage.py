# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.storage.media_storage import IMAGES_SUBDIR, MediaStorage, StorageMode


@pytest.fixture
def media_storage(tmp_path):
    """Create a MediaStorage instance with a temporary directory."""
    return MediaStorage(base_path=tmp_path)


@pytest.fixture
def sample_base64_png() -> str:
    """Create a valid 1x1 PNG as base64."""
    img = lazy.Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    return base64.b64encode(png_bytes).decode()


@pytest.fixture
def sample_base64_jpg() -> str:
    """Create a valid 1x1 JPEG as base64."""
    img = lazy.Image.new("RGB", (1, 1), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpg_bytes = buf.getvalue()
    return base64.b64encode(jpg_bytes).decode()


@pytest.mark.parametrize(
    "images_subdir,mode",
    [
        (IMAGES_SUBDIR, StorageMode.DISK),
        ("custom_images", StorageMode.DATAFRAME),
    ],
    ids=["defaults", "custom-subdir-dataframe"],
)
def test_media_storage_init(tmp_path, images_subdir: str, mode: StorageMode) -> None:
    """Test MediaStorage initialization with various configurations."""
    storage = MediaStorage(base_path=tmp_path, images_subdir=images_subdir, mode=mode)
    assert storage.base_path == tmp_path
    assert storage.images_subdir == images_subdir
    assert storage.images_dir == tmp_path / images_subdir
    assert storage.mode == mode
    # Directory should NOT exist until first save (lazy initialization)
    assert not storage.images_dir.exists()


@pytest.mark.parametrize(
    "image_fixture,expected_extension",
    [
        ("sample_base64_png", ".png"),
        ("sample_base64_jpg", ".jpg"),
    ],
)
def test_save_base64_image_format(media_storage, image_fixture, expected_extension, request):
    """Test saving images from base64 in different formats."""
    # Get the actual fixture value using request.getfixturevalue
    sample_base64 = request.getfixturevalue(image_fixture)

    relative_path = media_storage.save_base64_image(sample_base64, subfolder_name="test_column")

    # Check return value format (organized by column name)
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/test_column/")
    assert relative_path.endswith(expected_extension)

    # Check file exists on disk
    full_path = media_storage.base_path / relative_path
    assert full_path.exists()

    # Verify file content
    saved_bytes = full_path.read_bytes()
    expected_bytes = base64.b64decode(sample_base64)
    assert saved_bytes == expected_bytes


def test_save_base64_image_with_data_uri(media_storage, sample_base64_png):
    """Test saving image from data URI format."""
    data_uri = f"data:image/png;base64,{sample_base64_png}"
    relative_path = media_storage.save_base64_image(data_uri, subfolder_name="test_column")

    # Should successfully extract base64 and save (organized by column name)
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/test_column/")
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
        media_storage.save_base64_image("not-valid-base64!!!", subfolder_name="test_column")


def test_save_base64_image_multiple_images_unique_filenames(media_storage, sample_base64_png):
    """Test that multiple images get unique filenames."""
    path1 = media_storage.save_base64_image(sample_base64_png, subfolder_name="test_column")
    path2 = media_storage.save_base64_image(sample_base64_png, subfolder_name="test_column")

    # Paths should be different (different UUIDs)
    assert path1 != path2

    # Both files should exist
    assert (media_storage.base_path / path1).exists()
    assert (media_storage.base_path / path2).exists()


def test_save_base64_image_disk_mode_validates(tmp_path, sample_base64_png):
    """Test that DISK mode validates images."""
    storage = MediaStorage(base_path=tmp_path, mode=StorageMode.DISK)
    # Should succeed with valid image
    relative_path = storage.save_base64_image(sample_base64_png, subfolder_name="test_column")
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/test_column/")


def test_save_base64_image_disk_mode_corrupted_image_raises_error(tmp_path):
    """Test that DISK mode validates and rejects corrupted images."""
    storage = MediaStorage(base_path=tmp_path, mode=StorageMode.DISK)

    # Create base64 of invalid image data
    corrupted_bytes = b"not a valid image"
    corrupted_base64 = base64.b64encode(corrupted_bytes).decode()

    with pytest.raises(ValueError, match="Unable to detect image format"):
        storage.save_base64_image(corrupted_base64, subfolder_name="test_column")

    # Check that no files were left behind (cleanup on validation failure)
    column_dir = storage.images_dir / "test_column"
    if column_dir.exists():
        assert len(list(column_dir.iterdir())) == 0


@pytest.mark.parametrize("subfolder_name", ["test_column", "test_subfolder"], ids=["column", "subfolder"])
def test_save_base64_image_dataframe_mode_returns_base64(tmp_path, sample_base64_png, subfolder_name):
    """Test that DATAFRAME mode returns base64 directly regardless of subfolder name."""
    storage = MediaStorage(base_path=tmp_path, mode=StorageMode.DATAFRAME)

    result = storage.save_base64_image(sample_base64_png, subfolder_name=subfolder_name)
    assert result == sample_base64_png

    # Directory should not be created in DATAFRAME mode (lazy initialization)
    assert not storage.images_dir.exists()


def test_save_base64_image_with_subfolder_name(media_storage, sample_base64_png):
    """Test saving image with subfolder name organizes into subdirectory."""
    subfolder = "test_subfolder"
    relative_path = media_storage.save_base64_image(sample_base64_png, subfolder_name=subfolder)

    # Check return value format includes subfolder
    assert relative_path.startswith(f"{IMAGES_SUBDIR}/{subfolder}/")
    assert relative_path.endswith(".png")

    # Check file exists in correct subdirectory
    full_path = media_storage.base_path / relative_path
    assert full_path.exists()
    assert full_path.parent.name == subfolder

    # Verify file content
    saved_bytes = full_path.read_bytes()
    expected_bytes = base64.b64decode(sample_base64_png)
    assert saved_bytes == expected_bytes


def test_save_base64_image_with_different_subfolder_names(media_storage, sample_base64_png, sample_base64_jpg):
    """Test that images with different subfolder names are stored in separate subdirectories."""
    path1 = media_storage.save_base64_image(sample_base64_png, subfolder_name="subfolder_a")
    path2 = media_storage.save_base64_image(sample_base64_jpg, subfolder_name="subfolder_b")

    # Check paths are in different subdirectories
    assert "subfolder_a" in path1
    assert "subfolder_b" in path2

    # Check both directories exist
    subfolder_a_dir = media_storage.images_dir / "subfolder_a"
    subfolder_b_dir = media_storage.images_dir / "subfolder_b"
    assert subfolder_a_dir.exists()
    assert subfolder_b_dir.exists()

    # Check files exist in their respective directories
    assert (media_storage.base_path / path1).exists()
    assert (media_storage.base_path / path2).exists()


@pytest.mark.parametrize(
    "unsafe_name,expected_sanitized",
    [
        ("../evil", "__evil"),  # Parent directory traversal: .. -> _, / -> _
        ("foo/bar", "foo_bar"),  # Path separator (forward slash)
        ("foo\\bar", "foo_bar"),  # Path separator (backslash)
        ("test..name", "test_name"),  # Double dots in middle: .. -> _
    ],
)
def test_save_base64_image_sanitizes_subfolder_name(media_storage, sample_base64_png, unsafe_name, expected_sanitized):
    """Test that subfolder names are sanitized to prevent path traversal."""
    relative_path = media_storage.save_base64_image(sample_base64_png, subfolder_name=unsafe_name)

    # Check that path contains sanitized subfolder name
    assert expected_sanitized in relative_path
    assert "/" not in expected_sanitized  # No path separators
    assert "\\" not in expected_sanitized  # No backslashes
    assert ".." not in expected_sanitized  # No parent references

    # Verify file is inside images directory (not escaped via path traversal)
    full_path = media_storage.base_path / relative_path
    assert full_path.exists()
    assert media_storage.images_dir in full_path.parents


# ---------------------------------------------------------------------------
# delete_image
# ---------------------------------------------------------------------------


def test_delete_image_removes_saved_file(media_storage, sample_base64_png) -> None:
    """Test that delete_image removes a previously saved image."""
    relative_path = media_storage.save_base64_image(sample_base64_png, subfolder_name="col")
    full_path = media_storage.base_path / relative_path
    assert full_path.exists()

    result = media_storage.delete_image(relative_path)
    assert result is True
    assert not full_path.exists()


def test_delete_image_returns_false_for_nonexistent(media_storage) -> None:
    """Test that delete_image returns False when the file doesn't exist."""
    assert media_storage.delete_image(f"{IMAGES_SUBDIR}/col/nonexistent.png") is False


def test_delete_image_rejects_path_outside_images_dir(media_storage, tmp_path) -> None:
    """Test that delete_image refuses to delete files outside the images directory."""
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("should not be deleted")

    result = media_storage.delete_image("../outside.txt")
    assert result is False
    assert outside_file.exists()
