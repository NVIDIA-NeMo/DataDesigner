# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
from unittest.mock import Mock, patch

import pytest

from data_designer.config.models import ImageFormat
from data_designer.config.utils.image_helpers import (
    decode_base64_image,
    detect_image_format,
    extract_base64_from_data_uri,
    is_base64_image,
    is_image_diffusion_model,
    is_image_path,
    is_image_url,
    load_image_path_to_base64,
    validate_image,
)
from data_designer.lazy_heavy_imports import Image

# Tests for extract_base64_from_data_uri


def test_extract_base64_from_data_uri_with_prefix():
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANS"
    result = extract_base64_from_data_uri(data_uri)
    assert result == "iVBORw0KGgoAAAANS"


def test_extract_base64_plain_base64_without_prefix():
    plain_base64 = "iVBORw0KGgoAAAANS"
    result = extract_base64_from_data_uri(plain_base64)
    assert result == plain_base64


def test_extract_base64_invalid_data_uri_raises_error():
    with pytest.raises(ValueError, match="Invalid data URI format: missing comma separator"):
        extract_base64_from_data_uri("data:image/png;base64")


# Tests for decode_base64_image


def test_decode_base64_image_valid():
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    base64_data = base64.b64encode(png_bytes).decode()
    result = decode_base64_image(base64_data)
    assert result == png_bytes


def test_decode_base64_image_with_data_uri():
    png_bytes = b"\x89PNG\r\n\x1a\n"
    base64_data = base64.b64encode(png_bytes).decode()
    data_uri = f"data:image/png;base64,{base64_data}"
    result = decode_base64_image(data_uri)
    assert result == png_bytes


def test_decode_base64_image_invalid_raises_error():
    with pytest.raises(ValueError, match="Invalid base64 data"):
        decode_base64_image("not-valid-base64!!!")


# Tests for detect_image_format


def test_detect_image_format_png():
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
    assert detect_image_format(png_magic) == ImageFormat.PNG


def test_detect_image_format_jpg():
    jpg_magic = b"\xff\xd8\xff" + b"\x00" * 10
    assert detect_image_format(jpg_magic) == ImageFormat.JPG


def test_detect_image_format_webp():
    webp_magic = b"RIFF" + b"\x00" * 4 + b"WEBP"
    assert detect_image_format(webp_magic) == ImageFormat.WEBP


def test_detect_image_format_unknown_defaults_to_png():
    unknown_bytes = b"\x00\x00\x00\x00" + b"\x00" * 10
    assert detect_image_format(unknown_bytes) == ImageFormat.PNG


# Tests for is_image_path


def test_is_image_path_various_extensions():
    assert is_image_path("/path/to/image.png") is True
    assert is_image_path("image.PNG") is True
    assert is_image_path("image.jpg") is True
    assert is_image_path("image.jpeg") is True


def test_is_image_path_non_image():
    assert is_image_path("/path/to/file.txt") is False
    assert is_image_path("document.pdf") is False


def test_is_image_path_extension_in_directory():
    assert is_image_path("/some.png/file.txt") is False


# Tests for is_base64_image


def test_is_base64_image_data_uri():
    assert is_base64_image("data:image/png;base64,iVBORw0KGgo") is True


def test_is_base64_image_long_valid_base64():
    long_base64 = base64.b64encode(b"x" * 100).decode()
    assert is_base64_image(long_base64) is True


def test_is_base64_image_short_string():
    assert is_base64_image("short") is False


# Tests for is_image_url


def test_is_image_url_http_and_https():
    assert is_image_url("http://example.com/image.png") is True
    assert is_image_url("https://example.com/photo.jpg") is True


def test_is_image_url_with_query_params():
    assert is_image_url("https://example.com/image.png?size=large") is True


def test_is_image_url_without_image_extension():
    assert is_image_url("https://example.com/page.html") is False


def test_is_image_url_non_http():
    assert is_image_url("ftp://example.com/image.png") is False


# Tests for is_image_diffusion_model


def test_is_image_diffusion_model_dall_e():
    assert is_image_diffusion_model("dall-e-3") is True
    assert is_image_diffusion_model("DALL-E-2") is True
    assert is_image_diffusion_model("openai/dalle-2") is True


def test_is_image_diffusion_model_stable_diffusion():
    assert is_image_diffusion_model("stable-diffusion-xl") is True
    assert is_image_diffusion_model("sd-2.1") is True
    assert is_image_diffusion_model("sd_1.5") is True


def test_is_image_diffusion_model_imagen():
    assert is_image_diffusion_model("imagen-3") is True
    assert is_image_diffusion_model("google/imagen") is True


def test_is_image_diffusion_model_chat_completion_image_models():
    assert is_image_diffusion_model("gemini-3-pro-image-preview") is False
    assert is_image_diffusion_model("gpt-5-image") is False
    assert is_image_diffusion_model("flux.2-pro") is False


# Tests for validate_image


def test_validate_image_valid_png(tmp_path):
    # Create a valid 1x1 PNG using PIL
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    image_path = tmp_path / "test.png"
    image_path.write_bytes(png_bytes)

    # Should not raise
    validate_image(image_path)


def test_validate_image_corrupted_raises_error(tmp_path):
    # Create an invalid image file
    image_path = tmp_path / "corrupted.png"
    image_path.write_bytes(b"not a valid image")

    with pytest.raises(ValueError, match="Image validation failed"):
        validate_image(image_path)


def test_validate_image_nonexistent_raises_error(tmp_path):
    image_path = tmp_path / "nonexistent.png"

    with pytest.raises(ValueError, match="Image validation failed"):
        validate_image(image_path)


# Additional tests for uncovered lines


def test_detect_image_format_with_pil_fallback_unsupported_format(tmp_path):
    # Create a real GIF image that will trigger PIL fallback
    # (GIF has different magic bytes not in our fast-path detection)
    img = Image.new("RGB", (1, 1), color="red")
    gif_path = tmp_path / "test.gif"
    img.save(gif_path, format="GIF")

    gif_bytes = gif_path.read_bytes()
    # Should use PIL fallback and correctly detect GIF format
    result = detect_image_format(gif_bytes)
    assert result == ImageFormat.GIF


def test_detect_image_format_with_pil_fallback_jpeg():
    # Test PIL fallback path that converts "jpeg" format string to JPG enum
    # Use mock since we can't easily create valid JPEG bytes without magic bytes
    mock_img = Mock()
    mock_img.format = "JPEG"

    # Use bytes that don't match our magic bytes to trigger PIL fallback
    test_bytes = b"\x00\x00\x00\x00"

    with patch.object(Image, "open", return_value=mock_img):
        result = detect_image_format(test_bytes)
        # Should convert JPEG -> JPG via line 96
        assert result == ImageFormat.JPG


def test_is_image_path_non_string_input():
    assert is_image_path(123) is False
    assert is_image_path(None) is False
    assert is_image_path([]) is False


def test_is_base64_image_non_string_input():
    assert is_base64_image(123) is False
    assert is_base64_image(None) is False
    assert is_base64_image([]) is False


def test_is_base64_image_invalid_base64_decode():
    # String with valid base64 characters but incorrect padding that causes decode to fail
    # Single '=' in middle of string is invalid base64 (padding only allowed at end)
    invalid_base64 = "A" * 50 + "=" + "A" * 49 + "more text"
    assert is_base64_image(invalid_base64) is False


def test_is_image_url_non_string_input():
    assert is_image_url(123) is False
    assert is_image_url(None) is False
    assert is_image_url([]) is False


# Tests for load_image_path_to_base64


def test_load_image_path_to_base64_absolute_path(tmp_path):
    # Create a test image file
    img = Image.new("RGB", (1, 1), color="blue")
    image_path = tmp_path / "test.png"
    img.save(image_path)

    # Load with absolute path
    result = load_image_path_to_base64(str(image_path))
    assert result is not None
    assert len(result) > 0
    # Verify it's valid base64
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_load_image_path_to_base64_relative_with_base_path(tmp_path):
    # Create a test image file
    img = Image.new("RGB", (1, 1), color="green")
    image_path = tmp_path / "subdir" / "test.png"
    image_path.parent.mkdir(exist_ok=True)
    img.save(image_path)

    # Load with relative path and base_path
    result = load_image_path_to_base64("subdir/test.png", base_path=str(tmp_path))
    assert result is not None
    assert len(result) > 0


def test_load_image_path_to_base64_nonexistent_file():
    result = load_image_path_to_base64("/nonexistent/path/to/image.png")
    assert result is None


def test_load_image_path_to_base64_relative_with_cwd_fallback(tmp_path, monkeypatch):
    # Create test image in current working directory

    # Change to tmp_path as cwd
    monkeypatch.chdir(tmp_path)

    img = Image.new("RGB", (1, 1), color="yellow")
    image_path = tmp_path / "test_cwd.png"
    img.save(image_path)

    # Use relative path without base_path - should fall back to cwd
    result = load_image_path_to_base64("test_cwd.png")
    assert result is not None
    assert len(result) > 0


def test_load_image_path_to_base64_base_path_fallback_to_cwd(tmp_path, monkeypatch):
    # Test the case where base_path is provided but file isn't there, falls back to cwd
    monkeypatch.chdir(tmp_path)

    # Create image in cwd
    img = Image.new("RGB", (1, 1), color="red")
    image_path = tmp_path / "test.png"
    img.save(image_path)

    # Create a different base_path that doesn't have the image
    wrong_base = tmp_path / "wrong"
    wrong_base.mkdir()

    # Use relative path with wrong base_path - should fall back to cwd
    result = load_image_path_to_base64("test.png", base_path=str(wrong_base))
    assert result is not None
    assert len(result) > 0


def test_load_image_path_to_base64_exception_handling(tmp_path):
    # Create a directory (not a file) to trigger exception
    dir_path = tmp_path / "directory"
    dir_path.mkdir()

    result = load_image_path_to_base64(str(dir_path))
    assert result is None
