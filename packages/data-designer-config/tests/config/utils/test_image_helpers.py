# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

import pytest

from data_designer.config.models import ImageFormat
from data_designer.config.utils.image_helpers import (
    decode_base64_image,
    detect_image_format,
    extract_base64_from_data_uri,
    get_supported_image_extensions,
    is_base64_image,
    is_image_path,
    is_image_url,
)

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


# Tests for get_supported_image_extensions


def test_get_supported_image_extensions_matches_enum():
    result = get_supported_image_extensions()
    enum_values = [f".{fmt.value}" for fmt in ImageFormat]
    assert set(result) == set(enum_values)
