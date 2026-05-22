# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.media_helpers import (
    AudioFormat,
    ImageFormat,
    VideoFormat,
    audio_format_from_mime_type,
    audio_mime_type,
    decode_base64_image,
    detect_image_format,
    extract_base64_from_data_uri,
    get_media_base64_context,
    get_media_context,
    get_media_url_context,
    image_format_from_mime_type,
    is_audio_path,
    is_base64_image,
    is_image_diffusion_model,
    is_image_path,
    is_image_url,
    is_media_url,
    is_video_path,
    load_image_path_to_base64,
    normalize_media_context_values,
    parse_base64_data_uri,
    validate_image,
    video_format_from_mime_type,
    video_mime_type,
)


def test_media_context_builders() -> None:
    assert get_media_context("image", {"type": "url", "url": "https://example.com/image.png"}) == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/image.png"},
    }
    assert get_media_url_context("audio", "https://example.com/audio.mp3") == {
        "type": "audio",
        "source": {"type": "url", "url": "https://example.com/audio.mp3"},
    }
    assert get_media_base64_context("video", "video/mp4", "abc123") == {
        "type": "video",
        "source": {"type": "base64", "media_type": "video/mp4", "data": "abc123"},
    }


def test_normalize_media_context_values() -> None:
    assert normalize_media_context_values("single") == ["single"]
    assert normalize_media_context_values(["one", "two"]) == ["one", "two"]
    assert normalize_media_context_values(json.dumps(["one", "two"])) == ["one", "two"]
    assert normalize_media_context_values(json.dumps({"nested": "value"})) == [json.dumps({"nested": "value"})]
    assert normalize_media_context_values(lazy.np.array(["one", "two"])) == ["one", "two"]


def test_parse_base64_data_uri() -> None:
    assert parse_base64_data_uri("data:audio/mpeg;base64,abc123") == ("audio/mpeg", "abc123")
    assert parse_base64_data_uri("abc123") is None


def test_media_url_detection() -> None:
    assert is_media_url("https://example.com/download?id=123") is True
    assert is_media_url("http://example.com/media") is True
    assert is_media_url("ftp://example.com/media") is False
    assert is_media_url(123) is False  # type: ignore[arg-type]


def test_local_media_path_detection() -> None:
    assert is_audio_path("screen_recording.mp3") is True
    assert is_audio_path("nested/screen_recording.wav") is True
    assert is_audio_path("https://example.com/audio.mp3") is False
    assert is_video_path("screen_recording.mp4") is True
    assert is_video_path("nested/screen_recording.webm") is True
    assert is_video_path("https://example.com/video.mp4") is False


def test_media_format_mime_helpers() -> None:
    assert ImageFormat.PNG.value == "png"
    assert image_format_from_mime_type("image/png") == ImageFormat.PNG
    assert image_format_from_mime_type("image/jpeg") == ImageFormat.JPG
    assert audio_mime_type(AudioFormat.MP3) == "audio/mpeg"
    assert audio_format_from_mime_type("audio/mpeg") == AudioFormat.MP3
    assert audio_format_from_mime_type("audio/mp3") == AudioFormat.MP3
    assert audio_format_from_mime_type("audio/x-wav") == AudioFormat.WAV
    assert video_mime_type(VideoFormat.MP4) == "video/mp4"
    assert video_format_from_mime_type("video/mp4") == VideoFormat.MP4
    assert video_format_from_mime_type("VIDEO/MP4") == VideoFormat.MP4


def test_extract_base64_from_data_uri_with_prefix() -> None:
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANS"
    result = extract_base64_from_data_uri(data_uri)
    assert result == "iVBORw0KGgoAAAANS"


def test_extract_base64_plain_base64_without_prefix() -> None:
    plain_base64 = "iVBORw0KGgoAAAANS"
    result = extract_base64_from_data_uri(plain_base64)
    assert result == plain_base64


def test_extract_base64_invalid_data_uri_raises_error() -> None:
    with pytest.raises(ValueError, match="Invalid data URI format: missing comma separator"):
        extract_base64_from_data_uri("data:image/png;base64")


def test_decode_base64_image_valid() -> None:
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    base64_data = base64.b64encode(png_bytes).decode()
    result = decode_base64_image(base64_data)
    assert result == png_bytes


def test_decode_base64_image_with_data_uri() -> None:
    png_bytes = b"\x89PNG\r\n\x1a\n"
    base64_data = base64.b64encode(png_bytes).decode()
    data_uri = f"data:image/png;base64,{base64_data}"
    result = decode_base64_image(data_uri)
    assert result == png_bytes


def test_decode_base64_image_invalid_raises_error() -> None:
    with pytest.raises(ValueError, match="Invalid base64 data"):
        decode_base64_image("not-valid-base64!!!")


@pytest.mark.parametrize(
    "header_bytes,expected_format",
    [
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 10, ImageFormat.PNG),
        (b"\xff\xd8\xff" + b"\x00" * 10, ImageFormat.JPG),
        (b"RIFF" + b"\x00" * 4 + b"WEBP", ImageFormat.WEBP),
    ],
    ids=["png", "jpg", "webp"],
)
def test_detect_image_format_magic_bytes(header_bytes: bytes, expected_format: ImageFormat) -> None:
    assert detect_image_format(header_bytes) == expected_format


def test_detect_image_format_gif_magic_bytes(tmp_path: Path) -> None:
    img = lazy.Image.new("RGB", (1, 1), color="red")
    gif_path = tmp_path / "test.gif"
    img.save(gif_path, format="GIF")
    gif_bytes = gif_path.read_bytes()
    assert detect_image_format(gif_bytes) == ImageFormat.GIF


def test_detect_image_format_with_pil_fallback_jpeg() -> None:
    mock_img = Mock()
    mock_img.format = "JPEG"
    test_bytes = b"\x00\x00\x00\x00"

    with patch.object(lazy.Image, "open", return_value=mock_img):
        result = detect_image_format(test_bytes)
        assert result == ImageFormat.JPG


def test_detect_image_format_unknown_raises_error() -> None:
    unknown_bytes = b"\x00\x00\x00\x00" + b"\x00" * 10
    with pytest.raises(ValueError, match="Unable to detect image format"):
        detect_image_format(unknown_bytes)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("/path/to/image.png", True),
        ("image.PNG", True),
        ("image.jpg", True),
        ("image.jpeg", True),
        ("/path/to/file.txt", False),
        ("document.pdf", False),
        ("/some.png/file.txt", False),
    ],
    ids=["png", "png-upper", "jpg", "jpeg", "txt", "pdf", "ext-in-dir"],
)
def test_is_image_path(value: str, expected: bool) -> None:
    assert is_image_path(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://example.com/image.png", True),
        ("https://example.com/photo.jpg", True),
        ("https://example.com/image.png?size=large", True),
        ("https://example.com/page.html", False),
        ("ftp://example.com/image.png", False),
    ],
    ids=["http", "https", "query-params", "non-image-ext", "ftp"],
)
def test_is_image_url(value: str, expected: bool) -> None:
    assert is_image_url(value) is expected


def test_is_base64_image_data_uri() -> None:
    assert is_base64_image("data:image/png;base64,iVBORw0KGgo") is True


def test_is_base64_image_long_valid_base64() -> None:
    long_base64 = base64.b64encode(b"x" * 100).decode()
    assert is_base64_image(long_base64) is True


def test_is_base64_image_short_string() -> None:
    assert is_base64_image("short") is False


def test_is_base64_image_invalid_base64_decode() -> None:
    invalid_base64 = "A" * 50 + "=" + "A" * 49 + "more text"
    assert is_base64_image(invalid_base64) is False


@pytest.mark.parametrize(
    "func",
    [is_image_path, is_base64_image, is_image_url],
    ids=["is_image_path", "is_base64_image", "is_image_url"],
)
@pytest.mark.parametrize("value", [123, None, []], ids=["int", "none", "list"])
def test_image_media_helpers_non_string_input_returns_false(func: Callable[..., bool], value: object) -> None:
    assert func(value) is False


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("dall-e-3", True),
        ("DALL-E-2", True),
        ("openai/dalle-2", True),
        ("stable-diffusion-xl", True),
        ("sd-2.1", True),
        ("sd_1.5", True),
        ("imagen-3", True),
        ("google/imagen", True),
        ("gpt-image-1", True),
        ("gemini-3-pro-image-preview", False),
        ("gpt-5-image", False),
        ("flux.2-pro", False),
    ],
    ids=[
        "dall-e-3",
        "DALL-E-2",
        "dalle-2",
        "stable-diffusion-xl",
        "sd-2.1",
        "sd_1.5",
        "imagen-3",
        "google-imagen",
        "gpt-image-1",
        "gemini-not-diffusion",
        "gpt-5-not-diffusion",
        "flux-not-diffusion",
    ],
)
def test_is_image_diffusion_model(model_name: str, expected: bool) -> None:
    assert is_image_diffusion_model(model_name) is expected


def test_validate_image_valid_png(tmp_path: Path, sample_png_bytes: bytes) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(sample_png_bytes)
    validate_image(image_path)


def test_validate_image_corrupted_raises_error(tmp_path: Path) -> None:
    image_path = tmp_path / "corrupted.png"
    image_path.write_bytes(b"not a valid image")
    with pytest.raises(ValueError, match="Image validation failed"):
        validate_image(image_path)


def test_validate_image_nonexistent_raises_error(tmp_path: Path) -> None:
    image_path = tmp_path / "nonexistent.png"
    with pytest.raises(ValueError, match="Image validation failed"):
        validate_image(image_path)


def test_load_image_path_to_base64_absolute_path(tmp_path: Path) -> None:
    img = lazy.Image.new("RGB", (1, 1), color="blue")
    image_path = tmp_path / "test.png"
    img.save(image_path)

    result = load_image_path_to_base64(str(image_path))
    assert result is not None
    assert len(result) > 0
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_load_image_path_to_base64_relative_with_base_path(tmp_path: Path) -> None:
    img = lazy.Image.new("RGB", (1, 1), color="green")
    image_path = tmp_path / "subdir" / "test.png"
    image_path.parent.mkdir(exist_ok=True)
    img.save(image_path)

    result = load_image_path_to_base64("subdir/test.png", base_path=str(tmp_path))
    assert result is not None
    assert len(result) > 0


def test_load_image_path_to_base64_nonexistent_file() -> None:
    result = load_image_path_to_base64("/nonexistent/path/to/image.png")
    assert result is None


def test_load_image_path_to_base64_relative_with_cwd_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    img = lazy.Image.new("RGB", (1, 1), color="yellow")
    image_path = tmp_path / "test_cwd.png"
    img.save(image_path)

    result = load_image_path_to_base64("test_cwd.png")
    assert result is not None
    assert len(result) > 0


def test_load_image_path_to_base64_base_path_fallback_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    img = lazy.Image.new("RGB", (1, 1), color="red")
    image_path = tmp_path / "test.png"
    img.save(image_path)

    wrong_base = tmp_path / "wrong"
    wrong_base.mkdir()

    result = load_image_path_to_base64("test.png", base_path=str(wrong_base))
    assert result is not None
    assert len(result) > 0


def test_load_image_path_to_base64_exception_handling(tmp_path: Path) -> None:
    dir_path = tmp_path / "directory"
    dir_path.mkdir()

    result = load_image_path_to_base64(str(dir_path))
    assert result is None
