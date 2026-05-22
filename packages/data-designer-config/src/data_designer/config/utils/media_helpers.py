# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for multimodal media context values."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import requests

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.type_helpers import StrEnum


class ImageFormat(StrEnum):
    """Supported image formats for image modality."""

    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"


class AudioFormat(StrEnum):
    """Supported audio formats for audio context."""

    MP3 = "mp3"
    WAV = "wav"


class VideoFormat(StrEnum):
    """Supported video formats for video context."""

    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"


_SUPPORTED_IMAGE_EXTENSIONS = tuple(f".{fmt.value.lower()}" for fmt in ImageFormat)
_SUPPORTED_AUDIO_EXTENSIONS = tuple(f".{fmt.value.lower()}" for fmt in AudioFormat)
_SUPPORTED_VIDEO_EXTENSIONS = tuple(f".{fmt.value.lower()}" for fmt in VideoFormat)

_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/=]+$")
_DATA_URI_RE = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<data>.+)$")

_IMAGE_DIFFUSION_MODEL_PATTERNS = (
    "dall-e-",
    "dalle",
    "stable-diffusion",
    "sd-",
    "sd_",
    "imagen",
    "gpt-image-",
)

_IMAGE_FORMAT_MAGIC_BYTES = {
    ImageFormat.PNG: b"\x89PNG\r\n\x1a\n",
    ImageFormat.JPG: b"\xff\xd8\xff",
    ImageFormat.GIF: b"GIF8",
    # WEBP uses RIFF header - handled separately
}

# Maps PIL format name (lowercase) to our ImageFormat enum.
# PIL reports "JPEG" (not "JPG"), so we normalize it here.
_PIL_FORMAT_TO_IMAGE_FORMAT: dict[str, ImageFormat] = {
    "png": ImageFormat.PNG,
    "jpeg": ImageFormat.JPG,
    "jpg": ImageFormat.JPG,
    "gif": ImageFormat.GIF,
    "webp": ImageFormat.WEBP,
}

_AUDIO_FORMAT_TO_MIME_TYPE: dict[AudioFormat, str] = {
    AudioFormat.MP3: "audio/mpeg",
    AudioFormat.WAV: "audio/wav",
}
_VIDEO_FORMAT_TO_MIME_TYPE: dict[VideoFormat, str] = {
    VideoFormat.MP4: "video/mp4",
    VideoFormat.MOV: "video/quicktime",
    VideoFormat.WEBM: "video/webm",
}
_AUDIO_MIME_TYPE_TO_FORMAT: dict[str, AudioFormat] = {
    "audio/mpeg": AudioFormat.MP3,
    "audio/mp3": AudioFormat.MP3,
    "audio/wav": AudioFormat.WAV,
    "audio/wave": AudioFormat.WAV,
    "audio/x-wav": AudioFormat.WAV,
    "audio/vnd.wave": AudioFormat.WAV,
}
_VIDEO_MIME_TYPE_TO_FORMAT: dict[str, VideoFormat] = {
    "video/mp4": VideoFormat.MP4,
    "video/quicktime": VideoFormat.MOV,
    "video/webm": VideoFormat.WEBM,
}


def is_image_diffusion_model(model_name: str) -> bool:
    """Return True if the model is a diffusion-based image generation model."""
    return any(pattern in model_name.lower() for pattern in _IMAGE_DIFFUSION_MODEL_PATTERNS)


def extract_base64_from_data_uri(data: str) -> str:
    """Extract base64 from data URI or return as-is."""
    if data.startswith("data:"):
        if "," in data:
            return data.split(",", 1)[1]
        raise ValueError("Invalid data URI format: missing comma separator")
    return data


def decode_base64_image(base64_data: str) -> bytes:
    """Decode base64 string to image bytes."""
    base64_data = extract_base64_from_data_uri(base64_data)

    try:
        return base64.b64decode(base64_data, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}") from e


def detect_image_format(image_bytes: bytes) -> ImageFormat:
    """Detect image format from bytes."""
    if image_bytes.startswith(_IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.PNG]):
        return ImageFormat.PNG
    elif image_bytes.startswith(_IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.JPG]):
        return ImageFormat.JPG
    elif image_bytes.startswith(_IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.GIF]):
        return ImageFormat.GIF
    elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
        return ImageFormat.WEBP

    try:
        img = lazy.Image.open(io.BytesIO(image_bytes))
        format_str = img.format.lower() if img.format else None
        if format_str in _PIL_FORMAT_TO_IMAGE_FORMAT:
            return _PIL_FORMAT_TO_IMAGE_FORMAT[format_str]
    except Exception:
        pass

    raise ValueError(
        f"Unable to detect image format (first 8 bytes: {image_bytes[:8]!r}). "
        f"Supported formats: {', '.join(_SUPPORTED_IMAGE_EXTENSIONS)}."
    )


def is_image_path(value: str) -> bool:
    """Check if a string is an image file path."""
    if not isinstance(value, str):
        return False
    return any(value.lower().endswith(ext) for ext in _SUPPORTED_IMAGE_EXTENSIONS)


def is_base64_image(value: str) -> bool:
    """Check if a string is base64-encoded image data."""
    if not isinstance(value, str):
        return False
    if value.startswith("data:image/"):
        return True
    if len(value) > 100 and _BASE64_PATTERN.match(value[:100]):
        try:
            base64.b64decode(value[:100])
            return True
        except Exception:
            return False
    return False


def is_image_url(value: str) -> bool:
    """Check if a string is an image URL."""
    if not isinstance(value, str):
        return False
    return value.startswith(("http://", "https://")) and any(
        ext in value.lower() for ext in _SUPPORTED_IMAGE_EXTENSIONS
    )


def load_image_path_to_base64(image_path: str, base_path: str | None = None) -> str | None:
    """Load an image from a file path and return as base64."""
    try:
        path = Path(image_path)

        if not path.is_absolute():
            if base_path:
                path = Path(base_path) / path
            if not path.exists():
                path = Path.cwd() / image_path

        if not path.exists():
            return None

        with open(path, "rb") as f:
            image_bytes = f.read()
            return base64.b64encode(image_bytes).decode()
    except Exception:
        return None


def load_image_url_to_base64(url: str, timeout: int = 60) -> str:
    """Download an image from a URL and return as base64."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode()


async def aload_image_url_to_base64(url: str, timeout: int = 60) -> str:
    """Download an image from a URL asynchronously and return as base64."""
    async with lazy.httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=timeout)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode()


def validate_image(image_path: Path) -> None:
    """Validate that an image file is readable and not corrupted."""
    try:
        with lazy.Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Image validation failed: {e}") from e


def get_media_context(modality: str, source: dict[str, Any]) -> dict[str, Any]:
    """Build a canonical media context block."""
    return {"type": modality, "source": source}


def get_media_url_context(modality: str, url: Any) -> dict[str, Any]:
    """Build a canonical URL media context block."""
    return get_media_context(modality, {"type": "url", "url": url})


def get_media_base64_context(modality: str, media_type: str, data: Any) -> dict[str, Any]:
    """Build a canonical base64 media context block."""
    return get_media_context(modality, {"type": "base64", "media_type": media_type, "data": data})


def normalize_media_context_values(raw_value: Any) -> list[Any]:
    """Normalize scalar, JSON-list, list, and array-like media values."""
    if isinstance(raw_value, str):
        try:
            parsed_value = json.loads(raw_value)
            if isinstance(parsed_value, list):
                return parsed_value
        except (json.JSONDecodeError, TypeError):
            pass
        return [raw_value]

    if isinstance(raw_value, list):
        return raw_value

    if hasattr(raw_value, "__iter__") and not isinstance(raw_value, (str, bytes, dict)):
        return list(raw_value)

    return [raw_value]


def parse_base64_data_uri(value: str) -> tuple[str, str] | None:
    """Return ``(media_type, data)`` for a base64 data URI."""
    if not isinstance(value, str):
        return None
    match = _DATA_URI_RE.match(value)
    if match is None:
        return None
    return match.group("media_type"), match.group("data")


def is_media_url(value: str) -> bool:
    """Return whether a value is an HTTP(S) media URL."""
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def is_audio_path(value: str) -> bool:
    """Return whether a value looks like a local audio path."""
    return _has_path_extension(value, _SUPPORTED_AUDIO_EXTENSIONS)


def is_video_path(value: str) -> bool:
    """Return whether a value looks like a local video path."""
    return _has_path_extension(value, _SUPPORTED_VIDEO_EXTENSIONS)


def audio_mime_type(audio_format: AudioFormat) -> str:
    """Return the MIME type for an audio format."""
    return _AUDIO_FORMAT_TO_MIME_TYPE[audio_format]


def video_mime_type(video_format: VideoFormat) -> str:
    """Return the MIME type for a video format."""
    return _VIDEO_FORMAT_TO_MIME_TYPE[video_format]


def audio_format_from_mime_type(media_type: str) -> AudioFormat | None:
    """Infer an audio format from a MIME type."""
    return _AUDIO_MIME_TYPE_TO_FORMAT.get(media_type.lower())


def video_format_from_mime_type(media_type: str) -> VideoFormat | None:
    """Infer a video format from a MIME type."""
    return _VIDEO_MIME_TYPE_TO_FORMAT.get(media_type.lower())


def _has_path_extension(value: str, supported_extensions: tuple[str, ...]) -> bool:
    if not isinstance(value, str):
        return False
    return not is_media_url(value) and value.lower().endswith(supported_extensions)
