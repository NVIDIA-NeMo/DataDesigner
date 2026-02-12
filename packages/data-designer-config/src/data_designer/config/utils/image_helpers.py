# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities for working with images."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from data_designer.config.models import ImageFormat
from data_designer.lazy_heavy_imports import Image

if TYPE_CHECKING:
    from PIL import Image

# Magic bytes for image format detection
IMAGE_FORMAT_MAGIC_BYTES = {
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

_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/=]+$")

# Patterns for diffusion-based image models only (use image_generation API).
IMAGE_DIFFUSION_MODEL_PATTERNS = (
    "dall-e-",
    "dalle",
    "stable-diffusion",
    "sd-",
    "sd_",
    "imagen",
    "gpt-image-",
)

SUPPORTED_IMAGE_EXTENSIONS = [f".{fmt.value.lower()}" for fmt in ImageFormat]


def is_image_diffusion_model(model_name: str) -> bool:
    """Return True if the model is a diffusion-based image generation model.

    Args:
        model_name: Model name or identifier (e.g. from provider).

    Returns:
        True if the model is detected as diffusion-based, False otherwise.
    """
    return any(pattern in model_name.lower() for pattern in IMAGE_DIFFUSION_MODEL_PATTERNS)


def extract_base64_from_data_uri(data: str) -> str:
    """Extract base64 from data URI or return as-is.

    Handles data URIs like "data:image/png;base64,iVBORw0..." and returns
    just the base64 portion.

    Args:
        data: Data URI (e.g., "data:image/png;base64,XXX") or plain base64

    Returns:
        Base64 string without data URI prefix

    Raises:
        ValueError: If data URI format is invalid
    """
    if data.startswith("data:"):
        if "," in data:
            return data.split(",", 1)[1]
        raise ValueError("Invalid data URI format: missing comma separator")
    return data


def decode_base64_image(base64_data: str) -> bytes:
    """Decode base64 string to image bytes.

    Automatically handles data URIs by extracting the base64 portion first.

    Args:
        base64_data: Base64 string (with or without data URI prefix)

    Returns:
        Decoded image bytes

    Raises:
        ValueError: If base64 data is invalid
    """
    # Remove data URI prefix if present
    base64_data = extract_base64_from_data_uri(base64_data)

    try:
        return base64.b64decode(base64_data, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}") from e


def detect_image_format(image_bytes: bytes) -> ImageFormat:
    """Detect image format from bytes.

    Uses magic bytes for fast detection, falls back to PIL for robust detection.

    Args:
        image_bytes: Image data as bytes

    Returns:
        Detected ImageFormat

    Raises:
        ValueError: If the image format cannot be determined
    """
    # Check magic bytes first (fast)
    if image_bytes.startswith(IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.PNG]):
        return ImageFormat.PNG
    elif image_bytes.startswith(IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.JPG]):
        return ImageFormat.JPG
    elif image_bytes.startswith(IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.GIF]):
        return ImageFormat.GIF
    elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
        return ImageFormat.WEBP

    # Fallback to PIL for robust detection
    try:
        img = Image.open(io.BytesIO(image_bytes))
        format_str = img.format.lower() if img.format else None
        if format_str in _PIL_FORMAT_TO_IMAGE_FORMAT:
            return _PIL_FORMAT_TO_IMAGE_FORMAT[format_str]
    except Exception:
        pass

    raise ValueError(
        f"Unable to detect image format (first 8 bytes: {image_bytes[:8]!r}). "
        f"Supported formats: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}."
    )


def is_image_path(value: str) -> bool:
    """Check if a string is an image file path.

    Args:
        value: String to check

    Returns:
        True if the string looks like an image file path, False otherwise
    """
    if not isinstance(value, str):
        return False
    return any(value.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)


def is_base64_image(value: str) -> bool:
    """Check if a string is base64-encoded image data.

    Args:
        value: String to check

    Returns:
        True if the string looks like base64-encoded image data, False otherwise
    """
    if not isinstance(value, str):
        return False
    # Check if it starts with data URI scheme
    if value.startswith("data:image/"):
        return True
    # Check if it looks like base64 (at least 100 chars, contains only base64 chars)
    if len(value) > 100 and _BASE64_PATTERN.match(value[:100]):
        try:
            # Try to decode a small portion to verify it's valid base64
            base64.b64decode(value[:100])
            return True
        except Exception:
            return False
    return False


def is_image_url(value: str) -> bool:
    """Check if a string is an image URL.

    Args:
        value: String to check

    Returns:
        True if the string looks like an image URL, False otherwise
    """
    if not isinstance(value, str):
        return False
    return value.startswith(("http://", "https://")) and any(ext in value.lower() for ext in SUPPORTED_IMAGE_EXTENSIONS)


def load_image_path_to_base64(image_path: str, base_path: str | None = None) -> str | None:
    """Load an image from a file path and return as base64.

    Args:
        image_path: Relative or absolute path to the image file.
        base_path: Optional base path to resolve relative paths from.

    Returns:
        Base64-encoded image data or None if loading fails.
    """
    try:
        path = Path(image_path)

        # If path is not absolute, try to resolve it
        if not path.is_absolute():
            if base_path:
                path = Path(base_path) / path
            # If still not found, try current working directory
            if not path.exists():
                path = Path.cwd() / image_path

        # Check if file exists
        if not path.exists():
            return None

        # Read image file and convert to base64
        with open(path, "rb") as f:
            image_bytes = f.read()
            return base64.b64encode(image_bytes).decode()
    except Exception:
        return None


def load_image_url_to_base64(url: str, timeout: int = 60) -> str:
    """Download an image from a URL and return as base64.

    Args:
        url: HTTP(S) URL pointing to an image.
        timeout: Request timeout in seconds.

    Returns:
        Base64-encoded image data.

    Raises:
        requests.HTTPError: If the download fails with a non-2xx status.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode()


def validate_image(image_path: Path) -> None:
    """Validate that an image file is readable and not corrupted.

    Args:
        image_path: Path to image file

    Raises:
        ValueError: If image is corrupted or unreadable
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Image validation failed: {e}") from e
