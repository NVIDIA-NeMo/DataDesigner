# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities for working with images."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.models import ImageFormat
from data_designer.lazy_heavy_imports import Image

if TYPE_CHECKING:
    from PIL import Image

# Magic bytes for image format detection
IMAGE_FORMAT_MAGIC_BYTES = {
    ImageFormat.PNG: b"\x89PNG\r\n\x1a\n",
    ImageFormat.JPG: b"\xff\xd8\xff",
    # WEBP uses RIFF header - handled separately
}


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
        Detected format (defaults to PNG if unknown)
    """
    # Check magic bytes first (fast)
    if image_bytes.startswith(IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.PNG]):
        return ImageFormat.PNG
    elif image_bytes.startswith(IMAGE_FORMAT_MAGIC_BYTES[ImageFormat.JPG]):
        return ImageFormat.JPG
    elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
        return ImageFormat.WEBP

    # Fallback to PIL for robust detection
    try:
        img = Image.open(io.BytesIO(image_bytes))
        format_str = img.format.lower() if img.format else None
        if format_str in [ImageFormat.PNG, ImageFormat.JPG, ImageFormat.JPEG, ImageFormat.WEBP]:
            return ImageFormat(format_str if format_str != ImageFormat.JPEG else ImageFormat.JPG)
    except Exception:
        pass

    # Default to PNG
    return ImageFormat.PNG


def is_image_path(value: str) -> bool:
    """Check if a string is an image file path.

    Args:
        value: String to check

    Returns:
        True if the string looks like an image file path, False otherwise
    """
    if not isinstance(value, str):
        return False
    return any(value.lower().endswith(ext) for ext in get_supported_image_extensions())


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
    if len(value) > 100 and all(
        c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in value[:100]
    ):
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
    return value.startswith(("http://", "https://")) and any(
        ext in value.lower() for ext in get_supported_image_extensions()
    )


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


def get_supported_image_extensions() -> list[str]:
    """Get list of supported image extensions from ImageFormat enum.

    Returns:
        List of extensions with leading dot (e.g., [".png", ".jpg", ...])
    """
    return [f".{fmt.value}" for fmt in ImageFormat]
