# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities for working with images."""

from __future__ import annotations

import base64
from pathlib import Path

from data_designer.config.models import ImageFormat


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


def get_supported_image_extensions() -> list[str]:
    """Get list of supported image extensions from ImageFormat enum.

    Returns:
        List of extensions with leading dot (e.g., [".png", ".jpg", ...])
    """
    return [f".{fmt.value}" for fmt in ImageFormat]
