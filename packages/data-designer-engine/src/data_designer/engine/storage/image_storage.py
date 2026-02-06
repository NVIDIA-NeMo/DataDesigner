# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import uuid
from enum import Enum
from pathlib import Path


class ImageFormat(str, Enum):
    """Supported image formats."""

    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"


class ImageStorageManager:
    """Manages disk storage of generated images.

    Handles:
    - Creating images directory
    - Decoding base64 to bytes
    - Detecting image format
    - Saving with UUID filenames
    - Returning relative paths
    """

    def __init__(self, base_path: Path, images_subdir: str = "images", validate_images: bool = True) -> None:
        """Initialize image storage manager.

        Args:
            base_path: Base directory for dataset
            images_subdir: Subdirectory name for images (default: "images")
            validate_images: Whether to validate images after saving (default: True)
        """
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / images_subdir
        self.images_subdir = images_subdir
        self.validate_images = validate_images
        self._ensure_images_directory()

    def _ensure_images_directory(self) -> None:
        """Create images directory if it doesn't exist."""
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_base64_image(self, base64_data: str) -> str:
        """Save base64 image to disk and return relative path.

        Args:
            base64_data: Base64 encoded image string (with or without data URI prefix)

        Returns:
            Relative path to saved image (e.g., "images/f47ac10b-58cc.png")

        Raises:
            ValueError: If base64 data is invalid
            OSError: If disk write fails
        """
        # Decode base64 to bytes
        image_bytes = self._decode_base64(base64_data)

        # Detect format
        image_format = self._detect_format(image_bytes)

        # Generate unique filename
        image_id = uuid.uuid4()
        filename = f"{image_id}.{image_format.value}"
        full_path = self.images_dir / filename
        relative_path = f"{self.images_subdir}/{filename}"

        # Write to disk
        with open(full_path, "wb") as f:
            f.write(image_bytes)

        # Optional validation
        if self.validate_images:
            self._validate_image(full_path)

        return relative_path

    def _decode_base64(self, base64_data: str) -> bytes:
        """Decode base64 string to bytes.

        Args:
            base64_data: Base64 string (with or without data URI prefix)

        Returns:
            Decoded bytes

        Raises:
            ValueError: If base64 data is invalid
        """
        # Remove data URI prefix if present (e.g., "data:image/png;base64,")
        if base64_data.startswith("data:"):
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            else:
                raise ValueError("Invalid data URI format: missing comma separator")

        try:
            return base64.b64decode(base64_data, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}") from e

    def _detect_format(self, image_bytes: bytes) -> ImageFormat:
        """Detect image format from bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Detected format (defaults to PNG if unknown)
        """
        # Check magic bytes first (fast)
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ImageFormat.PNG
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            return ImageFormat.JPG
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
            return ImageFormat.WEBP

        # Fallback to PIL for robust detection
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            format_str = img.format.lower() if img.format else None
            if format_str in ["png", "jpeg", "jpg", "webp"]:
                return ImageFormat(format_str if format_str != "jpeg" else "jpg")
        except Exception:
            pass

        # Default to PNG
        return ImageFormat.PNG

    def _validate_image(self, image_path: Path) -> None:
        """Validate that saved image is readable.

        Args:
            image_path: Path to image file

        Raises:
            ValueError: If image is corrupted or unreadable
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            # Clean up invalid file
            image_path.unlink(missing_ok=True)
            raise ValueError(f"Saved image is invalid or corrupted: {e}") from e

    def cleanup(self) -> None:
        """Clean up image directory (for preview mode)."""
        import shutil

        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
