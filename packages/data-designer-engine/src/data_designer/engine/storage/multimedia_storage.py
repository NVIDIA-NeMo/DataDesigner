# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from pathlib import Path

from data_designer.config.utils.image_helpers import decode_base64_image, detect_image_format, validate_image

IMAGES_SUBDIR = "images"


class MultimediaStorage:
    """Manages disk storage of generated multimedia content.

    Currently supports:
    - Images (PNG, JPG, WEBP)

    Future support planned for:
    - Audio
    - Video

    Handles:
    - Creating storage directories
    - Decoding base64 to bytes
    - Detecting media format
    - Saving with UUID filenames
    - Returning relative paths
    """

    def __init__(self, base_path: Path, images_subdir: str = IMAGES_SUBDIR, validate_images: bool = True) -> None:
        """Initialize multimedia storage manager.

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
        image_bytes = decode_base64_image(base64_data)

        # Detect format
        image_format = detect_image_format(image_bytes)

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

    def _validate_image(self, image_path: Path) -> None:
        """Validate that saved image is readable.

        Args:
            image_path: Path to image file

        Raises:
            ValueError: If image is corrupted or unreadable
        """
        try:
            validate_image(image_path)
        except ValueError:
            # Clean up invalid file
            image_path.unlink(missing_ok=True)
            raise

    def cleanup(self) -> None:
        """Clean up image directory (for preview mode)."""
        import shutil

        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
