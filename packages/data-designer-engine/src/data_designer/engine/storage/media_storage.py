# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from pathlib import Path

from data_designer.config.utils.image_helpers import decode_base64_image, detect_image_format, validate_image
from data_designer.config.utils.type_helpers import StrEnum

IMAGES_SUBDIR = "images"


class StorageMode(StrEnum):
    """Storage mode for generated media content.

    - DISK: Save media to disk and store relative paths in dataframe (for dataset creation)
    - DATAFRAME: Store base64 data directly in dataframe (for preview mode)
    """

    DISK = "disk"
    DATAFRAME = "dataframe"


class MediaStorage:
    """Manages storage of generated media content.

    Currently supports:
    - Images (PNG, JPG, WEBP)

    Storage modes:
    - DISK: Save media to disk and return relative paths (for dataset creation)
    - DATAFRAME: Return base64 data directly (for preview mode)

    Handles:
    - Creating storage directories
    - Decoding base64 to bytes
    - Detecting media format
    - Saving with UUID filenames (DISK mode)
    - Returning relative paths or base64 data based on mode
    - Always validates images to ensure data quality
    """

    def __init__(
        self, base_path: Path, images_subdir: str = IMAGES_SUBDIR, mode: StorageMode = StorageMode.DISK
    ) -> None:
        """Initialize media storage manager.

        Args:
            base_path: Base directory for dataset
            images_subdir: Subdirectory name for images (default: "images")
            mode: Storage mode - DISK (save to disk) or DATAFRAME (return base64)
        """
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / images_subdir
        self.images_subdir = images_subdir
        self.mode = mode

    def _ensure_images_directory(self) -> None:
        """Create images directory if it doesn't exist (lazy initialization)."""
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_base64_image(self, base64_data: str, subfolder_name: str) -> str:
        """Save or return base64 image based on storage mode.

        Args:
            base64_data: Base64 encoded image string (with or without data URI prefix)
            subfolder_name: Subfolder name to organize images (e.g., "images/<subfolder_name>/")

        Returns:
            DISK mode: Relative path to saved image (e.g., "images/subfolder_name/f47ac10b-58cc.png")
            DATAFRAME mode: Original base64 data string

        Raises:
            ValueError: If base64 data is invalid (DISK mode only)
            OSError: If disk write fails (DISK mode only)
        """
        # DATAFRAME mode: return base64 directly without disk operations
        if self.mode == StorageMode.DATAFRAME:
            return base64_data

        # DISK mode: save to disk, validate, and return relative path
        # Determine the target directory (organized by subfolder)
        target_dir = self.images_dir / subfolder_name

        # Ensure target directory exists (lazy initialization)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Decode base64 to bytes
        image_bytes = decode_base64_image(base64_data)

        # Detect format
        image_format = detect_image_format(image_bytes)

        # Generate unique filename
        image_id = uuid.uuid4()
        filename = f"{image_id}.{image_format.value}"
        full_path = target_dir / filename

        # Build relative path
        relative_path = f"{self.images_subdir}/{subfolder_name}/{filename}"

        # Write to disk
        with open(full_path, "wb") as f:
            f.write(image_bytes)

        # Always validate in DISK mode to ensure data quality
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
