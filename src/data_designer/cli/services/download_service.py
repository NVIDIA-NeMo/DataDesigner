# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

from data_designer.config.utils.constants import LOCALES_WITH_MANAGED_DATASETS, NEMOTRON_PERSONAS_DATASET_PREFIX


class DownloadService:
    """Business logic for downloading assets via NGC CLI."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.managed_assets_dir = config_dir / "managed-assets" / "datasets"

    def get_available_locales(self) -> dict[str, str]:
        """Get dictionary of available persona locales (locale code -> locale code)."""
        return {locale: locale for locale in LOCALES_WITH_MANAGED_DATASETS}

    def download_persona_dataset(self, locale: str) -> Path:
        """Download persona dataset for a specific locale using NGC CLI and move to managed assets.

        Args:
            locale: Locale code (e.g., 'en_US', 'ja_JP')

        Returns:
            Path to the managed assets datasets directory

        Raises:
            ValueError: If locale is invalid
            subprocess.CalledProcessError: If NGC CLI command fails
        """
        if locale not in LOCALES_WITH_MANAGED_DATASETS:
            raise ValueError(f"Invalid locale: {locale}")

        self.managed_assets_dir.mkdir(parents=True, exist_ok=True)

        # Use temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run NGC CLI download command (without version to get latest)
            cmd = [
                "ngc",
                "registry",
                "resource",
                "download-version",
                f"nvidia/nemotron-personas/{_get_downloaded_dataset_name(locale)}",
                "--dest",
                temp_dir,
            ]

            subprocess.run(cmd, check=True)

            dataset_pattern = _get_downloaded_dataset_name(locale)
            download_pattern = f"{temp_dir}/{dataset_pattern}*/*.parquet"
            parquet_files = glob.glob(download_pattern)

            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found matching pattern: {download_pattern}")

            # Move each parquet file to managed assets
            for parquet_file in parquet_files:
                source = Path(parquet_file)
                dest = self.managed_assets_dir / source.name
                shutil.move(str(source), str(dest))

        return self.managed_assets_dir

    def get_managed_assets_directory(self) -> Path:
        """Get the directory where managed datasets are stored."""
        return self.managed_assets_dir

    def is_locale_downloaded(self, locale: str) -> bool:
        """Check if a locale has already been downloaded to managed assets.

        Args:
            locale: Locale code to check

        Returns:
            True if the locale dataset exists in managed assets
        """
        if locale not in LOCALES_WITH_MANAGED_DATASETS:
            return False

        if not self.managed_assets_dir.exists():
            return False

        # Look for any parquet files that start with the dataset pattern
        parquet_files = glob.glob(str(self.managed_assets_dir / f"{locale}.parquet"))

        return len(parquet_files) > 0


def _get_downloaded_dataset_name(locale: str) -> str:
    """Build the downloaded dataset name pattern for the given locale.

    Args:
        locale: Locale code (e.g., 'en_US', 'ja_JP')

    Returns:
        Dataset pattern (e.g., 'nemotron-personas-dataset-en_us')
    """
    return f"{NEMOTRON_PERSONAS_DATASET_PREFIX}{locale.lower()}"
