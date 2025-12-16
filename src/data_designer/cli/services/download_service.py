# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path


class DownloadService:
    """Business logic for downloading assets via NGC CLI."""

    # Persona dataset resources (without version - fetches latest)
    PERSONA_DATASETS = {
        "en_US": {
            "resource": "nvidia/nemotron-personas/nemotron-personas-dataset-en_us",
            "dataset_pattern": "nemotron-personas-dataset-en_us",
        },
        "en_IN": {
            "resource": "nvidia/nemotron-personas/nemotron-personas-dataset-en_in",
            "dataset_pattern": "nemotron-personas-dataset-en_in",
        },
        "hi_Deva_IN": {
            "resource": "nvidia/nemotron-personas/nemotron-personas-dataset-hi_deva_in",
            "dataset_pattern": "nemotron-personas-dataset-hi_deva_in",
        },
        "hi_Latn_IN": {
            "resource": "nvidia/nemotron-personas/nemotron-personas-dataset-hi_latn_in",
            "dataset_pattern": "nemotron-personas-dataset-hi_latn_in",
        },
        "ja_JP": {
            "resource": "nvidia/nemotron-personas/nemotron-personas-dataset-ja_jp",
            "dataset_pattern": "nemotron-personas-dataset-ja_jp",
        },
    }

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.managed_assets_dir = config_dir / "managed-assets" / "datasets"

    def check_ngc_cli_available(self) -> bool:
        """Check if NGC CLI is installed and available."""
        return shutil.which("ngc") is not None

    def get_ngc_version(self) -> str | None:
        """Get the NGC CLI version if available."""
        try:
            result = subprocess.run(
                ["ngc", "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def get_available_locales(self) -> dict[str, str]:
        """Get dictionary of available persona locales (locale code -> locale code)."""
        return {locale: locale for locale in self.PERSONA_DATASETS.keys()}

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
        if locale not in self.PERSONA_DATASETS:
            raise ValueError(f"Invalid locale: {locale}")

        dataset_info = self.PERSONA_DATASETS[locale]
        resource = dataset_info["resource"]

        # Ensure managed assets directory exists
        self.managed_assets_dir.mkdir(parents=True, exist_ok=True)

        # Use temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run NGC CLI download command (without version to get latest)
            cmd = [
                "ngc",
                "registry",
                "resource",
                "download-version",
                resource,
                "--dest",
                temp_dir,
            ]

            subprocess.run(cmd, check=True)

            # Find parquet files - NGC creates: <temp_dir>/<dataset-name-lowercase>_v<version>/*.parquet
            # Note: NGC uses lowercase for directory names
            # Use wildcard to match any version suffix
            locale_lower = locale.lower()
            download_pattern = f"{temp_dir}/nemotron-personas-dataset-{locale_lower}*/*.parquet"
            parquet_files = glob.glob(download_pattern)

            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found matching pattern: {download_pattern}")

            # Move each parquet file to managed assets
            for parquet_file in parquet_files:
                source = Path(parquet_file)
                dest = self.managed_assets_dir / source.name
                shutil.move(str(source), str(dest))

            # Temporary directory automatically cleaned up on context exit

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
        if locale not in self.PERSONA_DATASETS:
            return False

        if not self.managed_assets_dir.exists():
            return False

        # Check for parquet files matching this locale in managed assets
        dataset_pattern = self.PERSONA_DATASETS[locale]["dataset_pattern"]
        # Look for any parquet files that start with the dataset pattern
        parquet_files = glob.glob(str(self.managed_assets_dir / f"{dataset_pattern}*.parquet"))

        return len(parquet_files) > 0
