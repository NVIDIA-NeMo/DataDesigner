# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import webbrowser
from pathlib import Path

from data_designer.cli.services.download_service import DownloadService
from data_designer.cli.ui import (
    confirm_action,
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    print_text,
    print_warning,
    select_multiple_with_arrows,
)


class DownloadController:
    """Controller for asset download workflows."""

    NGC_CLI_INSTALL_URL = "https://org.ngc.nvidia.com/setup/installers/cli"

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.service = DownloadService(config_dir)

    def run_personas(self, locales: list[str] | None, all_locales: bool, dry_run: bool = False) -> None:
        """Main entry point for persona dataset downloads.

        Args:
            locales: List of locale codes to download (if provided via CLI flags)
            all_locales: If True, download all available locales
            dry_run: If True, only show what would be downloaded without actually downloading
        """
        header = "Download Nemotron-Persona Datasets (Dry Run)" if dry_run else "Download Nemotron-Persona Datasets"
        print_header(header)
        print_info(f"Datasets will be saved to: {self.service.get_managed_assets_directory()}")
        console.print()

        # Check NGC CLI availability (skip in dry run mode)
        if not dry_run and not self._check_ngc_cli():
            return

        # Determine which locales to download
        selected_locales = self._determine_locales(locales, all_locales)

        if not selected_locales:
            print_info("No locales selected")
            return

        # Show what will be downloaded
        console.print()
        action = "Would download" if dry_run else "Will download"
        print_text(f"📦 {action} {len(selected_locales)} Nemotron-Persona dataset(s):")
        for locale in selected_locales:
            already_downloaded = self.service.is_locale_downloaded(locale)
            status = " (already exists, will update)" if already_downloaded else ""
            print_text(f"  • {locale}{status}")

        console.print()

        # In dry run mode, exit here
        if dry_run:
            print_info("Dry run complete - no files were downloaded")
            return

        # Confirm download
        if not confirm_action("Proceed with download?", default=True):
            print_info("Download cancelled")
            return

        # Download each locale
        console.print()
        successful = []
        failed = []

        for locale in selected_locales:
            if self._download_locale(locale):
                successful.append(locale)
            else:
                failed.append(locale)

        # Summary
        console.print()
        if successful:
            print_success(f"Successfully downloaded {len(successful)} dataset(s): {', '.join(successful)}")
            print_info(f"Location: {self.service.get_managed_assets_directory()}")

        if failed:
            print_error(f"Failed to download {len(failed)} dataset(s): {', '.join(failed)}")

    def _check_ngc_cli(self) -> bool:
        """Check if NGC CLI is installed and guide user if not."""
        if self.service.check_ngc_cli_available():
            version = self.service.get_ngc_version()
            if version:
                print_info(f"NGC CLI: {version}")
            return True

        # NGC CLI not found
        print_error("NGC CLI not found!")
        console.print()
        print_text("The NGC CLI is required to download persona datasets.")
        print_text(f"Installation instructions: {self.NGC_CLI_INSTALL_URL}")
        console.print()

        if confirm_action("Open installation page in browser?", default=True):
            try:
                webbrowser.open(self.NGC_CLI_INSTALL_URL)
                print_success("Opened installation page in browser")
            except Exception as e:
                print_warning(f"Could not open browser: {e}")

        return False

    def _determine_locales(self, locales: list[str] | None, all_locales: bool) -> list[str]:
        """Determine which locales to download based on user input.

        Args:
            locales: List of locales from CLI flags (may be None)
            all_locales: Whether to download all locales

        Returns:
            List of locale codes to download
        """
        available_locales = self.service.get_available_locales()

        # If --all flag is set, return all locales
        if all_locales:
            return list(available_locales.keys())

        # If locales specified via flags, validate and return them
        if locales:
            invalid_locales = [loc for loc in locales if loc not in available_locales]
            if invalid_locales:
                print_error(f"Invalid locale(s): {', '.join(invalid_locales)}")
                print_info(f"Available locales: {', '.join(available_locales.keys())}")
                return []
            return locales

        # Interactive multi-select
        return self._select_locales_interactive(available_locales)

    def _select_locales_interactive(self, available_locales: dict[str, str]) -> list[str]:
        """Interactive multi-select for locales.

        Args:
            available_locales: Dictionary of {locale_code: description}

        Returns:
            List of selected locale codes
        """
        console.print()
        print_text("Select locales you want to download:")
        console.print()

        selected = select_multiple_with_arrows(
            options=available_locales,
            prompt_text="Use ↑/↓ to navigate, Space to toggle ✓, Enter to confirm:",
            default_keys=None,
            allow_empty=False,
        )

        return selected if selected else []

    def _download_locale(self, locale: str) -> bool:
        """Download a single locale using NGC CLI.

        Args:
            locale: Locale code to download

        Returns:
            True if download succeeded, False otherwise
        """
        # Print header before download (NGC CLI will show its own progress)
        print_text(f"📦 Downloading Nemotron-Persona dataset for {locale}...")
        console.print()

        try:
            self.service.download_persona_dataset(locale)
            console.print()
            print_success(f"✓ Downloaded Nemotron-Persona dataset for {locale}")
            return True

        except subprocess.CalledProcessError as e:
            console.print()
            print_error(f"✗ Failed to download Nemotron-Persona dataset for {locale}")
            print_error(f"NGC CLI error: {e}")
            return False

        except Exception as e:
            console.print()
            print_error(f"✗ Failed to download Nemotron-Persona dataset for {locale}")
            print_error(f"Unexpected error: {e}")
            return False
