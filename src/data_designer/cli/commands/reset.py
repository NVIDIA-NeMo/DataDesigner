# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import typer

from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.ui import (
    confirm_action,
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    print_text,
)
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def reset_command() -> None:
    """Reset configuration files by deleting them after confirmation."""
    print_header("Reset Configuration")

    # Determine configuration directory
    print_info(f"Configuration directory: {DATA_DESIGNER_HOME}")
    console.print()

    # Create repositories
    provider_repo = ProviderRepository(DATA_DESIGNER_HOME)
    model_repo = ModelRepository(DATA_DESIGNER_HOME)

    # Check which config files exist
    provider_exists = provider_repo.exists()
    model_exists = model_repo.exists()

    if not provider_exists and not model_exists:
        print_success("There are no configurations to reset! Nothing to do!")
        console.print()
        raise typer.Exit(0)

    # Show what configuration files exist
    print_text("Found the following configuration files:")
    console.print()

    if provider_exists:
        print_text(f"  |-- ⚙️  Model providers: {provider_repo.config_file}")

    if model_exists:
        print_text(f"  |-- 🤖 Model configs: {model_repo.config_file}")

    console.print()
    console.print()
    print_text("👀 You will be asked to confirm deletion for each file individually")
    console.print()

    # Track deletion results
    deleted_count = 0
    skipped_count = 0
    failed_count = 0

    # Ask for confirmation and delete model providers
    if provider_exists:
        if confirm_action(
            f"Delete model providers configuration in {str(provider_repo.config_file)!r}?", default=False
        ):
            try:
                provider_repo.delete()
                print_success("Deleted model providers configuration")
                deleted_count += 1
            except Exception as e:
                print_error(f"Failed to delete model providers configuration: {e}")
                failed_count += 1
        else:
            print_text("  |-- Skipped model providers configuration")
            skipped_count += 1
        console.print()

    # Ask for confirmation and delete model configs
    if model_exists:
        if confirm_action(f"Delete model configs configuration in {str(model_repo.config_file)!r}?", default=False):
            try:
                model_repo.delete()
                print_success("Deleted model configs configuration")
                deleted_count += 1
            except Exception as e:
                print_error(f"Failed to delete model configs configuration: {e}")
                failed_count += 1
        else:
            print_info("Skipped model configs configuration")
            skipped_count += 1
        console.print()

    # Summary
    if deleted_count > 0:
        print_success(f"Successfully deleted {deleted_count} configuration file(s)")
    if skipped_count > 0:
        print_info(f"Skipped {skipped_count} configuration file(s)")
    if failed_count > 0:
        print_error(f"Failed to delete {failed_count} configuration file(s)")
        raise typer.Exit(1)
