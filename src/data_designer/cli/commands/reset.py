# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.ui import (
    confirm_action,
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    print_text,
)
from data_designer.cli.utils import (
    get_default_config_dir,
    get_model_config_path,
    get_model_provider_path,
)


def reset_command(
    output_dir: str | None = typer.Option(None, "--output-dir", help="Custom output directory"),
) -> None:
    """Reset configuration files by deleting them after confirmation."""
    print_header("Reset Configuration")

    # Determine output directory
    if output_dir:
        config_dir = Path(output_dir).expanduser().resolve()
    else:
        config_dir = get_default_config_dir()

    print_info(f"Configuration directory: {config_dir}")
    console.print()

    # Check which config files exist
    provider_path = get_model_provider_path(config_dir)
    model_path = get_model_config_path(config_dir)

    provider_exists = provider_path.exists()
    model_exists = model_path.exists()

    if not provider_exists and not model_exists:
        print_success("There are no configurations to reset! Nothing to do!")
        console.print()
        raise typer.Exit(0)

    # Show what configuration files exist
    print_text("Found the following configuration files:")
    console.print()

    if provider_exists:
        print_text(f"  |-- ⚙️  Model providers: {provider_path}")

    if model_exists:
        print_text(f"  |-- 🤖 Model configs: {model_path}")

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
        if confirm_action(f"Delete model providers configuration in {str(provider_path)!r}?", default=False):
            try:
                provider_path.unlink()
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
        if confirm_action(f"Delete model configs configuration in {str(model_path)!r}?", default=False):
            try:
                model_path.unlink()
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
