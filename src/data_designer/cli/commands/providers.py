# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.constants import DEFAULT_CONFIG_DIR
from data_designer.cli.controllers.provider_controller import ProviderController
from data_designer.cli.utils import ensure_config_dir_exists


def providers_command(
    config_dir: str | None = typer.Option(None, "--config-dir", help="Custom configuration directory"),
) -> None:
    """Configure model providers interactively."""
    # Determine config directory
    config_path = Path(config_dir).expanduser().resolve() if config_dir else DEFAULT_CONFIG_DIR
    ensure_config_dir_exists(config_path)

    # Create and run controller
    controller = ProviderController(config_path)
    controller.run()
