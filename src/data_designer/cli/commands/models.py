# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.constants import DEFAULT_CONFIG_DIR
from data_designer.cli.controllers.model_controller import ModelController
from data_designer.cli.utils import ensure_config_dir_exists


def models_command(
    output_dir: str | None = typer.Option(None, "--output-dir", help="Custom output directory"),
) -> None:
    """Configure models interactively."""
    # Determine config directory
    config_dir = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_CONFIG_DIR
    ensure_config_dir_exists(config_dir)

    # Create and run controller
    controller = ModelController(config_dir)
    controller.run()
