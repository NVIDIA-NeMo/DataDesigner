# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.controllers.tool_controller import ToolController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def tools_command() -> None:
    """Configure tool configs interactively."""
    controller = ToolController(DATA_DESIGNER_HOME)
    controller.run()
