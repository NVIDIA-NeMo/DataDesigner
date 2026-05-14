# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.controllers.mcp_provider_controller import MCPProviderController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def mcp_command() -> None:
    """Configure MCP providers interactively."""
    controller = MCPProviderController(DATA_DESIGNER_HOME)
    controller.run()
