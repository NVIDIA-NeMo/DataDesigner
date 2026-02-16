# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.utils.constants import DATA_DESIGNER_HOME

# Controllers are imported inside command functions to avoid pulling in heavy
# dependencies (engine, models) at CLI startup time.


def mcp_command() -> None:
    """Configure MCP providers interactively."""
    from data_designer.cli.controllers.mcp_provider_controller import MCPProviderController

    controller = MCPProviderController(DATA_DESIGNER_HOME)
    controller.run()
