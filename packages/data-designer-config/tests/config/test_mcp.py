# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from data_designer.config.errors import InvalidConfigError
from data_designer.config.mcp import MCPServerConfig, MCPToolConfig


def test_mcp_server_config_validation() -> None:
    with pytest.raises(InvalidConfigError):
        MCPServerConfig(name="missing")

    with pytest.raises(InvalidConfigError):
        MCPServerConfig(name="both", command="python", url="http://localhost:8080")

    server = MCPServerConfig(name="stdio", command="python", args=["-m", "server"])
    assert server.command == "python"
    assert server.url is None

    server = MCPServerConfig(name="sse", url="http://localhost:8080")
    assert server.url == "http://localhost:8080"
    assert server.command is None

    with pytest.raises(InvalidConfigError):
        MCPServerConfig(name="invalid", url="http://localhost:8080", args=["--flag"])


def test_mcp_tool_config_defaults() -> None:
    tool_config = MCPToolConfig(server_name="tools")
    assert tool_config.tool_names is None
    assert tool_config.max_tool_calls == 5

    with pytest.raises(ValidationError):
        MCPToolConfig(server_name="tools", max_tool_calls=0)
