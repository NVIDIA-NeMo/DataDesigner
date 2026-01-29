# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, ToolConfig


def test_mcp_provider_requires_endpoint() -> None:
    with pytest.raises(ValidationError):
        MCPProvider(name="missing-endpoint")

    provider = MCPProvider(name="sse", endpoint="http://localhost:8080")
    assert provider.endpoint == "http://localhost:8080"
    assert provider.api_key is None

    provider_with_key = MCPProvider(name="sse-auth", endpoint="http://localhost:8080", api_key="secret")
    assert provider_with_key.api_key == "secret"


def test_local_stdio_mcp_provider_requires_command() -> None:
    with pytest.raises(ValidationError):
        LocalStdioMCPProvider(name="missing-command")

    provider = LocalStdioMCPProvider(name="stdio", command="python", args=["-m", "server"])
    assert provider.command == "python"
    assert provider.args == ["-m", "server"]
    assert provider.env == {}


def test_tool_config_defaults() -> None:
    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    assert tool_config.allow_tools is None
    assert tool_config.max_tool_call_turns == 5

    with pytest.raises(ValidationError):
        ToolConfig(tool_alias="search", providers=["tools"], max_tool_call_turns=0)


def test_tool_config_with_options() -> None:
    tool_config = ToolConfig(
        tool_alias="search",
        providers=["tools", "other-provider"],
        allow_tools=["lookup", "search"],
        max_tool_call_turns=10,
        timeout_sec=30.0,
    )
    assert tool_config.tool_alias == "search"
    assert tool_config.providers == ["tools", "other-provider"]
    assert tool_config.allow_tools == ["lookup", "search"]
    assert tool_config.max_tool_call_turns == 10
    assert tool_config.timeout_sec == 30.0
