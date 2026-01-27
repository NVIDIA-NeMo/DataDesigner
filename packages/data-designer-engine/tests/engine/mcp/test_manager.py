# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.config.mcp import MCPServerConfig, MCPToolConfig
from data_designer.engine.mcp.errors import MCPConfigurationError
from data_designer.engine.mcp.manager import MCPClientManager, MCPToolDefinition


def test_get_tool_schemas_filters_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MCPServerConfig(name="tools", command="python")
    manager = MCPClientManager(server_configs=[server])

    tools = [
        MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),
        MCPToolDefinition(name="other", description="Other", input_schema={"type": "object"}),
    ]

    def _list_tools(_: str, *, timeout_sec: float | None = None) -> list[MCPToolDefinition]:
        return tools

    monkeypatch.setattr(manager, "_list_tools", _list_tools)

    tool_config = MCPToolConfig(server_name="tools", tool_names=["lookup"])
    schemas = manager.get_tool_schemas(tool_config)

    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "lookup"


def test_get_tool_schemas_missing_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MCPServerConfig(name="tools", command="python")
    manager = MCPClientManager(server_configs=[server])

    def _list_tools(_: str, *, timeout_sec: float | None = None) -> list[MCPToolDefinition]:
        return [MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"})]

    monkeypatch.setattr(manager, "_list_tools", _list_tools)

    tool_config = MCPToolConfig(server_name="tools", tool_names=["missing"])
    with pytest.raises(MCPConfigurationError):
        manager.get_tool_schemas(tool_config)


def test_duplicate_server_names_rejected() -> None:
    with pytest.raises(MCPConfigurationError):
        MCPClientManager(
            server_configs=[
                MCPServerConfig(name="tools", command="python"),
                MCPServerConfig(name="tools", command="python"),
            ]
        )


def test_call_tool_missing_server() -> None:
    manager = MCPClientManager(server_configs=[MCPServerConfig(name="tools", command="python")])

    with pytest.raises(MCPConfigurationError):
        manager.call_tool("missing", "lookup", {"query": "x"})
