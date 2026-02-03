# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.mcp import LocalStdioMCPProvider, ToolConfig
from data_designer.engine.mcp.factory import create_mcp_registry
from data_designer.engine.mcp.registry import MCPRegistry
from data_designer.engine.model_provider import MCPProviderRegistry
from data_designer.engine.secret_resolver import PlaintextResolver


def test_create_mcp_registry_no_tool_configs() -> None:
    """Test creating MCPRegistry without tool configs."""
    secret_resolver = PlaintextResolver()
    mcp_provider_registry = MCPProviderRegistry()

    registry = create_mcp_registry(
        tool_configs=None,
        secret_resolver=secret_resolver,
        mcp_provider_registry=mcp_provider_registry,
    )

    assert isinstance(registry, MCPRegistry)
    assert len(registry.tool_configs) == 0


def test_create_mcp_registry_with_tool_configs() -> None:
    """Test creating MCPRegistry with tool configs."""
    secret_resolver = PlaintextResolver()

    provider = LocalStdioMCPProvider(name="test-provider", command="echo")
    mcp_provider_registry = MCPProviderRegistry(providers=[provider])

    tool_configs = [
        ToolConfig(
            tool_alias="test-tool",
            providers=["test-provider"],
        )
    ]

    registry = create_mcp_registry(
        tool_configs=tool_configs,
        secret_resolver=secret_resolver,
        mcp_provider_registry=mcp_provider_registry,
    )

    assert isinstance(registry, MCPRegistry)
    assert len(registry.tool_configs) == 1
    assert "test-tool" in registry.tool_configs


def test_create_mcp_registry_facade_factory_creates_facades() -> None:
    """Test that the registry's facade factory creates MCPFacade instances."""
    from unittest.mock import patch

    from data_designer.engine.mcp.registry import MCPToolDefinition

    secret_resolver = PlaintextResolver()

    provider = LocalStdioMCPProvider(name="test-provider", command="echo")
    mcp_provider_registry = MCPProviderRegistry(providers=[provider])

    tool_configs = [
        ToolConfig(
            tool_alias="test-tool",
            providers=["test-provider"],
        )
    ]

    registry = create_mcp_registry(
        tool_configs=tool_configs,
        secret_resolver=secret_resolver,
        mcp_provider_registry=mcp_provider_registry,
    )

    # Mock list_tools to avoid actual MCP connection (validation is deferred to first get_mcp call)
    mock_tools = (MCPToolDefinition(name="mock-tool", description="Mock", input_schema={}),)
    with patch("data_designer.engine.mcp.io.list_tools", return_value=mock_tools):
        # Get the facade through the registry
        facade = registry.get_mcp(tool_alias="test-tool")

    # Verify it's an MCPFacade
    from data_designer.engine.mcp.facade import MCPFacade

    assert isinstance(facade, MCPFacade)
