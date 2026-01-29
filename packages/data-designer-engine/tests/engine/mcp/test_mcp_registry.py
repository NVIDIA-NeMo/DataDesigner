# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, ToolConfig
from data_designer.engine.mcp.facade import MCPFacade
from data_designer.engine.mcp.registry import MCPRegistry
from data_designer.engine.model_provider import MCPProviderRegistry
from data_designer.engine.secret_resolver import SecretResolver


@pytest.fixture
def stub_mcp_provider_registry() -> MCPProviderRegistry:
    """Create a stub MCP provider registry with a single provider."""
    return MCPProviderRegistry(providers=[LocalStdioMCPProvider(name="tools", command="python")])


@pytest.fixture
def stub_secret_resolver() -> MagicMock:
    """Create a stub secret resolver for testing."""
    resolver = MagicMock(spec=SecretResolver)
    resolver.resolve.side_effect = lambda x: x  # Return the input as-is
    return resolver


def test_get_mcp_missing_alias(
    stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry
) -> None:
    registry = MCPRegistry(secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry)

    with pytest.raises(ValueError, match="No tool config with alias"):
        registry.get_mcp(tool_alias="missing")


def test_get_tool_config_missing_alias(
    stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry
) -> None:
    registry = MCPRegistry(secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry)

    with pytest.raises(ValueError, match="No tool config with alias"):
        registry.get_tool_config(tool_alias="missing")


def test_register_tool_configs(
    stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry
) -> None:
    registry = MCPRegistry(secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry)

    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    registry.register_tool_configs([tool_config])

    assert "search" in registry.tool_configs
    assert registry.get_tool_config(tool_alias="search") == tool_config


def test_get_mcp_creates_facade(
    stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry
) -> None:
    """Test that get_mcp creates and caches facades."""

    def facade_factory(
        tool_config: ToolConfig, secret_resolver: SecretResolver, provider_registry: MCPProviderRegistry
    ) -> MCPFacade:
        return MCPFacade(
            tool_config=tool_config, secret_resolver=secret_resolver, mcp_provider_registry=provider_registry
        )

    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        tool_configs=[tool_config],
        mcp_facade_factory=facade_factory,
    )

    facade = registry.get_mcp(tool_alias="search")
    assert facade is not None
    assert facade.tool_alias == "search"

    # Same facade should be returned on subsequent calls
    facade2 = registry.get_mcp(tool_alias="search")
    assert facade is facade2


def test_get_mcp_no_factory(stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry) -> None:
    """Test that get_mcp raises when no factory is provided."""
    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        tool_configs=[tool_config],
    )

    with pytest.raises(RuntimeError, match="not initialized with an mcp_facade_factory"):
        registry.get_mcp(tool_alias="search")


def test_mcp_provider_registry_property(
    stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry
) -> None:
    """Test that mcp_provider_registry property returns the registry."""
    registry = MCPRegistry(secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry)
    assert registry.mcp_provider_registry is stub_mcp_provider_registry


def test_facades_property(stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry) -> None:
    """Test that facades property returns the facades dict."""

    def facade_factory(
        tool_config: ToolConfig, secret_resolver: SecretResolver, provider_registry: MCPProviderRegistry
    ) -> MCPFacade:
        return MCPFacade(
            tool_config=tool_config, secret_resolver=secret_resolver, mcp_provider_registry=provider_registry
        )

    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        tool_configs=[tool_config],
        mcp_facade_factory=facade_factory,
    )

    # Initially empty
    assert len(registry.facades) == 0

    # After creating a facade
    registry.get_mcp(tool_alias="search")
    assert "search" in registry.facades


def test_tool_configs_property(
    stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry
) -> None:
    """Test that tool_configs property returns all registered configs."""
    tool_config1 = ToolConfig(tool_alias="search", providers=["tools"])
    tool_config2 = ToolConfig(tool_alias="lookup", providers=["tools"])
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        tool_configs=[tool_config1, tool_config2],
    )

    assert len(registry.tool_configs) == 2
    assert "search" in registry.tool_configs
    assert "lookup" in registry.tool_configs
