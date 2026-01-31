# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, ToolConfig
from data_designer.engine.mcp.registry import MCPRegistry
from data_designer.engine.model_provider import MCPProviderRegistry


def test_get_mcp_missing_alias(
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_mcp_facade_factory,
) -> None:
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=stub_mcp_facade_factory,
    )

    with pytest.raises(ValueError, match="No tool config with alias"):
        registry.get_mcp(tool_alias="missing")


def test_get_tool_config_missing_alias(
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_mcp_facade_factory,
) -> None:
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=stub_mcp_facade_factory,
    )

    with pytest.raises(ValueError, match="No tool config with alias"):
        registry.get_tool_config(tool_alias="missing")


def test_register_tool_configs(
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_mcp_facade_factory,
) -> None:
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=stub_mcp_facade_factory,
    )

    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    registry.register_tool_configs([tool_config])

    assert "search" in registry.tool_configs
    assert registry.get_tool_config(tool_alias="search") == tool_config


def test_get_mcp_creates_facade(
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_mcp_facade_factory,
) -> None:
    """Test that get_mcp creates and caches facades."""
    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=stub_mcp_facade_factory,
        tool_configs=[tool_config],
    )

    facade = registry.get_mcp(tool_alias="search")
    assert facade is not None
    assert facade.tool_alias == "search"

    # Same facade should be returned on subsequent calls
    facade2 = registry.get_mcp(tool_alias="search")
    assert facade is facade2


def test_mcp_provider_registry_property(
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_mcp_facade_factory,
) -> None:
    """Test that mcp_provider_registry property returns the registry."""
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=stub_mcp_facade_factory,
    )
    assert registry.mcp_provider_registry is stub_mcp_provider_registry


def test_facades_property(
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_mcp_facade_factory,
) -> None:
    """Test that facades property returns the facades dict."""
    tool_config = ToolConfig(tool_alias="search", providers=["tools"])
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=stub_mcp_facade_factory,
        tool_configs=[tool_config],
    )

    # Initially empty
    assert len(registry.facades) == 0

    # After creating a facade
    registry.get_mcp(tool_alias="search")
    assert "search" in registry.facades


def test_tool_configs_property(
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_mcp_facade_factory,
) -> None:
    """Test that tool_configs property returns all registered configs."""
    tool_config1 = ToolConfig(tool_alias="search", providers=["tools"])
    tool_config2 = ToolConfig(tool_alias="lookup", providers=["tools"])
    registry = MCPRegistry(
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
        mcp_facade_factory=stub_mcp_facade_factory,
        tool_configs=[tool_config1, tool_config2],
    )

    assert len(registry.tool_configs) == 2
    assert "search" in registry.tool_configs
    assert "lookup" in registry.tool_configs


class TestValidateNoDuplicateToolNames:
    """Tests for validate_no_duplicate_tool_names method."""

    def test_validate_no_duplicates_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_secret_resolver: MagicMock,
        stub_mcp_facade_factory,
    ) -> None:
        """Validation passes when tool names are unique across providers."""
        from data_designer.engine.mcp import io as mcp_io
        from data_designer.engine.mcp.registry import MCPToolDefinition

        providers = [
            LocalStdioMCPProvider(name="provider-1", command="python"),
            LocalStdioMCPProvider(name="provider-2", command="python"),
        ]
        provider_registry = MCPProviderRegistry(providers=providers)

        def mock_list_tools(
            provider: LocalStdioMCPProvider, timeout_sec: float | None = None
        ) -> tuple[MCPToolDefinition, ...]:
            if provider.name == "provider-1":
                return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)
            return (MCPToolDefinition(name="fetch", description="Fetch", input_schema={"type": "object"}),)

        monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

        tool_config = ToolConfig(tool_alias="tools", providers=["provider-1", "provider-2"])
        registry = MCPRegistry(
            secret_resolver=stub_secret_resolver,
            mcp_provider_registry=provider_registry,
            mcp_facade_factory=stub_mcp_facade_factory,
            tool_configs=[tool_config],
        )

        # Should not raise
        registry.validate_no_duplicate_tool_names()

    def test_validate_with_duplicates_raises_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_secret_resolver: MagicMock,
        stub_mcp_facade_factory,
    ) -> None:
        """Validation raises DuplicateToolNameError when duplicates found."""
        from data_designer.engine.mcp import io as mcp_io
        from data_designer.engine.mcp.errors import DuplicateToolNameError
        from data_designer.engine.mcp.registry import MCPToolDefinition

        providers = [
            LocalStdioMCPProvider(name="provider-1", command="python"),
            LocalStdioMCPProvider(name="provider-2", command="python"),
        ]
        provider_registry = MCPProviderRegistry(providers=providers)

        def mock_list_tools(
            provider: LocalStdioMCPProvider, timeout_sec: float | None = None
        ) -> tuple[MCPToolDefinition, ...]:
            # Both providers return "lookup" - duplicate
            return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

        monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

        tool_config = ToolConfig(tool_alias="tools", providers=["provider-1", "provider-2"])
        registry = MCPRegistry(
            secret_resolver=stub_secret_resolver,
            mcp_provider_registry=provider_registry,
            mcp_facade_factory=stub_mcp_facade_factory,
            tool_configs=[tool_config],
        )

        with pytest.raises(DuplicateToolNameError, match="Duplicate tool names found"):
            registry.validate_no_duplicate_tool_names()

    def test_validate_validates_all_tool_configs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_secret_resolver: MagicMock,
        stub_mcp_facade_factory,
    ) -> None:
        """Validation checks all registered ToolConfigs."""
        from data_designer.engine.mcp import io as mcp_io
        from data_designer.engine.mcp.errors import DuplicateToolNameError
        from data_designer.engine.mcp.registry import MCPToolDefinition

        providers = [
            LocalStdioMCPProvider(name="provider-1", command="python"),
            LocalStdioMCPProvider(name="provider-2", command="python"),
        ]
        provider_registry = MCPProviderRegistry(providers=providers)

        def mock_list_tools(
            provider: LocalStdioMCPProvider, timeout_sec: float | None = None
        ) -> tuple[MCPToolDefinition, ...]:
            # Both providers return "lookup" - duplicate
            return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

        monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

        # First tool config has no duplicates (single provider)
        tool_config1 = ToolConfig(tool_alias="tools-1", providers=["provider-1"])
        # Second tool config has duplicates (both providers, both have "lookup")
        tool_config2 = ToolConfig(tool_alias="tools-2", providers=["provider-1", "provider-2"])

        registry = MCPRegistry(
            secret_resolver=stub_secret_resolver,
            mcp_provider_registry=provider_registry,
            mcp_facade_factory=stub_mcp_facade_factory,
            tool_configs=[tool_config1, tool_config2],
        )

        with pytest.raises(DuplicateToolNameError):
            registry.validate_no_duplicate_tool_names()

    def test_validate_empty_tool_configs_passes(
        self,
        stub_secret_resolver: MagicMock,
        stub_mcp_provider_registry: MCPProviderRegistry,
        stub_mcp_facade_factory,
    ) -> None:
        """Validation passes when no tool configs are registered."""
        registry = MCPRegistry(
            secret_resolver=stub_secret_resolver,
            mcp_provider_registry=stub_mcp_provider_registry,
            mcp_facade_factory=stub_mcp_facade_factory,
            tool_configs=[],
        )

        # Should not raise
        registry.validate_no_duplicate_tool_names()
