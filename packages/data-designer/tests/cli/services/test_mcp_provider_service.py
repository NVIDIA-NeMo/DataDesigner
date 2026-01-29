# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRepository
from data_designer.cli.services.mcp_provider_service import MCPProviderService
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider

MCPProviderT = MCPProvider | LocalStdioMCPProvider


def test_list_all(stub_mcp_provider_service: MCPProviderService, stub_mcp_providers: list[MCPProviderT]) -> None:
    assert stub_mcp_provider_service.list_all() == stub_mcp_providers


def test_get_by_name(stub_mcp_provider_service: MCPProviderService, stub_mcp_providers: list[MCPProviderT]) -> None:
    assert stub_mcp_provider_service.get_by_name("mcp-provider-1") == stub_mcp_providers[0]
    assert stub_mcp_provider_service.get_by_name("nonexistent") is None


def test_add(
    stub_mcp_provider_service: MCPProviderService,
    stub_mcp_providers: list[MCPProviderT],
    stub_new_mcp_provider: MCPProvider,
) -> None:
    stub_mcp_provider_service.add(stub_new_mcp_provider)
    assert stub_mcp_provider_service.list_all() == stub_mcp_providers + [stub_new_mcp_provider]


def test_add_stdio_provider(
    stub_mcp_provider_service: MCPProviderService,
    stub_mcp_providers: list[MCPProviderT],
    stub_new_stdio_mcp_provider: LocalStdioMCPProvider,
) -> None:
    """Test adding a LocalStdioMCPProvider."""
    stub_mcp_provider_service.add(stub_new_stdio_mcp_provider)
    assert stub_mcp_provider_service.list_all() == stub_mcp_providers + [stub_new_stdio_mcp_provider]


def test_add_duplicate_name(stub_mcp_provider_service: MCPProviderService) -> None:
    """Test adding an MCP provider with a name that already exists."""
    duplicate_provider = MCPProvider(
        name="mcp-provider-1",
        endpoint="http://localhost:9999/sse",
    )
    with pytest.raises(ValueError, match="MCP provider 'mcp-provider-1' already exists"):
        stub_mcp_provider_service.add(duplicate_provider)


def test_update(stub_mcp_provider_service: MCPProviderService, stub_new_mcp_provider: MCPProvider) -> None:
    stub_mcp_provider_service.update("mcp-provider-1", stub_new_mcp_provider)
    assert stub_mcp_provider_service.get_by_name("mcp-provider-1") is None
    assert stub_mcp_provider_service.get_by_name("mcp-provider-3") == stub_new_mcp_provider


def test_update_no_registry(tmp_path: Path, stub_new_mcp_provider: MCPProvider) -> None:
    """Test updating when no registry exists."""
    service = MCPProviderService(MCPProviderRepository(tmp_path))
    with pytest.raises(ValueError, match="No MCP providers configured"):
        service.update("mcp-provider-1", stub_new_mcp_provider)


def test_update_nonexistent_provider(
    stub_mcp_provider_service: MCPProviderService, stub_new_mcp_provider: MCPProvider
) -> None:
    """Test updating an MCP provider that doesn't exist."""
    with pytest.raises(ValueError, match="MCP provider 'nonexistent' not found"):
        stub_mcp_provider_service.update("nonexistent", stub_new_mcp_provider)


def test_update_to_existing_name(
    stub_mcp_provider_service: MCPProviderService, stub_mcp_providers: list[MCPProviderT]
) -> None:
    """Test updating an MCP provider to a name that already exists."""
    updated_provider = MCPProvider(
        name="mcp-provider-2",  # Already exists
        endpoint="http://localhost:9999/sse",
    )
    with pytest.raises(ValueError, match="MCP provider name 'mcp-provider-2' already exists"):
        stub_mcp_provider_service.update("mcp-provider-1", updated_provider)


def test_delete(stub_mcp_provider_service: MCPProviderService, stub_mcp_providers: list[MCPProviderT]) -> None:
    stub_mcp_provider_service.delete("mcp-provider-1")
    assert len(stub_mcp_provider_service.list_all()) == len(stub_mcp_providers) - 1


def test_delete_no_registry(tmp_path: Path) -> None:
    """Test deleting when no registry exists."""
    service = MCPProviderService(MCPProviderRepository(tmp_path))
    with pytest.raises(ValueError, match="No MCP providers configured"):
        service.delete("mcp-provider-1")


def test_delete_nonexistent_provider(stub_mcp_provider_service: MCPProviderService) -> None:
    """Test deleting an MCP provider that doesn't exist."""
    with pytest.raises(ValueError, match="MCP provider 'nonexistent' not found"):
        stub_mcp_provider_service.delete("nonexistent")


def test_delete_all_providers(
    stub_mcp_provider_service: MCPProviderService, stub_mcp_providers: list[MCPProviderT]
) -> None:
    """Test deleting all MCP providers triggers repository.delete()."""
    for provider in stub_mcp_providers:
        stub_mcp_provider_service.delete(provider.name)
    assert stub_mcp_provider_service.list_all() == []
