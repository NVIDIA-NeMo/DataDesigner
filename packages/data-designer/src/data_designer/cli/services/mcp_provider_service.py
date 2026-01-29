# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRegistry, MCPProviderRepository
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider

MCPProviderT = MCPProvider | LocalStdioMCPProvider


class MCPProviderService:
    """Business logic for MCP provider management.

    Supports both MCPProvider (remote SSE) and LocalStdioMCPProvider (subprocess).
    """

    def __init__(self, repository: MCPProviderRepository):
        self.repository = repository

    def list_all(self) -> list[MCPProviderT]:
        """Get all configured MCP providers."""
        registry = self.repository.load()
        return list(registry.providers) if registry else []

    def get_by_name(self, name: str) -> MCPProviderT | None:
        """Get an MCP provider by name."""
        providers = self.list_all()
        return next((p for p in providers if p.name == name), None)

    def add(self, provider: MCPProviderT) -> None:
        """Add a new MCP provider.

        Raises:
            ValueError: If provider name already exists
        """
        registry = self.repository.load() or MCPProviderRegistry(providers=[])

        if any(p.name == provider.name for p in registry.providers):
            raise ValueError(f"MCP provider '{provider.name}' already exists")

        registry.providers.append(provider)
        self.repository.save(registry)

    def update(self, original_name: str, updated_provider: MCPProviderT) -> None:
        """Update an existing MCP provider.

        Raises:
            ValueError: If provider not found or new name already exists
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No MCP providers configured")

        # Find provider index
        index = next(
            (i for i, p in enumerate(registry.providers) if p.name == original_name),
            None,
        )
        if index is None:
            raise ValueError(f"MCP provider '{original_name}' not found")

        if updated_provider.name != original_name:
            if any(p.name == updated_provider.name for p in registry.providers):
                raise ValueError(f"MCP provider name '{updated_provider.name}' already exists")

        registry.providers[index] = updated_provider
        self.repository.save(registry)

    def delete(self, name: str) -> None:
        """Delete an MCP provider.

        Raises:
            ValueError: If provider not found
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No MCP providers configured")

        if not any(p.name == name for p in registry.providers):
            raise ValueError(f"MCP provider '{name}' not found")

        registry.providers = [p for p in registry.providers if p.name != name]

        if registry.providers:
            self.repository.save(registry)
        else:
            self.repository.delete()
