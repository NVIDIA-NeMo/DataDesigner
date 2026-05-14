# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.repositories.tool_repository import ToolConfigRegistry, ToolRepository
from data_designer.config.mcp import ToolConfig


class ToolService:
    """Business logic for tool configuration management."""

    def __init__(self, repository: ToolRepository):
        self.repository = repository

    def list_all(self) -> list[ToolConfig]:
        """Get all configured tool configs."""
        registry = self.repository.load()
        return list(registry.tool_configs) if registry else []

    def get_by_alias(self, alias: str) -> ToolConfig | None:
        """Get a tool config by alias."""
        configs = self.list_all()
        return next((c for c in configs if c.tool_alias == alias), None)

    def add(self, tool_config: ToolConfig) -> None:
        """Add a new tool config.

        Raises:
            ValueError: If tool alias already exists
        """
        registry = self.repository.load() or ToolConfigRegistry(tool_configs=[])

        if any(c.tool_alias == tool_config.tool_alias for c in registry.tool_configs):
            raise ValueError(f"Tool config with alias '{tool_config.tool_alias}' already exists")

        registry.tool_configs.append(tool_config)
        self.repository.save(registry)

    def update(self, original_alias: str, updated_config: ToolConfig) -> None:
        """Update an existing tool config.

        Raises:
            ValueError: If tool config not found or new alias already exists
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No tool configs configured")

        # Find config index
        index = next(
            (i for i, c in enumerate(registry.tool_configs) if c.tool_alias == original_alias),
            None,
        )
        if index is None:
            raise ValueError(f"Tool config with alias '{original_alias}' not found")

        if updated_config.tool_alias != original_alias:
            if any(c.tool_alias == updated_config.tool_alias for c in registry.tool_configs):
                raise ValueError(f"Tool config alias '{updated_config.tool_alias}' already exists")

        registry.tool_configs[index] = updated_config
        self.repository.save(registry)

    def delete(self, alias: str) -> None:
        """Delete a tool config.

        Raises:
            ValueError: If tool config not found
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No tool configs configured")

        if not any(c.tool_alias == alias for c in registry.tool_configs):
            raise ValueError(f"Tool config with alias '{alias}' not found")

        registry.tool_configs = [c for c in registry.tool_configs if c.tool_alias != alias]

        if registry.tool_configs:
            self.repository.save(registry)
        else:
            self.repository.delete()
