# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from data_designer.config.mcp import ToolConfig
from data_designer.engine.model_provider import MCPProviderRegistry
from data_designer.engine.secret_resolver import SecretResolver

if TYPE_CHECKING:
    from data_designer.engine.mcp.facade import MCPFacade


@dataclass(frozen=True)
class MCPToolDefinition:
    """Definition of an MCP tool with its schema."""

    name: str
    description: str | None
    input_schema: dict[str, Any] | None

    def to_openai_tool_schema(self) -> dict[str, Any]:
        """Convert this tool definition to OpenAI function calling format.

        Returns:
            A dictionary in OpenAI's tool schema format with 'type' set to
            'function' and nested 'function' containing name, description,
            and parameters.
        """
        schema = self.input_schema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": schema,
            },
        }


@dataclass(frozen=True)
class MCPToolResult:
    """Result from executing an MCP tool call."""

    content: str
    is_error: bool = False


class MCPRegistry:
    """Registry for MCP tool configurations and facades.

    MCPRegistry manages ToolConfig instances by tool_alias and lazily creates
    MCPFacade instances when requested. This is a config-only registry - all
    actual MCP operations are delegated to the MCPFacade and io module.

    This mirrors the ModelRegistry pattern for consistency across the codebase.
    """

    def __init__(
        self,
        *,
        secret_resolver: SecretResolver,
        mcp_provider_registry: MCPProviderRegistry,
        mcp_facade_factory: Callable[[ToolConfig, SecretResolver, MCPProviderRegistry], MCPFacade],
        tool_configs: list[ToolConfig] | None = None,
    ) -> None:
        """Initialize the MCPRegistry.

        Args:
            secret_resolver: Resolver for secrets referenced in provider configs.
            mcp_provider_registry: Registry of MCP provider configurations.
            mcp_facade_factory: Factory for creating MCPFacade instances.
            tool_configs: Optional list of tool configurations to register.
        """
        self._secret_resolver = secret_resolver
        self._mcp_provider_registry = mcp_provider_registry
        self._mcp_facade_factory = mcp_facade_factory
        self._tool_configs: dict[str, ToolConfig] = {}
        self._facades: dict[str, MCPFacade] = {}

        self._set_tool_configs(tool_configs)

    @property
    def tool_configs(self) -> dict[str, ToolConfig]:
        """Get all registered tool configurations."""
        return self._tool_configs

    @property
    def facades(self) -> dict[str, MCPFacade]:
        """Get all instantiated facades."""
        return self._facades

    @property
    def mcp_provider_registry(self) -> MCPProviderRegistry:
        """Get the MCP provider registry."""
        return self._mcp_provider_registry

    def register_tool_configs(self, tool_configs: list[ToolConfig]) -> None:
        """Register tool configurations at runtime.

        Args:
            tool_configs: List of tool configurations to register. If a configuration
                with the same alias already exists, it will be overwritten.
        """
        self._set_tool_configs(list(self._tool_configs.values()) + tool_configs)

    def get_mcp(self, *, tool_alias: str) -> MCPFacade:
        """Get or lazily create an MCPFacade for the given tool alias.

        Args:
            tool_alias: The alias of the tool configuration.

        Returns:
            An MCPFacade configured for the specified tool alias.

        Raises:
            ValueError: If no tool config with the given alias is found.
        """
        if tool_alias not in self._tool_configs:
            raise ValueError(f"No tool config with alias {tool_alias!r} found!")

        if tool_alias not in self._facades:
            self._facades[tool_alias] = self._create_facade(self._tool_configs[tool_alias])

        return self._facades[tool_alias]

    def get_tool_config(self, *, tool_alias: str) -> ToolConfig:
        """Get a tool configuration by alias.

        Args:
            tool_alias: The alias of the tool configuration.

        Returns:
            The tool configuration.

        Raises:
            ValueError: If no tool config with the given alias is found.
        """
        if tool_alias not in self._tool_configs:
            raise ValueError(f"No tool config with alias {tool_alias!r} found!")
        return self._tool_configs[tool_alias]

    def _set_tool_configs(self, tool_configs: list[ToolConfig] | None) -> None:
        """Set tool configurations from a list."""
        tool_configs = tool_configs or []
        self._tool_configs = {tc.tool_alias: tc for tc in tool_configs}

    def _create_facade(self, tool_config: ToolConfig) -> MCPFacade:
        """Create an MCPFacade for a tool configuration."""
        return self._mcp_facade_factory(tool_config, self._secret_resolver, self._mcp_provider_registry)

    def validate_no_duplicate_tool_names(self) -> None:
        """Validate that no ToolConfig has duplicate tool names across its providers.

        This method eagerly fetches tool schemas for all registered ToolConfigs,
        which triggers duplicate tool name detection. This catches cases where
        multiple providers in the same ToolConfig expose a tool with the same name.

        Raises:
            DuplicateToolNameError: If any ToolConfig has duplicate tool names across providers.
        """
        for tool_alias in self._tool_configs:
            facade = self.get_mcp(tool_alias=tool_alias)
            # get_tool_schemas() validates for duplicate tool names
            facade.get_tool_schemas()
