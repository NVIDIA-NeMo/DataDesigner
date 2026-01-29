# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.config.mcp import ToolConfig
from data_designer.engine.model_provider import MCPProviderRegistry
from data_designer.engine.secret_resolver import SecretResolver

if TYPE_CHECKING:
    from data_designer.engine.mcp.registry import MCPRegistry


def create_mcp_registry(
    *,
    tool_configs: list[ToolConfig] | None = None,
    secret_resolver: SecretResolver,
    mcp_provider_registry: MCPProviderRegistry,
) -> MCPRegistry:
    """Factory function for creating an MCPRegistry instance.

    This factory function creates an MCPRegistry with a facade factory that
    creates MCPFacade instances on demand. It follows the same pattern as
    create_model_registry for consistency.

    Args:
        tool_configs: Optional list of tool configurations to register.
        secret_resolver: Resolver for secrets referenced in provider configs.
        mcp_provider_registry: Registry of MCP provider configurations.

    Returns:
        A configured MCPRegistry instance.
    """
    from data_designer.engine.mcp.facade import MCPFacade
    from data_designer.engine.mcp.registry import MCPRegistry

    def mcp_facade_factory(
        tool_config: ToolConfig, secret_resolver: SecretResolver, provider_registry: MCPProviderRegistry
    ) -> MCPFacade:
        return MCPFacade(
            tool_config=tool_config, secret_resolver=secret_resolver, mcp_provider_registry=provider_registry
        )

    return MCPRegistry(
        secret_resolver=secret_resolver,
        mcp_provider_registry=mcp_provider_registry,
        mcp_facade_factory=mcp_facade_factory,
        tool_configs=tool_configs,
    )
