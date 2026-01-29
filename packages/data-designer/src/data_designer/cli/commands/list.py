# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rich.table import Table

from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRepository
from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.repositories.tool_repository import ToolRepository
from data_designer.cli.ui import console, print_error, print_header, print_info, print_warning
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider
from data_designer.config.utils.constants import DATA_DESIGNER_HOME, NordColor


def list_command() -> None:
    """List current Data Designer configurations.

    Returns:
        None
    """
    # Determine config directory
    print_header("Data Designer Configurations")
    print_info(f"Configuration directory: {DATA_DESIGNER_HOME}")
    console.print()

    # Display all configuration types
    display_providers(ProviderRepository(DATA_DESIGNER_HOME))
    display_models(ModelRepository(DATA_DESIGNER_HOME))
    display_mcp_providers(MCPProviderRepository(DATA_DESIGNER_HOME))
    display_tool_configs(ToolRepository(DATA_DESIGNER_HOME))


def display_providers(provider_repo: ProviderRepository) -> None:
    """Load and display model providers.

    Args:
        provider_repo: Provider repository

    Returns:
        None
    """
    try:
        provider_registry = provider_repo.load()

        if not provider_registry:
            print_warning("Providers have not been configured. Run 'data-designer config providers' to configure them.")
            console.print()
            return

        # Display as table
        table = Table(title="Model Providers", border_style=NordColor.NORD8.value)
        table.add_column("Name", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Endpoint", style=NordColor.NORD4.value)
        table.add_column("Type", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("API Key", style=NordColor.NORD7.value)
        table.add_column("Default", style=NordColor.NORD13.value, justify="center")

        default_name = provider_registry.default or provider_registry.providers[0].name

        for provider in provider_registry.providers:
            is_default = "✓" if provider.name == default_name else ""
            api_key_display = provider.api_key or "(not set)"

            # Mask actual API keys (keep env var names visible)
            if provider.api_key and not provider.api_key.isupper():
                api_key_display = "***" + provider.api_key[-4:] if len(provider.api_key) > 4 else "***"

            table.add_row(
                provider.name,
                provider.endpoint,
                provider.provider_type,
                api_key_display,
                is_default,
            )

        console.print(table)
        console.print()
    except Exception as e:
        print_error(f"Error loading provider configuration: {e}")
        console.print()


def display_models(model_repo: ModelRepository) -> None:
    """Load and display model configurations.

    Args:
        model_repo: Model repository

    Returns:
        None
    """
    try:
        registry = model_repo.load()

        if not registry:
            print_warning("Models have not been configured. Run 'data-designer config models' to configure them.")
            console.print()
            return

        # Display as table
        table = Table(title="Model Configurations", border_style=NordColor.NORD8.value)
        table.add_column("Alias", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Model", style=NordColor.NORD4.value)
        table.add_column("Provider", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("Inference Parameters", style=NordColor.NORD15.value)

        for mc in registry.model_configs:
            params_display = mc.inference_parameters.format_for_display()

            table.add_row(
                mc.alias,
                mc.model,
                mc.provider or "(default)",
                params_display,
            )

        console.print(table)
        console.print()
    except Exception as e:
        print_error(f"Error loading model configuration: {e}")
        console.print()


def display_mcp_providers(mcp_provider_repo: MCPProviderRepository) -> None:
    """Load and display MCP provider configurations.

    Handles both MCPProvider (remote SSE) and LocalStdioMCPProvider (subprocess).

    Args:
        mcp_provider_repo: MCP provider repository

    Returns:
        None
    """
    try:
        registry = mcp_provider_repo.load()

        if not registry:
            print_warning("MCP providers have not been configured. Run 'data-designer config mcp' to configure them.")
            console.print()
            return

        # Display as table
        table = Table(title="MCP Providers", border_style=NordColor.NORD8.value)
        table.add_column("Name", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Endpoint / Command", style=NordColor.NORD4.value)
        table.add_column("Type", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("API Key / Env", style=NordColor.NORD7.value)

        for provider in registry.providers:
            if isinstance(provider, MCPProvider):
                endpoint_display = provider.endpoint
                api_key_display = provider.api_key or "(not set)"
                # Mask actual API keys (keep env var names visible)
                if provider.api_key and not provider.api_key.isupper():
                    api_key_display = "***" + provider.api_key[-4:] if len(provider.api_key) > 4 else "***"
            elif isinstance(provider, LocalStdioMCPProvider):
                # Display command + args for stdio provider
                args_str = " ".join(provider.args) if provider.args else ""
                endpoint_display = f"{provider.command} {args_str}".strip()
                # Show env vars count for stdio provider
                api_key_display = f"{len(provider.env)} env vars" if provider.env else "(none)"
            else:
                endpoint_display = "(unknown)"
                api_key_display = "(unknown)"

            table.add_row(
                provider.name,
                endpoint_display,
                provider.provider_type,
                api_key_display,
            )

        console.print(table)
        console.print()
    except Exception as e:
        print_error(f"Error loading MCP provider configuration: {e}")
        console.print()


def display_tool_configs(tool_repo: ToolRepository) -> None:
    """Load and display tool configurations.

    Args:
        tool_repo: Tool repository

    Returns:
        None
    """
    try:
        registry = tool_repo.load()

        if not registry:
            print_warning("Tool configs have not been configured. Run 'data-designer config tools' to configure them.")
            console.print()
            return

        # Display as table
        table = Table(title="Tool Configurations", border_style=NordColor.NORD8.value)
        table.add_column("Alias", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Providers", style=NordColor.NORD4.value)
        table.add_column("Allowed Tools", style=NordColor.NORD9.value)
        table.add_column("Max Turns", style=NordColor.NORD7.value, justify="center")
        table.add_column("Timeout", style=NordColor.NORD15.value, justify="center")

        for tc in registry.tool_configs:
            providers_display = ", ".join(tc.providers)
            allow_tools_display = ", ".join(tc.allow_tools) if tc.allow_tools else "(all)"
            timeout_display = f"{tc.timeout_sec}s" if tc.timeout_sec else "(none)"

            table.add_row(
                tc.tool_alias,
                providers_display,
                allow_tools_display,
                str(tc.max_tool_call_turns),
                timeout_display,
            )

        console.print(table)
        console.print()
    except Exception as e:
        print_error(f"Error loading tool configuration: {e}")
        console.print()
