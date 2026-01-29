# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from pathlib import Path

from data_designer.cli.forms.mcp_provider_builder import MCPProviderFormBuilder, MCPProviderT
from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRepository
from data_designer.cli.services.mcp_provider_service import MCPProviderService
from data_designer.cli.ui import (
    confirm_action,
    console,
    display_config_preview,
    print_error,
    print_header,
    print_info,
    print_success,
    select_with_arrows,
)
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider


class MCPProviderController:
    """Controller for MCP provider configuration workflows."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.repository = MCPProviderRepository(config_dir)
        self.service = MCPProviderService(self.repository)

    def run(self) -> None:
        """Main entry point for MCP provider configuration."""
        print_header("Configure MCP Providers")
        print_info(f"Configuration directory: {self.config_dir}")
        console.print()

        # Check for existing configuration
        providers = self.service.list_all()

        if providers:
            self._show_existing_config()
            mode = self._select_mode()
        else:
            print_info("No MCP providers configured yet")
            console.print()
            mode = "add"

        if mode is None:
            print_info("No changes made")
            return

        # Execute selected mode
        mode_handlers = {
            "add": self._handle_add,
            "update": self._handle_update,
            "delete": self._handle_delete,
            "delete_all": self._handle_delete_all,
        }

        handler = mode_handlers.get(mode)
        if handler:
            handler()

    def _show_existing_config(self) -> None:
        """Display current configuration."""
        registry = self.repository.load()
        if not registry:
            return

        print_info(f"Found {len(registry.providers)} configured MCP provider(s)")
        console.print()

        # Display configuration (with masked API keys)
        config_dict = registry.model_dump(mode="json", exclude_none=True)
        masked_config = self._mask_api_keys(config_dict)
        display_config_preview(masked_config, "Current Configuration")
        console.print()

    def _mask_api_keys(self, config: dict) -> dict:
        """Mask API keys in configuration for display."""
        masked = copy.deepcopy(config)

        if "providers" in masked:
            for provider in masked["providers"]:
                if "api_key" in provider and provider["api_key"]:
                    api_key = provider["api_key"]
                    # Keep environment variable names visible
                    if not api_key.isupper():
                        provider["api_key"] = "***" + api_key[-4:] if len(api_key) > 4 else "***"

        return masked

    def _select_mode(self) -> str | None:
        """Prompt user to select operation mode."""
        options = {
            "add": "Add a new MCP provider",
            "update": "Update an existing MCP provider",
            "delete": "Delete an MCP provider",
            "delete_all": "Delete all MCP providers",
            "exit": "Exit without changes",
        }

        result = select_with_arrows(
            options,
            "What would you like to do?",
            default_key="add",
            allow_back=False,
        )

        return None if result == "exit" or result is None else result

    def _handle_add(self) -> None:
        """Handle adding new MCP providers."""
        existing_names = {p.name for p in self.service.list_all()}

        while True:
            # Create builder with current existing names
            builder = MCPProviderFormBuilder(existing_names)
            provider = builder.run()

            if provider is None:
                break

            # Attempt to add
            try:
                self.service.add(provider)
                print_success(f"MCP provider '{provider.name}' added successfully")
                existing_names.add(provider.name)
            except ValueError as e:
                print_error(f"Failed to add MCP provider: {e}")
                break

            # Ask if they want to add more
            if not self._confirm_add_another():
                break

    def _handle_update(self) -> None:
        """Handle updating an existing MCP provider."""
        providers = self.service.list_all()
        if not providers:
            print_error("No MCP providers to update")
            return

        # Select provider to update
        selected_name = self._select_provider(providers, "Select MCP provider to update")
        if selected_name is None:
            return

        provider = self.service.get_by_name(selected_name)
        if not provider:
            print_error(f"MCP provider '{selected_name}' not found")
            return

        # Run builder with existing data
        existing_names = {p.name for p in providers if p.name != selected_name}
        builder = MCPProviderFormBuilder(existing_names)
        initial_data = provider.model_dump(mode="json", exclude_none=True)
        updated_provider = builder.run(initial_data)

        if updated_provider:
            try:
                self.service.update(selected_name, updated_provider)
                print_success(f"MCP provider '{updated_provider.name}' updated successfully")
            except ValueError as e:
                print_error(f"Failed to update MCP provider: {e}")

    def _handle_delete(self) -> None:
        """Handle deleting an MCP provider."""
        providers = self.service.list_all()
        if not providers:
            print_error("No MCP providers to delete")
            return

        # Select provider to delete
        selected_name = self._select_provider(providers, "Select MCP provider to delete")
        if selected_name is None:
            return

        # Confirm deletion
        console.print()
        if confirm_action(f"Delete MCP provider '{selected_name}'?", default=False):
            try:
                self.service.delete(selected_name)
                print_success(f"MCP provider '{selected_name}' deleted successfully")
            except ValueError as e:
                print_error(f"Failed to delete MCP provider: {e}")

    def _handle_delete_all(self) -> None:
        """Handle deleting all MCP providers."""
        providers = self.service.list_all()
        if not providers:
            print_error("No MCP providers to delete")
            return

        # List providers to be deleted
        console.print()
        provider_count = len(providers)
        provider_names = ", ".join([f"'{p.name}'" for p in providers])

        if confirm_action(
            f"Delete ALL ({provider_count}) MCP provider(s): {provider_names}?\n   This action cannot be undone.",
            default=False,
        ):
            try:
                self.repository.delete()
                print_success(f"All ({provider_count}) MCP provider(s) deleted successfully")
            except Exception as e:
                print_error(f"Failed to delete all MCP providers: {e}")

    def _select_provider(self, providers: list[MCPProviderT], prompt: str, default: str | None = None) -> str | None:
        """Helper to select an MCP provider from list."""
        options = {}
        for p in providers:
            if isinstance(p, MCPProvider):
                options[p.name] = f"{p.name} (SSE: {p.endpoint})"
            elif isinstance(p, LocalStdioMCPProvider):
                options[p.name] = f"{p.name} (stdio: {p.command})"
            else:
                options[p.name] = p.name
        return select_with_arrows(
            options,
            prompt,
            default_key=default or providers[0].name,
            allow_back=False,
        )

    def _confirm_add_another(self) -> bool:
        """Ask if user wants to add another MCP provider."""
        result = select_with_arrows(
            {"yes": "Add another MCP provider", "no": "Finish"},
            "Add another MCP provider?",
            default_key="no",
            allow_back=False,
        )
        return result == "yes"
