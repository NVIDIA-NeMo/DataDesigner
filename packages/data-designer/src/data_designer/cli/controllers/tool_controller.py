# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.cli.forms.tool_builder import ToolFormBuilder
from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRepository
from data_designer.cli.repositories.tool_repository import ToolRepository
from data_designer.cli.services.mcp_provider_service import MCPProviderService
from data_designer.cli.services.tool_service import ToolService
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

if TYPE_CHECKING:
    from data_designer.config.mcp import ToolConfig


class ToolController:
    """Controller for tool configuration workflows."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.repository = ToolRepository(config_dir)
        self.service = ToolService(self.repository)
        self.mcp_provider_repository = MCPProviderRepository(config_dir)
        self.mcp_provider_service = MCPProviderService(self.mcp_provider_repository)

    def run(self) -> None:
        """Main entry point for tool configuration."""
        print_header("Configure Tool Configs")

        # Check if MCP providers are configured
        available_providers = self._get_available_providers()

        if not available_providers:
            print_error("No MCP providers available!")
            print_info("Please run 'data-designer config mcp' first to configure MCP providers.")
            return

        print_info(f"Configuration directory: {self.config_dir}")
        console.print()

        # Check for existing configuration
        tool_configs = self.service.list_all()

        if tool_configs:
            self._show_existing_config()
            mode = self._select_mode()
        else:
            print_info("No tool configs configured yet")
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
            handler(available_providers)

    def _get_available_providers(self) -> list[str]:
        """Get list of available MCP providers."""
        return [p.name for p in self.mcp_provider_service.list_all()]

    def _show_existing_config(self) -> None:
        """Display current configuration."""
        registry = self.repository.load()
        if not registry:
            return

        print_info(f"Found {len(registry.tool_configs)} configured tool config(s)")
        console.print()

        # Display configuration
        config_dict = registry.model_dump(mode="json", exclude_none=True)
        display_config_preview(config_dict, "Current Configuration")
        console.print()

    def _select_mode(self) -> str | None:
        """Prompt user to select operation mode."""
        options = {
            "add": "Add a new tool config",
            "update": "Update an existing tool config",
            "delete": "Delete a tool config",
            "delete_all": "Delete all tool configs",
            "exit": "Exit without changes",
        }

        result = select_with_arrows(
            options,
            "What would you like to do?",
            default_key="add",
            allow_back=False,
        )

        return None if result == "exit" or result is None else result

    def _handle_add(self, available_providers: list[str]) -> None:
        """Handle adding new tool configs."""
        existing_aliases = {c.tool_alias for c in self.service.list_all()}

        while True:
            # Create builder with current existing aliases
            builder = ToolFormBuilder(existing_aliases, available_providers)
            tool_config = builder.run()

            if tool_config is None:
                break

            # Attempt to add
            try:
                self.service.add(tool_config)
                print_success(f"Tool config '{tool_config.tool_alias}' added successfully")
                existing_aliases.add(tool_config.tool_alias)
            except ValueError as e:
                print_error(f"Failed to add tool config: {e}")
                break

            # Ask if they want to add more
            if not self._confirm_add_another():
                break

    def _handle_update(self, available_providers: list[str]) -> None:
        """Handle updating an existing tool config."""
        tool_configs = self.service.list_all()
        if not tool_configs:
            print_error("No tool configs to update")
            return

        # Select tool config to update
        selected_alias = self._select_tool_config(tool_configs, "Select tool config to update")
        if selected_alias is None:
            return

        tool_config = self.service.get_by_alias(selected_alias)
        if not tool_config:
            print_error(f"Tool config '{selected_alias}' not found")
            return

        # Run builder with existing data
        existing_aliases = {c.tool_alias for c in tool_configs if c.tool_alias != selected_alias}
        builder = ToolFormBuilder(existing_aliases, available_providers)
        initial_data = tool_config.model_dump(mode="json", exclude_none=True)
        updated_config = builder.run(initial_data)

        if updated_config:
            try:
                self.service.update(selected_alias, updated_config)
                print_success(f"Tool config '{updated_config.tool_alias}' updated successfully")
            except ValueError as e:
                print_error(f"Failed to update tool config: {e}")

    def _handle_delete(self, available_providers: list[str]) -> None:
        """Handle deleting a tool config."""
        tool_configs = self.service.list_all()
        if not tool_configs:
            print_error("No tool configs to delete")
            return

        # Select tool config to delete
        selected_alias = self._select_tool_config(tool_configs, "Select tool config to delete")
        if selected_alias is None:
            return

        # Confirm deletion
        console.print()
        if confirm_action(f"Delete tool config '{selected_alias}'?", default=False):
            try:
                self.service.delete(selected_alias)
                print_success(f"Tool config '{selected_alias}' deleted successfully")
            except ValueError as e:
                print_error(f"Failed to delete tool config: {e}")

    def _handle_delete_all(self, available_providers: list[str]) -> None:
        """Handle deleting all tool configs."""
        tool_configs = self.service.list_all()
        if not tool_configs:
            print_error("No tool configs to delete")
            return

        # List tool configs to be deleted
        console.print()
        config_count = len(tool_configs)
        config_aliases = ", ".join([f"'{c.tool_alias}'" for c in tool_configs])

        if confirm_action(
            f"Delete ALL ({config_count}) tool config(s): {config_aliases}?\n   This action cannot be undone.",
            default=False,
        ):
            try:
                self.repository.delete()
                print_success(f"All ({config_count}) tool config(s) deleted successfully")
            except Exception as e:
                print_error(f"Failed to delete all tool configs: {e}")

    def _select_tool_config(
        self, tool_configs: list[ToolConfig], prompt: str, default: str | None = None
    ) -> str | None:
        """Helper to select a tool config from list."""
        options = {c.tool_alias: f"{c.tool_alias} (providers: {', '.join(c.providers)})" for c in tool_configs}
        return select_with_arrows(
            options,
            prompt,
            default_key=default or tool_configs[0].tool_alias,
            allow_back=False,
        )

    def _confirm_add_another(self) -> bool:
        """Ask if user wants to add another tool config."""
        result = select_with_arrows(
            {"yes": "Add another tool config", "no": "Finish"},
            "Add another tool config?",
            default_key="no",
            allow_back=False,
        )
        return result == "yes"
