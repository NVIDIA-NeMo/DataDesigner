# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

from data_designer.cli.forms.provider_builder import ProviderFormBuilder
from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.services.model_service import ModelService
from data_designer.cli.services.provider_service import ProviderService
from data_designer.cli.ui import (
    confirm_action,
    console,
    display_config_preview,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    select_with_arrows,
)
from data_designer.engine.model_provider import ModelProvider


class ProviderController:
    """Controller for provider configuration workflows."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.repository = ProviderRepository(config_dir)
        self.service = ProviderService(self.repository)
        self.model_repository = ModelRepository(config_dir)
        self.model_service = ModelService(self.model_repository)

    def run(self) -> None:
        """Main entry point for provider configuration."""
        print_header("Configure Model Providers")
        print_info(f"Configuration directory: {self.config_dir}")
        console.print()

        # Check for existing configuration
        providers = self.service.list_all()

        if providers:
            self._show_existing_config()
            mode = self._select_mode()
        else:
            print_info("No providers configured yet")
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
            "change_default": self._handle_change_default,
        }

        handler = mode_handlers.get(mode)
        if handler:
            handler()

    def _show_existing_config(self) -> None:
        """Display current configuration."""
        registry = self.repository.load()
        if not registry:
            return

        print_info(f"Found {len(registry.providers)} configured provider(s)")
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
            "add": "Add a new provider",
            "update": "Update an existing provider",
            "delete": "Delete a provider",
            "delete_all": "Delete all providers",
        }

        # Only show change_default if multiple providers
        if len(self.service.list_all()) > 1:
            options["change_default"] = "Change default provider"

        options["exit"] = "Exit without changes"

        result = select_with_arrows(
            options,
            "What would you like to do?",
            default_key="add",
            allow_back=False,
        )

        return None if result == "exit" or result is None else result

    def _handle_add(self) -> None:
        """Handle adding new providers."""
        existing_names = {p.name for p in self.service.list_all()}

        while True:
            # Create builder with current existing names
            builder = ProviderFormBuilder(existing_names)
            provider = builder.run()

            if provider is None:
                break

            # Attempt to add
            try:
                self.service.add(provider)
                print_success(f"Provider '{provider.name}' added successfully")
                existing_names.add(provider.name)
            except ValueError as e:
                print_error(f"Failed to add provider: {e}")
                break

            # Ask if they want to add more
            if not self._confirm_add_another():
                break

    def _handle_update(self) -> None:
        """Handle updating an existing provider."""
        providers = self.service.list_all()
        if not providers:
            print_error("No providers to update")
            return

        # Select provider to update
        selected_name = self._select_provider(providers, "Select provider to update")
        if selected_name is None:
            return

        provider = self.service.get_by_name(selected_name)
        if not provider:
            print_error(f"Provider '{selected_name}' not found")
            return

        # Run builder with existing data
        existing_names = {p.name for p in providers if p.name != selected_name}
        builder = ProviderFormBuilder(existing_names)
        initial_data = provider.model_dump(mode="json", exclude_none=True)
        updated_provider = builder.run(initial_data)

        if updated_provider:
            try:
                self.service.update(selected_name, updated_provider)
                print_success(f"Provider '{updated_provider.name}' updated successfully")
            except ValueError as e:
                print_error(f"Failed to update provider: {e}")

    def _handle_delete(self) -> None:
        """Handle deleting a provider."""
        providers = self.service.list_all()
        if not providers:
            print_error("No providers to delete")
            return

        # Select provider to delete
        selected_name = self._select_provider(providers, "Select provider to delete")
        if selected_name is None:
            return

        # Check for associated models
        associated_models = self.model_service.find_by_provider(selected_name)

        # Confirm deletion
        console.print()

        if associated_models:
            # Notify user about associated models
            model_count = len(associated_models)
            model_aliases = ", ".join([f"'{m.alias}'" for m in associated_models])

            print_warning(f"Provider '{selected_name}' has {model_count} associated model config(s): {model_aliases}")
            console.print()

            # Ask if user wants to delete provider and associated models
            if confirm_action(
                f"Delete provider '{selected_name}' and its {model_count} associated model config(s)?", default=False
            ):
                try:
                    # Delete associated models first
                    model_aliases_to_delete = [m.alias for m in associated_models]
                    self.model_service.delete_by_aliases(model_aliases_to_delete)

                    # Then delete the provider
                    self.service.delete(selected_name)

                    print_success(
                        f"Provider '{selected_name}' and {model_count} associated model(s) deleted successfully"
                    )
                except ValueError as e:
                    print_error(f"Failed to delete provider and associated models: {e}")
        else:
            # No associated models, proceed with simple deletion
            if confirm_action(f"Delete provider '{selected_name}'?", default=False):
                try:
                    self.service.delete(selected_name)
                    print_success(f"Provider '{selected_name}' deleted successfully")
                except ValueError as e:
                    print_error(f"Failed to delete provider: {e}")

    def _handle_delete_all(self) -> None:
        """Handle deleting all providers."""
        providers = self.service.list_all()
        if not providers:
            print_error("No providers to delete")
            return

        # Check for associated models across all providers
        all_models = self.model_service.list_all()
        provider_names_set = {p.name for p in providers}
        associated_models = [m for m in all_models if m.provider in provider_names_set]

        # List providers to be deleted
        console.print()
        provider_count = len(providers)
        provider_names = ", ".join([f"'{p.name}'" for p in providers])

        if associated_models:
            model_count = len(associated_models)
            print_warning(f"Deleting all providers will also affect {model_count} associated model config(s)")
            console.print()

            if confirm_action(
                f"⚠️  Delete ALL ({provider_count}) provider(s): {provider_names} and {model_count} associated model(s)?\n   This action cannot be undone.",
                default=False,
            ):
                try:
                    # Delete all associated models first
                    model_aliases_to_delete = [m.alias for m in associated_models]
                    self.model_service.delete_by_aliases(model_aliases_to_delete)

                    # Then delete all providers
                    self.repository.delete()

                    print_success(
                        f"All ({provider_count}) provider(s) and {model_count} associated model(s) deleted successfully"
                    )
                except Exception as e:
                    print_error(f"Failed to delete all providers and associated models: {e}")
        else:
            if confirm_action(
                f"⚠️  Delete ALL ({provider_count}) provider(s): {provider_names}?\n   This action cannot be undone.",
                default=False,
            ):
                try:
                    self.repository.delete()
                    print_success(f"All ({provider_count}) provider(s) deleted successfully")
                except Exception as e:
                    print_error(f"Failed to delete all providers: {e}")

    def _handle_change_default(self) -> None:
        """Handle changing the default provider."""
        providers = self.service.list_all()
        current_default = self.service.get_default()

        print_info(f"Current default: {current_default}")
        console.print()

        # Select new default
        selected_name = self._select_provider(providers, "Select new default provider", default=current_default)
        if selected_name is None:
            return
        if selected_name and selected_name != current_default:
            try:
                self.service.set_default(selected_name)
                print_success(f"Default provider changed to '{selected_name}'")
            except ValueError as e:
                print_error(f"Failed to change default: {e}")

    def _select_provider(self, providers: list[ModelProvider], prompt: str, default: str | None = None) -> str | None:
        """Helper to select a provider from list."""
        options = {p.name: f"{p.name} ({p.endpoint})" for p in providers}
        return select_with_arrows(
            options,
            prompt,
            default_key=default or providers[0].name,
            allow_back=False,
        )

    def _confirm_add_another(self) -> bool:
        """Ask if user wants to add another provider."""
        result = select_with_arrows(
            {"yes": "Add another provider", "no": "Finish"},
            "Add another provider?",
            default_key="no",
            allow_back=False,
        )
        return result == "yes"
