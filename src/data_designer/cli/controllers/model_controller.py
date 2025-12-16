# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.cli.forms.model_builder import ModelFormBuilder
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
    print_text,
    print_warning,
    select_with_arrows,
)
from data_designer.config.models import ModelConfig


class ModelController:
    """Controller for model configuration workflows."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.model_repository = ModelRepository(config_dir)
        self.model_service = ModelService(self.model_repository)
        self.provider_repository = ProviderRepository(config_dir)
        self.provider_service = ProviderService(self.provider_repository)

    def run(self) -> None:
        """Main entry point for model configuration."""
        print_header("Configure Models")

        # Check if providers are configured
        available_providers = self._get_available_providers()

        if not available_providers:
            print_error("No providers available!")
            print_info("Please run 'data-designer config providers' first")
            return

        print_info(f"Configuration directory: {self.config_dir}")
        console.print()

        # Check for existing configuration
        models = self.model_service.list_all()

        if models:
            self._show_existing_config()
            mode = self._select_mode()
        else:
            print_info("No models configured yet")
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
        """Get list of available providers."""
        return [p.name for p in self.provider_service.list_all()]

    def _show_existing_config(self) -> None:
        """Display current configuration."""
        registry = self.model_repository.load()
        if not registry:
            return

        print_info(f"Found {len(registry.model_configs)} configured model(s)")
        console.print()

        # Display configuration
        config_dict = registry.model_dump(mode="json", exclude_none=True)
        display_config_preview(config_dict, "Current Configuration")
        console.print()

    def _select_mode(self) -> str | None:
        """Prompt user to select operation mode."""
        options = {
            "add": "Add a new model",
            "update": "Update an existing model",
            "delete": "Delete a model",
            "delete_all": "Delete all models",
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
        """Handle adding new models."""
        existing_aliases = {m.alias for m in self.model_service.list_all()}

        while True:
            # Print message before starting configuration
            console.print()
            print_text("🚀 Starting a new model configuration")
            console.print()

            # Create builder with current existing aliases
            builder = ModelFormBuilder(existing_aliases, available_providers)
            model = builder.run()

            if model is None:
                break

            # Attempt to add
            try:
                self.model_service.add(model)
                print_success(f"Model '{model.alias}' added successfully")
                existing_aliases.add(model.alias)
            except ValueError as e:
                print_error(f"Failed to add model: {e}")
                break

            # Ask if they want to add more
            if not self._confirm_add_another():
                break

    def _handle_update(self, available_providers: list[str]) -> None:
        """Handle updating an existing model."""
        models = self.model_service.list_all()
        if not models:
            print_error("No models to update")
            return

        # Select model to update
        selected_alias = self._select_model(models, "Select model to update")
        if selected_alias is None:
            return

        model = self.model_service.get_by_alias(selected_alias)
        if not model:
            print_error(f"Model '{selected_alias}' not found")
            return

        # Check if model has distribution-based parameters
        params_dict = model.inference_parameters.model_dump(mode="json", exclude_none=True)
        has_distribution = any(isinstance(v, dict) and "distribution_type" in v for v in params_dict.values())

        if has_distribution:
            print_warning(
                "This model uses distribution-based inference parameters, "
                "which cannot be edited via the CLI. Please edit the configuration file directly."
            )
            return

        # Run builder with existing data
        existing_aliases = {m.alias for m in models if m.alias != selected_alias}
        builder = ModelFormBuilder(existing_aliases, available_providers)
        initial_data = model.model_dump(mode="json", exclude_none=True)
        updated_model = builder.run(initial_data)

        if updated_model:
            try:
                self.model_service.update(selected_alias, updated_model)
                print_success(f"Model '{updated_model.alias}' updated successfully")
            except ValueError as e:
                print_error(f"Failed to update model: {e}")

    def _handle_delete(self, available_providers: list[str]) -> None:
        """Handle deleting a model."""
        models = self.model_service.list_all()
        if not models:
            print_error("No models to delete")
            return

        # Select model to delete
        selected_alias = self._select_model(models, "Select model to delete")
        if selected_alias is None:
            return

        # Confirm deletion
        console.print()
        if confirm_action(f"Delete model '{selected_alias}'?", default=False):
            try:
                self.model_service.delete(selected_alias)
                print_success(f"Model '{selected_alias}' deleted successfully")
            except ValueError as e:
                print_error(f"Failed to delete model: {e}")

    def _handle_delete_all(self, available_providers: list[str]) -> None:
        """Handle deleting all models."""
        models = self.model_service.list_all()
        if not models:
            print_error("No models to delete")
            return

        # List models to be deleted
        console.print()
        model_count = len(models)
        model_aliases = ", ".join([f"'{m.alias}'" for m in models])

        if confirm_action(
            f"⚠️  Delete ALL ({model_count}) model(s): {model_aliases}?\n   This action cannot be undone.", default=False
        ):
            try:
                # Delete the entire configuration file
                self.model_repository.delete()
                print_success(f"All ({model_count}) model(s) deleted successfully")
            except Exception as e:
                print_error(f"Failed to delete all models: {e}")

    def _select_model(self, models: list[ModelConfig], prompt: str, default: str | None = None) -> str | None:
        """Helper to select a model from list."""
        options = {m.alias: f"{m.alias} ({m.model})" for m in models}
        return select_with_arrows(
            options,
            prompt,
            default_key=default or models[0].alias,
            allow_back=False,
        )

    def _confirm_add_another(self) -> bool:
        """Ask if user wants to add another model."""
        result = select_with_arrows(
            {"yes": "Add another model", "no": "Finish"},
            "Add another model?",
            default_key="no",
            allow_back=False,
        )
        return result == "yes"
