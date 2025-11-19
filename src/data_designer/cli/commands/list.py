# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.table import Table

from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.ui import console, print_error, print_header, print_info, print_warning
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

    # Display providers
    display_providers(ProviderRepository(DATA_DESIGNER_HOME))
    display_models(ModelRepository(DATA_DESIGNER_HOME))


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
        table.add_column("Model ID", style=NordColor.NORD4.value)
        table.add_column("Provider", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("Temperature", style=NordColor.NORD15.value, justify="right")
        table.add_column("Top P", style=NordColor.NORD15.value, justify="right")
        table.add_column("Max Tokens", style=NordColor.NORD15.value, justify="right")

        for mc in registry.model_configs:
            # Handle distribution-based parameters
            temp_display = (
                f"{mc.inference_parameters.temperature:.2f}"
                if isinstance(mc.inference_parameters.temperature, (int, float))
                else "dist"
            )
            top_p_display = (
                f"{mc.inference_parameters.top_p:.2f}"
                if isinstance(mc.inference_parameters.top_p, (int, float))
                else "dist"
            )

            table.add_row(
                mc.alias,
                mc.model,
                mc.provider or "(default)",
                temp_display,
                top_p_display,
                str(mc.inference_parameters.max_tokens) if mc.inference_parameters.max_tokens else "(none)",
            )

        console.print(table)
        console.print()
    except Exception as e:
        print_error(f"Error loading model configuration: {e}")
        console.print()
