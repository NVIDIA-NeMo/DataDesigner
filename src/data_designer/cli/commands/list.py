# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from rich.table import Table
import typer

from data_designer.cli.constants import DEFAULT_CONFIG_DIR
from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.ui import console, print_error, print_header, print_info, print_success, print_warning
from data_designer.config.utils.constants import NordColor


def list_command(
    config_dir: str | None = typer.Option(None, "--config-dir", help="Custom configuration directory"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List current Data Designer configurations."""
    # Determine config directory
    config_path = Path(config_dir).expanduser().resolve() if config_dir else DEFAULT_CONFIG_DIR

    if not output_json:
        print_header("Data Designer Configurations")
        print_info(f"Configuration directory: {config_path}")
        console.print()

    # Create repositories
    provider_repo = ProviderRepository(config_path)
    model_repo = ModelRepository(config_path)

    # Load configurations
    providers_data = load_providers(provider_repo, output_json)
    models_data = load_models(model_repo, output_json)

    # Output as JSON if requested
    if output_json:
        output = {
            "config_directory": str(config_path),
            "providers": providers_data,
            "models": models_data,
        }
        console.print_json(json.dumps(output, indent=2))
    else:
        # Display summary
        console.print()
        if providers_data or models_data:
            print_success("Configuration loaded successfully")


def load_providers(provider_repo: ProviderRepository, as_json: bool) -> dict | None:
    """Load and display model providers.

    Args:
        provider_repo: Provider repository
        as_json: If True, return data for JSON output instead of displaying

    Returns:
        Provider data if as_json=True, None otherwise
    """
    try:
        registry = provider_repo.load()

        if not registry:
            if not as_json:
                print_warning(
                    "Providers have not been configured. Run 'data-designer config providers' to configure them."
                )
                console.print()
            return (
                {"file": str(provider_repo.config_file), "valid": False, "error": "Providers have not been configured"}
                if as_json
                else None
            )

        if as_json:
            return {
                "file": str(provider_repo.config_file),
                "providers": [
                    {
                        "name": p.name,
                        "endpoint": p.endpoint,
                        "provider_type": p.provider_type,
                        "api_key": p.api_key,
                    }
                    for p in registry.providers
                ],
                "default": registry.default or registry.providers[0].name,
                "valid": True,
            }

        # Display as table
        table = Table(title="Model Providers", border_style=NordColor.NORD8.value)
        table.add_column("Name", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Endpoint", style=NordColor.NORD4.value)
        table.add_column("Type", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("API Key", style=NordColor.NORD7.value)
        table.add_column("Default", style=NordColor.NORD13.value, justify="center")

        default_name = registry.default or registry.providers[0].name

        for provider in registry.providers:
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
        return None

    except Exception as e:
        if not as_json:
            print_error(f"Error loading provider configuration: {e}")
            console.print()
        return {"file": str(provider_repo.config_file), "valid": False, "error": str(e)} if as_json else None


def load_models(model_repo: ModelRepository, as_json: bool) -> dict | None:
    """Load and display model configurations.

    Args:
        model_repo: Model repository
        as_json: If True, return data for JSON output instead of displaying

    Returns:
        Model data if as_json=True, None otherwise
    """
    try:
        registry = model_repo.load()

        if not registry:
            if not as_json:
                print_warning("Models have not been configured. Run 'data-designer config models' to configure them.")
                console.print()
            return (
                {"file": str(model_repo.config_file), "valid": False, "error": "Models have not been configured"}
                if as_json
                else None
            )

        if as_json:
            return {
                "file": str(model_repo.config_file),
                "models": [
                    {
                        "alias": mc.alias,
                        "model": mc.model,
                        "provider": mc.provider,
                        "inference_parameters": {
                            "temperature": mc.inference_parameters.temperature,
                            "top_p": mc.inference_parameters.top_p,
                            "max_tokens": mc.inference_parameters.max_tokens,
                            "max_parallel_requests": mc.inference_parameters.max_parallel_requests,
                            "timeout": mc.inference_parameters.timeout,
                        },
                    }
                    for mc in registry.model_configs
                ],
                "valid": True,
            }

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
        return None

    except Exception as e:
        if not as_json:
            print_error(f"Error loading model configuration: {e}")
            console.print()
        return {"file": str(model_repo.config_file), "valid": False, "error": str(e)} if as_json else None
