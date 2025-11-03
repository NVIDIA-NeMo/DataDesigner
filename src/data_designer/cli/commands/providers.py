# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.ui import (
    BACK,
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    prompt_text_input,
    select_with_arrows,
)
from data_designer.cli.utils import (
    ensure_config_dir_exists,
    get_default_config_dir,
    get_model_provider_path,
    save_config_file,
    validate_url,
)
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry


def providers_command(
    output_dir: str | None = typer.Option(None, "--output-dir", help="Custom output directory"),
) -> None:
    """Configure model providers interactively."""
    print_header("Configure Model Providers")

    # Determine output directory
    if output_dir:
        config_dir = Path(output_dir).expanduser().resolve()
    else:
        config_dir = get_default_config_dir()

    print_info(f"Configuration will be saved to: {config_dir}")
    console.print()

    # Run the provider configuration wizard
    provider_config = configure_providers()
    if provider_config is None:
        print_error("Provider configuration cancelled")
        raise typer.Exit(1)

    # Save configuration
    try:
        ensure_config_dir_exists(config_dir)
        provider_path = get_model_provider_path(config_dir)
        save_config_file(provider_path, provider_config)
        print_success(f"Provider configuration saved to: {provider_path}")
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)


def configure_providers() -> dict | None:
    """Interactive configuration for model providers with back navigation.

    Returns:
        Provider configuration dictionary, or None if cancelled
    """
    # Step-based state machine for back navigation
    step = "num_providers"
    num_providers = 1
    providers: list[dict] = []
    provider_idx = 0

    # Storage for provider data as we build it
    current_provider: dict = {}

    while True:
        if step == "num_providers":
            # Ask how many providers
            from data_designer.cli.utils import validate_positive_int

            result = prompt_text_input(
                "How many model providers do you want to configure? (1-5)",
                default="1",
                validator=lambda x: validate_positive_int(x) if x else (True, None),
                allow_back=False,  # First step, can't go back
            )
            if result is None:
                return None

            num_providers = int(result) if result else 1
            num_providers = min(max(num_providers, 1), 5)  # Clamp to 1-5
            provider_idx = 0
            providers = []
            step = "provider_name"

        elif step == "provider_name":
            if provider_idx >= num_providers:
                # Done with all providers, move to default selection
                step = "select_default"
                continue

            console.print()
            print_info(f"Configuring provider {provider_idx + 1}/{num_providers}")

            # Get existing provider names for validation
            provider_names = {p["name"] for p in providers}

            # Provider name with common completions
            common_provider_names = [
                "nvidia",
                "openai",
                "anthropic",
                "together",
                "replicate",
                "local",
            ]
            result = prompt_text_input(
                "Provider name",
                default="nvidia" if provider_idx == 0 else current_provider.get("name"),
                validator=lambda x, names=provider_names: (
                    (False, "Provider name must not be empty")
                    if not x
                    else (False, f"Provider name '{x}' already used")
                    if x in names
                    else (True, None)
                ),
                completions=common_provider_names,
                allow_back=True,  # Always allow back - go to previous provider or num_providers
            )

            if result is None:
                return None
            elif result is BACK:
                # Go back to previous provider's last step, or to num_providers
                if provider_idx > 0 and len(providers) > 0:
                    # Go back to previous provider
                    provider_idx -= 1
                    current_provider = providers.pop()
                    step = "provider_api_key"  # Go to last step of previous provider
                    continue
                else:
                    # First provider, go back to num_providers question
                    step = "num_providers"
                    continue

            current_provider = {"name": result}
            step = "provider_endpoint"

        elif step == "provider_endpoint":
            # Endpoint URL
            result = prompt_text_input(
                "API endpoint URL",
                default="https://integrate.api.nvidia.com/v1"
                if provider_idx == 0
                else current_provider.get("endpoint"),
                validator=lambda x: (
                    (False, "Endpoint URL is required")
                    if not x
                    else (False, "Invalid URL format (must start with http:// or https://)")
                    if not validate_url(x)
                    else (True, None)
                ),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "provider_name"
                continue

            current_provider["endpoint"] = result
            step = "provider_type"

        elif step == "provider_type":
            # Provider type
            provider_type_options = {
                "openai": "OpenAI-compatible API",
            }
            result = select_with_arrows(
                provider_type_options,
                "Select provider type",
                default_key="openai",
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "provider_endpoint"
                continue

            current_provider["provider_type"] = result
            step = "provider_api_key"

        elif step == "provider_api_key":
            # API key
            result = prompt_text_input(
                "API key or environment variable name (optional, press Enter to skip)",
                default="NVIDIA_API_KEY" if provider_idx == 0 else current_provider.get("api_key"),
                mask=False,  # Don't mask since it's often an env var name
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "provider_type"
                continue

            if result:
                current_provider["api_key"] = result

            # Validate using Pydantic model
            try:
                ModelProvider.model_validate(current_provider)
                providers.append(current_provider)
                provider_idx += 1
                current_provider = {}
                step = "provider_name"  # Move to next provider or finish
            except Exception as e:
                print_error(f"Invalid provider configuration: {e}")
                return None

        elif step == "select_default":
            # Determine default provider
            if len(providers) == 1:
                default_provider = providers[0]["name"]
                step = "done"
            else:
                provider_options = {p["name"]: f"{p['name']} ({p['endpoint']})" for p in providers}
                console.print()
                result = select_with_arrows(
                    provider_options,
                    "Select default provider",
                    default_key=providers[0]["name"],
                    allow_back=True,
                )

                if result is None:
                    return None
                elif result is BACK:
                    # Go back to last provider
                    provider_idx = len(providers) - 1
                    current_provider = providers.pop()
                    step = "provider_api_key"
                    continue

                default_provider = result
                step = "done"

        elif step == "done":
            config = {
                "providers": providers,
                "default": default_provider,
            }

            # Validate entire configuration
            try:
                ModelProviderRegistry.model_validate(config)
                return config
            except Exception as e:
                print_error(f"Invalid provider registry configuration: {e}")
                return None
