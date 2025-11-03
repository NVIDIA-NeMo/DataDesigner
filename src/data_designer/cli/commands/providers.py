# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.ui import (
    BACK,
    confirm_action,
    console,
    display_config_preview,
    print_error,
    print_header,
    print_info,
    print_navigation_tip,
    print_success,
    print_warning,
    prompt_text_input,
    select_with_arrows,
)
from data_designer.cli.utils import (
    ensure_config_dir_exists,
    get_default_config_dir,
    get_model_provider_path,
    load_config_file,
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
    print_navigation_tip()

    # Check for existing configuration
    provider_path = get_model_provider_path(config_dir)
    existing_config = None
    mode = "create"  # "create", "add", or "rewrite"

    if provider_path.exists():
        try:
            existing_config = load_config_file(provider_path)
        except Exception as e:
            print_warning(f"Could not load existing configuration: {e}")
            print_info("Starting with new configuration")
        else:
            # Successfully loaded existing config
            print_info(
                f"Found existing provider configuration with {len(existing_config.get('providers', []))} provider(s)"
            )
            console.print()

            # Show existing configuration
            display_config_preview(existing_config, "Current Configuration")
            console.print()

            # Ask what to do
            action_options = {
                "add": "Add more providers to existing configuration",
                "rewrite": "Rewrite the entire configuration",
                "exit": "Exit without making changes",
            }
            action = select_with_arrows(
                action_options,
                "What would you like to do?",
                default_key="add",
                allow_back=False,
            )

            if action is None or action == "exit":
                print_info("No changes made to configuration")
                raise typer.Exit(0)
            elif action == "add":
                mode = "add"
            elif action == "rewrite":
                mode = "rewrite"

    # Run the provider configuration wizard
    provider_config = configure_providers(existing_config if mode == "add" else None)
    if provider_config is None:
        print_error("Provider configuration cancelled")
        raise typer.Exit(1)

    # Check if config actually changed (when adding to existing)
    if mode == "add" and provider_config == existing_config:
        print_info("No changes made to configuration")
        raise typer.Exit(0)

    # Save configuration
    try:
        ensure_config_dir_exists(config_dir)
        save_config_file(provider_path, provider_config)
        print_success(f"Provider configuration saved to: {provider_path}")
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)


# Predefined provider configurations
PREDEFINED_PROVIDERS = {
    "nvidia": {
        "name": "nvidia",
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "provider_type": "openai",
        "default_api_key": "NVIDIA_API_KEY",
    },
    "openai": {
        "name": "openai",
        "endpoint": "https://api.openai.com/v1",
        "provider_type": "openai",
        "default_api_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "name": "anthropic",
        "endpoint": "https://api.anthropic.com/v1",
        "provider_type": "openai",
        "default_api_key": "ANTHROPIC_API_KEY",
    },
    "together": {
        "name": "together",
        "endpoint": "https://api.together.xyz/v1",
        "provider_type": "openai",
        "default_api_key": "TOGETHER_API_KEY",
    },
    "replicate": {
        "name": "replicate",
        "endpoint": "https://api.replicate.com/v1",
        "provider_type": "openai",
        "default_api_key": "REPLICATE_API_KEY",
    },
}


def configure_providers(existing_config: dict | None = None) -> dict | None:
    """Interactive configuration for model providers with back navigation.

    Args:
        existing_config: Optional existing configuration to add to

    Returns:
        Provider configuration dictionary, or None if cancelled
    """
    # Step-based state machine for back navigation
    step = "provider_choice"
    providers: list[dict] = []

    # If we have existing config, load the providers
    num_existing = 0
    if existing_config:
        providers = existing_config.get("providers", []).copy()
        num_existing = len(providers)
        print_info(f"Adding to existing {num_existing} provider(s)")
        console.print()

    # Track how many NEW providers we've added (for numbering and back navigation)
    new_providers_count = 0

    # Storage for provider data as we build it
    current_provider: dict = {}
    # Track if current provider is predefined or manual
    is_predefined = False

    # History stack for proper back navigation
    # Each entry: (step_name, provider_data, is_predefined)
    history: list[tuple[str, dict, bool]] = []

    while True:
        if step == "provider_choice":
            console.print()
            total_count = num_existing + new_providers_count + 1
            print_info(f"Configuring provider {total_count}")

            # Ask whether to use predefined or manual configuration
            choice_options = {
                "predefined": "Use a predefined provider",
                "manual": "Configure a custom provider",
            }
            result = select_with_arrows(
                choice_options,
                "How would you like to configure this provider?",
                default_key="predefined",
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                # Go back using history if available
                if history:
                    # Pop from history and restore state to edit a completed provider
                    prev_step, prev_provider, prev_is_predefined = history.pop()
                    # Remove the last provider from the list so we can re-add it after editing
                    if len(providers) > num_existing:
                        providers.pop()
                        new_providers_count -= 1
                    current_provider = prev_provider
                    is_predefined = prev_is_predefined
                    step = prev_step
                    continue
                else:
                    # No history - check if there are existing providers
                    if num_existing > 0:
                        # There are existing providers, so canceling is OK
                        if confirm_action("Discard the new providers?", default=False):
                            return existing_config  # Return the original config unchanged
                        continue
                    else:
                        # No existing providers, confirm complete cancellation
                        if confirm_action("Cancel all provider configuration?", default=False):
                            return None
                        continue

            # Starting a new provider
            if result == "predefined":
                is_predefined = True
                step = "select_predefined"
            else:
                is_predefined = False
                step = "provider_name"

        elif step == "select_predefined":
            # Get existing provider names for validation
            provider_names = {p["name"] for p in providers}

            # Filter out already-used predefined providers
            available_predefined = {
                key: f"{value['name']} - {value['endpoint']}"
                for key, value in PREDEFINED_PROVIDERS.items()
                if value["name"] not in provider_names
            }

            if not available_predefined:
                print_warning("All predefined providers have been configured. Switching to manual configuration.")
                is_predefined = False
                step = "provider_name"
                continue

            result = select_with_arrows(
                available_predefined,
                "Select a predefined provider",
                default_key="nvidia" if "nvidia" in available_predefined else list(available_predefined.keys())[0],
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "provider_choice"
                continue

            # Load predefined configuration
            predefined = PREDEFINED_PROVIDERS[result]
            current_provider = {
                "name": predefined["name"],
                "endpoint": predefined["endpoint"],
                "provider_type": predefined["provider_type"],
            }
            # Store default API key for later prompt
            current_provider["_default_api_key"] = predefined["default_api_key"]
            step = "provider_api_key"

        elif step == "provider_name":
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
                default="nvidia" if new_providers_count == 0 and num_existing == 0 else current_provider.get("name"),
                validator=lambda x, names=provider_names: (
                    (False, "Provider name must not be empty")
                    if not x
                    else (False, f"Provider name '{x}' already used")
                    if x in names
                    else (True, None)
                ),
                completions=common_provider_names,
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "provider_choice"
                continue

            current_provider = {"name": result}
            step = "provider_endpoint"

        elif step == "provider_endpoint":
            # Endpoint URL
            result = prompt_text_input(
                "API endpoint URL",
                default="https://integrate.api.nvidia.com/v1"
                if new_providers_count == 0 and num_existing == 0
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
            # API key - use predefined default if available
            default_api_key = current_provider.pop("_default_api_key", None)
            if default_api_key is None:
                default_api_key = (
                    "NVIDIA_API_KEY"
                    if new_providers_count == 0 and num_existing == 0
                    else current_provider.get("api_key")
                )

            result = prompt_text_input(
                "API key or environment variable name",
                default=default_api_key,
                mask=False,  # Don't mask since it's often an env var name
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                # Go back based on whether this is predefined or manual
                if is_predefined:
                    step = "select_predefined"
                else:
                    step = "provider_type"
                continue

            if result:
                current_provider["api_key"] = result

            # Validate using Pydantic model
            try:
                ModelProvider.model_validate(current_provider)
                providers.append(current_provider)
                new_providers_count += 1
                # Save to history before moving on
                history.append(("provider_api_key", current_provider.copy(), is_predefined))
                current_provider = {}
                is_predefined = False
                step = "add_another"  # Ask if they want to add another provider
            except Exception as e:
                print_error(f"Invalid provider configuration: {e}")
                return None

        elif step == "add_another":
            # Ask if user wants to add another provider
            console.print()

            # Create options for selection with back support
            add_another_options = {
                "yes": "Add another provider",
                "no": "Finish configuring providers",
            }
            result = select_with_arrows(
                add_another_options,
                "Would you like to add another provider?",
                default_key="no",
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                # Go back to the last provider's API key step
                if history:
                    prev_step, prev_provider, prev_is_predefined = history.pop()
                    # Remove the last provider so we can re-add it after editing
                    if len(providers) > num_existing:
                        providers.pop()
                        new_providers_count -= 1
                    current_provider = prev_provider
                    is_predefined = prev_is_predefined
                    step = "provider_api_key"
                continue
            elif result == "yes":
                step = "provider_choice"
            else:  # "no"
                step = "select_default"

        elif step == "select_default":
            # Determine default provider
            if len(providers) == 0:
                print_error("No providers configured")
                return None
            elif len(providers) == 1:
                default_provider = providers[0]["name"]
                step = "done"
            else:
                provider_options = {p["name"]: f"{p['name']} ({p['endpoint']})" for p in providers}
                console.print()

                # Use existing default if available and still valid
                existing_default = existing_config.get("default") if existing_config else None
                default_key = (
                    existing_default
                    if existing_default and existing_default in provider_options
                    else providers[0]["name"]
                )

                result = select_with_arrows(
                    provider_options,
                    "Select default provider",
                    default_key=default_key,
                    allow_back=True,
                )

                if result is None:
                    return None
                elif result is BACK:
                    # Go back to "add another" question
                    step = "add_another"
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
