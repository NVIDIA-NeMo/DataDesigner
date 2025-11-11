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
    print_text,
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
}


def _mask_api_keys_in_config(config: dict) -> dict:
    """Create a copy of config with API keys masked.

    Args:
        config: Configuration dictionary with providers

    Returns:
        New dictionary with masked API keys
    """
    import copy

    masked = copy.deepcopy(config)
    if "providers" in masked:
        for provider in masked["providers"]:
            if "api_key" in provider and provider["api_key"]:
                api_key = provider["api_key"]
                # Keep env var names visible (typically uppercase)
                if api_key.isupper():
                    # Looks like an environment variable name, keep it
                    continue
                # Mask actual API keys (show last 4 chars)
                if len(api_key) > 4:
                    provider["api_key"] = "***" + api_key[-4:]
                else:
                    provider["api_key"] = "***"
    return masked


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
    existing_providers: list[ModelProvider] = []
    existing_default: str | None = None
    mode = "create"  # "create", "add", or "rewrite"

    if provider_path.exists():
        try:
            config_dict = load_config_file(provider_path)
            registry = ModelProviderRegistry.model_validate(config_dict)
            existing_providers = list(registry.providers)
            existing_default = registry.default
        except Exception as e:
            print_warning(f"Could not load existing configuration: {e}")
            print_info("Starting with new configuration")
        else:
            # Successfully loaded existing config
            print_info(f"Found existing provider configuration with {len(existing_providers)} provider(s)")
            console.print()

            # Show existing configuration with masked API keys
            masked_config = _mask_api_keys_in_config(config_dict)
            display_config_preview(masked_config, "Current Configuration")
            console.print()

            # Ask what to do
            action_options = {
                "add": "Add more providers to existing configuration",
                "update": "Update an existing provider",
                "delete": "Delete a provider",
            }

            # Only show "Change default provider" if there are multiple providers
            num_providers = len(existing_providers)
            if num_providers > 1:
                action_options["change_default"] = "Change the default provider"

            action_options.update(
                {
                    "rewrite": "Reset and start from scratch",
                    "exit": "Exit without making changes",
                }
            )

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
            elif action == "update":
                mode = "update"
            elif action == "change_default":
                mode = "change_default"
            elif action == "delete":
                mode = "delete"
            elif action == "rewrite":
                mode = "rewrite"

    # Run the provider configuration wizard
    if mode == "update":
        updated_providers, updated_default = update_provider(existing_providers, existing_default)
    elif mode == "change_default":
        updated_providers, updated_default = change_default_provider(existing_providers, existing_default)
    elif mode == "delete":
        updated_providers, updated_default = delete_provider(existing_providers, existing_default)
    else:
        updated_providers, updated_default = configure_providers(
            existing_providers if mode == "add" else None, existing_default if mode == "add" else None
        )

    if updated_providers is None:
        print_error("Provider configuration cancelled")
        raise typer.Exit(1)

    # Check if config actually changed
    if (
        mode in ("add", "update", "change_default", "delete")
        and updated_providers == existing_providers
        and updated_default == existing_default
    ):
        print_info("No changes made to configuration")
        raise typer.Exit(0)

    # Handle empty provider list (all providers deleted)
    if len(updated_providers) == 0:
        try:
            if provider_path.exists():
                provider_path.unlink()
                print_text("  |-- All providers deleted. Configuration file removed.")
            else:
                print_info("Configuration file already removed.")
        except Exception as e:
            print_error(f"Failed to remove configuration file: {e}")
            raise typer.Exit(1)
        raise typer.Exit(0)

    # Save configuration
    try:
        ensure_config_dir_exists(config_dir)
        registry = ModelProviderRegistry(providers=updated_providers, default=updated_default)
        save_config_file(provider_path, registry.model_dump(mode="json", exclude_none=True))
        print_success(f"Provider configuration saved to: {provider_path}")
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)


def update_provider(
    existing_providers: list[ModelProvider], existing_default: str | None
) -> tuple[list[ModelProvider], str] | tuple[None, None]:
    """Update an existing provider configuration.

    Args:
        existing_providers: List of existing ModelProvider objects
        existing_default: Current default provider name

    Returns:
        Tuple of (updated providers list, default name), or (None, None) if cancelled
    """
    if not existing_providers:
        print_error("No providers found in configuration")
        return None, None

    # Select which provider to update
    console.print()
    provider_options = {p.name: f"{p.name} ({p.endpoint})" for p in existing_providers}

    selected_name = select_with_arrows(
        provider_options,
        "Select a provider to update",
        default_key=existing_providers[0].name,
        allow_back=False,
    )

    if selected_name is None:
        return None, None

    # Find the provider to update
    provider_index = next(i for i, p in enumerate(existing_providers) if p.name == selected_name)
    current_provider = existing_providers[provider_index]
    original_name = current_provider.name

    # Check if this is a predefined provider
    is_predefined = original_name in PREDEFINED_PROVIDERS

    print_info(f"Updating provider: {original_name}")
    if is_predefined:
        print_info("This is a predefined provider. Only the API key can be updated.")
    console.print()

    # Step through each field to update
    # For predefined providers, skip directly to API key
    step = "provider_api_key" if is_predefined else "provider_name"
    updated_name = current_provider.name
    updated_endpoint = current_provider.endpoint
    updated_type = current_provider.provider_type
    updated_api_key = current_provider.api_key

    while True:
        if step == "provider_name":
            # Get other provider names (excluding current) for validation
            other_names = {p.name for i, p in enumerate(existing_providers) if i != provider_index}

            result = prompt_text_input(
                "Provider name",
                default=updated_name,
                validator=lambda x, names=other_names: (
                    (False, "Provider name must not be empty")
                    if not x
                    else (False, f"Provider name '{x}' already used")
                    if x in names
                    else (True, None)
                ),
                allow_back=True,
            )

            if result is None:
                return None, None
            elif result is BACK:
                # Confirm cancellation
                if confirm_action("Cancel updating this provider?", default=False):
                    return existing_providers, existing_default
                continue

            updated_name = result
            step = "provider_endpoint"

        elif step == "provider_endpoint":
            result = prompt_text_input(
                "API endpoint URL",
                default=updated_endpoint,
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
                return None, None
            elif result is BACK:
                step = "provider_name"
                continue

            updated_endpoint = result
            step = "provider_type"

        elif step == "provider_type":
            provider_type_options = {
                "openai": "OpenAI-compatible API",
            }
            result = select_with_arrows(
                provider_type_options,
                "Select provider type",
                default_key=updated_type,
                allow_back=True,
            )

            if result is None:
                return None, None
            elif result is BACK:
                step = "provider_endpoint"
                continue

            updated_type = result
            step = "provider_api_key"

        elif step == "provider_api_key":
            result = prompt_text_input(
                "API key or environment variable name (you must provide a new value)",
                default=None,
                mask=False,
                validator=lambda x: (
                    (False, "API key or environment variable name is required")
                    if not x or not x.strip()
                    else (True, None)
                ),
                allow_back=True,
            )

            if result is None:
                return None, None
            elif result is BACK:
                # For predefined providers, confirm cancellation instead of going back
                if is_predefined:
                    if confirm_action("Cancel updating this provider?", default=False):
                        return existing_providers, existing_default
                    continue
                else:
                    step = "provider_type"
                    continue

            updated_api_key = result

            # Create the updated provider
            try:
                updated_provider = ModelProvider(
                    name=updated_name,
                    endpoint=updated_endpoint,
                    provider_type=updated_type,
                    api_key=updated_api_key,
                )

                # Create new list with updated provider
                new_providers = list(existing_providers)
                new_providers[provider_index] = updated_provider

                # Handle default provider update if name changed
                default_provider = existing_default or existing_providers[0].name
                if default_provider == original_name and updated_name != original_name:
                    default_provider = updated_name
                    print_info(f"Default provider updated to: {default_provider}")

                print_success(f"Provider '{updated_name}' updated successfully")

                return new_providers, default_provider
            except Exception as e:
                print_error(f"Invalid provider configuration: {e}")
                # Ask if they want to try again
                if confirm_action("Try again?", default=True):
                    step = "provider_name"
                    updated_name = current_provider.name
                    updated_endpoint = current_provider.endpoint
                    updated_type = current_provider.provider_type
                    updated_api_key = current_provider.api_key
                    continue
                return None, None


def change_default_provider(
    existing_providers: list[ModelProvider], existing_default: str | None
) -> tuple[list[ModelProvider], str] | tuple[None, None]:
    """Change the default provider in existing configuration.

    Args:
        existing_providers: List of existing ModelProvider objects
        existing_default: Current default provider name

    Returns:
        Tuple of (providers list, new default name), or (None, None) if cancelled
    """
    if not existing_providers:
        print_error("No providers found in configuration")
        return None, None

    if len(existing_providers) == 1:
        print_info("Only one provider configured. Cannot change default.")
        return existing_providers, existing_default

    # Show current default
    current_default = existing_default or existing_providers[0].name
    console.print()
    print_info(f"Current default provider: {current_default}")
    console.print()

    # Select new default provider
    provider_options = {p.name: f"{p.name} ({p.endpoint})" for p in existing_providers}

    # Set current default as the selected option
    selected_name = select_with_arrows(
        provider_options,
        "Select new default provider",
        default_key=current_default,
        allow_back=False,
    )

    if selected_name is None:
        return None, None

    # Check if it's actually different
    if selected_name == current_default:
        print_info(f"Default provider remains: {current_default}")
        return existing_providers, existing_default

    # Update the default
    print_success(f"Default provider changed from '{current_default}' to '{selected_name}'")

    return existing_providers, selected_name


def delete_provider(
    existing_providers: list[ModelProvider], existing_default: str | None
) -> tuple[list[ModelProvider], str | None] | tuple[None, None]:
    """Delete a provider from existing configuration.

    Args:
        existing_providers: List of existing ModelProvider objects
        existing_default: Current default provider name

    Returns:
        Tuple of (updated providers list, default name), or (None, None) if cancelled
    """
    if not existing_providers:
        print_error("No providers found in configuration")
        return None, None

    # Select which provider to delete
    console.print()
    provider_options = {p.name: f"{p.name} ({p.endpoint})" for p in existing_providers}

    selected_name = select_with_arrows(
        provider_options,
        "Select a provider to delete",
        default_key=existing_providers[0].name,
        allow_back=False,
    )

    if selected_name is None:
        return None, None

    # Confirm deletion
    console.print()
    if not confirm_action(f"Delete provider '{selected_name}'?", default=False):
        print_info("Deletion cancelled")
        return existing_providers, existing_default

    # Find and remove the provider
    new_providers = [p for p in existing_providers if p.name != selected_name]

    # Handle default provider based on how many providers remain
    if len(new_providers) == 0:
        # All providers deleted
        print_success(f"Provider '{selected_name}' deleted successfully")
        return new_providers, None
    elif len(new_providers) == 1:
        # Only one provider left, it becomes the default automatically
        new_default = new_providers[0].name
        print_success(f"Provider '{selected_name}' deleted successfully")
        print_info(f"Default provider set to '{new_default}' (only remaining provider)")
        return new_providers, new_default
    else:
        # Multiple providers remain
        current_default = existing_default or ""
        if current_default == selected_name:
            # The deleted provider was the default, set first remaining as new default
            new_default = new_providers[0].name
            print_success(f"Provider '{selected_name}' deleted successfully")
            print_info(f"Default provider changed to '{new_default}' (deleted provider was default)")
            return new_providers, new_default
        else:
            # Default provider wasn't deleted, keep it
            print_success(f"Provider '{selected_name}' deleted successfully")
            return new_providers, current_default


def configure_providers(
    existing_providers: list[ModelProvider] | None = None, existing_default: str | None = None
) -> tuple[list[ModelProvider], str] | tuple[None, None]:
    """Interactive configuration for model providers with back navigation.

    Args:
        existing_providers: Optional list of existing ModelProvider objects to add to
        existing_default: Optional existing default provider name

    Returns:
        Tuple of (providers list, default name), or (None, None) if cancelled
    """
    # Step-based state machine for back navigation
    step = "provider_choice"
    providers: list[ModelProvider] = []

    # If we have existing providers, load them
    num_existing = 0
    if existing_providers:
        providers = list(existing_providers)
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
                return None, None
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
                            return existing_providers, existing_default  # Return the original unchanged
                        continue
                    else:
                        # No existing providers, confirm complete cancellation
                        if confirm_action("Cancel all provider configuration?", default=False):
                            return None, None
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
            provider_names = {p.name for p in providers}

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
                return None, None
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
            provider_names = {p.name for p in providers}

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
                return None, None
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
                return None, None
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
                return None, None
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
                return None, None
            elif result is BACK:
                # Go back based on whether this is predefined or manual
                if is_predefined:
                    step = "select_predefined"
                else:
                    step = "provider_type"
                continue

            if result:
                current_provider["api_key"] = result

            # Validate and create ModelProvider object
            try:
                provider_obj = ModelProvider.model_validate(current_provider)
                providers.append(provider_obj)
                new_providers_count += 1
                # Save to history before moving on
                history.append(("provider_api_key", current_provider.copy(), is_predefined))
                current_provider = {}
                is_predefined = False
                step = "add_another"  # Ask if they want to add another provider
            except Exception as e:
                print_error(f"Invalid provider configuration: {e}")
                return None, None

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
                return None, None
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
                return None, None
            elif len(providers) == 1:
                default_provider = providers[0].name
                step = "done"
            else:
                provider_options = {p.name: f"{p.name} ({p.endpoint})" for p in providers}
                console.print()

                # Use existing default if available and still valid
                default_key = (
                    existing_default if existing_default and existing_default in provider_options else providers[0].name
                )

                result = select_with_arrows(
                    provider_options,
                    "Select default provider",
                    default_key=default_key,
                    allow_back=True,
                )

                if result is None:
                    return None, None
                elif result is BACK:
                    # Go back to "add another" question
                    step = "add_another"
                    continue

                default_provider = result
                step = "done"

        elif step == "done":
            # Validate entire configuration
            try:
                ModelProviderRegistry(providers=providers, default=default_provider)
                return providers, default_provider
            except Exception as e:
                print_error(f"Invalid provider registry configuration: {e}")
                return None, None
