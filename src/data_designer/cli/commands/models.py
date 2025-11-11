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
    get_model_config_path,
    get_model_provider_path,
    load_config_file,
    save_config_file,
    validate_numeric_range,
    validate_positive_int,
)
from data_designer.config.models import ModelConfig
from data_designer.config.utils.constants import MAX_TEMPERATURE, MAX_TOP_P, MIN_TEMPERATURE, MIN_TOP_P


def models_command(
    output_dir: str | None = typer.Option(None, "--output-dir", help="Custom output directory"),
) -> None:
    """Configure model configs interactively."""
    print_header("Configure Models")

    # Determine output directory
    if output_dir:
        config_dir = Path(output_dir).expanduser().resolve()
    else:
        config_dir = get_default_config_dir()

    # Check if providers are configured
    provider_path = get_model_provider_path(config_dir)
    model_path = get_model_config_path(config_dir)

    # Built-in providers that can be used without explicit configuration
    BUILTIN_PROVIDERS = ["nvidia", "openai"]
    available_providers: list[str] = []

    if not provider_path.exists():
        # No providers configured - check if models are also not configured
        if not model_path.exists():
            print_warning("Model providers have not been configured yet!")
            console.print()
            print_info(
                "You can configure models using built-in providers (nvidia, openai) without setting up providers first."
            )
            print_info("For custom providers, you need to configure providers first.")
            console.print()

            # Ask what they want to do
            choice_options = {
                "builtin": "Continue with built-in providers (nvidia, openai)",
                "setup_providers": "Set up custom providers first",
                "exit": "Exit",
            }
            choice = select_with_arrows(
                choice_options,
                "What would you like to do?",
                default_key="builtin",
                allow_back=False,
            )

            if choice is None or choice == "exit":
                print_info("No changes made to configuration")
                raise typer.Exit(0)
            elif choice == "setup_providers":
                # Import here to avoid circular dependency
                from data_designer.cli.commands.providers import providers_command

                console.print()
                # Run providers configuration with same output_dir
                providers_command(output_dir=output_dir)
                console.print()
                print_success("Provider configuration complete!")
                console.print()
                print_info("Now continuing with model configuration...")
                console.print()

                # Reload provider config after setup
                try:
                    provider_config = load_config_file(provider_path)
                    available_providers = [p["name"] for p in provider_config.get("providers", [])]
                except Exception:
                    pass
            else:  # builtin
                print_info("Continuing with built-in providers...")
                console.print()
                available_providers = BUILTIN_PROVIDERS.copy()
        else:
            print_error("Model providers have not been configured yet!")
            print_info("You can only use built-in providers (nvidia, openai) without provider configuration.")
            print_info("For custom providers, please run 'data-designer config providers' first")
            raise typer.Exit(1)
    else:
        # Load configured providers
        try:
            provider_config = load_config_file(provider_path)
            configured_providers = [p["name"] for p in provider_config.get("providers", [])]
            if not configured_providers:
                print_warning("No providers found in configuration!")
                available_providers = BUILTIN_PROVIDERS.copy()
            else:
                # Combine configured providers with built-in ones (avoid duplicates)
                available_providers = list(set(configured_providers + BUILTIN_PROVIDERS))
        except Exception as e:
            print_error(f"Failed to load provider configuration: {e}")
            print_info("Falling back to built-in providers only")
            available_providers = BUILTIN_PROVIDERS.copy()

    if not available_providers:
        print_error("No providers available!")
        print_info("Please run 'data-designer config providers' first")
        raise typer.Exit(1)

    print_info(f"Configuration will be saved to: {config_dir}")
    print_navigation_tip()

    # Check for existing configuration
    model_path = get_model_config_path(config_dir)
    existing_models: list[ModelConfig] = []
    mode = "create"  # "create", "add", or "rewrite"

    if model_path.exists():
        try:
            config_dict = load_config_file(model_path)
            existing_models = [ModelConfig.model_validate(m) for m in config_dict.get("model_configs", [])]
        except Exception as e:
            print_warning(f"Could not load existing configuration: {e}")
            print_info("Starting with new configuration")
        else:
            # Successfully loaded existing config
            print_info(f"Found existing model configuration with {len(existing_models)} model(s)")
            console.print()

            # Show existing configuration
            display_config_preview(config_dict, "Current Configuration")
            console.print()

            # Ask what to do
            action_options = {
                "add": "Add more models to existing configuration",
                "update": "Update an existing model configuration",
                "delete": "Delete a model configuration",
            }

            # Only show these options if there are models
            num_models = len(existing_models)
            if num_models > 0:
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
            elif action == "delete":
                mode = "delete"
            elif action == "rewrite":
                mode = "rewrite"

    # Run the model configuration wizard
    if mode == "update":
        updated_models = update_model(existing_models)
    elif mode == "delete":
        updated_models = delete_model(existing_models)
    else:
        updated_models = configure_models(available_providers, existing_models if mode == "add" else None)

    if updated_models is None:
        print_error("Model configuration cancelled")
        raise typer.Exit(1)

    # Check if config actually changed
    if mode in ("add", "update", "delete") and updated_models == existing_models:
        print_info("No changes made to configuration")
        raise typer.Exit(0)

    # Handle empty model list (all models deleted)
    if len(updated_models) == 0:
        try:
            if model_path.exists():
                model_path.unlink()
                print_success("All models deleted. Configuration file removed.")
            else:
                print_info("Configuration file already removed.")
        except Exception as e:
            print_error(f"Failed to remove configuration file: {e}")
            raise typer.Exit(1)
        raise typer.Exit(0)

    # Save configuration
    try:
        ensure_config_dir_exists(config_dir)
        model_path = get_model_config_path(config_dir)
        # Convert ModelConfig objects to dict format for saving
        config_dict = {"model_configs": [m.model_dump(mode="json", exclude_none=True) for m in updated_models]}
        save_config_file(model_path, config_dict)
        print_success(f"Model configuration saved to: {model_path}")
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)


def update_model(existing_models: list[ModelConfig]) -> list[ModelConfig] | None:
    """Update an existing model configuration.

    Args:
        existing_models: List of existing ModelConfig objects

    Returns:
        Updated list of ModelConfig objects, or None if cancelled
    """
    if not existing_models:
        print_error("No models found in configuration")
        return None

    # Select which model to update
    console.print()
    model_options = {m.alias: f"{m.alias} ({m.model})" for m in existing_models}

    selected_alias = select_with_arrows(
        model_options,
        "Select a model to update",
        default_key=existing_models[0].alias,
        allow_back=False,
    )

    if selected_alias is None:
        return None

    # Find the model to update
    model_index = next(i for i, m in enumerate(existing_models) if m.alias == selected_alias)
    current_model = existing_models[model_index]
    original_alias = current_model.alias

    print_info(f"Updating model: {original_alias}")
    console.print()

    # Get other model aliases for validation (excluding current)
    other_aliases = {m.alias for i, m in enumerate(existing_models) if i != model_index}

    # Check if model has distribution-based inference parameters (not supported in CLI)
    current_params = current_model.inference_parameters
    if current_params:
        if hasattr(current_params.temperature, "sample") or hasattr(current_params.top_p, "sample"):
            print_error(
                "This model uses distribution-based inference parameters, "
                "which cannot be edited via the CLI. Please edit the configuration file directly."
            )
            return existing_models

    # Step-based state machine for back navigation
    step = "alias"

    # Extract current inference parameters (only float/int values supported)
    if current_params:
        temp = current_params.temperature if current_params.temperature is not None else 0.7
        top_p = current_params.top_p if current_params.top_p is not None else 0.9
        max_tokens = current_params.max_tokens if current_params.max_tokens is not None else 2048
        max_parallel_requests = current_params.max_parallel_requests
    else:
        temp = 0.7
        top_p = 0.9
        max_tokens = 2048
        max_parallel_requests = 4

    updated_data = {
        "alias": current_model.alias,
        "model": current_model.model,
        "provider": current_model.provider,
        "inference_parameters": {
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_parallel_requests": max_parallel_requests,
        },
    }

    while True:
        if step == "alias":
            result = prompt_text_input(
                "Model alias (used in your configs)",
                default=updated_data["alias"],
                validator=lambda x, aliases=other_aliases: (
                    (False, "Model alias must not be empty")
                    if not x
                    else (False, f"Model alias '{x}' already used")
                    if x in aliases
                    else (True, None)
                ),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                if confirm_action("Cancel updating this model?", default=False):
                    return existing_models
                continue

            updated_data["alias"] = result
            step = "model_id"

        elif step == "model_id":
            result = prompt_text_input(
                "Model ID",
                default=updated_data["model"],
                validator=lambda x: (False, "Model ID is required") if not x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "alias"
                continue

            updated_data["model"] = result
            step = "temperature"

        elif step == "temperature":
            console.print()
            print_info("Inference Parameters")

            current_temp = updated_data["inference_parameters"]["temperature"]
            result = prompt_text_input(
                f"Temperature <dim>({MIN_TEMPERATURE}-{MAX_TEMPERATURE})</dim>",
                default=str(current_temp),
                validator=lambda x: validate_numeric_range(x, MIN_TEMPERATURE, MAX_TEMPERATURE) if x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "model_id"
                continue

            updated_data["inference_parameters"]["temperature"] = float(result) if result else current_temp
            step = "top_p"

        elif step == "top_p":
            current_top_p = updated_data["inference_parameters"]["top_p"]
            result = prompt_text_input(
                f"Top P <dim>({MIN_TOP_P}-{MAX_TOP_P})</dim>",
                default=str(current_top_p),
                validator=lambda x: validate_numeric_range(x, MIN_TOP_P, MAX_TOP_P) if x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "temperature"
                continue

            updated_data["inference_parameters"]["top_p"] = float(result) if result else current_top_p
            step = "max_tokens"

        elif step == "max_tokens":
            current_max_tokens = updated_data["inference_parameters"]["max_tokens"]
            result = prompt_text_input(
                "Max tokens",
                default=str(current_max_tokens),
                validator=lambda x: validate_positive_int(x) if x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "top_p"
                continue

            updated_data["inference_parameters"]["max_tokens"] = int(result) if result else current_max_tokens
            step = "done"

        elif step == "done":
            # Create updated ModelConfig
            try:
                updated_model = ModelConfig(
                    alias=updated_data["alias"],
                    model=updated_data["model"],
                    provider=updated_data["provider"],
                    inference_parameters=updated_data["inference_parameters"],
                )

                new_models = list(existing_models)
                new_models[model_index] = updated_model
                print_success(f"Model '{updated_data['alias']}' updated successfully")

                return new_models
            except Exception as e:
                print_error(f"Invalid model configuration: {e}")
                if confirm_action("Try again?", default=True):
                    step = "alias"
                    continue
                return None


def delete_model(existing_models: list[ModelConfig]) -> list[ModelConfig] | None:
    """Delete a model from existing configuration.

    Args:
        existing_models: List of existing ModelConfig objects

    Returns:
        Updated list of ModelConfig objects, or None if cancelled
    """
    if not existing_models:
        print_error("No models found in configuration")
        return None

    # Select which model to delete
    console.print()
    model_options = {m.alias: f"{m.alias} ({m.model})" for m in existing_models}

    selected_alias = select_with_arrows(
        model_options,
        "Select a model to delete",
        default_key=existing_models[0].alias,
        allow_back=False,
    )

    if selected_alias is None:
        return None

    # Confirm deletion
    console.print()
    if not confirm_action(f"Delete model '{selected_alias}'?", default=False):
        print_info("Deletion cancelled")
        return existing_models

    # Find and remove the model
    new_models = [m for m in existing_models if m.alias != selected_alias]

    print_success(f"Model '{selected_alias}' deleted successfully")

    if len(new_models) == 0:
        print_info("All models deleted.")
        return []

    return new_models


def configure_models(
    available_providers: list[str], existing_models: list[ModelConfig] | None = None
) -> list[ModelConfig] | None:
    """Interactive configuration for model configs with back navigation.

    Args:
        available_providers: List of available provider names
        existing_models: Optional existing list of ModelConfig objects to add to

    Returns:
        List of ModelConfig objects, or None if cancelled
    """
    # Step-based state machine for back navigation
    step = "model_alias"
    model_configs: list[ModelConfig] = []

    # If we have existing models, load them
    num_existing = 0
    if existing_models:
        model_configs = list(existing_models)
        num_existing = len(model_configs)
        print_info(f"Adding to existing {num_existing} model(s)")
        console.print()

    # Track how many NEW models we've added (for numbering and back navigation)
    new_models_count = 0

    # Storage for model data as we build it
    current_model: dict = {}

    # History stack for proper back navigation
    # Each entry: (step_name, model_data)
    history: list[tuple[str, dict]] = []

    while True:
        if step == "model_alias":
            console.print()
            total_count = num_existing + new_models_count + 1
            print_info(f"Configuring model {total_count}")

            # Get existing model aliases for validation
            model_aliases = {m.alias for m in model_configs}

            # Model alias
            result = prompt_text_input(
                "Model alias (used in your configs)",
                default=current_model.get("alias") if current_model.get("alias") else None,
                validator=lambda x, aliases=model_aliases: (
                    (False, "Model alias must not be empty")
                    if not x
                    else (False, f"Model alias '{x}' already used")
                    if x in aliases
                    else (True, None)
                ),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                # Go back using history if available
                if history:
                    # Pop from history and restore state to edit a completed model
                    prev_step, prev_model = history.pop()
                    # Remove the last model from the list so we can re-add it after editing
                    if len(model_configs) > num_existing:
                        model_configs.pop()
                        new_models_count -= 1
                    current_model = prev_model
                    step = prev_step
                    continue
                else:
                    # No history - check if there are existing models
                    if num_existing > 0:
                        # There are existing models, so canceling is OK
                        if confirm_action("Discard the new models?", default=False):
                            return existing_models  # Return the original models unchanged
                        continue
                    else:
                        # No existing models, confirm complete cancellation
                        if confirm_action("Cancel all model configuration?", default=False):
                            return None
                        continue

            current_model = {"alias": result}
            step = "model_id"

        elif step == "model_id":
            # Model ID
            result = prompt_text_input(
                "Model ID",
                default=current_model.get("model") if current_model.get("model") else None,
                validator=lambda x: (False, "Model ID is required") if not x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "model_alias"
                continue

            current_model["model"] = result
            step = "model_provider"

        elif step == "model_provider":
            # Provider (required) - must match against configured providers
            if len(available_providers) == 1:
                # Only one provider available, use it automatically
                provider = available_providers[0]
                print_info(f"Using provider: {provider}")
                current_model["provider"] = provider
                step = "model_temperature"
            else:
                # Multiple providers - let user select (required)
                provider_options = {p: p for p in available_providers}

                result = select_with_arrows(
                    provider_options,
                    "Select provider for this model (required)",
                    default_key=available_providers[0],
                    allow_back=True,
                )

                if result is None:
                    return None
                elif result is BACK:
                    step = "model_id"
                    continue

                current_model["provider"] = result
                step = "model_temperature"

        elif step == "model_temperature":
            # Temperature
            console.print()
            print_info("Inference Parameters")

            result = prompt_text_input(
                f"Temperature <dim>({MIN_TEMPERATURE}-{MAX_TEMPERATURE})</dim>",
                default="0.7",
                validator=lambda x: validate_numeric_range(x, MIN_TEMPERATURE, MAX_TEMPERATURE) if x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "model_provider"
                continue

            temperature = float(result) if result else 0.7
            current_model.setdefault("inference_parameters", {})["temperature"] = temperature
            step = "model_top_p"

        elif step == "model_top_p":
            # Top P
            result = prompt_text_input(
                f"Top P <dim>({MIN_TOP_P}-{MAX_TOP_P})</dim>",
                default="0.9",
                validator=lambda x: validate_numeric_range(x, MIN_TOP_P, MAX_TOP_P) if x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "model_temperature"
                continue

            top_p = float(result) if result else 0.9
            current_model["inference_parameters"]["top_p"] = top_p
            step = "model_max_tokens"

        elif step == "model_max_tokens":
            # Max tokens
            result = prompt_text_input(
                "Max tokens",
                default="2048",
                validator=lambda x: validate_positive_int(x) if x else (True, None),
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                step = "model_top_p"
                continue

            max_tokens = int(result) if result else 2048
            current_model["inference_parameters"]["max_tokens"] = max_tokens
            current_model["inference_parameters"]["max_parallel_requests"] = 4

            # Validate using Pydantic model and create ModelConfig object
            try:
                model_obj = ModelConfig.model_validate(current_model)
                model_configs.append(model_obj)
                new_models_count += 1
                # Save to history before moving on
                history.append(("model_max_tokens", current_model.copy()))
                current_model = {}
                step = "add_another"  # Ask if they want to add another model
            except Exception as e:
                print_error(f"Invalid model configuration: {e}")
                return None

        elif step == "add_another":
            # Ask if user wants to add another model
            console.print()

            # Create options for selection with back support
            add_another_options = {
                "yes": "Add another model",
                "no": "Finish configuring models",
            }
            result = select_with_arrows(
                add_another_options,
                "Would you like to add another model?",
                default_key="no",
                allow_back=True,
            )

            if result is None:
                return None
            elif result is BACK:
                # Go back to the last model's max_tokens step
                if history:
                    prev_step, prev_model = history.pop()
                    # Remove the last model so we can re-add it after editing
                    if len(model_configs) > num_existing:
                        model_configs.pop()
                        new_models_count -= 1
                    current_model = prev_model
                    step = "model_max_tokens"
                continue
            elif result == "yes":
                step = "model_alias"
            else:  # "no"
                step = "done"

        elif step == "done":
            if len(model_configs) == 0:
                print_error("No models configured")
                return None

            return model_configs
