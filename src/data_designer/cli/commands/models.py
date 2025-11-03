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
    if not provider_path.exists():
        print_error("Model providers have not been configured yet!")
        print_info("Please run 'data-designer config providers' first")
        raise typer.Exit(1)

    # Load provider names
    try:
        provider_config = load_config_file(provider_path)
        available_providers = [p["name"] for p in provider_config.get("providers", [])]
        if not available_providers:
            print_error("No providers found in configuration!")
            print_info("Please run 'data-designer config providers' first")
            raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to load provider configuration: {e}")
        print_info("Please run 'data-designer config providers' first")
        raise typer.Exit(1)

    print_info(f"Configuration will be saved to: {config_dir}")
    print_navigation_tip()

    # Check for existing configuration
    model_path = get_model_config_path(config_dir)
    existing_config = None
    mode = "create"  # "create", "add", or "rewrite"

    if model_path.exists():
        try:
            existing_config = load_config_file(model_path)
        except Exception as e:
            print_warning(f"Could not load existing configuration: {e}")
            print_info("Starting with new configuration")
        else:
            # Successfully loaded existing config
            print_info(
                f"Found existing model configuration with {len(existing_config.get('model_configs', []))} model(s)"
            )
            console.print()

            # Show existing configuration
            display_config_preview(existing_config, "Current Configuration")
            console.print()

            # Ask what to do
            action_options = {
                "add": "Add more models to existing configuration",
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

    # Run the model configuration wizard
    model_config = configure_models(available_providers, existing_config if mode == "add" else None)
    if model_config is None:
        print_error("Model configuration cancelled")
        raise typer.Exit(1)

    # Check if config actually changed (when adding to existing)
    if mode == "add" and model_config == existing_config:
        print_info("No changes made to configuration")
        raise typer.Exit(0)

    # Save configuration
    try:
        ensure_config_dir_exists(config_dir)
        model_path = get_model_config_path(config_dir)
        save_config_file(model_path, model_config)
        print_success(f"Model configuration saved to: {model_path}")
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)


def configure_models(available_providers: list[str], existing_config: dict | None = None) -> dict | None:
    """Interactive configuration for model configs with back navigation.

    Args:
        available_providers: List of available provider names
        existing_config: Optional existing configuration to add to

    Returns:
        Model configuration dictionary, or None if cancelled
    """
    # Step-based state machine for back navigation
    step = "model_alias"
    model_configs: list[dict] = []

    # If we have existing config, load the models
    num_existing = 0
    if existing_config:
        model_configs = existing_config.get("model_configs", []).copy()
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
            model_aliases = {m["alias"] for m in model_configs}

            # Model alias
            result = prompt_text_input(
                "Model alias (used in your configs)",
                default="llama-3-70b" if new_models_count == 0 and num_existing == 0 else current_model.get("alias"),
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
                            return existing_config  # Return the original config unchanged
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
                default="meta/llama-3.3-70b-instruct"
                if new_models_count == 0 and num_existing == 0
                else current_model.get("model"),
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
            # Provider (optional)
            if len(available_providers) > 1:
                provider_options = {"(default)": "Use default provider"}
                provider_options.update({p: p for p in available_providers})

                result = select_with_arrows(
                    provider_options,
                    "Select provider for this model",
                    default_key="(default)",
                    allow_back=True,
                )

                if result is None:
                    return None
                elif result is BACK:
                    step = "model_id"
                    continue

                provider = None if result == "(default)" else result
            else:
                provider = None

            if provider:
                current_model["provider"] = provider
            elif "provider" in current_model:
                del current_model["provider"]

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

            # Validate using Pydantic model
            try:
                ModelConfig.model_validate(current_model)
                model_configs.append(current_model)
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

            config = {"model_configs": model_configs}
            return config
