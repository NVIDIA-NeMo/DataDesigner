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
    console.print()

    # Run the model configuration wizard
    model_config = configure_models(available_providers)
    if model_config is None:
        print_error("Model configuration cancelled")
        raise typer.Exit(1)

    # Save configuration
    try:
        ensure_config_dir_exists(config_dir)
        model_path = get_model_config_path(config_dir)
        save_config_file(model_path, model_config)
        print_success(f"Model configuration saved to: {model_path}")
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)


def configure_models(available_providers: list[str]) -> dict | None:
    """Interactive configuration for model configs with back navigation.

    Args:
        available_providers: List of available provider names

    Returns:
        Model configuration dictionary, or None if cancelled
    """
    # Step-based state machine for back navigation
    step = "num_models"
    num_models = 1
    model_configs: list[dict] = []
    model_idx = 0

    # Storage for model data as we build it
    current_model: dict = {}

    while True:
        if step == "num_models":
            # Ask how many models
            result = prompt_text_input(
                "How many models do you want to configure? (1-10)",
                default="1",
                validator=lambda x: validate_positive_int(x) if x else (True, None),
                allow_back=False,  # First step, can't go back
            )

            if result is None:
                return None

            num_models = int(result) if result else 1
            num_models = min(max(num_models, 1), 10)  # Clamp to 1-10
            model_idx = 0
            model_configs = []
            step = "model_alias"

        elif step == "model_alias":
            if model_idx >= num_models:
                # Done with all models
                step = "done"
                continue

            console.print()
            print_info(f"Configuring model {model_idx + 1}/{num_models}")

            # Get existing model aliases for validation
            model_aliases = {m["alias"] for m in model_configs}

            # Model alias
            result = prompt_text_input(
                "Model alias (used in your configs)",
                default="llama-3-70b" if model_idx == 0 else current_model.get("alias"),
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
                # Go back to previous model or num_models
                if model_idx > 0 and len(model_configs) > 0:
                    # Go back to previous model
                    model_idx -= 1
                    current_model = model_configs.pop()
                    step = "model_max_tokens"  # Go to last step of previous model
                    continue
                else:
                    # First model, go back to num_models question
                    step = "num_models"
                    continue

            current_model = {"alias": result}
            step = "model_id"

        elif step == "model_id":
            # Model ID
            result = prompt_text_input(
                "Model ID (e.g., meta/llama-3.3-70b-instruct)",
                default="meta/llama-3.3-70b-instruct" if model_idx == 0 else current_model.get("model"),
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
                f"Temperature ({MIN_TEMPERATURE}-{MAX_TEMPERATURE})",
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
                f"Top P ({MIN_TOP_P}-{MAX_TOP_P})",
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
                "Max tokens (press Enter for default: 2048)",
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
                model_idx += 1
                current_model = {}
                step = "model_alias"  # Move to next model or finish
            except Exception as e:
                print_error(f"Invalid model configuration: {e}")
                return None

        elif step == "done":
            config = {"model_configs": model_configs}
            return config
