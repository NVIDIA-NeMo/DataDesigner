# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.interactive import (
    BACK,
    confirm_action,
    console,
    display_config_preview,
    display_welcome,
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
    save_config_file,
    validate_numeric_range,
    validate_positive_int,
    validate_url,
)
from data_designer.config.models import ModelConfig
from data_designer.config.utils.constants import MAX_TEMPERATURE, MAX_TOP_P, MIN_TEMPERATURE, MIN_TOP_P
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry


def init_command(
    provider_only: bool = typer.Option(False, "--provider-only", help="Only configure model providers"),
    model_only: bool = typer.Option(False, "--model-only", help="Only configure models"),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Custom output directory"),
) -> None:
    """Initialize Data Designer configuration files interactively."""
    # Display welcome message
    display_welcome()

    # Determine what to configure
    configure_providers = not model_only
    configure_models = not provider_only

    # Determine output directory
    if output_dir:
        config_dir = Path(output_dir).expanduser().resolve()
    else:
        config_dir = get_default_config_dir()

    print_info(f"Configuration will be saved to: {config_dir}")
    console.print()

    # Initialize configuration dictionaries
    provider_config = None
    model_config = None

    # Configure providers
    if configure_providers:
        provider_config = configure_model_providers()
        if provider_config is None:
            print_error("Provider configuration cancelled")
            raise typer.Exit(1)

    # Configure models
    if configure_models:
        # Get available providers for model configuration
        if provider_config:
            provider_names = [p["name"] for p in provider_config["providers"]]
        else:
            # Try to load existing providers
            provider_path = get_model_provider_path(config_dir)
            if provider_path.exists():
                from data_designer.cli.utils import load_config_file

                try:
                    existing_config = load_config_file(provider_path)
                    provider_names = [p["name"] for p in existing_config.get("providers", [])]
                except Exception:
                    provider_names = ["nvidia"]  # Default fallback
            else:
                provider_names = ["nvidia"]  # Default fallback

        model_config = configure_model_configs(provider_names)
        if model_config is None:
            print_error("Model configuration cancelled")
            raise typer.Exit(1)

    # Preview configurations
    console.print()
    print_header("Configuration Preview")

    if provider_config:
        display_config_preview(provider_config, "Model Providers")

    if model_config:
        display_config_preview(model_config, "Model Configs")

    # Confirm save
    console.print()
    if not confirm_action("Save these configurations?", default=True):
        print_info("Configuration not saved")
        raise typer.Exit(0)

    # Save configurations
    try:
        ensure_config_dir_exists(config_dir)

        if provider_config:
            provider_path = get_model_provider_path(config_dir)
            save_config_file(provider_path, provider_config)
            print_success(f"Model providers saved to: {provider_path}")

        if model_config:
            model_path = get_model_config_path(config_dir)
            save_config_file(model_path, model_config)
            print_success(f"Model configs saved to: {model_path}")

        console.print()
        print_success("Configuration complete!")
        print_info("You can now use these configurations in your Data Designer projects")

    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)


def configure_model_providers() -> dict | None:
    """Interactive configuration for model providers with back navigation.

    Returns:
        Provider configuration dictionary, or None if cancelled
    """
    print_header("Configure Model Providers")

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
            common_provider_names = ["nvidia", "openai", "anthropic", "together", "replicate", "local"]
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


def configure_model_configs(available_providers: list[str]) -> dict | None:
    """Interactive configuration for model configs.

    Args:
        available_providers: List of available provider names

    Returns:
        Model configuration dictionary, or None if cancelled
    """
    print_header("Configure Models")

    # Ask how many models
    num_models_str = prompt_text_input(
        "How many models do you want to configure? (1-10)",
        default="1",
        validator=lambda x: validate_positive_int(x) if x else (True, None),
    )

    if num_models_str is None:
        return None

    num_models = int(num_models_str) if num_models_str else 1
    num_models = min(max(num_models, 1), 10)  # Clamp to 1-10

    model_configs = []
    model_aliases = set()

    for i in range(num_models):
        console.print()
        print_info(f"Configuring model {i + 1}/{num_models}")

        # Model alias
        alias = prompt_text_input(
            "Model alias (used in your configs)",
            default="llama-3-70b" if i == 0 else None,
            validator=lambda x, aliases=model_aliases: (
                (False, "Model alias must not be empty")
                if not x
                else (False, f"Model alias '{x}' already used")
                if x in aliases
                else (True, None)
            ),
        )
        if alias is None:
            return None
        model_aliases.add(alias)

        # Model ID
        model_id = prompt_text_input(
            "Model ID (e.g., meta/llama-3.3-70b-instruct)",
            default="meta/llama-3.3-70b-instruct" if i == 0 else None,
            validator=lambda x: (False, "Model ID is required") if not x else (True, None),
        )
        if model_id is None:
            return None

        # Provider (optional)
        if len(available_providers) > 1:
            provider_options = {"(default)": "Use default provider"}
            provider_options.update({p: p for p in available_providers})

            selected_provider = select_with_arrows(
                provider_options,
                "Select provider for this model",
                default_key="(default)",
            )
            if selected_provider is None:
                return None

            provider = None if selected_provider == "(default)" else selected_provider
        else:
            provider = None

        # Temperature
        console.print()
        print_info("Inference Parameters")

        temperature_str = prompt_text_input(
            f"Temperature ({MIN_TEMPERATURE}-{MAX_TEMPERATURE})",
            default="0.7",
            validator=lambda x: validate_numeric_range(x, MIN_TEMPERATURE, MAX_TEMPERATURE) if x else (True, None),
        )
        if temperature_str is None:
            return None
        temperature = float(temperature_str) if temperature_str else 0.7

        # Top P
        top_p_str = prompt_text_input(
            f"Top P ({MIN_TOP_P}-{MAX_TOP_P})",
            default="0.9",
            validator=lambda x: validate_numeric_range(x, MIN_TOP_P, MAX_TOP_P) if x else (True, None),
        )
        if top_p_str is None:
            return None
        top_p = float(top_p_str) if top_p_str else 0.9

        # Max tokens
        max_tokens_str = prompt_text_input(
            "Max tokens (press Enter for default: 2048)",
            default="2048",
            validator=lambda x: validate_positive_int(x) if x else (True, None),
        )
        if max_tokens_str is None:
            return None
        max_tokens = int(max_tokens_str) if max_tokens_str else 2048

        # Build model config
        model_config = {
            "alias": alias,
            "model": model_id,
            "inference_parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "max_parallel_requests": 4,
            },
        }

        if provider:
            model_config["provider"] = provider

        # Validate using Pydantic model
        try:
            ModelConfig.model_validate(model_config)
            model_configs.append(model_config)
        except Exception as e:
            print_error(f"Invalid model configuration: {e}")
            return None

    config = {"model_configs": model_configs}
    return config
