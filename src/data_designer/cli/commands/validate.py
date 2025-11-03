# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import typer

from data_designer.cli.ui import console, print_error, print_header, print_info, print_success, print_warning
from data_designer.cli.utils import get_default_config_dir, get_model_config_path, get_model_provider_path
from data_designer.config.errors import InvalidConfigError, InvalidFileFormatError, InvalidFilePathError
from data_designer.config.models import ModelConfig
from data_designer.engine.model_provider import ModelProviderRegistry


def validate_command(
    config_dir: str | None = typer.Option(None, "--config-dir", help="Custom configuration directory"),
) -> None:
    """Validate Data Designer configuration files."""
    # Determine config directory
    if config_dir:
        config_path = Path(config_dir).expanduser().resolve()
    else:
        config_path = get_default_config_dir()

    print_header("Configuration Validation")
    print_info(f"Validating configurations in: {config_path}")
    console.print()

    # Validate providers
    provider_path = get_model_provider_path(config_path)
    provider_valid = validate_providers(provider_path)

    # Validate models
    model_path = get_model_config_path(config_path)
    model_valid = validate_models(model_path)

    # Summary
    console.print()
    print_header("Validation Summary")

    if provider_valid and model_valid:
        print_success("All configurations are valid!")
        raise typer.Exit(0)
    elif not provider_valid and not model_valid:
        print_error("Both provider and model configurations are invalid")
        raise typer.Exit(1)
    elif not provider_valid:
        print_error("Provider configuration is invalid")
        raise typer.Exit(1)
    else:
        print_error("Model configuration is invalid")
        raise typer.Exit(1)


def validate_providers(provider_path: Path) -> bool:
    """Validate model provider configuration.

    Args:
        provider_path: Path to provider configuration file

    Returns:
        True if valid, False otherwise
    """
    print_info(f"Validating providers: {provider_path}")

    try:
        from data_designer.cli.utils import load_config_file

        config = load_config_file(provider_path)

        # Validate with Pydantic
        registry = ModelProviderRegistry.model_validate(config)

        # Additional checks
        provider_count = len(registry.providers)
        default_name = registry.default or registry.providers[0].name

        print_success("Provider configuration is valid")
        print_info(f"  - {provider_count} provider(s) configured")
        print_info(f"  - Default provider: {default_name}")

        # Check for API keys
        providers_without_keys = [p.name for p in registry.providers if not p.api_key]
        if providers_without_keys:
            print_warning(f"  - Providers without API keys: {', '.join(providers_without_keys)}")
            print_info("    (API keys can be set via environment variables)")

        console.print()
        return True

    except InvalidFilePathError:
        print_error("Provider configuration file not found")
        print_info("Run 'data-designer init' to create it")
        console.print()
        return False

    except InvalidFileFormatError as e:
        print_error(f"Invalid YAML format: {e}")
        console.print()
        return False

    except InvalidConfigError as e:
        print_error(f"Invalid configuration: {e}")
        console.print()
        return False

    except ValueError as e:
        print_error(f"Validation error: {e}")
        console.print()
        return False

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        console.print()
        return False


def validate_models(model_path: Path) -> bool:
    """Validate model configuration.

    Args:
        model_path: Path to model configuration file

    Returns:
        True if valid, False otherwise
    """
    print_info(f"Validating models: {model_path}")

    try:
        from data_designer.cli.utils import load_config_file

        config = load_config_file(model_path)

        # Check for model_configs key
        if "model_configs" not in config:
            raise InvalidConfigError("Missing 'model_configs' key in configuration")

        if not config["model_configs"]:
            raise InvalidConfigError("'model_configs' list is empty")

        # Validate each model config
        model_configs = [ModelConfig.model_validate(mc) for mc in config["model_configs"]]

        # Check for duplicate aliases
        aliases = [mc.alias for mc in model_configs]
        duplicates = [alias for alias in aliases if aliases.count(alias) > 1]
        if duplicates:
            raise InvalidConfigError(f"Duplicate model aliases found: {set(duplicates)}")

        print_success("Model configuration is valid")
        print_info(f"  - {len(model_configs)} model(s) configured")
        print_info(f"  - Aliases: {', '.join(mc.alias for mc in model_configs)}")

        # Check for custom providers
        custom_providers = [mc.provider for mc in model_configs if mc.provider]
        if custom_providers:
            print_info(f"  - Models with custom providers: {len(set(custom_providers))}")

        # Temperature/Top-P warnings
        for mc in model_configs:
            params = mc.inference_parameters

            # Check for distribution-based parameters
            if hasattr(params.temperature, "sample"):
                print_info(f"  - {mc.alias}: uses distribution-based temperature")

            if hasattr(params.top_p, "sample"):
                print_info(f"  - {mc.alias}: uses distribution-based top_p")

            # Check for extreme values
            if isinstance(params.temperature, (int, float)):
                if params.temperature < 0.1:
                    print_warning(f"  - {mc.alias}: very low temperature ({params.temperature})")
                elif params.temperature > 1.5:
                    print_warning(f"  - {mc.alias}: very high temperature ({params.temperature})")

        console.print()
        return True

    except InvalidFilePathError:
        print_error("Model configuration file not found")
        print_info("Run 'data-designer init' to create it")
        console.print()
        return False

    except InvalidFileFormatError as e:
        print_error(f"Invalid YAML format: {e}")
        console.print()
        return False

    except InvalidConfigError as e:
        print_error(f"Invalid configuration: {e}")
        console.print()
        return False

    except ValueError as e:
        print_error(f"Validation error: {e}")
        console.print()
        return False

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        console.print()
        return False
