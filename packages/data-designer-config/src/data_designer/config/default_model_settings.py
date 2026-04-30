# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from data_designer.config.models import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    InferenceParamsT,
    ModelConfig,
    ModelProvider,
)
from data_designer.config.utils.constants import (
    MANAGED_ASSETS_PATH,
    MODEL_CONFIGS_FILE_PATH,
    MODEL_PROVIDERS_FILE_PATH,
    PREDEFINED_PROVIDERS,
    PREDEFINED_PROVIDERS_MODEL_MAP,
)
from data_designer.config.utils.io_helpers import load_config_file, save_config_file

logger = logging.getLogger(__name__)


def get_default_inference_parameters(
    model_alias: Literal["text", "reasoning", "vision", "embedding"],
    inference_parameters: dict[str, Any],
) -> InferenceParamsT:
    if model_alias == "reasoning":
        return ChatCompletionInferenceParams(**inference_parameters)
    elif model_alias == "vision":
        return ChatCompletionInferenceParams(**inference_parameters)
    elif model_alias == "embedding":
        return EmbeddingInferenceParams(**inference_parameters)
    else:
        return ChatCompletionInferenceParams(**inference_parameters)


def get_builtin_model_configs() -> list[ModelConfig]:
    model_configs = []
    for provider, model_alias_map in PREDEFINED_PROVIDERS_MODEL_MAP.items():
        for model_alias, settings in model_alias_map.items():
            model_configs.append(
                ModelConfig(
                    alias=f"{provider}-{model_alias}",
                    model=settings["model"],
                    provider=provider,
                    inference_parameters=get_default_inference_parameters(
                        model_alias, settings["inference_parameters"]
                    ),
                )
            )
    return model_configs


def get_builtin_model_providers() -> list[ModelProvider]:
    return [ModelProvider.model_validate(provider) for provider in PREDEFINED_PROVIDERS]


def get_default_model_configs() -> list[ModelConfig]:
    if MODEL_CONFIGS_FILE_PATH.exists():
        config_dict = load_config_file(MODEL_CONFIGS_FILE_PATH)
        if "model_configs" in config_dict:
            return [ModelConfig.model_validate(mc) for mc in config_dict["model_configs"]]
    return []


def get_providers_with_missing_api_keys(providers: list[ModelProvider]) -> list[ModelProvider]:
    providers_with_missing_keys = []

    for provider in providers:
        if provider.api_key is None:
            # No API key specified at all
            providers_with_missing_keys.append(provider)
        elif provider.api_key.isupper() and "_" in provider.api_key:
            # Looks like an environment variable name, check if it's set
            if os.environ.get(provider.api_key) is None:
                providers_with_missing_keys.append(provider)
        # else: It's an actual API key value (not an env var), so it's valid

    return providers_with_missing_keys


def get_default_providers() -> list[ModelProvider]:
    config_dict = _get_default_providers_file_content(MODEL_PROVIDERS_FILE_PATH)
    if "providers" in config_dict:
        return [ModelProvider.model_validate(p) for p in config_dict["providers"]]
    return []


def get_default_provider_name() -> str | None:
    """Return the YAML's ``default:`` provider name, if set.

    Deprecated: this function and the underlying YAML key are deprecated and
    will be removed in a future release. Specify ``provider=`` explicitly on
    each ``ModelConfig`` instead. See issue #589.
    """
    default = _get_default_providers_file_content(MODEL_PROVIDERS_FILE_PATH).get("default")
    if default is not None:
        warnings.warn(
            f"The 'default:' key in {MODEL_PROVIDERS_FILE_PATH} is deprecated and will "
            "be removed in a future release. Remove it and specify provider= explicitly "
            "on each ModelConfig instead. See issue #589.",
            DeprecationWarning,
            stacklevel=2,
        )
    return default


def resolve_seed_default_model_settings() -> None:
    if not MODEL_CONFIGS_FILE_PATH.exists():
        logger.debug(
            f"🍾 Default model configs were not found, so writing the following to {str(MODEL_CONFIGS_FILE_PATH)!r}"
        )
        save_config_file(
            MODEL_CONFIGS_FILE_PATH,
            {"model_configs": [mc.model_dump(mode="json") for mc in get_builtin_model_configs()]},
        )

    if not MODEL_PROVIDERS_FILE_PATH.exists():
        logger.debug(
            f"🪄  Default model providers were not found, so writing the following to {str(MODEL_PROVIDERS_FILE_PATH)!r}"
        )
        save_config_file(
            MODEL_PROVIDERS_FILE_PATH, {"providers": [p.model_dump(mode="json") for p in get_builtin_model_providers()]}
        )

    if not MANAGED_ASSETS_PATH.exists():
        logger.debug(f"🏗️ Default managed assets path was not found, so creating it at {str(MANAGED_ASSETS_PATH)!r}")
        MANAGED_ASSETS_PATH.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _get_default_providers_file_content(file_path: Path) -> dict[str, Any]:
    """Load and cache the default providers file content."""
    if file_path.exists():
        return load_config_file(file_path)
    raise FileNotFoundError(f"Default model providers file not found at {str(file_path)!r}")
