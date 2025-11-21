# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

from .models import InferenceParameters, ModelConfig, ModelProvider
from .utils.constants import (
    MANAGED_ASSETS_PATH,
    MODEL_CONFIGS_FILE_PATH,
    MODEL_PROVIDERS_FILE_PATH,
    PREDEFINED_PROVIDERS,
    PREDEFINED_PROVIDERS_MODEL_MAP,
)
from .utils.io_helpers import load_config_file, save_config_file

logger = logging.getLogger(__name__)


def get_default_text_alias_inference_parameters() -> InferenceParameters:
    return InferenceParameters(
        temperature=0.85,
        top_p=0.95,
    )


def get_default_reasoning_alias_inference_parameters() -> InferenceParameters:
    return InferenceParameters(
        temperature=0.35,
        top_p=0.95,
    )


def get_default_vision_alias_inference_parameters() -> InferenceParameters:
    return InferenceParameters(
        temperature=0.85,
        top_p=0.95,
    )


def get_default_inference_parameters(model_alias: Literal["text", "reasoning", "vision"]) -> InferenceParameters:
    if model_alias == "reasoning":
        return get_default_reasoning_alias_inference_parameters()
    elif model_alias == "vision":
        return get_default_vision_alias_inference_parameters()
    else:
        return get_default_text_alias_inference_parameters()


def get_builtin_model_configs() -> list[ModelConfig]:
    model_configs = []
    for provider, model_alias_map in PREDEFINED_PROVIDERS_MODEL_MAP.items():
        for model_alias, model_id in model_alias_map.items():
            model_configs.append(
                ModelConfig(
                    alias=f"{provider}-{model_alias}",
                    model=model_id,
                    provider=provider,
                    inference_parameters=get_default_inference_parameters(model_alias),
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


def get_defaul_model_providers_missing_api_keys() -> list[str]:
    missing_api_keys = []
    for predefined_provider in PREDEFINED_PROVIDERS:
        if os.environ.get(predefined_provider["api_key"]) is None:
            missing_api_keys.append(predefined_provider["api_key"])
    return missing_api_keys


def get_default_providers() -> list[ModelProvider]:
    config_dict = _get_default_providers_file_content(MODEL_PROVIDERS_FILE_PATH)
    if "providers" in config_dict:
        return [ModelProvider.model_validate(p) for p in config_dict["providers"]]
    return []


def get_default_provider_name() -> Optional[str]:
    return _get_default_providers_file_content(MODEL_PROVIDERS_FILE_PATH).get("default")


def resolve_seed_default_model_settings() -> None:
    if not MODEL_CONFIGS_FILE_PATH.exists():
        logger.debug(
            f"🍾 Default model configs were not found, so writing the following to {str(MODEL_CONFIGS_FILE_PATH)!r}"
        )
        save_config_file(
            MODEL_CONFIGS_FILE_PATH, {"model_configs": [mc.model_dump() for mc in get_builtin_model_configs()]}
        )

    if not MODEL_PROVIDERS_FILE_PATH.exists():
        logger.debug(
            f"🪄  Default model providers were not found, so writing the following to {str(MODEL_PROVIDERS_FILE_PATH)!r}"
        )
        save_config_file(
            MODEL_PROVIDERS_FILE_PATH, {"providers": [p.model_dump() for p in get_builtin_model_providers()]}
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
