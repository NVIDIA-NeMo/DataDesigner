# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Literal, Optional

from ..cli.utils import load_config_file, save_config_file
from .models import InferenceParameters, ModelConfig, ModelProvider
from .utils.constants import (
    MODEL_CONFIGS_FILE_PATH,
    MODEL_PROVIDERS_FILE_PATH,
    NVIDIA_API_KEY_ENV_VAR_NAME,
    OPENAI_API_KEY_ENV_VAR_NAME,
    PREDEFINED_PROVIDERS,
    PREDEFINED_PROVIDERS_MODEL_MAP,
)

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
            logger.info(f"♻️ Found default model configs in {str(MODEL_CONFIGS_FILE_PATH)!r}")
            return [ModelConfig.model_validate(mc) for mc in config_dict["model_configs"]]
    raise FileNotFoundError(f"Default model configs file not found at {str(MODEL_CONFIGS_FILE_PATH)!r}")


def get_default_providers() -> list[ModelProvider]:
    if MODEL_PROVIDERS_FILE_PATH.exists():
        config_dict = load_config_file(MODEL_PROVIDERS_FILE_PATH)
        if "providers" in config_dict:
            logger.info(f"♻️ Found default model providers in {str(MODEL_PROVIDERS_FILE_PATH)!r}")
            return [ModelProvider.model_validate(p) for p in config_dict["providers"]]
    raise FileNotFoundError(f"Default model providers file not found at {str(MODEL_PROVIDERS_FILE_PATH)!r}")


def get_nvidia_api_key() -> Optional[str]:
    return os.getenv(NVIDIA_API_KEY_ENV_VAR_NAME)


def get_openai_api_key() -> Optional[str]:
    return os.getenv(OPENAI_API_KEY_ENV_VAR_NAME)


def resolve_seed_default_model_settings() -> None:
    if not MODEL_CONFIGS_FILE_PATH.exists():
        logger.debug(f"Creating default model configs file at {str(MODEL_CONFIGS_FILE_PATH)!r}")
        save_config_file(
            MODEL_CONFIGS_FILE_PATH, {"model_configs": [mc.model_dump() for mc in get_builtin_model_configs()]}
        )

    if not MODEL_PROVIDERS_FILE_PATH.exists():
        logger.debug(f"Creating default model providers file at {str(MODEL_PROVIDERS_FILE_PATH)!r}")
        save_config_file(
            MODEL_PROVIDERS_FILE_PATH, {"providers": [p.model_dump() for p in get_builtin_model_providers()]}
        )
