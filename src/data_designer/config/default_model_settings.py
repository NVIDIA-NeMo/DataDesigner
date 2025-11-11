# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Optional

from data_designer.cli.utils import get_model_config_path, get_model_provider_path, load_config_file

from .models import InferenceParameters, ModelConfig, ModelProvider
from .utils.constants import (
    NVIDIA_API_KEY_ENV_VAR_NAME,
    NVIDIA_PROVIDER_NAME,
    OPENAI_API_KEY_ENV_VAR_NAME,
    OPENAI_PROVIDER_NAME,
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


def get_default_nvidia_model_configs() -> list[ModelConfig]:
    if not get_nvidia_api_key():
        logger.warning(
            f"🔑 {NVIDIA_API_KEY_ENV_VAR_NAME!r} environment variable is not set. Please set it to your API key from 'https://build.nvidia.com' if you want to use the default NVIDIA model configs."
        )
        return []
    return [
        ModelConfig(
            alias=f"{NVIDIA_PROVIDER_NAME}-text",
            model="nvidia/nvidia-nemotron-nano-9b-v2",
            provider=NVIDIA_PROVIDER_NAME,
            inference_parameters=get_default_text_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{NVIDIA_PROVIDER_NAME}-reasoning",
            model="openai/gpt-oss-20b",
            provider=NVIDIA_PROVIDER_NAME,
            inference_parameters=get_default_reasoning_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{NVIDIA_PROVIDER_NAME}-vision",
            model="nvidia/nemotron-nano-12b-v2-vl",
            provider=NVIDIA_PROVIDER_NAME,
            inference_parameters=get_default_vision_alias_inference_parameters(),
        ),
    ]


def get_default_openai_model_configs() -> list[ModelConfig]:
    if not get_openai_api_key():
        logger.warning(
            f"🔑 {OPENAI_API_KEY_ENV_VAR_NAME!r} environment variable is not set. Please set it to your API key from 'https://platform.openai.com/api-keys' if you want to use the default OpenAI model configs."
        )
        return []
    return [
        ModelConfig(
            alias=f"{OPENAI_PROVIDER_NAME}-text",
            model="gpt-4.1",
            provider=OPENAI_PROVIDER_NAME,
            inference_parameters=get_default_text_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{OPENAI_PROVIDER_NAME}-reasoning",
            model="gpt-5",
            provider=OPENAI_PROVIDER_NAME,
            inference_parameters=get_default_reasoning_alias_inference_parameters(),
        ),
        ModelConfig(
            alias=f"{OPENAI_PROVIDER_NAME}-vision",
            model="gpt-5",
            provider=OPENAI_PROVIDER_NAME,
            inference_parameters=get_default_vision_alias_inference_parameters(),
        ),
    ]


def get_user_defined_default_model_configs() -> list[ModelConfig]:
    pre_defined_model_config_path = get_model_config_path()
    if pre_defined_model_config_path.exists():
        config_dict = load_config_file(pre_defined_model_config_path)
        if "model_configs" in config_dict:
            logger.info(f"♻️ Found user-defined default model configs in {str(pre_defined_model_config_path)!r}")
            return [ModelConfig.model_validate(mc) for mc in config_dict["model_configs"]]
    return []


def get_default_model_configs() -> list[ModelConfig]:
    user_defined_default_model_configs = get_user_defined_default_model_configs()
    if len(user_defined_default_model_configs) > 0:
        return user_defined_default_model_configs
    return get_default_nvidia_model_configs() + get_default_openai_model_configs()


def get_default_providers() -> list[ModelProvider]:
    user_defined_default_providers = get_user_defined_default_providers()
    if len(user_defined_default_providers) > 0:
        return user_defined_default_providers
    return [
        ModelProvider(
            name=NVIDIA_PROVIDER_NAME,
            endpoint="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY_ENV_VAR_NAME,
        ),
        ModelProvider(
            name=OPENAI_PROVIDER_NAME,
            endpoint="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY_ENV_VAR_NAME,
        ),
    ]


def get_user_defined_default_providers() -> list[ModelProvider]:
    pre_defined_model_provider_path = get_model_provider_path()
    if pre_defined_model_provider_path.exists():
        config_dict = load_config_file(pre_defined_model_provider_path)
        if "providers" in config_dict:
            logger.info(f"♻️ Found user-defined default model providers in {str(pre_defined_model_provider_path)!r}")
            return [ModelProvider.model_validate(p) for p in config_dict["providers"]]
    return []


def get_nvidia_api_key() -> Optional[str]:
    return os.getenv(NVIDIA_API_KEY_ENV_VAR_NAME)


def get_openai_api_key() -> Optional[str]:
    return os.getenv(OPENAI_API_KEY_ENV_VAR_NAME)
