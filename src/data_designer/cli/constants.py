# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

# Built-in providers that can be used without explicit configuration
BUILTIN_PROVIDERS = ["nvidia", "openai"]

DEFAULT_CONFIG_DIR = Path.home() / ".data-designer"
MODEL_CONFIGS_FILE_NAME = "model_configs.yaml"
MODEL_PROVIDERS_FILE_NAME = "model_providers.yaml"

# Predefined provider templates
PREDEFINED_PROVIDERS = {
    "nvidia": {
        "name": "nvidia",
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "provider_type": "openai",
        "api_key": "NVIDIA_API_KEY",
    },
    "openai": {
        "name": "openai",
        "endpoint": "https://api.openai.com/v1",
        "provider_type": "openai",
        "api_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "name": "anthropic",
        "endpoint": "https://api.anthropic.com/v1",
        "provider_type": "openai",
        "api_key": "ANTHROPIC_API_KEY",
    },
}
