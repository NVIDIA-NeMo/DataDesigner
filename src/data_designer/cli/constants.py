# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

# Built-in providers that can be used without explicit configuration
BUILTIN_PROVIDERS = ["nvidia", "openai"]

DEFAULT_CONFIG_DIR = Path.home() / ".data-designer"
MODEL_CONFIGS_FILE_NAME = "model_configs.yaml"
MODEL_PROVIDERS_FILE_NAME = "model_providers.yaml"
