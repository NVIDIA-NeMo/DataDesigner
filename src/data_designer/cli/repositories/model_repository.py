# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.cli.constants import MODEL_CONFIGS_FILE_NAME
from data_designer.cli.repositories.base import ConfigRepository
from data_designer.cli.utils import load_config_file, save_config_file
from data_designer.config.models import ModelConfig


class ModelConfigRegistry:
    """Registry for model configurations."""

    def __init__(self, model_configs: list[ModelConfig]):
        self.model_configs = model_configs

    def model_dump(self, **kwargs) -> dict:
        """Dump to dictionary format."""
        return {"model_configs": [mc.model_dump(**kwargs) for mc in self.model_configs]}


class ModelRepository(ConfigRepository[ModelConfigRegistry]):
    """Repository for model configurations."""

    @property
    def config_file(self) -> Path:
        """Get the model configuration file path."""
        return self.config_dir / MODEL_CONFIGS_FILE_NAME

    def load(self) -> ModelConfigRegistry | None:
        """Load model configuration from file."""
        if not self.exists():
            return None

        try:
            config_dict = load_config_file(self.config_file)
            if "model_configs" not in config_dict:
                return None
            model_configs = [ModelConfig.model_validate(mc) for mc in config_dict["model_configs"]]
            return ModelConfigRegistry(model_configs)
        except Exception:
            return None

    def save(self, config: ModelConfigRegistry) -> None:
        """Save model configuration to file."""
        config_dict = config.model_dump(mode="json", exclude_none=True)
        save_config_file(self.config_file, config_dict)
