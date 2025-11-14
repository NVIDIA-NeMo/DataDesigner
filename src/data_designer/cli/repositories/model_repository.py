# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import BaseModel

from data_designer.cli.repositories.base import ConfigRepository
from data_designer.config.models import ModelConfig
from data_designer.config.utils.constants import MODEL_CONFIGS_FILE_NAME
from data_designer.config.utils.io_helpers import load_config_file, save_config_file


class ModelConfigRegistry(BaseModel):
    """Registry for model configurations."""

    model_configs: list[ModelConfig]


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
            return ModelConfigRegistry.model_validate(config_dict)
        except Exception:
            return None

    def save(self, config: ModelConfigRegistry) -> None:
        """Save model configuration to file."""
        config_dict = config.model_dump(mode="json", exclude_none=True)
        save_config_file(self.config_file, config_dict)
