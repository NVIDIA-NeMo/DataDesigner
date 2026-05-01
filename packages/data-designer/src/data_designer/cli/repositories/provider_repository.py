# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from pathlib import Path

from pydantic import BaseModel

from data_designer.cli.repositories.base import ConfigRepository
from data_designer.config.models import ModelProvider
from data_designer.config.utils.constants import MODEL_PROVIDERS_FILE_NAME
from data_designer.config.utils.io_helpers import load_config_file, save_config_file


class ModelProviderRegistry(BaseModel):
    """Registry for model provider configurations."""

    providers: list[ModelProvider]
    default: str | None = None


class ProviderRepository(ConfigRepository[ModelProviderRegistry]):
    """Repository for provider configurations."""

    @property
    def config_file(self) -> Path:
        """Get the provider configuration file path."""
        return self.config_dir / MODEL_PROVIDERS_FILE_NAME

    def load(self) -> ModelProviderRegistry | None:
        """Load provider configuration from file."""
        if not self.exists():
            return None

        try:
            config_dict = load_config_file(self.config_file)
        except Exception:
            return None

        # Emit the deprecation warning *outside* the validation try/except below.
        # ``DeprecationWarning`` is an ``Exception`` subclass, so under
        # ``filterwarnings("error", DeprecationWarning)`` a warn raised inside
        # the catch-all would be silently swallowed and ``load`` would drop the
        # registry. See PR #594 review.
        if config_dict.get("default") is not None:
            warnings.warn(
                f"The 'default:' key in {self.config_file} is deprecated and will "
                "be removed in a future release. Remove it and specify provider= "
                "explicitly on each ModelConfig instead. See issue #589.",
                DeprecationWarning,
                stacklevel=2,
            )

        try:
            return ModelProviderRegistry.model_validate(config_dict)
        except Exception:
            return None

    def save(self, config: ModelProviderRegistry) -> None:
        """Save provider configuration to file."""
        config_dict = config.model_dump(mode="json", exclude_none=True)
        save_config_file(self.config_file, config_dict)
