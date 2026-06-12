# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from data_designer.cli.repositories.base import ConfigRepository
from data_designer.config.models import ModelConfig
from data_designer.config.utils.constants import MODEL_CONFIGS_FILE_NAME
from data_designer.config.utils.io_helpers import load_config_file, save_config_file


class LegacyModelConfigMigrationError(ValueError):
    """Raised when on-disk model configs omit the required ``provider`` field."""


class ModelConfigRegistry(BaseModel):
    """Registry for model configurations."""

    model_configs: list[ModelConfig]


def _aliases_missing_provider(config_dict: Any) -> list[str]:
    if not isinstance(config_dict, dict):
        return []
    entries = config_dict.get("model_configs")
    if not isinstance(entries, list):
        return []
    missing: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("provider") is None:
            alias = entry.get("alias")
            missing.append(str(alias) if alias is not None else "<unknown>")
    return missing


def _format_missing_provider_message(aliases: list[str]) -> str:
    alias_list = ", ".join(f"'{alias}'" for alias in aliases)
    return (
        f"model_configs.yaml contains model alias(es) missing a required 'provider' field: {alias_list}. "
        "Add an explicit provider name to each alias before saving changes "
        "(edit the file directly or run `data-designer config models` after updating it)."
    )


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
        except Exception:
            return None

        missing_providers = _aliases_missing_provider(config_dict)
        if missing_providers:
            raise LegacyModelConfigMigrationError(_format_missing_provider_message(missing_providers))

        try:
            return ModelConfigRegistry.model_validate(config_dict)
        except Exception:
            return None

    def save(self, config: ModelConfigRegistry) -> None:
        """Save model configuration to file."""
        config_dict = config.model_dump(mode="json", exclude_none=True)
        save_config_file(self.config_file, config_dict)
