# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel

from data_designer.config.utils.constants import (
    NEMOTRON_PERSONAS_DATASET_PREFIX,
    NEMOTRON_PERSONAS_DATASET_SIZES,
)


class PersonaLocale(BaseModel):
    """Metadata for a single persona locale."""

    code: str
    size: str
    dataset_name: str


class PersonaLocaleRegistry(BaseModel):
    """Registry for available persona locales."""

    locales: list[PersonaLocale]
    dataset_prefix: str = NEMOTRON_PERSONAS_DATASET_PREFIX


class PersonaRepository:
    """Repository for persona locale metadata.

    This repository provides access to built-in persona locale metadata.
    Unlike ConfigRepository subclasses, this is read-only reference data
    about what's available in NGC, not user configuration.
    """

    def __init__(self) -> None:
        """Initialize repository with built-in locale metadata."""
        self._registry = self._initialize_registry()

    def _initialize_registry(self) -> PersonaLocaleRegistry:
        """Initialize registry from constants."""
        locales = [
            PersonaLocale(
                code=code,
                size=size,
                dataset_name=f"{NEMOTRON_PERSONAS_DATASET_PREFIX}{code.lower()}",
            )
            for code, size in NEMOTRON_PERSONAS_DATASET_SIZES.items()
        ]
        return PersonaLocaleRegistry(locales=locales)

    def list_all(self) -> list[PersonaLocale]:
        """Get all available persona locales.

        Returns:
            List of all available persona locales
        """
        return list(self._registry.locales)

    def get_by_code(self, code: str) -> PersonaLocale | None:
        """Get a specific locale by code.

        Args:
            code: Locale code (e.g., 'en_US', 'ja_JP')

        Returns:
            PersonaLocale if found, None otherwise
        """
        return next((locale for locale in self._registry.locales if locale.code == code), None)

    def get_dataset_name(self, code: str) -> str | None:
        """Get the NGC dataset name for a locale.

        Args:
            code: Locale code (e.g., 'en_US', 'ja_JP')

        Returns:
            Dataset name if locale exists, None otherwise
        """
        locale = self.get_by_code(code)
        return locale.dataset_name if locale else None

    def get_dataset_prefix(self) -> str:
        """Get the dataset prefix for all persona datasets.

        Returns:
            Dataset prefix string
        """
        return self._registry.dataset_prefix
