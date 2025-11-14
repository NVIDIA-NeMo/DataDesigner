# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ConfigRepository(ABC, Generic[T]):
    """Abstract base for configuration persistence."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir

    @property
    @abstractmethod
    def config_file(self) -> Path:
        """Get the configuration file path."""

    @abstractmethod
    def load(self) -> T | None:
        """Load configuration from file."""

    @abstractmethod
    def save(self, config: T) -> None:
        """Save configuration to file."""

    def exists(self) -> bool:
        """Check if configuration file exists."""
        return self.config_file.exists()

    def delete(self) -> None:
        """Delete configuration file."""
        if self.exists():
            self.config_file.unlink()
